#!/usr/bin/env python3
"""Camera Profile Studio — small local web app to tune UVC controls live.

Usage:
    python -m tools.camera_profile [--host 127.0.0.1] [--port 8770] \
        [--config port_config.env] [--profiles-dir profiles/]

Then open http://127.0.0.1:8770 in a browser.

Notes:
  * Cameras must be *free* (close any running recorder / Guvcview) before
    starting this tool, because the preview holds them open.
  * V4L2 control changes are sent on a separate fd, so the preview keeps
    streaming while you move sliders. Changes apply on the *next* frame.
  * "Save profile" writes a JSON file under `--profiles-dir`. The recorder
    can later apply it via `python -m tools.camera_profile apply <path>`.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import threading
import time
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qs, unquote, urlsplit

import cv2
import numpy as np

from .profile import (
    CameraProfile,
    apply_profile,
    apply_to_device,
    load_device_paths_from_env,
    load_profile,
    reset_to_defaults,
    save_profile,
    snapshot_to_profile,
)
from .v4l2_controls import (
    ControlInfo,
    enumerate_controls,
    get_control,
    set_control,
    snapshot_values,
)

log = logging.getLogger("camera_profile.app")

HERE = Path(__file__).resolve().parent
INDEX_HTML = HERE / "index.html"

# Rotation per camera key (matches examples/cam_viewer.py and the recorder).
# Only affects the *preview* — actual recordings are rotated at capture time.
CAMERA_ROTATION = {
    "WRIST_LEFT": 180,
    "WRIST_RIGHT": 180,
    "CONTEXT_CAM": 0,
}

# Preview capture geometry. Kept modest so the browser can stream all three
# feeds smoothly even over the local network.
PREVIEW_WIDTH = 640
PREVIEW_HEIGHT = 360
PREVIEW_FPS = 15


def _render_offline_jpeg(cam_key: str, message: str) -> bytes:
    """Pre-render a 'comm-loss' placeholder JPEG used when the camera is
    disconnected, so the browser stops showing a stale frozen frame.
    """
    img = np.zeros((PREVIEW_HEIGHT, PREVIEW_WIDTH, 3), dtype=np.uint8)
    img[:] = (28, 28, 36)
    cv2.rectangle(img, (8, 8), (PREVIEW_WIDTH - 8, PREVIEW_HEIGHT - 8),
                  (60, 60, 90), 2)
    cv2.putText(img, cam_key, (24, 56),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (220, 220, 240), 2, cv2.LINE_AA)
    cv2.putText(img, "communication lost", (24, 96),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (120, 120, 255), 2, cv2.LINE_AA)
    cv2.putText(img, message, (24, 132),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 200), 1, cv2.LINE_AA)
    cv2.putText(img, "reconnecting...", (24, PREVIEW_HEIGHT - 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 200, 160), 1, cv2.LINE_AA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buf.tobytes() if ok else b""


# ---------------------------------------------------------------------------
# Capture thread
# ---------------------------------------------------------------------------


class CameraStream:
    """Holds one VideoCapture, encodes frames to JPEG, fans out to clients."""

    def __init__(self, cam_key: str, device_path: str, rotation: int = 0):
        self.cam_key = cam_key
        self.device_path = device_path
        self.rotation = rotation
        self._lock = threading.Lock()
        self._frame_cond = threading.Condition(self._lock)
        self._latest_jpeg: Optional[bytes] = None
        self._frame_idx = 0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._cap: Optional[cv2.VideoCapture] = None
        self.connect_error: Optional[str] = None
        # Set true while a successful read loop is running. /api/cameras
        # surfaces this so the UI can tell "wired but currently disconnected".
        self.streaming: bool = False
        self.last_frame_ts: float = 0.0
        # Bumped every time `_open` succeeds. The UI watches this to refresh
        # control values after a swap/reconnect even if it missed the offline
        # window (e.g. inactive browser tab throttling the poller).
        self.open_count: int = 0

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._run, name=f"cam-{self.cam_key}", daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _open(self) -> bool:
        # The device-by-path symlink only exists while the camera is plugged
        # in; checking up-front gives a cleaner error than letting OpenCV
        # spam its own stderr warnings on every reopen attempt.
        if not os.path.exists(self.device_path):
            self.connect_error = f"device path missing: {self.device_path}"
            return False
        cap = cv2.VideoCapture(self.device_path, cv2.CAP_V4L2)
        if not cap.isOpened():
            self.connect_error = f"VideoCapture failed to open {self.device_path}"
            return False
        # MJPG fourcc + small size for snappy preview.
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, PREVIEW_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, PREVIEW_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, PREVIEW_FPS)
        self._cap = cap
        self.connect_error = None
        self.open_count += 1
        return True

    def _release(self) -> None:
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None
        self.streaming = False

    def _run(self) -> None:
        # Reconnect loop: a USB unplug/replug, a transient ENODEV, or even an
        # initial-startup race (device not yet enumerated) all funnel through
        # the same retry path so the preview self-heals.
        backoff = 0.5  # seconds, doubled on each consecutive failure
        while not self._stop.is_set():
            if not self._open():
                if backoff <= 0.5:
                    log.warning(
                        "[%s] %s — retrying every %.1fs",
                        self.cam_key, self.connect_error, backoff,
                    )
                if self._stop.wait(backoff):
                    break
                backoff = min(backoff * 2.0, 5.0)
                continue
            log.info("[%s] streaming preview from %s", self.cam_key, self.device_path)
            backoff = 0.5
            self.streaming = True
            empty_streak = 0
            while not self._stop.is_set():
                ok, frame = self._cap.read()
                if not ok or frame is None:
                    empty_streak += 1
                    if empty_streak > 50:
                        log.warning(
                            "[%s] no frames for 50 reads — will try to reopen",
                            self.cam_key,
                        )
                        break
                    time.sleep(0.02)
                    continue
                empty_streak = 0
                if self.rotation == 180:
                    frame = cv2.rotate(frame, cv2.ROTATE_180)
                elif self.rotation == 90:
                    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                elif self.rotation == 270:
                    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if not ok:
                    continue
                with self._frame_cond:
                    self._latest_jpeg = buf.tobytes()
                    self._frame_idx += 1
                    self.last_frame_ts = time.time()
                    self._frame_cond.notify_all()
            self._release()
        self._release()
        log.info("[%s] preview thread exiting", self.cam_key)

    def wait_for_frame(self, last_idx: int, timeout: float = 1.0) -> tuple[int, Optional[bytes]]:
        with self._frame_cond:
            if self._frame_idx == last_idx:
                self._frame_cond.wait(timeout=timeout)
            return self._frame_idx, self._latest_jpeg

    def snapshot_jpeg(self) -> Optional[bytes]:
        with self._lock:
            return self._latest_jpeg


# ---------------------------------------------------------------------------
# Server state
# ---------------------------------------------------------------------------


class AppState:
    def __init__(
        self,
        device_paths: dict[str, str],
        profiles_dir: Path,
        env_path: Path,
    ):
        self.device_paths = device_paths
        self.profiles_dir = profiles_dir
        self.env_path = env_path
        self.streams: dict[str, CameraStream] = {}
        for key, path in device_paths.items():
            if not os.path.exists(path):
                log.warning("Device path missing for %s: %s — skipping", key, path)
                continue
            stream = CameraStream(key, path, rotation=CAMERA_ROTATION.get(key, 0))
            stream.start()
            self.streams[key] = stream

    def shutdown(self) -> None:
        for s in self.streams.values():
            s.stop()


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------


def _control_info_to_json(info: ControlInfo, current: Optional[int]) -> dict:
    return {
        "id": info.id,
        "name": info.name,
        "type": info.type,
        "minimum": info.minimum,
        "maximum": info.maximum,
        "step": info.step,
        "default": info.default,
        "current": current,
        "writable": info.is_writable,
        "menu": info.menu if info.menu else None,
    }


class Handler(BaseHTTPRequestHandler):
    state: AppState  # set on the class before serve_forever

    # Quiet down access log noise.
    def log_message(self, fmt, *args):
        log.debug("%s - %s", self.address_string(), fmt % args)

    # --- routing helpers ----------------------------------------------------

    def _send_json(self, payload, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_error_json(self, status: int, msg: str) -> None:
        self._send_json({"error": msg}, status=status)

    def _read_json_body(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        return json.loads(raw)

    # --- routes -------------------------------------------------------------

    def do_GET(self):  # noqa: N802 — stdlib API
        try:
            url = urlsplit(self.path)
            path = url.path
            if path == "/" or path == "/index.html":
                return self._serve_index()
            if path == "/api/cameras":
                return self._api_cameras()
            if path.startswith("/api/controls/"):
                cam_key = unquote(path[len("/api/controls/"):])
                return self._api_controls(cam_key)
            if path == "/api/profile":
                return self._api_profile_snapshot()
            if path == "/api/profile/list":
                return self._api_profile_list()
            if path.startswith("/stream/"):
                cam_key = unquote(path[len("/stream/"):])
                return self._stream_mjpeg(cam_key)
            return self._send_error_json(404, f"unknown path {path}")
        except Exception as e:
            log.exception("GET %s failed", self.path)
            try:
                self._send_error_json(500, repr(e))
            except Exception:
                pass

    def do_POST(self):  # noqa: N802
        try:
            url = urlsplit(self.path)
            path = url.path
            if path.startswith("/api/controls/"):
                rest = unquote(path[len("/api/controls/"):])
                cam_key, _, ctrl_name = rest.partition("/")
                if not cam_key or not ctrl_name:
                    return self._send_error_json(400, "expected /api/controls/<cam>/<control>")
                return self._api_set_control(cam_key, ctrl_name)
            if path.startswith("/api/reset/"):
                cam_key = unquote(path[len("/api/reset/"):])
                return self._api_reset(cam_key)
            if path == "/api/reset":
                return self._api_reset_all()
            if path == "/api/profile/save":
                return self._api_profile_save()
            if path == "/api/profile/load":
                return self._api_profile_load()
            return self._send_error_json(404, f"unknown POST {path}")
        except Exception as e:
            log.exception("POST %s failed", self.path)
            try:
                self._send_error_json(500, repr(e))
            except Exception:
                pass

    # --- /index.html --------------------------------------------------------

    def _serve_index(self):
        try:
            data = INDEX_HTML.read_bytes()
        except FileNotFoundError:
            return self._send_error_json(500, "index.html missing")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    # --- /api/cameras -------------------------------------------------------

    def _api_cameras(self):
        now = time.time()
        cams = []
        for key, path in self.state.device_paths.items():
            stream = self.state.streams.get(key)
            # "live" = thread is currently inside the read loop AND we got a
            # frame in the last 2 seconds. Disconnect kicks `streaming` False
            # within a second; replug picks it back up via the reconnect loop.
            live = bool(
                stream
                and stream.streaming
                and stream.last_frame_ts > 0
                and (now - stream.last_frame_ts) < 2.0
            )
            cams.append({
                "key": key,
                "device": path,
                "available": stream is not None,
                "live": live,
                "connect_error": stream.connect_error if stream else "device missing",
                "rotation": CAMERA_ROTATION.get(key, 0),
                "open_count": stream.open_count if stream else 0,
            })
        self._send_json({"cameras": cams})

    # --- /api/controls/<cam_key> -------------------------------------------

    def _api_controls(self, cam_key: str):
        path = self.state.device_paths.get(cam_key)
        if not path:
            return self._send_error_json(404, f"unknown camera {cam_key}")
        try:
            controls = enumerate_controls(path)
        except OSError as e:
            return self._send_error_json(500, f"enumerate failed: {e}")
        out = []
        fd = os.open(path, os.O_RDWR | os.O_NONBLOCK)
        try:
            for info in controls:
                current = None
                if info.is_writable:
                    try:
                        current = get_control(fd, info.id)
                    except OSError:
                        current = None
                out.append(_control_info_to_json(info, current))
        finally:
            os.close(fd)
        self._send_json({"camera": cam_key, "device": path, "controls": out})

    # --- POST /api/controls/<cam_key>/<control_name> -----------------------

    def _api_set_control(self, cam_key: str, ctrl_name: str):
        path = self.state.device_paths.get(cam_key)
        if not path:
            return self._send_error_json(404, f"unknown camera {cam_key}")
        body = self._read_json_body()
        if "value" not in body:
            return self._send_error_json(400, "expected JSON body {'value': <int>}")
        target = int(body["value"])
        from .v4l2_controls import control_index
        idx = control_index(path)
        info = idx.get(ctrl_name)
        if info is None:
            return self._send_error_json(404, f"unknown control {ctrl_name}")
        if not info.is_writable:
            return self._send_error_json(400, f"control {ctrl_name} not writable")
        target = max(info.minimum, min(info.maximum, target))
        try:
            set_control(path, info.id, target)
        except OSError as e:
            return self._send_error_json(500, f"set failed: {e}")
        # Re-read to confirm.
        try:
            now = get_control(path, info.id)
        except OSError:
            now = target
        self._send_json({"camera": cam_key, "control": ctrl_name, "value": now})

    # --- POST /api/reset and /api/reset/<cam_key> --------------------------

    def _api_reset(self, cam_key: str):
        path = self.state.device_paths.get(cam_key)
        if not path:
            return self._send_error_json(404, f"unknown camera {cam_key}")
        changes = reset_to_defaults(path)
        self._send_json({"camera": cam_key, "changes": changes})

    def _api_reset_all(self):
        out: dict[str, dict] = {}
        for cam_key, path in self.state.device_paths.items():
            if not os.path.exists(path):
                continue
            try:
                out[cam_key] = reset_to_defaults(path)
            except OSError as e:
                out[cam_key] = {"error": str(e)}
        self._send_json({"reset": out})

    # --- /api/profile ------------------------------------------------------

    def _api_profile_snapshot(self):
        cams: dict[str, dict[str, int]] = {}
        for cam_key, path in self.state.device_paths.items():
            if not os.path.exists(path):
                continue
            try:
                cams[cam_key] = snapshot_values(path)
            except OSError as e:
                cams[cam_key] = {"_error": str(e)}  # type: ignore[dict-item]
        self._send_json({"name": "current", "description": "live snapshot", "cameras": cams})

    def _api_profile_list(self):
        items = []
        if self.state.profiles_dir.is_dir():
            for p in sorted(self.state.profiles_dir.glob("*.json")):
                try:
                    d = json.loads(p.read_text())
                    items.append({
                        "filename": p.name,
                        "path": str(p),
                        "name": d.get("name", p.stem),
                        "description": d.get("description", ""),
                    })
                except Exception as e:
                    items.append({"filename": p.name, "path": str(p), "error": str(e)})
        self._send_json({"profiles_dir": str(self.state.profiles_dir), "profiles": items})

    def _api_profile_save(self):
        body = self._read_json_body()
        filename = body.get("filename")
        if not filename:
            return self._send_error_json(400, "expected JSON body with 'filename'")
        # Sanitize: only allow basenames, .json extension forced.
        filename = os.path.basename(filename)
        if not filename.endswith(".json"):
            filename = filename + ".json"
        target_path = self.state.profiles_dir / filename
        name = body.get("name", target_path.stem)
        description = body.get("description", "")
        # Snapshot current camera values into the profile.
        profile = snapshot_to_profile(name, description, self.state.device_paths)
        save_profile(profile, target_path)
        self._send_json({
            "saved": str(target_path),
            "profile": profile.to_dict(),
        })

    def _api_profile_load(self):
        body = self._read_json_body()
        filename = body.get("filename") or body.get("path")
        if not filename:
            return self._send_error_json(400, "expected JSON body with 'filename' or 'path'")
        # Allow either an absolute path or a basename inside profiles_dir.
        p = Path(filename)
        if not p.is_absolute():
            p = self.state.profiles_dir / os.path.basename(filename)
        if not p.is_file():
            return self._send_error_json(404, f"profile not found: {p}")
        profile = load_profile(p)
        changes = apply_profile(profile, self.state.device_paths)
        self._send_json({"loaded": str(p), "profile": profile.to_dict(), "changes": changes})

    # --- /stream/<cam_key> --------------------------------------------------

    def _stream_mjpeg(self, cam_key: str):
        stream = self.state.streams.get(cam_key)
        if stream is None:
            return self._send_error_json(404, f"no stream for {cam_key}")
        boundary = "frame"
        self.send_response(200)
        self.send_header("Cache-Control", "no-store, private")
        self.send_header("Pragma", "no-cache")
        self.send_header("Connection", "close")
        self.send_header(
            "Content-Type", f"multipart/x-mixed-replace; boundary={boundary}",
        )
        self.end_headers()
        last_idx = -1
        offline_jpeg_cache: Optional[bytes] = None
        offline_sent_for_idx = -1  # send the placeholder once per outage
        try:
            while True:
                last_idx, jpeg = stream.wait_for_frame(last_idx, timeout=2.0)
                # If the capture thread is currently not streaming (camera
                # unplugged, ENODEV, or initial enumeration race) we send a
                # rendered "communication lost" frame instead of leaving the
                # browser stuck on the last live image. Resent every ~2s so
                # if the user reloads, they still see the warning.
                if jpeg is None or not stream.streaming:
                    if offline_jpeg_cache is None:
                        offline_jpeg_cache = _render_offline_jpeg(
                            cam_key, stream.connect_error or "no frames",
                        )
                    if last_idx != offline_sent_for_idx:
                        offline_sent_for_idx = last_idx
                    jpeg_to_send = offline_jpeg_cache
                else:
                    offline_jpeg_cache = None
                    offline_sent_for_idx = -1
                    jpeg_to_send = jpeg
                if not jpeg_to_send:
                    continue
                header = (
                    f"--{boundary}\r\n"
                    f"Content-Type: image/jpeg\r\n"
                    f"Content-Length: {len(jpeg_to_send)}\r\n\r\n"
                ).encode("ascii")
                self.wfile.write(header)
                self.wfile.write(jpeg_to_send)
                self.wfile.write(b"\r\n")
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception:
            log.exception("[%s] mjpeg stream error", cam_key)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _serve(args) -> int:
    device_paths = load_device_paths_from_env(args.config)
    if not device_paths:
        log.error("No cameras found in %s", args.config)
        return 1
    profiles_dir = Path(args.profiles_dir).resolve()
    profiles_dir.mkdir(parents=True, exist_ok=True)
    state = AppState(
        device_paths=device_paths,
        profiles_dir=profiles_dir,
        env_path=Path(args.config).resolve(),
    )
    Handler.state = state
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    log.info("Camera Profile Studio: http://%s:%d/", args.host, args.port)
    log.info("Profiles dir: %s", profiles_dir)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("shutting down")
    finally:
        state.shutdown()
        server.server_close()
    return 0


def _cli_apply(args) -> int:
    device_paths = load_device_paths_from_env(args.config)
    if not device_paths:
        log.error("No cameras found in %s", args.config)
        return 1
    profile = load_profile(args.profile)
    log.info("Applying profile %r from %s", profile.name, args.profile)
    changes = apply_profile(profile, device_paths)
    for cam_key, c in changes.items():
        if not c:
            log.info("  %s: no change", cam_key)
        else:
            for name, (old, new) in c.items():
                log.info("  %s.%s: %s -> %s", cam_key, name, old, new)
    return 0


def _cli_reset(args) -> int:
    device_paths = load_device_paths_from_env(args.config)
    targets = args.cameras or list(device_paths.keys())
    for key in targets:
        path = device_paths.get(key)
        if not path:
            log.error("Unknown camera %r", key)
            continue
        changes = reset_to_defaults(path)
        log.info("%s: %s", key, changes or "already at defaults")
    return 0


def _cli_snapshot(args) -> int:
    device_paths = load_device_paths_from_env(args.config)
    profile = snapshot_to_profile(args.name, args.description, device_paths)
    if args.output:
        save_profile(profile, args.output)
        log.info("wrote %s", args.output)
    else:
        print(json.dumps(profile.to_dict(), indent=2))
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m tools.camera_profile",
        description="Tune and persist UVC camera settings (Innomaker U30CAM-4K-S1).",
    )
    parser.add_argument(
        "--config", default="port_config.env",
        help="Path to port_config.env (default: %(default)s)",
    )
    parser.add_argument(
        "--profiles-dir", default="profiles/cameras",
        help="Directory for JSON profiles (default: %(default)s)",
    )
    parser.add_argument("--log-level", default="INFO")

    sub = parser.add_subparsers(dest="cmd")
    # default: serve the web UI
    sp_serve = sub.add_parser("serve", help="Launch the web UI (default)")
    sp_serve.add_argument("--host", default="127.0.0.1")
    sp_serve.add_argument("--port", type=int, default=8770)

    sp_apply = sub.add_parser("apply", help="Apply a JSON profile to all cameras")
    sp_apply.add_argument("profile", help="Path to profile JSON")

    sp_reset = sub.add_parser("reset", help="Reset cameras to driver defaults")
    sp_reset.add_argument(
        "cameras", nargs="*",
        help="Camera keys (default: all). Choices: WRIST_LEFT WRIST_RIGHT CONTEXT_CAM",
    )

    sp_snap = sub.add_parser("snapshot", help="Print/save current values as a profile")
    sp_snap.add_argument("--output", "-o", help="Write to this path; otherwise stdout")
    sp_snap.add_argument("--name", default="snapshot")
    sp_snap.add_argument("--description", default="")

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    cmd = args.cmd or "serve"
    if cmd == "serve":
        # serve flags may live on the subparser; copy defaults if absent.
        if not hasattr(args, "host"):
            args.host = "127.0.0.1"
        if not hasattr(args, "port"):
            args.port = 8770
        return _serve(args)
    if cmd == "apply":
        return _cli_apply(args)
    if cmd == "reset":
        return _cli_reset(args)
    if cmd == "snapshot":
        return _cli_snapshot(args)
    parser.error(f"unknown command {cmd}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
