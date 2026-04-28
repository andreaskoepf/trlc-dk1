#!/usr/bin/env python3
"""DK1 dataset inspector — small web app to view recorded LeRobot v3 datasets.

Run:
    python tools/dk1_inspector/app.py [--host 127.0.0.1] [--port 8765]

Then open http://127.0.0.1:8765 and enter a dataset path.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import logging
import re
import site
import threading
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit


def _preload_video_reader_libs() -> None:
    """Preload video-reader-rs's bundled FFmpeg libs via ctypes.

    The wheel ships its own libavcodec/libavformat/etc. in
    ``site-packages/video_reader_rs.libs/``, but the bundled libavcodec lacks
    an RPATH/RUNPATH, so ``libsharpyuv`` and similar deps fail to resolve at
    import time. Preloading every bundled .so with RTLD_GLOBAL satisfies the
    transitive symbols regardless of dlopen order.
    """
    candidates: list[Path] = []
    for sp in site.getsitepackages() + [site.getusersitepackages()]:
        p = Path(sp) / "video_reader_rs.libs"
        if p.is_dir():
            candidates.append(p)
            break
    if not candidates:
        return
    libs = sorted(candidates[0].glob("*.so*"))
    # Two passes so libs whose deps were loaded later still bind cleanly.
    for _ in range(2):
        for lib in libs:
            try:
                ctypes.CDLL(str(lib), mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass


_preload_video_reader_libs()

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
import video_reader as vrlib  # noqa: E402

log = logging.getLogger("dk1_inspector")

HERE = Path(__file__).resolve().parent
INDEX_HTML = HERE / "index.html"

FRAME_CACHE_SIZE = 512
META_CACHE_SIZE = 8


# ---------------------------------------------------------------------------
# Dataset access
# ---------------------------------------------------------------------------


class DatasetCache:
    """Caches dataset info.json + episode-meta parquet per dataset path."""

    def __init__(self, max_entries: int = META_CACHE_SIZE):
        self._lock = threading.Lock()
        self._entries: OrderedDict[str, dict] = OrderedDict()
        self._max = max_entries

    def get(self, dataset_path: Path) -> dict:
        key = str(dataset_path)
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None:
                self._entries.move_to_end(key)
                return entry
        entry = self._load(dataset_path)
        with self._lock:
            self._entries[key] = entry
            self._entries.move_to_end(key)
            while len(self._entries) > self._max:
                self._entries.popitem(last=False)
        return entry

    @staticmethod
    def _load(dataset_path: Path) -> dict:
        info_path = dataset_path / "meta" / "info.json"
        if not info_path.is_file():
            raise FileNotFoundError(f"meta/info.json not found under {dataset_path}")
        info = json.loads(info_path.read_text())

        ep_meta_dir = dataset_path / "meta" / "episodes"
        ep_files = sorted(ep_meta_dir.glob("chunk-*/file-*.parquet"))
        if not ep_files:
            raise FileNotFoundError(f"No episode metadata under {ep_meta_dir}")
        ep_table = pq.read_table(ep_files[0]) if len(ep_files) == 1 else pq.concat_tables(
            [pq.read_table(p) for p in ep_files]
        )
        episodes = ep_table.to_pylist()
        episodes.sort(key=lambda r: r["episode_index"])

        cameras = [k for k, v in info["features"].items() if v.get("dtype") == "video"]

        return {
            "info": info,
            "episodes": episodes,
            "cameras": cameras,
        }


DATASET_CACHE = DatasetCache()


def resolve_dataset_path(raw: str) -> Path:
    p = Path(raw).expanduser()
    if not p.is_absolute():
        raise ValueError("dataset path must be absolute")
    p = p.resolve()
    if not p.is_dir():
        raise ValueError(f"dataset path is not a directory: {p}")
    return p


def episode_meta(ds: dict, ep_index: int) -> dict:
    for row in ds["episodes"]:
        if int(row["episode_index"]) == ep_index:
            return row
    raise KeyError(f"episode {ep_index} not in dataset")


def episode_data_path(dataset_path: Path, ds: dict, ep_index: int) -> Path:
    row = episode_meta(ds, ep_index)
    chunk_idx = int(row["data/chunk_index"])
    file_idx = int(row["data/file_index"])
    template = ds["info"]["data_path"]
    rel = template.format(chunk_index=chunk_idx, file_index=file_idx)
    return dataset_path / rel


def episode_video_path(dataset_path: Path, ds: dict, ep_index: int, cam: str) -> Path:
    row = episode_meta(ds, ep_index)
    chunk_idx = int(row[f"videos/{cam}/chunk_index"])
    file_idx = int(row[f"videos/{cam}/file_index"])
    template = ds["info"]["video_path"]
    rel = template.format(video_key=cam, chunk_index=chunk_idx, file_index=file_idx)
    return dataset_path / rel


# ---------------------------------------------------------------------------
# Frame decoding (video-reader-rs) + LRU caches
# ---------------------------------------------------------------------------


class _LRU:
    def __init__(self, max_entries: int):
        self._lock = threading.Lock()
        self._entries: OrderedDict = OrderedDict()
        self._max = max_entries

    def get(self, key):
        with self._lock:
            v = self._entries.get(key)
            if v is not None:
                self._entries.move_to_end(key)
            return v

    def put(self, key, value) -> None:
        with self._lock:
            self._entries[key] = value
            self._entries.move_to_end(key)
            while len(self._entries) > self._max:
                self._entries.popitem(last=False)


FRAME_CACHE = _LRU(FRAME_CACHE_SIZE)


class ReaderPool:
    """Caches PyVideoReader instances keyed by absolute mp4 path.

    Each reader has its own lock since the underlying decoder isn't safe to
    poke at from multiple threads concurrently.
    """

    def __init__(self, max_entries: int = 8):
        self._lock = threading.Lock()
        self._entries: OrderedDict[str, tuple[object, threading.Lock]] = OrderedDict()
        self._max = max_entries

    def get(self, video_path: Path):
        key = str(video_path)
        with self._lock:
            existing = self._entries.get(key)
            if existing is not None:
                self._entries.move_to_end(key)
                return existing
            reader = vrlib.PyVideoReader(key)
            entry = (reader, threading.Lock())
            self._entries[key] = entry
            self._entries.move_to_end(key)
            while len(self._entries) > self._max:
                self._entries.popitem(last=False)
            return entry


READER_POOL = ReaderPool()


def decode_frame_jpeg(
    video_path: Path,
    target_video_frame: int,
    jpeg_quality: int = 85,
    max_long_side: int | None = 960,
) -> bytes:
    """Decode the (target_video_frame)-th frame of an MP4 and JPEG-encode it.

    target_video_frame is the absolute frame index inside the MP4 (so caller
    must add the per-camera priming/from_timestamp offset).
    """
    reader, lock = READER_POOL.get(video_path)
    with lock:
        rgb = reader[int(target_video_frame)]  # (H, W, 3) uint8 RGB

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    if max_long_side is not None:
        h, w = bgr.shape[:2]
        long_side = max(h, w)
        if long_side > max_long_side:
            scale = max_long_side / long_side
            bgr = cv2.resize(
                bgr, (int(round(w * scale)), int(round(h * scale))),
                interpolation=cv2.INTER_AREA,
            )

    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return bytes(buf)


# ---------------------------------------------------------------------------
# HTTP server
# ---------------------------------------------------------------------------


ROUTE_FRAME = re.compile(
    r"^/api/frame/(?P<cam>[^/]+)/(?P<ep>\d+)/(?P<frame>\d+)/?$"
)
ROUTE_SERIES = re.compile(r"^/api/episode/(?P<ep>\d+)/series/?$")


class Handler(BaseHTTPRequestHandler):
    server_version = "DK1Inspector/0.1"

    # Silence default noisy access log.
    def log_message(self, fmt: str, *args) -> None:  # noqa: A003
        log.debug("%s - %s", self.address_string(), fmt % args)

    def do_GET(self) -> None:  # noqa: N802
        try:
            self._dispatch()
        except FileNotFoundError as e:
            self._json_error(404, str(e))
        except (KeyError, ValueError) as e:
            self._json_error(400, str(e))
        except Exception as e:
            log.exception("internal error handling %s", self.path)
            self._json_error(500, f"{type(e).__name__}: {e}")

    # ---- routing -----------------------------------------------------------
    def _dispatch(self) -> None:
        parts = urlsplit(self.path)
        path = parts.path
        query = parse_qs(parts.query)

        if path == "/" or path == "/index.html":
            self._serve_static(INDEX_HTML, "text/html; charset=utf-8")
            return

        if path == "/api/info":
            self._handle_info(query)
            return

        m = ROUTE_SERIES.match(path)
        if m:
            self._handle_series(query, int(m.group("ep")))
            return

        m = ROUTE_FRAME.match(path)
        if m:
            self._handle_frame(
                query,
                cam=unquote(m.group("cam")),
                ep=int(m.group("ep")),
                frame=int(m.group("frame")),
            )
            return

        self._json_error(404, f"unknown route {path}")

    # ---- handlers ----------------------------------------------------------
    def _handle_info(self, query: dict) -> None:
        dataset_path = self._get_dataset_path(query)
        ds = DATASET_CACHE.get(dataset_path)
        info = ds["info"]
        episodes_meta = ds["episodes"]

        eps = []
        for row in episodes_meta:
            tasks = row.get("tasks") or []
            if isinstance(tasks, np.ndarray):
                tasks = tasks.tolist()
            eps.append({
                "index": int(row["episode_index"]),
                "length": int(row["length"]),
                "task": tasks[0] if tasks else "",
            })

        action_names = info["features"]["action"].get("names") or []
        state_names = info["features"]["observation.state"].get("names") or []

        payload = {
            "path": str(dataset_path),
            "fps": int(info["fps"]),
            "robot_type": info.get("robot_type", ""),
            "total_episodes": int(info.get("total_episodes", len(eps))),
            "total_frames": int(info.get("total_frames", 0)),
            "cameras": ds["cameras"],
            "action_names": action_names,
            "state_names": state_names,
            "episodes": eps,
        }
        self._json_ok(payload)

    def _handle_series(self, query: dict, ep: int) -> None:
        dataset_path = self._get_dataset_path(query)
        ds = DATASET_CACHE.get(dataset_path)
        row = episode_meta(ds, ep)
        parquet_path = episode_data_path(dataset_path, ds, ep)
        if not parquet_path.is_file():
            raise FileNotFoundError(f"episode parquet missing: {parquet_path}")

        table = pq.read_table(
            parquet_path, columns=["frame_index", "timestamp", "action", "observation.state"]
        )
        frame_index = table["frame_index"].to_numpy(zero_copy_only=False)
        timestamp = table["timestamp"].to_numpy(zero_copy_only=False)

        action = _stack_list_column(table["action"])
        state = _stack_list_column(table["observation.state"])

        info = ds["info"]
        action_names = info["features"]["action"].get("names") or [
            f"action[{i}]" for i in range(action.shape[1])
        ]
        state_names = info["features"]["observation.state"].get("names") or [
            f"observation.state[{i}]" for i in range(state.shape[1])
        ]

        videos = {}
        for cam in ds["cameras"]:
            from_key = f"videos/{cam}/from_timestamp"
            to_key = f"videos/{cam}/to_timestamp"
            video_path = episode_video_path(dataset_path, ds, ep, cam)
            videos[cam] = {
                "from_timestamp": float(row[from_key]) if from_key in row else 0.0,
                "to_timestamp": float(row[to_key]) if to_key in row else 0.0,
                "path": str(video_path),
                "filename": video_path.name,
                "rel_path": str(video_path.relative_to(dataset_path)) if video_path.is_relative_to(dataset_path) else str(video_path),
            }

        # Send per-channel arrays (transposed) for compact JSON.
        action_per_ch = action.T.tolist()
        state_per_ch = state.T.tolist()

        payload = {
            "episode_index": ep,
            "length": int(len(frame_index)),
            "fps": int(info["fps"]),
            "frame_index": [int(x) for x in frame_index.tolist()],
            "timestamp": [float(x) for x in timestamp.tolist()],
            "action_names": action_names,
            "action": action_per_ch,
            "state_names": state_names,
            "state": state_per_ch,
            "cameras": ds["cameras"],
            "videos": videos,
            # Back-compat: keep flat from_timestamp map for any existing client.
            "video_offsets": {cam: videos[cam]["from_timestamp"] for cam in ds["cameras"]},
            "video_path_template": info["video_path"],
        }
        self._json_ok(payload)

    def _handle_frame(self, query: dict, cam: str, ep: int, frame: int) -> None:
        dataset_path = self._get_dataset_path(query)
        ds = DATASET_CACHE.get(dataset_path)
        if cam not in ds["cameras"]:
            raise ValueError(f"unknown camera {cam!r}")
        row = episode_meta(ds, ep)
        if not (0 <= frame < int(row["length"])):
            raise ValueError(
                f"frame {frame} out of range for episode {ep} (length={row['length']})"
            )

        from_ts = float(row.get(f"videos/{cam}/from_timestamp", 0.0))
        fps = int(ds["info"]["fps"])
        priming = int(round(from_ts * fps))
        target_video_frame = priming + frame

        cache_key = (str(dataset_path), cam, ep, frame)
        cached = FRAME_CACHE.get(cache_key)
        if cached is None:
            video_path = episode_video_path(dataset_path, ds, ep, cam)
            cached = decode_frame_jpeg(video_path, target_video_frame)
            FRAME_CACHE.put(cache_key, cached)

        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(cached)))
        self.send_header("Cache-Control", "public, max-age=3600")
        self.end_headers()
        self.wfile.write(cached)

    # ---- helpers -----------------------------------------------------------
    def _get_dataset_path(self, query: dict) -> Path:
        raw = query.get("path", [None])[0]
        if not raw:
            raise ValueError("missing required query parameter 'path'")
        return resolve_dataset_path(raw)

    def _serve_static(self, path: Path, content_type: str) -> None:
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _json_ok(self, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json_error(self, code: int, message: str) -> None:
        body = json.dumps({"error": message}).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _stack_list_column(col) -> np.ndarray:
    """Stack a pyarrow list/list-of-float column into a 2D numpy array (N, K)."""
    arr = col.to_numpy(zero_copy_only=False)
    if len(arr) == 0:
        return np.zeros((0, 0), dtype=np.float32)
    rows = [np.asarray(r, dtype=np.float32) for r in arr]
    return np.stack(rows, axis=0)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="DK1 dataset inspector web UI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--path", default=None,
                        help="Optional default dataset path to preload in the UI")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.daemon_threads = True
    Handler.default_path = args.path  # informational only; not consumed yet
    log.info("DK1 inspector listening on http://%s:%d", args.host, args.port)
    if args.path:
        log.info("Suggest opening with ?path=%s", args.path)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("shutting down")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
