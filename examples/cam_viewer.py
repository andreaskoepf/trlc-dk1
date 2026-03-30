#!/usr/bin/env python3
"""Simple multi-camera viewer for wrist and overview cameras."""

import argparse
import os
import sys
import cv2
import numpy as np


DEFAULT_CAM_CONFIG = os.path.join(os.path.dirname(__file__), "..", "port_config.env")

CAMERAS = {
    "left_wrist": {"env": "WRIST_LEFT", "rotation": 180},
    "right_wrist": {"env": "WRIST_RIGHT", "rotation": 180},
    "overview": {"env": "CONTEXT_CAM", "rotation": None},
}


def load_cam_config(config_path: str) -> dict[str, str]:
    """Parse port_config.env and return env var -> path mapping."""
    paths = {}
    with open(config_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or not line:
                continue
            if line.startswith("export "):
                line = line[len("export "):]
            key, _, val = line.partition("=")
            paths[key.strip()] = val.strip().strip('"')
    return paths


def open_camera(path: str, width: int = 1280, height: int = 720, fps: int = 30):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    return cap


def main():
    parser = argparse.ArgumentParser(description="Multi-camera viewer")
    parser.add_argument(
        "-c", "--config", default=DEFAULT_CAM_CONFIG,
        help="Path to port_config.env (default: %(default)s)",
    )
    parser.add_argument(
        "--width", type=int, default=1280, help="Capture width (default: 1280)",
    )
    parser.add_argument(
        "--height", type=int, default=720, help="Capture height (default: 720)",
    )
    parser.add_argument(
        "--fps", type=int, default=30, help="Capture FPS (default: 30)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"Config not found: {args.config}")
        print("Run: python examples/identify_ports.py -m bimanual")
        sys.exit(1)

    env = load_cam_config(args.config)

    caps = {}
    for name, info in CAMERAS.items():
        path = env.get(info["env"])
        if not path:
            print(f"Warning: {info['env']} not set, skipping {name}")
            continue
        if not os.path.exists(path):
            print(f"Warning: {path} does not exist, skipping {name}")
            continue
        cap = open_camera(path, args.width, args.height, args.fps)
        if cap is None:
            print(f"Warning: could not open {name} at {path}")
            continue
        caps[name] = (cap, info["rotation"])
        print(f"Opened {name}: {path}")

    if not caps:
        print("No cameras available.")
        sys.exit(1)

    print(f"\nShowing {len(caps)} camera(s). Press 'q' to quit.")

    window_name = "Camera Viewer"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    while True:
        frames = []
        for name, (cap, rotation) in caps.items():
            ret, frame = cap.read()
            if not ret:
                frame = np.zeros((args.height, args.width, 3), dtype=np.uint8)
                cv2.putText(
                    frame, f"{name}: no frame", (30, args.height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2,
                )
            else:
                if rotation == 180:
                    frame = cv2.rotate(frame, cv2.ROTATE_180)
            # Add label
            cv2.putText(
                frame, name, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2,
            )
            frames.append(frame)

        # Layout: wrist cams side-by-side on top, overview below (or just tile)
        if len(frames) == 3:
            top = np.hstack(frames[:2])
            # Center the bottom frame
            bot = frames[2]
            pad_w = top.shape[1] - bot.shape[1]
            if pad_w > 0:
                left_pad = pad_w // 2
                right_pad = pad_w - left_pad
                bot = np.pad(
                    bot, ((0, 0), (left_pad, right_pad), (0, 0)),
                    mode="constant",
                )
            mosaic = np.vstack([top, bot])
        elif len(frames) == 2:
            mosaic = np.hstack(frames)
        else:
            mosaic = frames[0]

        cv2.imshow(window_name, mosaic)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    for cap, _ in caps.values():
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
