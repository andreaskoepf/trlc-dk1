#!/usr/bin/env python3
"""Benchmark USB camera capture throughput.

Opens all cameras simultaneously and measures max capture FPS
at different resolutions and formats.
"""

import argparse
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import cv2

CAMERA_DEVICES = [0, 2, 4]

MJPG_RESOLUTIONS = [
    (640, 480),
    (1280, 720),
    (1920, 1080),
    (2560, 1440),
    (3840, 2160),
]

YUYV_RESOLUTIONS = [
    (640, 480),
    (1280, 720),
    (1920, 1080),
]


@dataclass
class BenchResult:
    device: int
    codec: str
    requested_res: tuple[int, int]
    actual_res: tuple[int, int]
    fps: float
    frames: int
    duration: float
    bytes_per_frame: int


def benchmark_camera(
    device: int, codec: str, width: int, height: int, duration: float
) -> BenchResult:
    fourcc = cv2.VideoWriter_fourcc(*codec)
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, fourcc)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Warm up: grab a few frames to let the camera settle
    for _ in range(5):
        cap.read()

    frames = 0
    total_bytes = 0
    t0 = time.perf_counter()
    deadline = t0 + duration
    while time.perf_counter() < deadline:
        ret, frame = cap.read()
        if not ret:
            break
        frames += 1
        total_bytes += frame.nbytes

    elapsed = time.perf_counter() - t0
    cap.release()

    return BenchResult(
        device=device,
        codec=codec,
        requested_res=(width, height),
        actual_res=(aw, ah),
        fps=frames / elapsed if elapsed > 0 else 0,
        frames=frames,
        duration=elapsed,
        bytes_per_frame=total_bytes // frames if frames > 0 else 0,
    )


def benchmark_single(device, codec, w, h, duration):
    """Run benchmark for a single camera (used for sequential mode)."""
    r = benchmark_camera(device, codec, w, h, duration)
    mbps = r.fps * r.bytes_per_frame * 8 / 1e6
    print(
        f"  /dev/video{r.device}: {r.actual_res[0]}x{r.actual_res[1]} "
        f"{r.fps:6.1f} fps  {mbps:8.1f} Mbps  ({r.frames} frames in {r.duration:.1f}s)"
    )
    return r


def benchmark_all_concurrent(devices, codec, w, h, duration):
    """Run benchmark for all cameras concurrently."""
    with ThreadPoolExecutor(max_workers=len(devices)) as pool:
        futures = {
            dev: pool.submit(benchmark_camera, dev, codec, w, h, duration)
            for dev in devices
        }
        results = []
        total_mbps = 0
        for dev in devices:
            r = futures[dev].result()
            mbps = r.fps * r.bytes_per_frame * 8 / 1e6
            total_mbps += mbps
            print(
                f"  /dev/video{r.device}: {r.actual_res[0]}x{r.actual_res[1]} "
                f"{r.fps:6.1f} fps  {mbps:8.1f} Mbps  ({r.frames} frames in {r.duration:.1f}s)"
            )
            results.append(r)
        print(f"  {'TOTAL':>54s}: {total_mbps:8.1f} Mbps")
        return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-d", "--duration", type=float, default=5.0, help="Seconds per test (default: 5)"
    )
    parser.add_argument(
        "-c", "--cameras", type=int, nargs="+", default=CAMERA_DEVICES,
        help=f"Video device indices (default: {CAMERA_DEVICES})",
    )
    parser.add_argument(
        "--no-yuyv", action="store_true", help="Skip YUYV (uncompressed) tests"
    )
    args = parser.parse_args()

    print(f"Cameras: {[f'/dev/video{d}' for d in args.cameras]}")
    print(f"Duration per test: {args.duration}s\n")

    for codec, resolutions in [("MJPG", MJPG_RESOLUTIONS), ("YUYV", YUYV_RESOLUTIONS)]:
        if codec == "YUYV" and args.no_yuyv:
            continue

        print(f"{'='*72}")
        print(f" {codec} — Single camera")
        print(f"{'='*72}")
        for w, h in resolutions:
            print(f"\n[{w}x{h}]")
            for dev in args.cameras:
                benchmark_single(dev, codec, w, h, args.duration)

        print(f"\n{'='*72}")
        print(f" {codec} — All cameras concurrent")
        print(f"{'='*72}")
        for w, h in resolutions:
            print(f"\n[{w}x{h}]")
            benchmark_all_concurrent(args.cameras, codec, w, h, args.duration)


if __name__ == "__main__":
    main()
