#!/usr/bin/env python3
"""
Measure USB-serial write latency and round-trip time for the DM motor protocol.

Tests how long serial_.write() blocks on this system, and measures the
time from sending a command to receiving a response. This helps diagnose
whether the C++ RT loop's 84 Hz cap is caused by USB-serial latency.

Usage:
    python examples/measure_serial_latency.py --port /dev/ttyACM0
    python examples/measure_serial_latency.py --port /dev/ttyACM0 --burst 7
"""
import argparse
import os
import time

import numpy as np
import serial


# DM motor protocol: 30-byte frame
def build_refresh_frame(slave_id: int) -> bytes:
    """Build a 30-byte DM protocol frame that queries motor status."""
    frame = bytearray(30)
    frame[0] = 0xAA
    frame[1] = 0x11
    frame[2] = 0x00
    can_id = 0x7FF
    frame[3] = can_id & 0xFF
    frame[4] = (can_id >> 8) & 0xFF
    frame[5] = (can_id >> 16) & 0xFF
    frame[6] = (can_id >> 24) & 0xFF
    frame[7] = slave_id & 0xFF
    frame[8] = (slave_id >> 8) & 0xFF
    frame[9] = 0xCC
    frame[28] = 0x00
    frame[29] = 0xFD
    return bytes(frame)


def measure_write_latency(ser: serial.Serial, frame: bytes, n: int) -> np.ndarray:
    """Measure how long each serial.write() call takes."""
    latencies_us = np.zeros(n, dtype=np.float64)
    for i in range(n):
        t0 = time.monotonic()
        ser.write(frame)
        t1 = time.monotonic()
        latencies_us[i] = (t1 - t0) * 1e6
        time.sleep(0.002)
    return latencies_us


def measure_burst_write_latency(ser: serial.Serial, frames: list[bytes], n_cycles: int) -> tuple[np.ndarray, np.ndarray]:
    """Measure time for burst of N writes (like the RT loop does per cycle)."""
    n_frames = len(frames)
    per_write = np.zeros((n_cycles, n_frames), dtype=np.float64)
    per_burst = np.zeros(n_cycles, dtype=np.float64)

    for c in range(n_cycles):
        ser.reset_input_buffer()

        t_burst_start = time.monotonic()
        for i, frame in enumerate(frames):
            t0 = time.monotonic()
            ser.write(frame)
            t1 = time.monotonic()
            per_write[c, i] = (t1 - t0) * 1e6
        t_burst_end = time.monotonic()
        per_burst[c] = (t_burst_end - t_burst_start) * 1e6

        time.sleep(0.004)

    return per_write, per_burst


def measure_full_cycle_nonblocking(ser: serial.Serial, frames: list[bytes],
                                    n_cycles: int, period_s: float) -> np.ndarray:
    """Simulate the C++ RT loop pattern with non-blocking reads.

    Each cycle:
      1. Non-blocking read (drain whatever arrived from previous cycle)
      2. Send all frames
      3. Sleep until next period

    Returns per-cycle time in us (excluding sleep).
    """
    cycle_us = np.zeros(n_cycles, dtype=np.float64)

    # Use timeout=0 for non-blocking reads (like C++ VMIN=0 VTIME=0)
    old_timeout = ser.timeout
    ser.timeout = 0

    next_wakeup = time.monotonic()

    for c in range(n_cycles):
        next_wakeup += period_s
        t0 = time.monotonic()

        # 1. Non-blocking read (drain previous responses)
        try:
            ser.read(4096)
        except Exception:
            pass

        # 2. Send all frames
        for frame in frames:
            ser.write(frame)

        t1 = time.monotonic()
        cycle_us[c] = (t1 - t0) * 1e6

        # 3. Sleep until next period
        sleep_for = next_wakeup - time.monotonic()
        if sleep_for > 0:
            time.sleep(sleep_for)
        else:
            next_wakeup = time.monotonic()

    ser.timeout = old_timeout
    return cycle_us


def measure_write_tcdrain(ser: serial.Serial, frame: bytes, n: int) -> np.ndarray:
    """Measure write + tcdrain (flush to hardware) latency.
    This shows actual USB transfer time, not just kernel buffer copy."""
    import ctypes
    import ctypes.util
    libc = ctypes.CDLL(ctypes.util.find_library("c"))

    latencies_us = np.zeros(n, dtype=np.float64)
    for i in range(n):
        t0 = time.monotonic()
        ser.write(frame)
        ser.flush()  # calls tcdrain
        t1 = time.monotonic()
        latencies_us[i] = (t1 - t0) * 1e6
        time.sleep(0.002)
    return latencies_us


def measure_burst_tcdrain(ser: serial.Serial, frames: list[bytes], n_cycles: int) -> np.ndarray:
    """Measure burst write + tcdrain total time.
    Shows how long it takes for all 7 frames to actually leave the USB port."""
    burst_us = np.zeros(n_cycles, dtype=np.float64)

    for c in range(n_cycles):
        ser.reset_input_buffer()
        t0 = time.monotonic()
        for frame in frames:
            ser.write(frame)
        ser.flush()  # tcdrain — wait until all bytes sent to hardware
        t1 = time.monotonic()
        burst_us[c] = (t1 - t0) * 1e6
        time.sleep(0.004)

    return burst_us


def print_stats(label: str, values_us: np.ndarray):
    valid = values_us[~np.isnan(values_us)]
    if len(valid) == 0:
        print(f"  {label}: no valid samples")
        return
    print(f"  {label} ({len(valid)} samples):")
    print(f"    min:  {np.min(valid):>10.1f} us")
    print(f"    max:  {np.max(valid):>10.1f} us")
    print(f"    mean: {np.mean(valid):>10.1f} us")
    print(f"    std:  {np.std(valid):>10.1f} us")
    for p in [50, 90, 95, 99]:
        print(f"    p{p:<3d}: {np.percentile(valid, p):>10.1f} us")


def main():
    parser = argparse.ArgumentParser(description="Measure USB-serial latency")
    parser.add_argument("--port", default="/dev/ttyACM0", help="Serial port")
    parser.add_argument("--baud", type=int, default=921600, help="Baud rate")
    parser.add_argument("--n", type=int, default=500, help="Number of samples")
    parser.add_argument("--burst", type=int, default=7,
                        help="Number of frames per burst (simulates RT loop)")
    parser.add_argument("--hz", type=float, default=250.0,
                        help="Target Hz for full cycle simulation")
    args = parser.parse_args()

    print(f"Serial latency test")
    print(f"  Port: {args.port}")
    print(f"  Baud: {args.baud}")
    print(f"  Samples: {args.n}")
    print()

    # Check USB speed
    tty_name = os.path.basename(os.path.realpath(args.port))
    if tty_name.startswith("ttyACM"):
        try:
            dev = os.path.realpath(f"/sys/class/tty/{tty_name}/device/..")
            speed = open(f"{dev}/speed").read().strip()
            product = open(f"{dev}/product").read().strip()
            print(f"  USB device: {product}")
            print(f"  USB speed:  {speed} Mbps")
        except Exception:
            pass
    print()

    ser = serial.Serial(args.port, args.baud, timeout=0)
    time.sleep(0.5)
    ser.reset_input_buffer()

    # Test 1: Single write latency (kernel buffer copy only)
    print("=" * 60)
    print("Test 1: Single write() latency (30-byte frame, kernel copy)")
    print("=" * 60)
    frame = build_refresh_frame(0x01)
    lat = measure_write_latency(ser, frame, args.n)
    print_stats("write()", lat)

    # Test 2: Single write + tcdrain (actual USB transfer time)
    print()
    print("=" * 60)
    print("Test 2: Single write() + flush/tcdrain (actual USB transfer)")
    print("=" * 60)
    lat_drain = measure_write_tcdrain(ser, frame, args.n)
    print_stats("write+drain", lat_drain)

    # Test 3: Burst write latency
    print()
    print("=" * 60)
    print(f"Test 3: Burst write latency ({args.burst} frames, kernel copy)")
    print("=" * 60)
    frames = [build_refresh_frame(i + 1) for i in range(args.burst)]
    per_write, per_burst = measure_burst_write_latency(ser, frames, args.n)
    for i in range(args.burst):
        print_stats(f"  frame[{i}]", per_write[:, i])
    print()
    print_stats(f"  total burst ({args.burst} writes)", per_burst)
    max_hz_write = 1e6 / np.mean(per_burst)
    print(f"\n  => Max Hz (write-copy only): {max_hz_write:.0f} Hz")

    # Test 4: Burst write + tcdrain (actual USB transfer for all 7)
    print()
    print("=" * 60)
    print(f"Test 4: Burst write + flush/tcdrain ({args.burst} frames)")
    print("=" * 60)
    burst_drain = measure_burst_tcdrain(ser, frames, args.n)
    print_stats(f"burst+drain ({args.burst} frames)", burst_drain)
    max_hz_drain = 1e6 / np.mean(burst_drain)
    print(f"\n  => Max Hz (USB-transfer limited): {max_hz_drain:.0f} Hz")

    # Test 5: Full cycle simulation (non-blocking read + burst write)
    print()
    print("=" * 60)
    print(f"Test 5: Full RT loop simulation @ {args.hz} Hz (non-blocking read + {args.burst} writes)")
    print("=" * 60)
    period_s = 1.0 / args.hz
    cycle = measure_full_cycle_nonblocking(ser, frames, args.n, period_s)
    print_stats("cycle work time", cycle)
    work_pct = np.mean(cycle) / (period_s * 1e6) * 100
    print(f"\n  Work takes {np.mean(cycle):.0f}us of {period_s*1e6:.0f}us period ({work_pct:.1f}% utilization)")
    print(f"  Headroom: {period_s*1e6 - np.mean(cycle):.0f}us")
    if np.max(cycle) > period_s * 1e6:
        overruns = np.sum(cycle > period_s * 1e6)
        print(f"  WARNING: {overruns} cycles exceeded period ({overruns/args.n*100:.1f}%)")

    ser.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
