#!/usr/bin/env python3
"""
Measure scheduling jitter at a target frequency — no hardware required.

This mimics what the C++ RT loop does: clock_nanosleep(CLOCK_MONOTONIC,
TIMER_ABSTIME, ...) in a tight loop, and records how late each wakeup is.

Runs in both Python (time.sleep) and optionally via a tiny C helper that
uses clock_nanosleep directly, to match the C++ control loop behaviour.

Usage:
    python examples/measure_jitter.py [--hz 250] [--duration 10] [--method clock_nanosleep]
"""
import argparse
import ctypes
import ctypes.util
import os
import struct
import sys
import tempfile
import time

import numpy as np


def measure_python(hz: float, duration: float) -> np.ndarray:
    """Measure jitter using Python's time.sleep (relative sleep)."""
    period = 1.0 / hz
    n = int(hz * duration) + 1
    latencies_ns = np.zeros(n, dtype=np.int64)

    next_wakeup = time.monotonic()
    for i in range(n):
        next_wakeup += period
        now = time.monotonic()
        sleep_for = next_wakeup - now
        if sleep_for > 0:
            time.sleep(sleep_for)
        else:
            # We're behind — reset to avoid cascade
            next_wakeup = time.monotonic() + period
        actual = time.monotonic()
        latencies_ns[i] = int((actual - (next_wakeup - period + period)) * 1e9)

    return latencies_ns


def measure_clock_nanosleep(hz: float, duration: float) -> np.ndarray:
    """Measure jitter using clock_nanosleep TIMER_ABSTIME via ctypes."""
    CLOCK_MONOTONIC = 1
    TIMER_ABSTIME = 1

    libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)

    class timespec(ctypes.Structure):
        _fields_ = [("tv_sec", ctypes.c_long), ("tv_nsec", ctypes.c_long)]

    clock_gettime = libc.clock_gettime
    clock_nanosleep_fn = libc.clock_nanosleep

    period_ns = int(1e9 / hz)
    n = int(hz * duration) + 1
    latencies_us = np.zeros(n, dtype=np.float64)

    # Get current time
    ts = timespec()
    clock_gettime(CLOCK_MONOTONIC, ctypes.byref(ts))
    next_ns = ts.tv_sec * 1_000_000_000 + ts.tv_nsec

    for i in range(n):
        next_ns += period_ns
        target = timespec(next_ns // 1_000_000_000, next_ns % 1_000_000_000)
        clock_nanosleep_fn(CLOCK_MONOTONIC, TIMER_ABSTIME, ctypes.byref(target), None)

        # Measure actual wakeup time
        clock_gettime(CLOCK_MONOTONIC, ctypes.byref(ts))
        actual_ns = ts.tv_sec * 1_000_000_000 + ts.tv_nsec
        latencies_us[i] = (actual_ns - next_ns) / 1000.0  # overshoot in us

    return latencies_us


def print_stats(latencies_us: np.ndarray, hz: float, method: str):
    target_us = 1e6 / hz
    n = len(latencies_us)

    print(f"\n{'=' * 60}")
    print(f"Jitter test: {method}  @  {hz:.0f} Hz  ({n} samples)")
    print(f"{'=' * 60}")
    print(f"  Target period:  {target_us:.0f} us")
    print()
    print(f"  Wakeup overshoot (how late the thread woke up):")
    print(f"    min:  {np.min(latencies_us):>10.1f} us")
    print(f"    max:  {np.max(latencies_us):>10.1f} us")
    print(f"    mean: {np.mean(latencies_us):>10.1f} us")
    print(f"    std:  {np.std(latencies_us):>10.1f} us")
    print()

    percentiles = [50, 90, 95, 99, 99.9]
    print(f"  Percentiles:")
    for p in percentiles:
        val = np.percentile(latencies_us, p)
        print(f"    p{p:<5}: {val:>10.1f} us")

    # Histogram of overshoot
    bins = [0, 10, 50, 100, 500, 1000, 2000, 5000, 10000, float('inf')]
    labels = ["0-10us", "10-50us", "50-100us", "100-500us", "500us-1ms",
              "1-2ms", "2-5ms", "5-10ms", ">10ms"]
    counts, _ = np.histogram(latencies_us, bins=bins)
    print(f"\n  Overshoot histogram:")
    for label, count in zip(labels, counts):
        pct = 100.0 * count / n
        bar = "#" * int(pct / 2)
        print(f"    {label:>10s}: {count:>8d} ({pct:5.1f}%)  {bar}")

    # Count deadline misses (overshoot > 50% of period)
    miss_threshold = target_us * 0.5
    misses = np.sum(latencies_us > miss_threshold)
    print(f"\n  Deadline misses (>{miss_threshold:.0f}us late): {misses} ({100.0 * misses / n:.2f}%)")

    # Effective Hz
    # If we were late by overshoot on average, effective period is target + mean_overshoot
    eff_period = target_us + np.mean(latencies_us)
    eff_hz = 1e6 / eff_period
    print(f"  Effective Hz:    {eff_hz:.1f} (target {hz:.0f})")


def main():
    parser = argparse.ArgumentParser(description="Measure scheduling jitter")
    parser.add_argument("--hz", type=float, default=250.0, help="Target frequency")
    parser.add_argument("--duration", type=float, default=10.0, help="Test duration (seconds)")
    parser.add_argument("--method", default="clock_nanosleep",
                        choices=["clock_nanosleep", "python", "both"],
                        help="Sleep method to test")
    args = parser.parse_args()

    print(f"Scheduling jitter test")
    print(f"  Target: {args.hz} Hz  Duration: {args.duration}s")
    print(f"  Kernel: ", end="")
    os.system("uname -r")
    print(f"  Scheduler: SCHED_OTHER (no RT privileges)")
    print(f"  ulimit -r: ", end="")
    os.system("ulimit -r")

    if args.method in ("clock_nanosleep", "both"):
        print(f"\nRunning clock_nanosleep test ({args.duration}s)...")
        lat = measure_clock_nanosleep(args.hz, args.duration)
        print_stats(lat, args.hz, "clock_nanosleep (TIMER_ABSTIME)")

    if args.method in ("python", "both"):
        print(f"\nRunning Python time.sleep test ({args.duration}s)...")
        lat = measure_python(args.hz, args.duration)
        lat_us = lat / 1000.0  # convert ns to us
        print_stats(lat_us, args.hz, "Python time.sleep (relative)")


if __name__ == "__main__":
    main()
