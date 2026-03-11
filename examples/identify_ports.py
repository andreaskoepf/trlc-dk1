#!/usr/bin/env python3
"""Identify USB serial ports and cameras by physical disconnection.

Walks you through unplugging each device one at a time to map
logical names to stable /dev/serial/by-path and /dev/v4l/by-path entries.

Outputs a port_config.env file for use with the recording scripts.
"""

import argparse
import glob
import os
import time

SINGLE_ARM_SERIAL = [
    "left_leader",
    "left_follower",
]

SINGLE_ARM_CAMERAS = [
    "wrist_left",
    "context_cam",
]

BIMANUAL_SERIAL = [
    "left_leader",
    "right_leader",
    "left_follower",
    "right_follower",
]

BIMANUAL_CAMERAS = [
    "wrist_left",
    "wrist_right",
    "context_cam",
]


def get_serial_paths() -> set[str]:
    """Get current serial by-path symlinks (excluding usbv2 duplicates)."""
    return {
        p for p in glob.glob("/dev/serial/by-path/*")
        if "usbv2-" not in p
    }


def get_camera_paths() -> set[str]:
    """Get current camera by-path symlinks (excluding usbv3 duplicates and metadata nodes)."""
    return {
        p for p in glob.glob("/dev/v4l/by-path/*")
        if "usbv3-" not in p and "video-index0" in p
    }


def wait_for_removal(get_paths_fn, baseline: set[str], name: str) -> str:
    """Wait for exactly one device to disappear and return its path."""
    print(f"\n  >>> Unplug: {name}")
    print(f"      Waiting for disconnection...", end="", flush=True)

    while True:
        current = get_paths_fn()
        disappeared = baseline - current
        if len(disappeared) == 1:
            path = disappeared.pop()
            print(f"\n      Detected: {path}")
            return path
        if len(disappeared) > 1:
            print(f"\n  !! Multiple devices disappeared: {disappeared}")
            print(f"     Please reconnect all and try again.")
            input("     Press Enter when ready...")
            baseline.update(disappeared)
            print(f"      Waiting for disconnection...", end="", flush=True)
        time.sleep(0.3)


def wait_for_reconnection(get_paths_fn, expected_count: int, name: str):
    """Wait for the device to be reconnected."""
    print(f"  <<< Reconnect: {name}")
    print(f"      Waiting for reconnection...", end="", flush=True)

    while True:
        current = get_paths_fn()
        if len(current) >= expected_count:
            print(" OK")
            time.sleep(1)  # let it settle
            return
        time.sleep(0.3)


def identify_devices(names: list[str], get_paths_fn, device_type: str) -> dict[str, str]:
    """Identify a list of devices by unplugging each one."""
    mapping = {}

    baseline = get_paths_fn()
    expected_count = len(baseline)
    print(f"\n{'='*60}")
    print(f"  Identifying {device_type} ({len(names)} devices)")
    print(f"  Currently detected: {expected_count} device(s)")
    print(f"{'='*60}")

    if expected_count < len(names):
        print(f"\n  WARNING: Expected at least {len(names)} devices but only {expected_count} found.")
        print(f"  Make sure all devices are connected before continuing.")
        input("  Press Enter when ready...")
        baseline = get_paths_fn()
        expected_count = len(baseline)

    for i, name in enumerate(names):
        print(f"\n  [{i+1}/{len(names)}] {name}")

        path = wait_for_removal(get_paths_fn, baseline, name)
        mapping[name] = path

        wait_for_reconnection(get_paths_fn, expected_count, name)
        baseline = get_paths_fn()

    return mapping


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-o", "--output", default="port_config.env",
        help="Output env file (default: port_config.env)",
    )
    parser.add_argument(
        "-m", "--mode", choices=["single", "bimanual"], default="single",
        help="Arm mode: single (default) or bimanual",
    )
    parser.add_argument(
        "--serial-only", action="store_true", help="Only identify serial ports",
    )
    parser.add_argument(
        "--cameras-only", action="store_true", help="Only identify cameras",
    )
    args = parser.parse_args()

    if args.mode == "bimanual":
        serial_devices = BIMANUAL_SERIAL
        camera_devices = BIMANUAL_CAMERAS
    else:
        serial_devices = SINGLE_ARM_SERIAL
        camera_devices = SINGLE_ARM_CAMERAS

    print("USB Device Port Identifier")
    print("=" * 60)
    print(f"  Mode: {args.mode}")
    print(f"  Serial ports: {', '.join(serial_devices)}")
    print(f"  Cameras: {', '.join(camera_devices)}")
    print("=" * 60)
    print("This script identifies devices by physical disconnection.")
    print("You will be asked to unplug and replug each device.")
    print("Make sure ALL devices are connected before starting.")
    input("\nPress Enter to begin...")

    mapping = {}

    if not args.cameras_only:
        serial_map = identify_devices(serial_devices, get_serial_paths, "serial ports")
        mapping.update(serial_map)

    if not args.serial_only:
        camera_map = identify_devices(camera_devices, get_camera_paths, "cameras")
        mapping.update(camera_map)

    # Summary
    print(f"\n{'='*60}")
    print("  Results")
    print(f"{'='*60}")
    for name, path in mapping.items():
        print(f"    {name:20s} -> {path}")

    # Write env file
    env_path = args.output
    with open(env_path, "w") as f:
        f.write(f"# Generated by: python examples/identify_ports.py -m {args.mode}\n")
        f.write(f"# Re-run after changing USB port assignments.\n")
        for name, path in mapping.items():
            f.write(f'export {name.upper()}="{path}"\n')

    print(f"\n  Saved to: {env_path}")
    print(f"  Usage: source {env_path}")


if __name__ == "__main__":
    main()
