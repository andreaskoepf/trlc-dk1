#!/usr/bin/env python3
"""Identify which physical hub ports are USB 3.0 vs USB 2.0.

Plug a USB 3.0 device into each port one at a time.
The script detects the new device and reports if it connected at SuperSpeed.
"""

import subprocess
import time


def get_usb_devices():
    result = subprocess.run(["lsusb", "-t"], capture_output=True, text=True)
    return result.stdout


def get_device_list():
    result = subprocess.run(["lsusb"], capture_output=True, text=True)
    return set(result.stdout.strip().splitlines())


def main():
    print("USB 3.0 Port Identifier")
    print("=" * 50)
    print("Plug a USB 3.0 device into each port one at a time.")
    print("Wait for the result, then move to the next port.")
    print("Press Ctrl+C to stop.\n")

    port_num = 1
    baseline = get_device_list()

    while True:
        print(f"Waiting for device on port #{port_num}...")
        try:
            # Wait for a new device
            while True:
                current = get_device_list()
                new_devices = current - baseline
                if new_devices:
                    break
                time.sleep(0.3)

            time.sleep(0.5)  # let it settle
            current = get_device_list()
            new_devices = current - baseline

            for dev in new_devices:
                # Parse bus number
                parts = dev.split()
                bus = int(parts[1])

                # Check bus speed from lsusb -t
                tree = get_usb_devices()
                for line in tree.splitlines():
                    if f"Bus {bus:03d}.Port 001" in line:
                        if "5000M" in line or "10000M" in line or "20000M" in line:
                            speed = "USB 3.x"
                        else:
                            speed = "USB 2.0"
                        break
                else:
                    speed = "unknown"

                label = "USB 3.0" if "3.x" in speed else "USB 2.0 ONLY"
                icon = "OK" if "3.x" in speed else "!!"
                print(f"  [{icon}] Port #{port_num}: {label} (Bus {bus:03d}) — {dev.split(':', 1)[1].strip()}")

            # Wait for device removal
            print(f"       Unplug the device to test the next port...")
            while True:
                current = get_device_list()
                if current == baseline or len(current) <= len(baseline):
                    break
                time.sleep(0.3)

            time.sleep(0.5)
            baseline = get_device_list()
            port_num += 1
            print()

        except KeyboardInterrupt:
            print("\n\nDone!")
            break


if __name__ == "__main__":
    main()
