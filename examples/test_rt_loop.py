#!/usr/bin/env python3
"""
Step-by-step hardware smoke test for the C++ RT control loop.

Connects to the follower arm, calibrates the gripper, holds position
for a few seconds, then prints jitter stats and shuts down.

Usage:
    python examples/test_rt_loop.py --port /dev/ttyACM1
"""

import argparse
import sys
import time

import numpy as np


def make_config(port, hz=250.0, gravity_comp=True, gravity_scale=1.0):
    from trlc_dk1_control._trlc_dk1_rt import RtLoopConfig, MotorDescriptor, MotorType

    cfg = RtLoopConfig()
    cfg.serial_port = port
    cfg.loop_hz = hz
    cfg.gravity_comp_scale = gravity_scale
    cfg.min_motors_required = 6
    cfg.gripper_cal_timeout_s = 10.0

    motor_defs = [
        ("joint_1", MotorType.DM4340, 0x01, 0x11),
        ("joint_2", MotorType.DM4340, 0x02, 0x12),
        ("joint_3", MotorType.DM4340, 0x03, 0x13),
        ("joint_4", MotorType.DM4310, 0x04, 0x14),
        ("joint_5", MotorType.DM4310, 0x05, 0x15),
        ("joint_6", MotorType.DM4310, 0x06, 0x16),
        ("gripper", MotorType.DM4310, 0x07, 0x17),
    ]
    motors = []
    for name, mtype, sid, mid in motor_defs:
        desc = MotorDescriptor()
        desc.name = name
        desc.type = mtype
        desc.slave_id = sid
        desc.master_id = mid
        motors.append(desc)
    cfg.motors = motors

    if gravity_comp:
        import pathlib
        urdf = pathlib.Path(__file__).parent.parent / "urdf" / "follower" / "TRLC-DK1-Follower.urdf"
        cfg.model_path = str(urdf.resolve())
    else:
        cfg.model_path = ""

    return cfg


def main():
    parser = argparse.ArgumentParser(description="RT control loop hardware smoke test")
    parser.add_argument("--port", default="/dev/ttyACM1", help="Serial port for follower arm")
    parser.add_argument("--hz", type=float, default=250.0, help="Loop frequency")
    parser.add_argument("--duration", type=float, default=10.0, help="Hold duration in seconds")
    parser.add_argument("--no-gravity-comp", action="store_true", help="Disable gravity compensation")
    parser.add_argument("--gravity-scale", type=float, default=1.0, help="Gravity compensation scale (0.0-1.0)")
    args = parser.parse_args()

    try:
        from trlc_dk1_control._trlc_dk1_rt import RtControlLoop, detect_rt_kernel
    except ImportError:
        print("Error: C++ RT extension not found. Reinstall the package:")
        print("  uv sync --reinstall-package lerobot-robot-trlc-dk1")
        sys.exit(1)

    print(f"RT kernel: {detect_rt_kernel()}")
    print(f"Serial port: {args.port}")
    print(f"Loop Hz: {args.hz}")
    print(f"Gravity comp: {not args.no_gravity_comp}")
    print()

    cfg = make_config(args.port, args.hz, gravity_comp=not args.no_gravity_comp,
                      gravity_scale=args.gravity_scale)

    # --- Step 1: Create loop ---
    print("Step 1: Creating RtControlLoop...")
    loop = RtControlLoop(cfg)
    print("  OK")

    # --- Step 2: Start (configure motors, calibrate gripper, launch RT thread) ---
    print("Step 2: Starting (motor config + gripper calibration)...")
    try:
        loop.start()
    except RuntimeError as e:
        error_msg = str(e)
        if "Motor initialization failed" in error_msg:
            print(f"  FAILED during motor initialization:\n  {e}")
            print("\n  Troubleshooting:")
            print("  1. Check that the power supply is ON")
            print("  2. Verify the serial port is correct (--port flag)")
            print("  3. Check CAN bus cable connections")
        elif "Gripper calibration timed out" in error_msg:
            print(f"  FAILED during gripper calibration:\n  {e}")
            print("\n  Troubleshooting:")
            print("  1. Check that the gripper motor is powered on")
            print("  2. Ensure the gripper can move freely to its open stop")
        else:
            print(f"  FAILED: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"  FAILED (invalid config): {e}")
        sys.exit(1)
    print(f"  OK (RT active: {loop.is_rt_active()})")

    # --- Step 3: Read initial state ---
    print("Step 3: Reading initial state...")
    state = loop.get_joint_state()
    gripper = loop.get_gripper_state()
    print(f"  Joint pos (deg): {np.degrees(np.array(state.pos))}")
    print(f"  Joint vel:       {np.array(state.vel)}")
    print(f"  Joint torque:    {np.array(state.torque)}")
    print(f"  Gripper pos:     {gripper.pos:.3f} (normalized)")
    print(f"  Gripper torque:  {gripper.torque:.3f} Nm")

    # --- Step 4: Hold position ---
    print(f"Step 4: Holding position for {args.duration}s...")
    print(f"  (The arm should hold steady with gravity compensation)")
    print(f"  You can gently push the arm — it should resist like a spring.")
    print()

    hold_pos = np.array(state.pos)
    loop.command_joint_pos(hold_pos)
    loop.command_gripper(0.0)  # open

    try:
        start = time.monotonic()
        last_print = start
        while time.monotonic() - start < args.duration:
            time.sleep(0.1)

            now = time.monotonic()
            if now - last_print >= 2.0:
                last_print = now
                elapsed = now - start

                st = loop.get_joint_state()
                perf = loop.get_perf()
                health = loop.get_health()

                pos_err = np.array(st.pos) - hold_pos
                status_flags = []
                if health.damping_mode:
                    status_flags.append("DAMPING")
                if health.comm_loss:
                    status_flags.append("COMM_LOSS")
                status = " [" + ", ".join(status_flags) + "]" if status_flags else ""

                print(
                    f"  [{elapsed:5.1f}s] "
                    f"pos_err(deg)=[{', '.join(f'{np.degrees(e):+5.2f}' for e in pos_err)}]  "
                    f"mean={perf.mean_cycle_us:.0f}us  "
                    f"max={perf.max_cycle_us:.0f}us  "
                    f"misses={perf.deadline_misses}"
                    f"{status}"
                )
    except KeyboardInterrupt:
        print("\n  Interrupted.")

    # --- Step 5: Test gripper ---
    print("Step 5: Testing gripper (close 50%, hold 2s, open)...")
    loop.command_gripper(0.5)
    time.sleep(2.0)
    gs = loop.get_gripper_state()
    print(f"  Gripper at {gs.pos:.3f} (target 0.5), torque={gs.torque:.3f} Nm")
    loop.command_gripper(0.0)
    time.sleep(1.0)

    # --- Step 6: Final jitter stats ---
    print("Step 6: Jitter statistics...")
    perf = loop.get_perf()
    print(f"  Total loops:     {perf.loop_count}")
    print(f"  Min cycle:       {perf.min_cycle_us:.1f} us")
    print(f"  Max cycle:       {perf.max_cycle_us:.1f} us")
    print(f"  Mean cycle:      {perf.mean_cycle_us:.1f} us")
    print(f"  Deadline misses: {perf.deadline_misses}")

    target_us = 1e6 / args.hz
    hist = np.array(perf.histogram)
    labels = ["0-100us", "100-500us", "500us-1ms", "1-2ms", "2-4ms", ">4ms"]
    print(f"  Target:          {target_us:.0f} us")
    print("  Histogram:")
    for label, count in zip(labels, hist):
        pct = 100.0 * count / max(perf.loop_count, 1)
        print(f"    {label:>10s}: {count:>8d} ({pct:5.1f}%)")

    cycle_times = loop.read_cycle_times(1000)
    if len(cycle_times) > 0:
        print(f"  Last {len(cycle_times)} cycles: "
              f"p50={np.percentile(cycle_times, 50):.0f}us  "
              f"p95={np.percentile(cycle_times, 95):.0f}us  "
              f"p99={np.percentile(cycle_times, 99):.0f}us")

    # --- Step 7: Health summary ---
    print("Step 7: Health state...")
    health = loop.get_health()
    print(f"  Damping mode:    {health.damping_mode}")
    print(f"  Overcurrent:     {health.overcurrent_count}")
    print(f"  Overspeed:       {health.overspeed_count}")
    print(f"  Comm loss:       {health.comm_loss}")
    print(f"  Empty cycles:    {health.consecutive_empty_cycles}")
    print(f"  Total RX bytes:  {health.total_rx_bytes}")
    print(f"  Total TX frames: {health.total_tx_frames}")
    print(f"  Write errors:    {health.total_write_errors}")
    print(f"  Motor stale:     {health.motor_stale}")

    # --- Step 8: Stop ---
    print("Step 8: Stopping...")
    loop.stop()
    print("  Done. All motors disabled.")


if __name__ == "__main__":
    main()
