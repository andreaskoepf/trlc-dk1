#!/usr/bin/env python3
"""
Gravity compensation measurement with friction cancellation.

Uses bi-directional approach: for each pose, commands a small offset
above and below, then averages to cancel out motor friction.

    From above: kp*(q_des - q_above) + τ_model - friction = τ_real
    From below: kp*(q_des - q_below) + τ_model + friction = τ_real
    Average:    τ_model_error = kp * (q_des - (q_above + q_below)/2)
    Friction:   friction = kp * (q_below - q_above) / 2

Usage:
    python examples/measure_gravity.py --port /dev/ttyACM1
"""
import argparse
import sys
import time

import numpy as np

JOINT_NAMES = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]
DEFAULT_KP = np.array([80.0, 70.0, 60.0, 20.0, 20.0, 10.0])


def make_config(port, hz=250.0, gravity_scale=1.0):
    from _trlc_dk1_rt import RtLoopConfig, MotorDescriptor, MotorType

    cfg = RtLoopConfig()
    cfg.serial_port = port
    cfg.loop_hz = hz
    cfg.gravity_comp_scale = gravity_scale

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

    import pathlib
    urdf = pathlib.Path(__file__).parent.parent / "urdf" / "follower" / "TRLC-DK1-Follower.urdf"
    cfg.model_path = str(urdf.resolve())

    return cfg


def compute_mujoco_gravity(urdf_path, q):
    try:
        from trlc_dk1_control.gravity_comp import GravityCompensator
        gc = GravityCompensator(urdf_path, num_dofs=6)
        return gc.compute(q)
    except Exception as e:
        print(f"Warning: could not compute MuJoCo gravity: {e}")
        return None


def sample_position(loop, n_samples=50, dt=0.01):
    """Collect position samples and return mean."""
    positions = []
    for _ in range(n_samples):
        st = loop.get_joint_state()
        positions.append(np.array(st.pos))
        time.sleep(dt)
    return np.mean(positions, axis=0)


def main():
    parser = argparse.ArgumentParser(description="Bi-directional gravity measurement")
    parser.add_argument("--port", default="/dev/ttyACM1", help="Serial port")
    parser.add_argument("--hz", type=float, default=250.0, help="Loop frequency")
    parser.add_argument("--gravity-scale", type=float, default=1.0, help="Gravity comp scale")
    parser.add_argument("--settle-time", type=float, default=2.0,
                        help="Seconds to wait after each offset command")
    parser.add_argument("--offset-deg", type=float, default=5.0,
                        help="Offset in degrees for bi-directional test")
    args = parser.parse_args()

    try:
        from _trlc_dk1_rt import RtControlLoop
    except ImportError:
        print("Error: C++ RT extension not found. Build it first.")
        sys.exit(1)

    import pathlib
    urdf_path = str((pathlib.Path(__file__).parent.parent / "urdf" / "follower" / "TRLC-DK1-Follower.urdf").resolve())

    cfg = make_config(args.port, args.hz, gravity_scale=args.gravity_scale)
    loop = RtControlLoop(cfg)

    print(f"Connecting to {args.port}...")
    loop.start()
    print(f"Connected. Gravity comp scale = {args.gravity_scale}")

    # Hold current position
    state = loop.get_joint_state()
    hold_pos = np.array(state.pos)
    loop.command_joint_pos(hold_pos)
    loop.command_gripper(0.0)

    offset_rad = np.radians(args.offset_deg)
    measurements = []

    print()
    print("=" * 70)
    print("  BI-DIRECTIONAL GRAVITY MEASUREMENT")
    print("=" * 70)
    print()
    print("  This tool cancels motor friction by measuring from both directions.")
    print()
    print(f"  For each pose, it commands ±{args.offset_deg}° offsets on each joint,")
    print("  measures where the arm actually settles, and averages to cancel")
    print("  friction. This reveals the true gravity model error.")
    print()
    print("  How to use:")
    print("    1. Move the arm by hand to a pose")
    print("    2. Press Enter — the script runs the bi-directional test")
    print("       (takes ~30 seconds per pose — don't touch the arm)")
    print("    3. Repeat for different poses")
    print("    4. Type 'q' + Enter to quit and see results")
    print()
    print("  SUGGESTED POSES:")
    print("    1. Rest position (as-is)")
    print("    2. Arm extended horizontally (joints 2+3)")
    print("    3. Wrist tilted down (joint 4 or 5)")
    print("    4. Various arm angles")
    print()
    print("=" * 70)
    print()

    while True:
        try:
            cmd = input(f"  [{len(measurements)} recorded] Move arm, press Enter (q=quit) > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            break

        if cmd == 'q':
            break

        # Read current position as the nominal pose
        state = loop.get_joint_state()
        nominal_pos = np.array(state.pos)
        loop.command_joint_pos(nominal_pos)
        print(f"  Nominal pose: [{', '.join(f'{d:+7.1f}' for d in np.degrees(nominal_pos))}] deg")
        print(f"  Running bi-directional test (±{args.offset_deg}°)...")

        # For each joint, command offset above and below, measure settled position
        settled_above = np.zeros(6)  # position when approached from above
        settled_below = np.zeros(6)  # position when approached from below

        for j in range(6):
            # Test offset above: command nominal + offset, then back to nominal
            pos_high = nominal_pos.copy()
            pos_high[j] += offset_rad
            loop.command_joint_pos(pos_high)
            time.sleep(args.settle_time)

            # Now command nominal — arm approaches from above
            loop.command_joint_pos(nominal_pos)
            time.sleep(args.settle_time)
            settled_above[j] = sample_position(loop)[j]

            # Test offset below: command nominal - offset, then back to nominal
            pos_low = nominal_pos.copy()
            pos_low[j] -= offset_rad
            loop.command_joint_pos(pos_low)
            time.sleep(args.settle_time)

            # Now command nominal — arm approaches from below
            loop.command_joint_pos(nominal_pos)
            time.sleep(args.settle_time)
            settled_below[j] = sample_position(loop)[j]

            # Return to nominal
            loop.command_joint_pos(nominal_pos)

            above_err = np.degrees(settled_above[j] - nominal_pos[j])
            below_err = np.degrees(settled_below[j] - nominal_pos[j])
            print(f"    {JOINT_NAMES[j]}: from_above={above_err:+.4f}° from_below={below_err:+.4f}°")

        # Compute gravity model error (friction-cancelled)
        midpoint = (settled_above + settled_below) / 2.0
        grav_model_error = DEFAULT_KP * (midpoint - nominal_pos)

        # Compute friction estimate
        friction = DEFAULT_KP * (settled_below - settled_above) / 2.0

        # MuJoCo prediction
        mj_gravity = compute_mujoco_gravity(urdf_path, nominal_pos)

        m = {
            'nominal_pos': nominal_pos.copy(),
            'settled_above': settled_above.copy(),
            'settled_below': settled_below.copy(),
            'grav_model_error': grav_model_error.copy(),
            'friction': friction.copy(),
            'mj_gravity': mj_gravity.copy() if mj_gravity is not None else None,
        }
        measurements.append(m)

        # Print results
        print()
        header = f"  {'Joint':<10s} {'GravErr(Nm)':>12s} {'Friction(Nm)':>13s}"
        if mj_gravity is not None:
            header += f" {'MJ_model(Nm)':>13s} {'Real_grav(Nm)':>14s}"
        print(header)

        for i in range(6):
            line = f"  {JOINT_NAMES[i]:<10s} {grav_model_error[i]:>+12.3f} {friction[i]:>13.3f}"
            if mj_gravity is not None:
                real_grav = mj_gravity[i] - grav_model_error[i]
                line += f" {mj_gravity[i]:>13.3f} {real_grav:>+14.3f}"
            print(line)

        print()
        print("  GravErr: model - real gravity (+ = overcompensates, - = undercompensates)")
        print("  Friction: estimated static friction per joint")
        print()

    # Summary
    if measurements:
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)

        has_mj = measurements[0]['mj_gravity'] is not None

        for idx, m in enumerate(measurements):
            print(f"\n--- Pose #{idx+1} ([{', '.join(f'{d:+6.1f}' for d in np.degrees(m['nominal_pos']))}] deg) ---")
            print(f"  Grav err (Nm):  [{', '.join(f'{e:+7.3f}' for e in m['grav_model_error'])}]")
            print(f"  Friction (Nm):  [{', '.join(f'{f:7.3f}' for f in m['friction'])}]")
            if has_mj and m['mj_gravity'] is not None:
                real_grav = m['mj_gravity'] - m['grav_model_error']
                print(f"  MJ model (Nm):  [{', '.join(f'{t:+7.3f}' for t in m['mj_gravity'])}]")
                print(f"  Real grav (Nm): [{', '.join(f'{t:+7.3f}' for t in real_grav)}]")
                with np.errstate(divide='ignore', invalid='ignore'):
                    scale = np.where(np.abs(m['mj_gravity']) > 0.5,
                                     real_grav / m['mj_gravity'],
                                     np.nan)
                print(f"  Scale factor:   [{', '.join(f'{s:+7.3f}' if not np.isnan(s) else '    N/A' for s in scale)}]")

        if len(measurements) > 1:
            print("\n--- Per-Joint Average ---")
            all_errors = np.array([m['grav_model_error'] for m in measurements])
            all_friction = np.array([m['friction'] for m in measurements])
            mean_err = np.mean(all_errors, axis=0)
            mean_friction = np.mean(all_friction, axis=0)

            print(f"  {'Joint':<10s} {'MeanGravErr':>12s} {'MeanFriction':>13s} {'Interpretation':>30s}")
            for i in range(6):
                if abs(mean_err[i]) > 1.0:
                    direction = "OVER" if mean_err[i] > 0 else "UNDER"
                    interp = f"{direction} by {abs(mean_err[i]):.1f} Nm"
                elif abs(mean_err[i]) > 0.3:
                    direction = "slightly over" if mean_err[i] > 0 else "slightly under"
                    interp = f"{direction} by {abs(mean_err[i]):.1f} Nm"
                else:
                    interp = "OK (< 0.3 Nm)"
                print(f"  {JOINT_NAMES[i]:<10s} {mean_err[i]:>+12.3f} {mean_friction[i]:>13.3f} {interp:>30s}")

    print("\nStopping...")
    loop.stop()
    print("Done.")


if __name__ == "__main__":
    main()
