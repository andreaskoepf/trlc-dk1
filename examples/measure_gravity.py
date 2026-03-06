#!/usr/bin/env python3
"""
Gravity compensation measurement and URDF calibration tool.

Measures the gravity model error by holding position with gravity comp
enabled and observing the steady-state position drift. Since:

    At equilibrium: kp*(q_des - q_eq) + τ_model(q) = τ_real(q)
    => gravity_model_error = kp * (q_actual - q_desired)

Positive error means URDF overestimates gravity (model pushes up too much).
Negative error means URDF underestimates gravity (arm droops).

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


def print_measurement(idx, hold_pos, avg_pos, avg_torque, std_pos, mj_gravity):
    pos_error = avg_pos - hold_pos
    pos_error_deg = np.degrees(pos_error)
    grav_model_error = DEFAULT_KP * pos_error

    print(f"\n  Measurement #{idx}:")
    print(f"  Hold pos (deg):   [{', '.join(f'{d:+7.1f}' for d in np.degrees(hold_pos))}]")
    print(f"  Actual pos (deg): [{', '.join(f'{d:+7.1f}' for d in np.degrees(avg_pos))}]")
    print(f"  Motor tau (Nm):   [{', '.join(f'{t:+7.3f}' for t in avg_torque)}]")
    print()
    header = f"  {'Joint':<10s} {'PosErr(deg)':>12s} {'GravErr(Nm)':>12s}"
    if mj_gravity is not None:
        header += f" {'MJ_grav(Nm)':>12s} {'Inferred(Nm)':>13s}"
    print(header)

    for i in range(6):
        line = f"  {JOINT_NAMES[i]:<10s} {pos_error_deg[i]:>+12.3f} {grav_model_error[i]:>+12.3f}"
        if mj_gravity is not None:
            inferred_real = mj_gravity[i] - grav_model_error[i]
            line += f" {mj_gravity[i]:>12.3f} {inferred_real:>+13.3f}"
        print(line)

    print()
    print("  GravErr > 0: model overcompensates (pushes up)")
    print("  GravErr < 0: model undercompensates (droops)")
    print("  Inferred = what gravity really is at this pose")

    return {
        'hold_pos': hold_pos.copy(),
        'actual_pos': avg_pos.copy(),
        'pos_error_deg': pos_error_deg.copy(),
        'motor_torque': avg_torque.copy(),
        'grav_model_error': grav_model_error.copy(),
        'mj_gravity': mj_gravity.copy() if mj_gravity is not None else None,
    }


def main():
    parser = argparse.ArgumentParser(description="Measure gravity torques for URDF calibration")
    parser.add_argument("--port", default="/dev/ttyACM1", help="Serial port for follower arm")
    parser.add_argument("--hz", type=float, default=250.0, help="Loop frequency")
    parser.add_argument("--gravity-scale", type=float, default=1.0,
                        help="Gravity comp scale")
    parser.add_argument("--settle-time", type=float, default=3.0,
                        help="Seconds to wait after setting hold position")
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

    measurements = []

    print()
    print("=" * 60)
    print("  GRAVITY COMPENSATION MEASUREMENT")
    print("=" * 60)
    print()
    print("  How it works:")
    print("    1. Move the arm by hand to a pose")
    print("    2. Press Enter — the script will:")
    print("       a) Lock the current position as the target")
    print(f"       b) Wait {args.settle_time}s for the arm to settle")
    print("       c) Measure how far the arm drifted (= gravity error)")
    print("    3. Repeat for different poses")
    print("    4. Type 'q' + Enter to quit and see results")
    print()
    print("  SUGGESTED POSES:")
    print("    1. Rest position (don't move, just press Enter)")
    print("    2. Arm extended horizontally (push joints 2+3 out)")
    print("    3. Arm pointing up (vertical)")
    print("    4. Wrist tilted down (bend joint 4 or 5)")
    print("    5. Joint 2 at ~45 degrees")
    print()
    print("=" * 60)
    print()

    while True:
        try:
            cmd = input(f"  [{len(measurements)} recorded] Move arm to a pose, then press Enter (q=quit) > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            break

        if cmd == 'q':
            break

        # Step 1: Read current position and set as hold target
        state = loop.get_joint_state()
        hold_pos = np.array(state.pos)
        loop.command_joint_pos(hold_pos)
        print(f"  Target locked at: [{', '.join(f'{d:+7.1f}' for d in np.degrees(hold_pos))}] deg")

        # Step 2: Wait for arm to settle
        print(f"  Settling for {args.settle_time}s...")
        time.sleep(args.settle_time)

        # Step 3: Collect samples over 0.5s
        n_samples = 50
        torques_list = []
        positions_list = []
        for _ in range(n_samples):
            st = loop.get_joint_state()
            torques_list.append(np.array(st.torque))
            positions_list.append(np.array(st.pos))
            time.sleep(0.01)

        avg_torque = np.mean(torques_list, axis=0)
        avg_pos = np.mean(positions_list, axis=0)
        std_pos = np.std(positions_list, axis=0)

        # Step 4: Compute MuJoCo prediction at this pose
        mj_gravity = compute_mujoco_gravity(urdf_path, hold_pos)

        # Step 5: Print and store
        m = print_measurement(len(measurements) + 1, hold_pos, avg_pos, avg_torque, std_pos, mj_gravity)
        measurements.append(m)
        print()

    # Final summary
    if measurements:
        print("\n" + "=" * 80)
        print("SUMMARY: All Measurements")
        print("=" * 80)

        has_mj = measurements[0]['mj_gravity'] is not None

        for idx, m in enumerate(measurements):
            print(f"\n--- Pose #{idx+1} (hold: [{', '.join(f'{d:+6.1f}' for d in np.degrees(m['hold_pos']))}] deg) ---")
            print(f"  Pos err (deg): [{', '.join(f'{e:+7.3f}' for e in m['pos_error_deg'])}]")
            print(f"  Grav err (Nm): [{', '.join(f'{e:+7.3f}' for e in m['grav_model_error'])}]")
            if has_mj and m['mj_gravity'] is not None:
                inferred = m['mj_gravity'] - m['grav_model_error']
                print(f"  MJ model (Nm): [{', '.join(f'{t:+7.3f}' for t in m['mj_gravity'])}]")
                print(f"  Inferred (Nm): [{', '.join(f'{t:+7.3f}' for t in inferred)}]")

        if len(measurements) > 1 and has_mj:
            print("\n--- Per-Joint Analysis ---")
            print(f"  {'Joint':<10s} {'MeanErr(Nm)':>12s} {'Interpretation':>40s}")
            all_errors = np.array([m['grav_model_error'] for m in measurements])
            mean_err = np.mean(all_errors, axis=0)
            for i in range(6):
                if abs(mean_err[i]) > 1.0:
                    direction = "OVER (pushes up)" if mean_err[i] > 0 else "UNDER (droops)"
                    interp = f"{direction} by {abs(mean_err[i]):.1f} Nm"
                elif abs(mean_err[i]) > 0.3:
                    direction = "slightly over" if mean_err[i] > 0 else "slightly under"
                    interp = f"{direction} by {abs(mean_err[i]):.1f} Nm"
                else:
                    interp = "OK (< 0.3 Nm)"
                print(f"  {JOINT_NAMES[i]:<10s} {mean_err[i]:>+12.3f} {interp:>40s}")

    print("\nStopping...")
    loop.stop()
    print("Done.")


if __name__ == "__main__":
    main()
