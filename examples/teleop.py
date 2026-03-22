#!/usr/bin/env python3
"""
Teleoperation: Leader arm (Dynamixel) controls Follower arm (DM motors).

Usage:
    python examples/teleop.py [--leader /dev/ttyACM0] [--follower /dev/ttyACM1]
                              [--mode rt_impedance] [--gravity-scale 0.5] [--hz 200]
"""
import argparse
import time

from lerobot_robot_trlc_dk1.follower import DK1Follower, DK1FollowerConfig
from lerobot_robot_trlc_dk1.leader import DK1Leader, DK1LeaderConfig


def main():
    parser = argparse.ArgumentParser(description="DK1 Teleop: Leader → Follower")
    parser.add_argument("--leader", default="/dev/ttyACM0", help="Leader serial port")
    parser.add_argument("--follower", default="/dev/ttyACM1", help="Follower serial port")
    parser.add_argument("--mode", default="rt_impedance",
                        choices=["rt_impedance", "impedance", "pos_vel"],
                        help="Follower control mode")
    parser.add_argument("--gravity-scale", type=float, default=0.5,
                        help="Gravity compensation scale")
    parser.add_argument("--hz", type=float, default=200.0, help="Teleop loop frequency")
    parser.add_argument("--disable-torque-on-disconnect", action="store_true",
                        help="Disable motor torque on shutdown")
    args = parser.parse_args()

    leader_config = DK1LeaderConfig(port=args.leader)
    follower_config = DK1FollowerConfig(
        port=args.follower,
        control_mode=args.mode,
        gravity_comp_scale=args.gravity_scale,
        disable_torque_on_disconnect=args.disable_torque_on_disconnect,
    )

    print(f"Leader:   {args.leader}")
    print(f"Follower: {args.follower} (mode={args.mode}, gravity_scale={args.gravity_scale})")
    print(f"Loop Hz:  {args.hz}")
    print()

    leader = DK1Leader(leader_config)
    leader.connect()
    print("Leader connected")

    follower = DK1Follower(follower_config)
    follower.connect()
    print("Follower connected")
    print()
    print("Teleop running — move the leader arm. Press Ctrl+C to stop.")

    has_rt = hasattr(follower, '_robot') and follower._robot is not None
    try:
        last_print_time = time.monotonic()
        while True:
            action = leader.get_action()
            follower.send_action(action)
            time.sleep(1.0 / args.hz)

            now = time.monotonic()
            if now - last_print_time >= 1.0 and has_rt:
                last_print_time = now
                state = follower._robot.get_joint_state()
                grip = follower._robot.get_gripper_state()
                perf = follower._robot.get_perf() if hasattr(follower._robot, 'get_perf') else None

                tau = state["torque"]
                tau_str = " ".join(f"{t:+6.2f}" for t in tau)
                grip_str = f"{grip['torque']:+5.2f}"

                perf_str = ""
                if perf is not None:
                    perf_str = (f"  | RT: {perf.mean_cycle_us:.0f}/{perf.max_cycle_us:.0f}us"
                                f" miss={perf.deadline_misses}")

                print(f"  tau: {tau_str}  grip:{grip_str}{perf_str}")
    except KeyboardInterrupt:
        print("\nStopping teleop...")

    leader.disconnect()
    follower.disconnect()
    print("Done.")


if __name__ == "__main__":
    main()
