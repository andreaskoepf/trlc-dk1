#!/usr/bin/env python3
"""
Bimanual Teleoperation: Two Leader arms control two Follower arms.

Usage:
    python examples/bi_teleop.py [--mode rt_impedance] [--gravity-scale 1.0] [--hz 200]
"""
import argparse
import time

from lerobot_robot_trlc_dk1.bi_follower import BiDK1Follower, BiDK1FollowerConfig
from lerobot_robot_trlc_dk1.bi_leader import BiDK1Leader, BiDK1LeaderConfig


def main():
    parser = argparse.ArgumentParser(description="DK1 Bimanual Teleop")
    parser.add_argument("--left-leader", default="/dev/ttyACM3", help="Left leader port")
    parser.add_argument("--right-leader", default="/dev/ttyACM2", help="Right leader port")
    parser.add_argument("--left-follower", default="/dev/ttyACM1", help="Left follower port")
    parser.add_argument("--right-follower", default="/dev/ttyACM0", help="Right follower port")
    parser.add_argument("--mode", default="rt_impedance",
                        choices=["rt_impedance", "impedance", "pos_vel"],
                        help="Follower control mode")
    parser.add_argument("--gravity-scale", type=float, default=1.0,
                        help="Gravity compensation scale")
    parser.add_argument("--hz", type=float, default=200.0, help="Teleop loop frequency")
    parser.add_argument("--disable-torque-on-disconnect", action="store_true",
                        help="Disable motor torque on shutdown")
    args = parser.parse_args()

    leader_config = BiDK1LeaderConfig(
        left_arm_port=args.left_leader,
        right_arm_port=args.right_leader,
    )
    follower_config = BiDK1FollowerConfig(
        left_arm_port=args.left_follower,
        right_arm_port=args.right_follower,
        control_mode=args.mode,
        gravity_comp_scale=args.gravity_scale,
        disable_torque_on_disconnect=args.disable_torque_on_disconnect,
    )

    print(f"Left:  leader={args.left_leader}  follower={args.left_follower}")
    print(f"Right: leader={args.right_leader}  follower={args.right_follower}")
    print(f"Mode:  {args.mode}  gravity_scale={args.gravity_scale}  hz={args.hz}")
    print()

    leader = BiDK1Leader(leader_config)
    leader.connect()
    print("Leaders connected")

    follower = BiDK1Follower(follower_config)
    follower.connect()
    print("Followers connected")
    print()
    print("Bimanual teleop running — move the leader arms. Press Ctrl+C to stop.")

    has_rt = args.mode in ("impedance", "rt_impedance")
    try:
        last_print_time = time.monotonic()
        while True:
            action = leader.get_action()
            follower.send_action(action)
            time.sleep(1.0 / args.hz)

            now = time.monotonic()
            if now - last_print_time >= 1.0 and has_rt:
                last_print_time = now

                def _arm_stats(arm, label):
                    if arm._robot is None:
                        return ""
                    state = arm._robot.get_joint_state()
                    grip = arm._robot.get_gripper_state()
                    perf = arm._robot.get_perf()
                    tau = state["torque"]
                    tau_str = " ".join(f"{t:+5.1f}" for t in tau)
                    grip_str = f"{grip['torque']:+5.2f}"
                    perf_str = ""
                    if perf is not None:
                        perf_str = (f"  | RT: {perf.mean_cycle_us:.0f}/{perf.max_cycle_us:.0f}us"
                                    f" miss={perf.deadline_misses}")
                    return f"  {label} tau: {tau_str}  grip:{grip_str}{perf_str}"

                print(_arm_stats(follower.left_arm, " L"))
                print(_arm_stats(follower.right_arm, "R"))
    except KeyboardInterrupt:
        print("\nStopping teleop...")

    leader.disconnect()
    follower.disconnect()
    print("Done.")


if __name__ == "__main__":
    main()
