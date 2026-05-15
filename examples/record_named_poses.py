#!/usr/bin/env python3
"""
Record a set of named joint-space poses by teleop.

Drives the followers from the leader arms (standard bimanual teleop). When you
have the arms where you want them, press `b` to capture the current 14-DoF
*commanded joint target* into the active slot. Navigate slots with `a` (prev)
/ `c` (next). The output JSON is updated on every capture and on graceful
exit.

The 14-DoF layout matches what fastwam uses:
    [L_j1, L_j2, L_j3, L_j4, L_j5, L_j6, L_gripper,
     R_j1, R_j2, R_j3, R_j4, R_j5, R_j6, R_gripper]
i.e. 7 left + 7 right, gripper after the six joints on each arm.

What gets captured: the most recent leader-provided action that was sent to
the follower (i.e. the joint command), NOT the follower's observed reached
position. This matters for `rt_impedance` mode — under that controller the
follower tracks the commanded target with a small steady-state error, so the
"reached" position is offset from the "command". When these poses are later
replayed as commanded targets (e.g. by the fastwam eval transfer planner,
also in rt_impedance), saving the original command keeps the closed-loop
behaviour consistent: command-in → command-out, with the same tracking
offset applied at replay time.

Knows nothing about prompts. The set of slot names is taken from either:
    * the existing keys of the output JSON file (re-recording / filling in
      missing slots), or
    * --names a,b,c,... (creating a fresh file).

Foot-pedal-friendly: only three keys are used during recording, and the
binding is printed at startup so you can wire the pedal channels however you
like.

Usage:
    python examples/record_named_poses.py poses.json \
        --names clear_area,place_base,place_body_left,...

    # Re-open existing file to fill gaps or re-record specific slots:
    python examples/record_named_poses.py poses.json
"""

from __future__ import annotations

import argparse
import json
import os
import select
import signal
import sys
import termios
import time
import tty
from pathlib import Path
from typing import Optional

from lerobot_robot_trlc_dk1.bi_follower import BiDK1Follower, BiDK1FollowerConfig
from lerobot_robot_trlc_dk1.bi_leader import BiDK1Leader, BiDK1LeaderConfig


# Joint order in the output vector (must match the fastwam DK-1 14-DoF layout).
_LEFT_KEYS = [f"left_joint_{i}.pos" for i in range(1, 7)] + ["left_gripper.pos"]
_RIGHT_KEYS = [f"right_joint_{i}.pos" for i in range(1, 7)] + ["right_gripper.pos"]
_POSE_KEYS: list[str] = _LEFT_KEYS + _RIGHT_KEYS  # length 14


# Foot-pedal-friendly bindings. Edit here if you want a different layout —
# the bound keys are printed at startup so the pedal's channel mapping can
# match.
KEY_PREV = "a"
KEY_CAPTURE = "b"
KEY_NEXT = "c"


_stop_requested = False


def _sigint_handler(signum, frame):
    global _stop_requested
    if _stop_requested:
        raise KeyboardInterrupt  # second Ctrl-C forces exit
    _stop_requested = True


def _read_key_nonblocking() -> Optional[str]:
    """Return the next character from stdin if one is available, else None.

    Assumes the caller has already put stdin into cbreak mode.
    """
    r, _, _ = select.select([sys.stdin], [], [], 0)
    if not r:
        return None
    return sys.stdin.read(1)


def _action_pose_vector(action: dict[str, float]) -> list[float]:
    """Project a leader/follower action dict into the 14-float pose vector.

    The action dict — as emitted by `leader.get_action()` and passed to
    `follower.send_action()` — contains the commanded joint targets in the
    rt_impedance / impedance control modes. That's what we want to store
    (see module docstring: command-in / command-out).
    """
    missing = [k for k in _POSE_KEYS if k not in action]
    if missing:
        raise RuntimeError(
            f"Action dict missing expected keys: {missing}. "
            f"Got keys: {sorted(action.keys())}"
        )
    return [float(action[k]) for k in _POSE_KEYS]


def _format_pose_short(pose: list[float]) -> str:
    """Compact human-readable summary of a 14-DoF pose."""
    parts = [f"{v:+.3f}" for v in pose]
    return f"L[{','.join(parts[:7])}]  R[{','.join(parts[7:])}]"


def _load_existing(path: Path) -> dict[str, Optional[list[float]]]:
    """Load JSON if it exists, else return empty dict."""
    if not path.exists():
        return {}
    with path.open() as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object (dict)")
    return data


def _save(path: Path, data: dict[str, Optional[list[float]]]) -> None:
    """Atomic save: write to .tmp then rename."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def _resolve_names(
    existing: dict[str, Optional[list[float]]],
    names_arg: Optional[str],
) -> list[str]:
    """Determine the ordered slot names for this session.

    Precedence:
      1. --names a,b,c,... (always wins; merges with anything already in the
         file — preserves existing values for matching names; new names start
         empty).
      2. existing JSON keys, in file order.
    """
    if names_arg:
        names = [n.strip() for n in names_arg.split(",") if n.strip()]
        if not names:
            raise ValueError("--names parsed empty")
        # Stable order: as given on the CLI.
        for n in names:
            if n not in existing:
                existing[n] = None
        return names
    if not existing:
        raise ValueError(
            "Output file is empty / missing and --names was not given. "
            "Pass --names name1,name2,... to seed the slot list."
        )
    return list(existing.keys())


def main():
    parser = argparse.ArgumentParser(
        description="Record named bimanual joint poses by teleop.",
    )
    parser.add_argument("output", help="Path to the JSON file to write/update.")
    parser.add_argument("--names", default=None,
                        help="Comma-separated slot names (creates new slots; "
                             "preserves existing). If omitted, names are "
                             "taken from the existing file's keys.")
    parser.add_argument("--left-leader", default="/dev/ttyACM3")
    parser.add_argument("--right-leader", default="/dev/ttyACM2")
    parser.add_argument("--left-follower", default="/dev/ttyACM1")
    parser.add_argument("--right-follower", default="/dev/ttyACM0")
    parser.add_argument("--mode", default="rt_impedance",
                        choices=["rt_impedance", "impedance", "pos_vel"],
                        help="Follower control mode (teleop runs through this).")
    parser.add_argument("--gravity-scale", type=float, default=1.0)
    parser.add_argument("--hz", type=float, default=200.0,
                        help="Teleop dispatch rate.")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = _load_existing(output_path)
    names = _resolve_names(data, args.names)

    # Persist any seeded keys (so a subsequent run with --names left off can
    # still find them in the file).
    _save(output_path, data)

    # ----- Banner -----
    print("=" * 70)
    print(" record_named_poses — bimanual joint-pose teleop recorder")
    print("=" * 70)
    print(f"  output:   {output_path}")
    print(f"  slots:    {len(names)} ({sum(1 for n in names if data.get(n))}"
          f" already recorded)")
    print(f"  leaders:  L={args.left_leader}  R={args.right_leader}")
    print(f"  follower: L={args.left_follower}  R={args.right_follower}")
    print(f"  mode:     {args.mode}  gravity_scale={args.gravity_scale}"
          f"  hz={args.hz}")
    print("")
    print(f"  KEYS:  {KEY_PREV!r} = previous slot   "
          f"{KEY_CAPTURE!r} = CAPTURE current pose   "
          f"{KEY_NEXT!r} = next slot")
    print("         Ctrl-C twice to exit (data saved on every capture).")
    print("=" * 70)
    print("")

    # ----- Connect leaders + followers -----
    leader_config = BiDK1LeaderConfig(
        left_arm_port=args.left_leader,
        right_arm_port=args.right_leader,
    )
    follower_config = BiDK1FollowerConfig(
        left_arm_port=args.left_follower,
        right_arm_port=args.right_follower,
        control_mode=args.mode,
        gravity_comp_scale=args.gravity_scale,
        disable_torque_on_disconnect=False,
        cameras={},
    )

    leader = BiDK1Leader(leader_config)
    leader.connect()
    print("Leaders connected")

    follower = BiDK1Follower(follower_config)
    follower.connect()
    print("Followers connected")
    print("")
    print("Teleop running. Drive the followers with the leader arms.")
    print("")

    signal.signal(signal.SIGINT, _sigint_handler)

    # ----- Teleop loop with cbreak stdin -----
    fd = sys.stdin.fileno()
    is_tty = sys.stdin.isatty()
    saved_termios = None
    if is_tty:
        saved_termios = termios.tcgetattr(fd)
        tty.setcbreak(fd)
    else:
        print("[!] stdin is not a TTY — key capture is line-buffered "
              "(press Enter after each key).")

    cursor = 0  # index into `names`
    dt = 1.0 / args.hz
    last_status_t = 0.0
    # Most recent commanded action (leader → follower). Used by _capture().
    last_action: dict[str, float] = {}

    def _print_status(force: bool = False) -> None:
        nonlocal last_status_t
        now = time.monotonic()
        if not force and now - last_status_t < 0.5:
            return
        last_status_t = now
        name = names[cursor]
        recorded = data.get(name)
        tag = "RECORDED" if recorded else "  empty "
        # \r + clear-to-end so the line redraws cleanly without scrolling.
        sys.stdout.write(
            f"\r[{cursor + 1:>3}/{len(names)}] {tag}  {name:<32}"
            "\x1b[K"
        )
        sys.stdout.flush()

    def _capture() -> None:
        name = names[cursor]
        if not last_action:
            sys.stdout.write(
                f"\r[{cursor + 1:>3}/{len(names)}] capture SKIPPED: "
                f"no action sent to follower yet\n"
            )
            sys.stdout.flush()
            return
        pose = _action_pose_vector(last_action)
        data[name] = pose
        _save(output_path, data)
        # Push a full line above the live status line.
        sys.stdout.write(
            f"\r[{cursor + 1:>3}/{len(names)}] CAPTURED {name}\n"
            f"        {_format_pose_short(pose)}\n"
        )
        sys.stdout.flush()

    def _move(delta: int) -> None:
        nonlocal cursor
        cursor = (cursor + delta) % len(names)

    _print_status(force=True)

    try:
        while not _stop_requested:
            # 1) Drive followers from leaders. Remember the commanded action
            #    so a subsequent capture saves the joint *target*, not the
            #    follower's tracking-offset observation.
            action = leader.get_action()
            follower.send_action(action)
            last_action = action

            # 2) Drain pending keystrokes.
            while True:
                ch = _read_key_nonblocking()
                if ch is None:
                    break
                if ch == KEY_PREV:
                    _move(-1)
                    _print_status(force=True)
                elif ch == KEY_NEXT:
                    _move(+1)
                    _print_status(force=True)
                elif ch == KEY_CAPTURE:
                    _capture()
                    _print_status(force=True)
                elif ch in ("\x03",):  # Ctrl-C in cbreak
                    _sigint_handler(None, None)

            # 3) Update status (rate-limited).
            _print_status()
            time.sleep(dt)
    except (KeyboardInterrupt, OSError):
        pass
    finally:
        if saved_termios is not None:
            termios.tcsetattr(fd, termios.TCSADRAIN, saved_termios)
        sys.stdout.write("\n")
        sys.stdout.flush()

    print("Stopping teleop...")

    # Match bi_teleop.py shutdown pattern: stop followers via RT loop, close
    # leader serial ports directly.
    for arm in [follower.left_arm, follower.right_arm]:
        if hasattr(arm, "_robot") and arm._robot is not None:
            try:
                arm._robot.disconnect()
            except Exception as e:
                print(f"  Follower disconnect error: {e}")
    for arm in [leader.left_arm, leader.right_arm]:
        try:
            arm.bus.port_handler.ser.close()
        except Exception:
            pass

    # Final save (mostly redundant, but cheap).
    _save(output_path, data)

    n_recorded = sum(1 for n in names if data.get(n))
    print(f"Saved {output_path} — {n_recorded}/{len(names)} slots recorded.")
    missing = [n for n in names if not data.get(n)]
    if missing:
        print(f"  Missing: {', '.join(missing)}")
    print("Done.")


if __name__ == "__main__":
    main()
