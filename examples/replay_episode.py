#!/usr/bin/env python3
"""
Replay a recorded LeRobot episode on the real DK1 follower arms.

Reads the recorded `action` stream (the 14-DoF commanded joint targets) for a
single episode from a local LeRobot v3 dataset and plays it back through the
bimanual follower — by default in `rt_impedance` mode, the same controller the
recorder uses. No leader arms are needed.

The replay rate is taken from the dataset's own `fps` (info.json). An optional
velocity scale slows playback down: --velocity-scale 0.1 replays at 10 % of the
recorded rate (e.g. a 30 Hz dataset plays back at 3 Hz). The joint *targets*
are unchanged — only the time between frames is stretched — so the arm visits
exactly the same poses, just more slowly.

Before playback the arms are smoothly ramped from their current pose to the
episode's first frame over --ramp-time seconds (mirrors the recorder's startup
sync) so nothing snaps.

Action layout (matches the recorder / info.json `action.names`):
    [left_joint_1.pos … left_joint_6.pos, left_gripper.pos,
     right_joint_1.pos … right_joint_6.pos, right_gripper.pos]

Usage:
    python examples/replay_episode.py data/dk1_duplo_stack_simple --episode 0

    # Slow, 10 % speed:
    python examples/replay_episode.py data/dk1_duplo_stack_simple \
        --episode 3 --velocity-scale 0.1
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from lerobot.utils.robot_utils import precise_sleep

from lerobot_robot_trlc_dk1.bi_follower import BiDK1Follower, BiDK1FollowerConfig


_stop_requested = False


def _sigint_handler(signum, frame):
    global _stop_requested
    if _stop_requested:
        raise KeyboardInterrupt  # second Ctrl-C forces exit
    _stop_requested = True


# ---------------------------------------------------------------------------
# Port config (shared format with the recorder / port_config.env)
# ---------------------------------------------------------------------------

def load_env_config(env_file: Path) -> dict[str, str]:
    """Load shell-style `export VAR=value` lines from an env file."""
    config: dict[str, str] = {}
    if not env_file.exists():
        return config
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if line.startswith("#") or not line:
            continue
        if line.startswith("export "):
            line = line[7:]
        if "=" in line:
            key, _, value = line.partition("=")
            config[key.strip()] = value.strip().strip('"').strip("'")
    return config


# ---------------------------------------------------------------------------
# Dataset reading
# ---------------------------------------------------------------------------

def load_episode_actions(
    dataset_dir: Path, episode_index: int
) -> tuple[float, list[str], np.ndarray]:
    """Load the recorded action stream for a single episode.

    Returns (fps, action_names, actions) where `actions` has shape (T, 14),
    one commanded target per recorded frame, in `frame_index` order.

    The whole `action` column is read across all data shards and filtered by
    `episode_index`; this is robust to whatever per-file episode packing the
    writer used (the scalar data is only a few MB even for large datasets).
    """
    info_path = dataset_dir / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(
            f"{info_path} not found — is {dataset_dir} a LeRobot v3 dataset root?"
        )
    info = json.loads(info_path.read_text())

    fps = float(info["fps"])
    action_names = list(info["features"]["action"]["names"])

    total_episodes = info.get("total_episodes")
    if total_episodes is not None and not (0 <= episode_index < total_episodes):
        raise ValueError(
            f"Episode {episode_index} out of range — dataset has "
            f"{total_episodes} episodes (0..{total_episodes - 1})."
        )

    data_files = sorted((dataset_dir / "data").glob("**/*.parquet"))
    if not data_files:
        raise FileNotFoundError(f"No data parquet files under {dataset_dir / 'data'}")

    chunks: list[pd.DataFrame] = []
    for f in data_files:
        df = pd.read_parquet(f, columns=["action", "episode_index", "frame_index"])
        sub = df[df["episode_index"] == episode_index]
        if len(sub):
            chunks.append(sub)

    if not chunks:
        raise ValueError(f"No frames found for episode {episode_index} in {dataset_dir}")

    df = pd.concat(chunks).sort_values("frame_index")
    actions = np.stack(df["action"].to_numpy()).astype(np.float64)

    if actions.shape[1] != len(action_names):
        raise ValueError(
            f"Action width {actions.shape[1]} != number of action names "
            f"{len(action_names)} — dataset schema mismatch."
        )

    return fps, action_names, actions


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------

# Cap the ramp's interpolation speed (rad/s). When the start pose is far from
# frame 0 the ramp duration is stretched so the approach never exceeds this —
# well under the RT loop's slew limit (5 rad/s on joints 1-3).
RAMP_SAFE_RATE = 1.5


def _action_dict(action_names: list[str], vec: np.ndarray) -> dict[str, float]:
    return {name: float(v) for name, v in zip(action_names, vec)}


def _joint_keys(action_names: list[str]) -> list[str]:
    """Action keys that are joint angles in radians (excludes the grippers,
    which are normalized 0..1 and on a different scale)."""
    return [k for k in action_names if k.endswith(".pos") and "gripper" not in k]


def check_start_gate(
    start_pose: dict[str, float] | None,
    target0: dict[str, float],
    action_names: list[str],
    max_start_distance: float,
    ramp_time: float,
    force: bool,
) -> tuple[bool, float]:
    """Report the start-pose offset and decide whether playback may proceed.

    Returns (proceed, effective_ramp_time). When the start pose can't be read,
    or any joint is more than `max_start_distance` rad from frame 0, playback
    is refused unless `force` is set. The ramp time is extended so the approach
    to frame 0 stays at or below RAMP_SAFE_RATE.
    """
    if start_pose is None:
        print("[replay] WARNING: could not read current follower pose — "
              "cannot verify start distance.")
        if not force:
            print("[replay] ABORTING — rerun with --force to skip the start-distance check.")
            return False, ramp_time
        print("[replay] --force set: proceeding (RT slew limiter is the only guard).")
        return True, ramp_time

    joint_keys = _joint_keys(action_names)
    print("[replay] start pose -> episode frame 0:")
    max_jd = 0.0
    for k in action_names:
        d = target0[k] - start_pose[k]
        unit = "rad" if k in joint_keys else "   "
        if k in joint_keys:
            max_jd = max(max_jd, abs(d))
        print(f"    {k:<22} {start_pose[k]:+.3f} -> {target0[k]:+.3f}  (Δ {d:+.3f} {unit})")
    print(f"[replay] max joint offset: {max_jd:.3f} rad  (limit {max_start_distance:.3f})")

    if max_jd > max_start_distance and not force:
        print(f"[replay] ABORTING — start pose is {max_jd:.3f} rad from frame 0, "
              f"exceeding --max-start-distance ({max_start_distance:.3f}).\n"
              f"           Move the arms closer to the episode start, raise "
              f"--max-start-distance, or rerun with --force.")
        return False, ramp_time

    needed = max_jd / RAMP_SAFE_RATE
    effective = max(ramp_time, needed)
    if effective > ramp_time + 1e-3:
        print(f"[replay] extending ramp {ramp_time:.1f}s -> {effective:.1f}s "
              f"to keep approach ≤ {RAMP_SAFE_RATE:g} rad/s")
    return True, effective


def _current_pose_action(
    follower: BiDK1Follower, action_names: list[str]
) -> dict[str, float] | None:
    """Read the follower's current `.pos` for every action key, or None."""
    try:
        obs = follower.get_observation()
    except Exception as e:
        print(f"[replay] could not read follower pose for ramp: {e}", file=sys.stderr)
        return None
    missing = [k for k in action_names if k not in obs]
    if missing:
        print(f"[replay] observation missing keys {missing}; skipping ramp", file=sys.stderr)
        return None
    return {k: float(obs[k]) for k in action_names}


def ramp_to(
    follower: BiDK1Follower,
    action_names: list[str],
    target: dict[str, float],
    duration: float,
    start: dict[str, float] | None = None,
    hz: float = 200.0,
) -> None:
    """Smoothly interpolate from the follower's current pose to `target`.

    `start` may be supplied by the caller (e.g. the safety gate already read
    it); otherwise the current pose is read here.
    """
    if start is None:
        start = _current_pose_action(follower, action_names)
    if start is None or duration <= 0:
        # No usable start pose — send the target directly (RT loop's
        # slew-rate limiter still prevents an instantaneous jump).
        follower.send_action(target)
        return

    print(f"[replay] ramping to first frame over {duration:.1f}s ...")
    period = 1.0 / hz
    t0 = time.perf_counter()
    while not _stop_requested:
        t = (time.perf_counter() - t0) / duration
        if t >= 1.0:
            break
        action = {k: start[k] * (1.0 - t) + target[k] * t for k in action_names}
        follower.send_action(action)
        precise_sleep(period)
    follower.send_action(target)


def replay(
    follower: BiDK1Follower,
    action_names: list[str],
    actions: np.ndarray,
    fps: float,
    velocity_scale: float,
) -> None:
    effective_fps = fps * velocity_scale
    period = 1.0 / effective_fps
    n = len(actions)
    duration = n * period

    print(
        f"[replay] playing {n} frames @ {effective_fps:.2f} Hz "
        f"({fps:.1f} Hz × {velocity_scale:g})  ~{duration:.1f}s"
    )

    t_start = time.perf_counter()
    for i in range(n):
        if _stop_requested:
            print("\n[replay] interrupted — stopping.")
            break
        follower.send_action(_action_dict(action_names, actions[i]))

        # Absolute-time pacing so per-frame jitter does not accumulate.
        next_t = t_start + (i + 1) * period
        remaining = next_t - time.perf_counter()
        if remaining > 0:
            precise_sleep(remaining)

        if i % max(1, int(effective_fps)) == 0 or i == n - 1:
            sys.stdout.write(f"\r[replay] frame {i + 1}/{n}\x1b[K")
            sys.stdout.flush()
    sys.stdout.write("\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Replay a recorded LeRobot episode on the DK1 follower arms.",
    )
    parser.add_argument("dataset", type=Path, help="Path to the LeRobot dataset root.")
    parser.add_argument("--episode", type=int, default=0, help="Episode index to replay.")
    parser.add_argument(
        "--velocity-scale", type=float, default=1.0,
        help="Playback speed as a fraction of the dataset fps "
             "(0.1 = 10%% speed / 10× slower). Default 1.0.",
    )
    parser.add_argument(
        "--ramp-time", type=float, default=2.0,
        help="Minimum seconds to smoothly move from the current pose to the "
             "episode's first frame before playback. Auto-extended for large "
             "offsets. Default 2.0.",
    )
    parser.add_argument(
        "--max-start-distance", type=float, default=0.5,
        help="Refuse to start if any joint is more than this many radians from "
             "the episode's first frame (override with --force). Default 0.5.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Skip the start-distance safety gate.",
    )
    parser.add_argument(
        "--port-config", type=Path, default=Path("port_config.env"),
        help="Port config env file (default: port_config.env).",
    )
    parser.add_argument("--left-follower", default=None, help="Left follower port (overrides env).")
    parser.add_argument("--right-follower", default=None, help="Right follower port (overrides env).")
    parser.add_argument(
        "--mode", default="rt_impedance",
        choices=["rt_impedance", "impedance", "pos_vel"],
        help="Follower control mode (default: rt_impedance).",
    )
    parser.add_argument("--gravity-scale", type=float, default=1.0,
                        help="Gravity compensation scale (impedance modes).")
    parser.add_argument("--disable-torque-on-disconnect", action="store_true",
                        help="Disable motor torque when the script exits.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Load and summarize the episode without connecting to hardware.")
    args = parser.parse_args()

    if args.velocity_scale <= 0:
        parser.error("--velocity-scale must be > 0")
    if args.velocity_scale > 1.0:
        print(f"[replay] WARNING: --velocity-scale {args.velocity_scale:g} > 1.0 "
              "replays FASTER than recorded.", file=sys.stderr)

    # --- Load episode -----------------------------------------------------
    fps, action_names, actions = load_episode_actions(args.dataset, args.episode)
    print(f"[replay] dataset:  {args.dataset}")
    print(f"[replay] episode:  {args.episode}  ({len(actions)} frames)")
    print(f"[replay] fps:      {fps:g}  ->  effective {fps * args.velocity_scale:g} Hz")
    print(f"[replay] action:   {len(action_names)} DoF")

    if args.dry_run:
        print("[replay] dry-run: not connecting to hardware.")
        first = _action_dict(action_names, actions[0])
        print("[replay] first frame:")
        for k, v in first.items():
            print(f"    {k:<22} {v:+.4f}")
        return

    # --- Resolve follower ports ------------------------------------------
    cfg = load_env_config(args.port_config)
    left_follower = args.left_follower or cfg.get("LEFT_FOLLOWER", os.environ.get("LEFT_FOLLOWER", ""))
    right_follower = args.right_follower or cfg.get("RIGHT_FOLLOWER", os.environ.get("RIGHT_FOLLOWER", ""))
    if not left_follower or not right_follower:
        parser.error(
            "Follower ports not set. Provide --left-follower/--right-follower or a "
            "port_config.env with LEFT_FOLLOWER/RIGHT_FOLLOWER."
        )

    print(f"[replay] follower: L={left_follower}  R={right_follower}")
    print(f"[replay] mode:     {args.mode}  gravity_scale={args.gravity_scale}")
    print()

    follower_config = BiDK1FollowerConfig(
        left_arm_port=left_follower,
        right_arm_port=right_follower,
        control_mode=args.mode,
        gravity_comp_scale=args.gravity_scale,
        disable_torque_on_disconnect=args.disable_torque_on_disconnect,
        cameras={},
    )

    signal.signal(signal.SIGINT, _sigint_handler)

    follower = BiDK1Follower(follower_config)
    print("[replay] connecting follower arms (ensure E-Stop is released)...")
    follower.connect()
    print("[replay] follower connected.")
    print()

    try:
        # Safety gate: read the current pose, report the offset to frame 0,
        # and refuse / extend the ramp as needed before any motion.
        start_pose = _current_pose_action(follower, action_names)
        target0 = _action_dict(action_names, actions[0])
        proceed, ramp_time = check_start_gate(
            start_pose, target0, action_names,
            args.max_start_distance, args.ramp_time, args.force,
        )
        if proceed and not _stop_requested:
            ramp_to(follower, action_names, target0, ramp_time, start=start_pose)
            if not _stop_requested:
                replay(follower, action_names, actions, fps, args.velocity_scale)
    except (KeyboardInterrupt, OSError):
        pass
    finally:
        print("[replay] stopping follower...")
        for arm in (follower.left_arm, follower.right_arm):
            if getattr(arm, "_robot", None) is not None:
                try:
                    arm._robot.disconnect()
                except Exception as e:
                    print(f"  follower disconnect error: {e}")
        print("[replay] done.")


if __name__ == "__main__":
    main()
