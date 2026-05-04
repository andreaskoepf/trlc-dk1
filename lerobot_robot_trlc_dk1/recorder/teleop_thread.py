# Copyright 2025 The Robot Learning Company UG (haftungsbeschränkt). All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Always-on high-rate teleop thread.

Reads leader arm positions and commands follower arms at ~200 Hz,
decoupled from the recording rate (60 fps). The latest action is stored
for the recorder thread to snapshot via atomic reference read.
"""

from __future__ import annotations

import logging
import threading
import time

from lerobot.utils.robot_utils import precise_sleep

logger = logging.getLogger(__name__)


class TeleopThread:
    """Always-on teleop: reads leader arms, commands follower arms.

    Runs at ``target_hz`` (~200 Hz by default), limited by Dynamixel
    sync_read latency. Stores latest action for the recorder thread
    to snapshot. Never touches cameras, queues, or disk I/O.

    The latest action dict is shared with the recorder thread via an
    atomic reference swap (safe under CPython's GIL since
    ``leader.get_action()`` returns a fresh dict each call).
    """

    ARMS: tuple[str, ...] = ("left", "right")

    def __init__(self, follower, leader, target_hz: float = 250.0,
                 auto_home_threshold: float = 0.0,
                 startup_sync_duration: float = 1.5):
        self.follower = follower
        self.leader = leader
        self.target_hz = target_hz

        # Startup sync: at thread start, capture the follower's actual pose
        # and ramp linearly from there to the (live) leader pose over
        # ``startup_sync_duration`` seconds, instead of snapping the follower
        # to the leader on the very first send_action. Set to 0.0 to disable
        # (preserves the legacy snap behaviour).
        self._startup_sync_duration = max(0.0, float(startup_sync_duration))
        self._startup_sync_active = self._startup_sync_duration > 0.0
        self._startup_sync_t0: float = 0.0
        self._startup_sync_start_action: dict[str, float] | None = None

        # Auto-home: when active and an arm's joint positions fall within
        # ``auto_home_threshold`` of zero, ramp that arm's commands from
        # leader to zero over ramp_duration. Per-arm state so one arm can
        # hold at zero while the other follows the leader. Hysteresis:
        # operator must first move away before the ramp arms.
        self._auto_home_threshold = auto_home_threshold
        self._auto_home_ramp_duration = 1.0  # seconds for ramp in/out
        self._auto_home_active = False

        # Per-arm state dicts, keyed by arm name ("left", "right").
        self._auto_home_departed: dict[str, bool] = {a: False for a in self.ARMS}
        self._auto_home_ramping: dict[str, bool] = {a: False for a in self.ARMS}
        self._auto_home_ramp_start: dict[str, float] = {a: 0.0 for a in self.ARMS}
        # Leader snapshot (arm's joint-pos keys only) captured when ramp starts.
        self._auto_home_ramp_action: dict[str, dict[str, float] | None] = {
            a: None for a in self.ARMS
        }
        # Ramp-out: smooth return from "held" pose back to leader. Held is
        # whatever the follower was last commanded to (not necessarily zero,
        # since release may fire mid-ramp when leader exits threshold).
        self._auto_home_releasing: dict[str, bool] = {a: False for a in self.ARMS}
        self._auto_home_release_start: dict[str, float] = {a: 0.0 for a in self.ARMS}
        self._auto_home_release_from: dict[str, dict[str, float] | None] = {
            a: None for a in self.ARMS
        }

        # Latest action — read by recorder thread (atomic reference swap).
        self._latest_action: dict[str, float] | None = None

        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._actual_hz: float = 0.0
        self._jitter_count: int = 0

    @property
    def latest_action(self) -> dict[str, float] | None:
        """Latest action dict (atomic read, may be one iteration stale)."""
        return self._latest_action

    @property
    def actual_hz(self) -> float:
        """Exponential-moving-average of actual loop rate."""
        return self._actual_hz

    @property
    def startup_sync_active(self) -> bool:
        """True while the startup ramp from follower-current-pose to leader is in progress."""
        return self._startup_sync_active

    @property
    def auto_home_active(self) -> bool:
        """Whether auto-home snapping is currently active."""
        return self._auto_home_active

    @property
    def auto_home_ramping(self) -> bool:
        """True while any arm is ramping toward / holding at zero."""
        return any(self._auto_home_ramping.values())

    @property
    def auto_home_at_rest(self) -> bool:
        """True only when ALL arms have completed their ramp to zero."""
        now = time.perf_counter()
        for arm in self.ARMS:
            if not self._auto_home_ramping[arm]:
                return False
            if now - self._auto_home_ramp_start[arm] < self._auto_home_ramp_duration:
                return False
        return True

    @auto_home_active.setter
    def auto_home_active(self, value: bool):
        if not value:
            # Clear ramping flags; keep any in-flight release so the
            # smooth ramp-back to leader still completes.
            for arm in self.ARMS:
                self._auto_home_ramping[arm] = False
        self._auto_home_active = value

    def reset_auto_home(self):
        """Re-arm auto-home for a new episode (clears departure + ramp flags).

        Does not touch in-flight releases — they must complete uninterrupted.
        """
        for arm in self.ARMS:
            self._auto_home_departed[arm] = False
            self._auto_home_ramping[arm] = False

    def release_auto_home(self):
        """Start smooth ramp back to leader for every ramping arm.

        Call after episode save. Each ramping arm gets a per-arm release
        that lerps from its current held pose (not necessarily zero — it
        may have been mid-ramp) back to the current leader command.
        """
        now = time.perf_counter()
        for arm in self.ARMS:
            if self._auto_home_ramping[arm]:
                self._begin_release(arm, now)
            # Re-arm departure hysteresis for the next episode
            self._auto_home_departed[arm] = False

    def _begin_release(self, arm: str, now: float):
        """Snapshot the currently-commanded held pose and start the release ramp."""
        start = self._auto_home_ramp_action[arm] or {}
        ramp_t = min(1.0, (now - self._auto_home_ramp_start[arm])
                     / self._auto_home_ramp_duration)
        # Wherever the follower actually is right now: start * (1 - ramp_t).
        # At ramp_t=1 this is zero; at ramp_t<1 it's partway to zero.
        held = {k: v * (1.0 - ramp_t) for k, v in start.items()}
        self._auto_home_ramping[arm] = False
        self._auto_home_releasing[arm] = True
        self._auto_home_release_start[arm] = now
        self._auto_home_release_from[arm] = held
        logger.debug("Auto-home[%s]: releasing from ramp_t=%.2f", arm, ramp_t)

    @staticmethod
    def _is_arm_joint(key: str, arm: str) -> bool:
        """True for arm-prefixed non-gripper position keys (e.g. ``left_joint_1.pos``)."""
        return (key.startswith(f"{arm}_")
                and ".pos" in key
                and "gripper" not in key)

    def _apply_auto_home(self, arm: str, action: dict, now: float) -> dict:
        """Apply per-arm auto-home / release to ``action``. Returns modified dict."""
        # Release ramp: smooth lerp from held snapshot back to current leader.
        if self._auto_home_releasing[arm]:
            t = min(1.0, (now - self._auto_home_release_start[arm])
                    / self._auto_home_ramp_duration)
            if t >= 1.0:
                self._auto_home_releasing[arm] = False
                self._auto_home_release_from[arm] = None
                logger.debug("Auto-home[%s]: release complete", arm)
                return action  # leader pass-through
            held = self._auto_home_release_from[arm] or {}
            return {
                k: (held[k] * (1.0 - t) + v * t
                    if self._is_arm_joint(k, arm) and k in held
                    else v)
                for k, v in action.items()
            }

        if not (self._auto_home_active and self._auto_home_threshold > 0):
            return action

        arm_joints = {k: v for k, v in action.items() if self._is_arm_joint(k, arm)}
        if not arm_joints:
            return action
        max_err = max(abs(v) for v in arm_joints.values())
        threshold = self._auto_home_threshold

        if not self._auto_home_departed[arm]:
            # Hysteresis: arm must first move away (fixed at 0.5 rad, matches
            # rest-pose departure threshold — independent of ``threshold``).
            if max_err > 0.5:
                self._auto_home_departed[arm] = True
                logger.debug("Auto-home[%s]: departed (max_err=%.2f rad)",
                             arm, max_err)
            return action

        if max_err >= threshold:
            # Leader outside threshold — if we were ramping, start a smooth
            # release from the current held pose instead of jumping to leader.
            if self._auto_home_ramping[arm]:
                self._begin_release(arm, now)
                held = self._auto_home_release_from[arm] or {}
                # First release frame: t=0, command = held (no discontinuity).
                return {
                    k: (held[k] if self._is_arm_joint(k, arm) and k in held else v)
                    for k, v in action.items()
                }
            return action

        # Leader within threshold — engage / continue ramp toward zero
        if not self._auto_home_ramping[arm]:
            self._auto_home_ramping[arm] = True
            self._auto_home_ramp_start[arm] = now
            self._auto_home_ramp_action[arm] = dict(arm_joints)
            logger.debug("Auto-home[%s]: ramp started (max_err=%.3f rad)",
                         arm, max_err)

        t = min(1.0, (now - self._auto_home_ramp_start[arm])
                / self._auto_home_ramp_duration)
        start = self._auto_home_ramp_action[arm] or {}
        return {
            k: (start[k] * (1.0 - t)
                if self._is_arm_joint(k, arm) and k in start
                else v)
            for k, v in action.items()
        }

    @property
    def jitter_count(self) -> int:
        """Number of cycles that exceeded 2× target period."""
        return self._jitter_count

    def start(self):
        """Start the teleop thread."""
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="teleop"
        )
        self._thread.start()
        logger.info("Teleop thread started (target %.0f Hz)", self.target_hz)

    def stop(self):
        """Signal the teleop thread to stop and wait for it."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        logger.info("Teleop thread stopped")

    def _run(self):
        period = 1.0 / self.target_hz
        jitter_threshold = period * 2.0  # warn when cycle > 2× target
        hz_filter = 0.0
        consecutive_errors = 0
        last_jitter_log = 0.0  # throttle: max 1 warning per second

        while not self._stop_event.is_set():
            t0 = time.perf_counter()

            try:
                action = self.leader.get_action()

                # Startup sync: on the first cycle, snapshot the follower's
                # actual pose and ramp linearly from there to the (live) leader
                # pose over startup_sync_duration. Replaces the legacy "snap
                # follower to leader on first send_action" behaviour. Per-arm
                # auto-home is gated below so it does not compose with this.
                if self._startup_sync_active:
                    if self._startup_sync_start_action is None:
                        try:
                            obs = self.follower.get_observation()
                        except Exception:
                            logger.exception(
                                "Startup sync: follower.get_observation() failed; "
                                "disabling ramp (legacy snap behaviour)"
                            )
                            self._startup_sync_active = False
                            obs = None
                        if obs is not None:
                            self._startup_sync_start_action = {
                                k: float(v) for k, v in obs.items()
                                if isinstance(k, str) and k.endswith(".pos")
                                and isinstance(v, (int, float))
                            }
                            self._startup_sync_t0 = time.perf_counter()
                            logger.info(
                                "Startup sync: ramping follower to leader over %.2fs "
                                "(captured %d .pos joints)",
                                self._startup_sync_duration,
                                len(self._startup_sync_start_action),
                            )

                    if self._startup_sync_active and self._startup_sync_start_action is not None:
                        t = (time.perf_counter() - self._startup_sync_t0) / self._startup_sync_duration
                        if t >= 1.0:
                            self._startup_sync_active = False
                            self._startup_sync_start_action = None
                            logger.info("Startup sync: complete; pass-through resumed")
                            # action stays as the live leader pose
                        else:
                            start = self._startup_sync_start_action
                            # Ramp ALL .pos keys (joints AND grippers) so
                            # nothing snaps. Non-.pos values pass through.
                            action = {
                                k: (start[k] * (1.0 - t) + v * t
                                    if k in start
                                    else v)
                                for k, v in action.items()
                            }

                # Per-arm auto-home (release / engage / hold). Gated off
                # while the startup-sync ramp is still running so the two
                # don't compose. Each arm runs its own state machine, so
                # one arm can hold at zero while the other teleops.
                if not self._startup_sync_active:
                    now = time.perf_counter()
                    for arm in self.ARMS:
                        action = self._apply_auto_home(arm, action, now)

                self.follower.send_action(action)
                self._latest_action = action  # atomic reference swap
                if consecutive_errors > 0:
                    logger.info("Teleop recovered after %d errors", consecutive_errors)
                consecutive_errors = 0
            except ConnectionError:
                # Transient serial bus error (common under USB load)
                consecutive_errors += 1
                if consecutive_errors <= 3:
                    logger.warning("Teleop: serial read error (%d)", consecutive_errors)
                elif consecutive_errors % 100 == 0:
                    logger.error("Teleop: %d consecutive serial errors", consecutive_errors)
                time.sleep(0.005)
                continue
            except Exception:
                consecutive_errors += 1
                logger.exception("Teleop loop error")
                time.sleep(0.01)
                continue

            elapsed = time.perf_counter() - t0
            remaining = period - elapsed
            if remaining > 0:
                precise_sleep(remaining)

            dt = time.perf_counter() - t0
            hz = 1.0 / dt if dt > 0 else 0
            hz_filter = 0.95 * hz_filter + 0.05 * hz
            self._actual_hz = hz_filter

            # Jitter detection: flag cycles that exceed 2× target period
            if dt > jitter_threshold:
                self._jitter_count += 1
                now = time.monotonic()
                if now - last_jitter_log > 1.0:
                    logger.warning(
                        "Teleop jitter: %.1f ms (target %.1f ms, total %d)",
                        dt * 1000, period * 1000, self._jitter_count,
                    )
                    last_jitter_log = now
