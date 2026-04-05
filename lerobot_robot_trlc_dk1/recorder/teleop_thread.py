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
decoupled from the recording rate (30 fps). The latest action is stored
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

    def __init__(self, follower, leader, target_hz: float = 250.0,
                 auto_home_threshold: float = 0.0):
        self.follower = follower
        self.leader = leader
        self.target_hz = target_hz

        # Auto-home: when active and all joint positions within threshold
        # of zero, ramp commands from leader to zero over ramp_duration.
        # Uses hysteresis: operator must first move away before ramp activates.
        self._auto_home_threshold = auto_home_threshold
        self._auto_home_ramp_duration = 1.0  # seconds for ramp in/out
        self._auto_home_active = False
        self._auto_home_departed = False
        self._auto_home_ramp_start: float = 0.0  # perf_counter when ramp began
        self._auto_home_ramp_action: dict[str, float] | None = None  # leader snapshot at ramp start
        self._auto_home_ramping = False
        # Ramp-out: smooth return from zero back to leader after episode save
        self._auto_home_releasing = False
        self._auto_home_release_start: float = 0.0

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
    def auto_home_active(self) -> bool:
        """Whether auto-home snapping is currently active."""
        return self._auto_home_active

    @auto_home_active.setter
    def auto_home_active(self, value: bool):
        if not value:
            self._auto_home_ramping = False
            self._auto_home_releasing = False
        self._auto_home_active = value

    def release_auto_home(self):
        """Start smooth ramp from zero back to leader (call after episode save).

        If the robot was held at zero (ramping), starts a smooth release ramp.
        If not (e.g. gesture-triggered end away from home), just resets state
        so auto-home can re-arm for the next episode.
        """
        if self._auto_home_ramping:
            # Was holding at zero — smooth ramp back to leader
            self._auto_home_ramping = False
            self._auto_home_releasing = True
            self._auto_home_release_start = time.perf_counter()
            logger.debug("Auto-home: releasing (ramp back to teleop)")
        # Reset departure so auto-home must re-arm
        self._auto_home_departed = False

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

                # Auto-home release: smooth ramp from zero back to leader
                if self._auto_home_releasing:
                    now = time.perf_counter()
                    t = min(1.0, (now - self._auto_home_release_start)
                            / self._auto_home_ramp_duration)
                    if t >= 1.0:
                        self._auto_home_releasing = False
                        logger.debug("Auto-home: release complete, teleop resumed")
                        # action is already the leader — pass through
                    else:
                        # Lerp: cmd = zero * (1-t) + leader * t = leader * t
                        action = {
                            k: (v * t
                                if ".pos" in k and "gripper" not in k
                                else v)
                            for k, v in action.items()
                        }

                # Auto-home: when active and leader joints are near zero,
                # ramp follower commands from leader to zero over ramp_duration.
                # Once ramped, hold at zero. Cancel if leader exits threshold.
                # Skip while releasing (ramp-out has priority).
                elif (self._auto_home_active
                        and self._auto_home_threshold > 0):
                    threshold = self._auto_home_threshold
                    max_joint_error = max(
                        (abs(v) for k, v in action.items()
                         if ".pos" in k and "gripper" not in k),
                        default=0.0,
                    )

                    if not self._auto_home_departed:
                        # Must move away from zero first — fixed at 0.5 rad
                        # (matches rest pose departure), independent of threshold.
                        if max_joint_error > 0.5:
                            self._auto_home_departed = True
                            logger.debug("Auto-home: departed (max_err=%.2f rad)",
                                        max_joint_error)
                    elif max_joint_error >= threshold:
                        # Leader moved outside threshold — cancel ramp
                        if self._auto_home_ramping:
                            self._auto_home_ramping = False
                            logger.debug("Auto-home: cancelled (leader left threshold)")
                    else:
                        # Leader within threshold
                        now = time.perf_counter()
                        if not self._auto_home_ramping:
                            # Start ramp: snapshot current leader as start point
                            self._auto_home_ramping = True
                            self._auto_home_ramp_start = now
                            self._auto_home_ramp_action = {
                                k: v for k, v in action.items()
                                if ".pos" in k and "gripper" not in k
                            }
                            logger.debug("Auto-home: ramp started (max_err=%.3f rad)",
                                        max_joint_error)

                        # Ramp progress: 0 → 1 over ramp_duration, then hold at zero
                        t = min(1.0, (now - self._auto_home_ramp_start)
                                / self._auto_home_ramp_duration)
                        # Lerp: cmd = start * (1 - t); at t=1 this is zero (hold)
                        start = self._auto_home_ramp_action
                        action = {
                            k: (start[k] * (1.0 - t)
                                if k in start
                                else v)
                            for k, v in action.items()
                        }

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
