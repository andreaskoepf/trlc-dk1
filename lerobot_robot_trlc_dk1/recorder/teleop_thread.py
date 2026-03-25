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

    def __init__(self, follower, leader, target_hz: float = 250.0):
        self.follower = follower
        self.leader = leader
        self.target_hz = target_hz

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
