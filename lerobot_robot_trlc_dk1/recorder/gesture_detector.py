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

"""Gripper double-close gesture detector for episode boundary signaling.

Operates on the **leader** gripper position (what the operator squeezes,
not the follower which lags behind). A double-close is detected when the
operator fully closes the gripper, opens it, and closes it again within
a configurable time window — analogous to a mouse double-click.
"""

from __future__ import annotations

import time


class GripperGestureDetector:
    """Detect double-close gripper gesture.

    Gripper position convention: 0 = fully open, 1 = fully closed.

    A "close" event is a rising edge: the gripper crosses threshold_close
    while previously below threshold_open. Two close events within
    ``double_close_window_s`` trigger a detection.

    Args:
        threshold_close: Position above which the gripper counts as closed.
        threshold_open: Position below which the gripper counts as open
            (hysteresis band to avoid spurious edge detection).
        double_close_window_s: Maximum time between two close events
            for a double-close detection.
    """

    def __init__(
        self,
        threshold_close: float = 0.85,
        threshold_open: float = 0.3,
        double_close_window_s: float = 0.8,
    ):
        self.threshold_close = threshold_close
        self.threshold_open = threshold_open
        self.double_close_window_s = double_close_window_s
        self._reset()

    def _reset(self):
        self._was_closed: bool = False
        self._last_close_time: float | None = None
        self._close_count: int = 0

    def update(self, gripper_pos: float) -> bool:
        """Feed a gripper position sample.

        Args:
            gripper_pos: Current gripper position (0=open, 1=closed).

        Returns:
            True when a double-close gesture is detected.
        """
        is_closed = gripper_pos >= self.threshold_close
        now = time.monotonic()

        # Detect rising edge: gripper just closed
        if is_closed and not self._was_closed:
            if (
                self._last_close_time is not None
                and now - self._last_close_time < self.double_close_window_s
            ):
                self._close_count += 1
                if self._close_count >= 2:
                    self._reset()
                    return True  # Double-close detected!
            else:
                self._close_count = 1
            self._last_close_time = now

        # Track open/closed state (hysteresis)
        if gripper_pos <= self.threshold_open:
            self._was_closed = False
        if is_closed:
            self._was_closed = True

        # Expire stale single-close
        if (
            self._last_close_time is not None
            and now - self._last_close_time > self.double_close_window_s * 1.5
        ):
            self._close_count = 0

        return False
