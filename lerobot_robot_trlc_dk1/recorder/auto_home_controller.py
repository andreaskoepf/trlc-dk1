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

"""Self-contained auto-home controller.

Manages the smooth ramp-to-zero when the leader arms return near the
home position, and the release ramp back to leader after an episode ends.

Four phases:
  1. DEPART  — operator must move >0.5 rad from zero (hysteresis)
  2. RAMP    — leader within threshold → lerp from leader to zero over 1s
  3. HOLD    — at zero, waiting for settle detection to trigger episode end
  4. RELEASE — after episode end, lerp from zero back to leader over 1s

Previously this logic was split across TeleopThread (ramp state machine),
RecorderThread (settle detection), and dk1_recorder.py (activation).
"""

from __future__ import annotations

import logging
import time

import numpy as np

logger = logging.getLogger(__name__)

# Departure threshold — must move this far from zero before auto-home
# can activate. Fixed at 0.5 rad, independent of the ramp threshold.
_DEPARTURE_THRESHOLD = 0.5

# Settle detection constants for auto-home at-rest check.
_SETTLE_POS_TOLERANCE = 0.1    # rad — tight tolerance against absolute zero
_SETTLE_VEL_TOLERANCE = 0.1    # rad/s


class AutoHomeController:
    """Auto-home ramp: smooth follower-to-zero when leader near zero.

    Thread safety: ``apply()`` is called from the teleop thread (~250 Hz),
    ``check_settle()`` from the recorder thread (~60 Hz), and property
    setters from the main event loop (~20 Hz). All state mutations are
    simple scalar writes (GIL-safe atomic under CPython).

    Args:
        threshold: Max leader joint error from zero to trigger ramp (radians).
            Set to 0 to disable auto-home entirely.
        ramp_duration: Duration of ramp in/out in seconds.
    """

    def __init__(self, threshold: float, ramp_duration: float = 1.0):
        self._threshold = threshold
        self._ramp_duration = ramp_duration

        # Activation — only active during RECORDING state
        self._enabled = False

        # Phase state
        self._departed = False
        self._ramping = False
        self._ramp_start: float = 0.0
        self._ramp_action: dict[str, float] | None = None  # leader snapshot at ramp start

        # Release phase (ramp from zero back to leader)
        self._releasing = False
        self._release_start: float = 0.0

        # Settle detection (for auto-home at-rest check)
        self._settle_count: int = 0

    # -- Properties (read by RecorderThread and event loop) -----------------

    @property
    def threshold(self) -> float:
        return self._threshold

    @property
    def enabled(self) -> bool:
        """Whether auto-home is currently active (only during RECORDING)."""
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool):
        if not value:
            self._ramping = False
            # Don't clear _releasing — the release ramp must continue
            # even after auto_home is deactivated between episodes.
        self._enabled = value

    @property
    def ramping(self) -> bool:
        """True while ramp-to-zero is in progress (including hold at zero)."""
        return self._ramping

    @property
    def at_rest(self) -> bool:
        """True when ramp completed and robot is held at zero."""
        if not self._ramping:
            return False
        elapsed = time.perf_counter() - self._ramp_start
        return elapsed >= self._ramp_duration

    # -- Teleop integration (called every tick at ~250 Hz) ------------------

    def apply(self, action: dict[str, float]) -> dict[str, float]:
        """Transform action through auto-home logic.

        Called every teleop tick. Returns the (possibly modified) action.
        Handles release ramp, active ramp, and departure detection.
        """
        # Release ramp: smooth return from zero back to leader
        if self._releasing:
            now = time.perf_counter()
            t = min(1.0, (now - self._release_start) / self._ramp_duration)
            if t >= 1.0:
                self._releasing = False
                logger.debug("Auto-home: release complete, teleop resumed")
                return action
            # Lerp: cmd = zero * (1-t) + leader * t = leader * t
            return {
                k: (v * t if ".pos" in k and "gripper" not in k else v)
                for k, v in action.items()
            }

        # Active ramp logic (only when enabled and threshold > 0)
        if not self._enabled or self._threshold <= 0:
            return action

        max_joint_error = max(
            (abs(v) for k, v in action.items()
             if ".pos" in k and "gripper" not in k),
            default=0.0,
        )

        if not self._departed:
            # Must move away from zero first
            if max_joint_error > _DEPARTURE_THRESHOLD:
                self._departed = True
                logger.debug("Auto-home: departed (max_err=%.2f rad)",
                             max_joint_error)
            return action

        if max_joint_error >= self._threshold:
            # Leader moved outside threshold — cancel ramp
            if self._ramping:
                self._ramping = False
                logger.debug("Auto-home: cancelled (leader left threshold)")
            return action

        # Leader within threshold — ramp to zero
        now = time.perf_counter()
        if not self._ramping:
            # Start ramp: snapshot current leader as start point
            self._ramping = True
            self._ramp_start = now
            self._ramp_action = {
                k: v for k, v in action.items()
                if ".pos" in k and "gripper" not in k
            }
            logger.debug("Auto-home: ramp started (max_err=%.3f rad)",
                         max_joint_error)

        # Ramp progress: 0 → 1 over ramp_duration, then hold at zero
        t = min(1.0, (now - self._ramp_start) / self._ramp_duration)
        start = self._ramp_action
        return {
            k: (start[k] * (1.0 - t) if k in start else v)
            for k, v in action.items()
        }

    # -- Episode boundary methods (called from event loop) ------------------

    def release(self):
        """Start smooth ramp from zero back to leader (call after episode save).

        If the robot was held at zero (ramping), starts a smooth release
        ramp. Otherwise just resets departure for the next episode.
        """
        if self._ramping:
            self._ramping = False
            self._releasing = True
            self._release_start = time.perf_counter()
            logger.debug("Auto-home: releasing (ramp back to teleop)")
        self._departed = False

    def reset_departure(self):
        """Reset departure flag for a new episode.

        Call at episode start so the operator must move away from zero
        before auto-home can activate again.
        """
        self._departed = False
        self._ramping = False
        self._settle_count = 0

    # -- Settle detection (called from recorder thread at ~60 Hz) -----------

    def check_settle(self, obs_state: np.ndarray, joint_pos_indices,
                     vel_indices, fps: int) -> bool:
        """Check if robot has settled at zero after ramp.

        Uses tight tolerance (0.1 rad position, 0.1 rad/s velocity) and
        short settle window (~100ms at recording FPS).

        Args:
            obs_state: Full 40-element observation state vector.
            joint_pos_indices: Indices of joint positions (no grippers) in obs_state.
            vel_indices: Indices of joint velocities in obs_state.
            fps: Recording FPS (for settle frame count).

        Returns:
            True when settled at zero for long enough.
        """
        pos_err = float(np.max(np.abs(obs_state[joint_pos_indices])))
        max_vel = float(np.max(np.abs(obs_state[vel_indices])))

        if pos_err < _SETTLE_POS_TOLERANCE and max_vel < _SETTLE_VEL_TOLERANCE:
            self._settle_count += 1
        else:
            self._settle_count = 0

        settle_target = max(1, fps // 10)  # ~100ms
        if self._settle_count >= settle_target:
            logger.info(
                "Auto-home settled (pos_err=%.3f rad, max_vel=%.3f rad/s, "
                "%d frames)",
                pos_err, max_vel, self._settle_count,
            )
            return True

        return False

    def reset_settle(self):
        """Reset settle counter (call when ramp is in progress but not done)."""
        self._settle_count = 0
