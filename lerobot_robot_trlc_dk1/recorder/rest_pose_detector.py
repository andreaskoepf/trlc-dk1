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

"""Auto-detect episode end when arms return to resting position.

Captures a reference "rest pose" at the start of each episode, then
monitors whether the follower has returned to that pose with grippers
open and joints settled (low velocity). Uses a debounce window to
avoid false triggers when passing through the rest position mid-task.
"""

from __future__ import annotations

import logging

import numpy as np

from lerobot_robot_trlc_dk1.recorder.recorder_thread import OBS_STATE_KEYS

logger = logging.getLogger(__name__)

# Index maps into the 40-element observation state vector
_POS_INDICES = [i for i, k in enumerate(OBS_STATE_KEYS) if ".pos" in k]
_VEL_INDICES = [i for i, k in enumerate(OBS_STATE_KEYS) if ".vel" in k]
_JOINT_POS_INDICES = [i for i, k in enumerate(OBS_STATE_KEYS)
                      if ".pos" in k and "gripper" not in k]
_GRIPPER_POS_INDICES = [i for i, k in enumerate(OBS_STATE_KEYS)
                        if "gripper.pos" in k]


class RestPoseDetector:
    """Detect when the robot has returned to its resting position.

    Triggers when ALL of the following hold for ``settle_frames``
    consecutive frames:
    - All joint positions within ``pos_tolerance_rad`` of the rest pose
    - Both grippers below ``gripper_open_threshold`` (fully open)
    - All joint velocities below ``vel_tolerance_rad_s``

    Args:
        pos_tolerance_rad: Max joint position deviation from rest (radians).
        vel_tolerance_rad_s: Max joint velocity magnitude (rad/s).
        gripper_open_threshold: Gripper position below this = "open" (0=open, 1=closed).
        settle_frames: How many consecutive frames the condition must hold.
        min_episode_frames: Don't trigger before this many frames into the episode.
        fps: Recording FPS (for logging only).
    """

    def __init__(
        self,
        pos_tolerance_rad: float = 0.25,       # ~14 degrees — your actual return error is ~0.15
        departure_threshold_rad: float = 0.5,  # ~29 degrees — must move this far to arm detection
        vel_tolerance_rad_s: float = 0.8,       # impedance controller oscillates during settle
        gripper_open_threshold: float = 0.25,   # grippers must be mostly open
        settle_frames: int = 30,                # ~1 second at 30fps
        min_episode_frames: int = 90,           # ~3 seconds minimum episode
        fps: int = 30,
    ):
        self.pos_tolerance_rad = pos_tolerance_rad
        self.departure_threshold_rad = departure_threshold_rad
        self.vel_tolerance_rad_s = vel_tolerance_rad_s
        self.gripper_open_threshold = gripper_open_threshold
        self.settle_frames = settle_frames
        self.min_episode_frames = min_episode_frames
        self.fps = fps

        self._rest_pose: np.ndarray | None = None  # joint positions at episode start
        self._settle_count: int = 0
        self._has_departed: bool = False  # True once robot moved away from rest
        self._enabled: bool = True

    def capture_rest_pose(self, obs_state: np.ndarray):
        """Snapshot the current joint positions as the rest pose.

        Call this at the start of each episode (first frame).
        """
        self._rest_pose = obs_state[_JOINT_POS_INDICES].copy()
        self._settle_count = 0
        self._has_departed = False
        logger.debug(
            "Rest pose captured (departure threshold=%.2f rad): %s",
            self.departure_threshold_rad,
            np.array2string(self._rest_pose, precision=2, separator=", "),
        )

    def update(self, obs_state: np.ndarray, frame_index: int) -> bool:
        """Feed an observation state vector. Returns True when rest pose detected.

        Args:
            obs_state: float32[40] observation state vector.
            frame_index: Current frame index within the episode.

        Returns:
            True when the robot has settled at rest pose for long enough.
        """
        if not self._enabled or self._rest_pose is None:
            return False

        current_pos = obs_state[_JOINT_POS_INDICES]
        pos_error = np.abs(current_pos - self._rest_pose)
        max_pos_error = float(np.max(pos_error))

        # Hysteresis: robot must first DEPART from the rest region before
        # auto-end can activate. This prevents triggering when the operator
        # is still at/near the start position at the beginning of the episode.
        if not self._has_departed:
            if max_pos_error > self.departure_threshold_rad:
                self._has_departed = True
                logger.info(
                    "Rest pose: DEPARTED at frame %d (max_error=%.2f rad > threshold %.2f)",
                    frame_index, max_pos_error, self.departure_threshold_rad,
                )
            else:
                # Log periodically so we can see values are updating
                if frame_index % 100 == 0:
                    logger.info(
                        "Rest pose: waiting for departure at frame %d "
                        "(max_error=%.3f rad, need > %.2f)",
                        frame_index, max_pos_error, self.departure_threshold_rad,
                    )
            return False

        # Don't trigger too early in the episode
        if frame_index < self.min_episode_frames:
            self._settle_count = 0
            return False

        # Check joint positions
        pos_ok = np.all(pos_error < self.pos_tolerance_rad)

        # Check grippers are open
        gripper_pos = obs_state[_GRIPPER_POS_INDICES]
        grippers_ok = np.all(gripper_pos < self.gripper_open_threshold)

        # Check velocities are low
        velocities = obs_state[_VEL_INDICES]
        vel_ok = np.all(np.abs(velocities) < self.vel_tolerance_rad_s)

        # Log which conditions are failing (every 100 frames after departure)
        if frame_index % 100 == 0:
            worst_joint_idx = int(np.argmax(pos_error))
            worst_joint = [k for k in OBS_STATE_KEYS if ".pos" in k and "gripper" not in k][worst_joint_idx]
            logger.info(
                "Rest pose check frame %d: pos=%s(%.3f/%s) grip=%s(%.2f,%.2f) vel=%s(%.3f) | %s",
                frame_index,
                "OK" if pos_ok else "FAIL", float(np.max(pos_error)), worst_joint,
                "OK" if grippers_ok else "FAIL",
                float(gripper_pos[0]),
                float(gripper_pos[1]) if len(gripper_pos) > 1 else 0,
                "OK" if vel_ok else "FAIL", float(np.max(np.abs(velocities))),
                "ALL OK" if (pos_ok and grippers_ok and vel_ok) else "NOT MET",
            )

        if pos_ok and grippers_ok and vel_ok:
            self._settle_count += 1
            if self._settle_count == 1:
                logger.info(
                    "Rest pose: settling started at frame %d "
                    "(max_pos_err=%.3f, max_vel=%.3f, grippers=%.2f/%.2f)",
                    frame_index,
                    float(np.max(pos_error)),
                    float(np.max(np.abs(velocities))),
                    float(gripper_pos[0]),
                    float(gripper_pos[1]) if len(gripper_pos) > 1 else 0,
                )
            if self._settle_count >= self.settle_frames:
                logger.info(
                    "Rest pose detected at frame %d (settled for %d frames = %.1fs)",
                    frame_index, self.settle_frames,
                    self.settle_frames / self.fps,
                )
                self._enabled = False  # don't trigger again until reset
                return True
        else:
            if self._settle_count > 5:
                logger.debug(
                    "Rest pose approach interrupted at frame %d after %d frames "
                    "(pos_ok=%s grippers_ok=%s vel_ok=%s)",
                    frame_index, self._settle_count, pos_ok, grippers_ok, vel_ok,
                )
            self._settle_count = 0

        return False

    def reset(self):
        """Reset state for a new episode."""
        self._rest_pose = None
        self._settle_count = 0
        self._has_departed = False
        self._enabled = True

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool):
        self._enabled = value
