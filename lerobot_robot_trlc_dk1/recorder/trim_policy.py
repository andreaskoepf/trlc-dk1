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

"""Trim policy for episode tail trimming.

Pure function that computes how many frames to drop from the end of an
episode's scalar buffer, based on how the episode was stopped. No side
effects, no dependencies — trivially testable.
"""

from __future__ import annotations

from enum import Enum


class StopTrigger(Enum):
    """How an episode was stopped — determines trim strategy."""
    KEYBOARD = "keyboard"
    GESTURE = "gesture"
    REST_POSE = "rest_pose"
    REST_POSE_AUTO_HOME = "rest_pose_auto_home"
    QUIT = "quit"


def compute_trim(
    trigger: StopTrigger,
    fps: int,
    frame_index: int = 0,
    gesture_first_close_frame: int = 0,
) -> int:
    """Return number of frames to trim from episode tail.

    Args:
        trigger: How the episode was stopped.
        fps: Recording FPS (used to convert time margins to frames).
        frame_index: Current frame index at stop time.
        gesture_first_close_frame: Frame index of the first close in the
            double-close gesture (only used for GESTURE trigger).

    Returns:
        Number of frames to drop from the end of the scalar buffer.
    """
    if trigger is StopTrigger.GESTURE:
        # Trim everything from the first close of the double-close gesture,
        # plus a 250ms margin before it to remove the start of the grab.
        frames_since_first_close = frame_index - gesture_first_close_frame
        margin = int(0.25 * fps)
        return frames_since_first_close + margin

    if trigger is StopTrigger.REST_POSE:
        # Classic rest-pose settle — trim the idle tail (~500ms).
        return fps // 2

    # REST_POSE_AUTO_HOME: ramp-to-zero is intentional training data.
    # KEYBOARD / QUIT: operator chose the exact stop point.
    return 0
