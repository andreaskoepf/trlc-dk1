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

"""Table-driven state machine for the DK1 recorder.

Replaces the nested if/elif chain in the event loop with a declarative
transition table. Each transition maps (State, InputEvent) to a target
state and an action method name to call on the RecorderApp.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum


class State(Enum):
    """Recorder state machine states."""
    IDLE = "idle"
    STARTING = "starting"       # gesture detected, waiting for grippers to open
    COUNTDOWN = "countdown"     # 3-2-1-GO
    RECORDING = "recording"
    SAVING = "saving"
    WAITING = "waiting"         # post-first-episode idle


class InputEvent(Enum):
    """Events that can trigger state transitions."""
    SPACE = "space"
    RERECORD = "rerecord"
    QUIT = "quit"
    TASK = "task"
    GESTURE = "gesture"
    GRIPPERS_OPEN = "grippers_open"
    COUNTDOWN_DONE = "countdown_done"
    REST_POSE = "rest_pose"


@dataclass(frozen=True)
class Transition:
    """A state machine transition: target state + action to execute."""
    target: State
    action: str  # method name to call on RecorderApp


class StateMachine:
    """Table-driven state machine with transition-time tracking.

    The transition table maps (current_state, event) pairs to Transition
    objects. When a transition fires, the state is updated and the
    transition time is recorded (used for gesture cooldown).

    Action methods may override the target state by setting ``self.state``
    directly — this handles dynamic transitions like COUNTDOWN cancel
    going to IDLE or WAITING depending on context.
    """

    def __init__(
        self,
        initial: State,
        transitions: dict[tuple[State, InputEvent], Transition],
    ):
        self.state = initial
        self._transitions = transitions
        self._transition_time: float = 0.0

    @property
    def transition_time(self) -> float:
        """Monotonic time of the last state transition."""
        return self._transition_time

    def transition(self, event: InputEvent) -> Transition | None:
        """Process an event. Returns the Transition if one fires, else None.

        Updates state and transition_time when a transition is found.
        """
        key = (self.state, event)
        t = self._transitions.get(key)
        if t is not None:
            self.state = t.target
            self._transition_time = time.monotonic()
        return t


def build_transition_table() -> dict[tuple[State, InputEvent], Transition]:
    """Build the standard recorder transition table.

    Returns a dict mapping (State, InputEvent) -> Transition.
    """
    S, E, T = State, InputEvent, Transition
    return {
        # -- IDLE: waiting for first episode --
        (S.IDLE, E.SPACE):           T(S.COUNTDOWN, "start_countdown"),
        (S.IDLE, E.GESTURE):         T(S.STARTING,  "on_start_gesture"),

        # -- STARTING: gesture detected, waiting for grippers to open --
        (S.STARTING, E.GRIPPERS_OPEN): T(S.COUNTDOWN, "start_countdown"),
        (S.STARTING, E.SPACE):         T(S.COUNTDOWN, "start_countdown"),
        (S.STARTING, E.RERECORD):      T(S.IDLE,      "on_cancel"),
        (S.STARTING, E.QUIT):          T(S.IDLE,      "on_cancel"),

        # -- COUNTDOWN: 3-2-1-GO (cancelable) --
        (S.COUNTDOWN, E.COUNTDOWN_DONE): T(S.RECORDING, "begin_recording"),
        (S.COUNTDOWN, E.RERECORD):       T(S.IDLE,      "on_cancel_countdown"),
        (S.COUNTDOWN, E.QUIT):           T(S.IDLE,      "on_cancel_countdown"),

        # -- RECORDING: capturing frames --
        (S.RECORDING, E.SPACE):      T(S.SAVING,  "end_episode_keyboard"),
        (S.RECORDING, E.GESTURE):    T(S.SAVING,  "end_episode_gesture"),
        (S.RECORDING, E.REST_POSE):  T(S.SAVING,  "end_episode_rest_pose"),
        (S.RECORDING, E.RERECORD):   T(S.WAITING, "discard_episode"),

        # -- WAITING: between episodes --
        (S.WAITING, E.SPACE):    T(S.COUNTDOWN, "start_countdown"),
        (S.WAITING, E.GESTURE):  T(S.STARTING,  "on_start_gesture"),
    }
