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

"""Centralized episode lifecycle management.

Owns the full episode lifecycle: prepare (pre-roll + GC), begin (mark
encoder boundary + start recording), end (stop recording + trim + save),
and discard (drain encoders + delete orphan MP4s).

Previously this logic was split across RecorderThread.prepare_episode /
begin_episode / end_episode, the _save_episode() free function in
dk1_recorder.py, and inline trim decisions in the event loop.
"""

from __future__ import annotations

import logging
import queue
import time
from pathlib import Path

from lerobot_robot_trlc_dk1.recorder.nvenc_encoder import EncoderResult, NvencEncoder
from lerobot_robot_trlc_dk1.recorder.trim_policy import StopTrigger, compute_trim

logger = logging.getLogger(__name__)


class CountdownTimer:
    """Manages the 3-2-1-GO countdown timing and beep scheduling.

    Call ``start()`` to begin, then ``tick()`` each event loop iteration.
    ``tick()`` returns the current count, whether to beep, and whether
    the countdown is done.
    """

    def __init__(self, duration: float = 3.0):
        self._duration = duration
        self._start_time: float = 0.0
        self._beeped: set[int] = set()

    @property
    def active(self) -> bool:
        return self._start_time > 0.0

    def start(self):
        """Start the countdown."""
        self._start_time = time.monotonic()
        self._beeped = set()

    def reset(self):
        """Reset to inactive state."""
        self._start_time = 0.0
        self._beeped = set()

    def tick(self) -> tuple[int, bool, bool]:
        """Advance the countdown.

        Returns:
            (count, should_beep, done):
            - count: current countdown value (3, 2, 1, or 0 for GO)
            - should_beep: True if this count hasn't been beeped yet
            - done: True when countdown is complete
        """
        elapsed = time.monotonic() - self._start_time
        if elapsed >= self._duration:
            return 0, 0 not in self._beeped, True

        count = int(self._duration - elapsed)  # 3 → 2 → 1
        should_beep = count not in self._beeped
        if should_beep:
            self._beeped.add(count)
        return count, should_beep, False


class EpisodeManager:
    """Centralized episode lifecycle management.

    Args:
        recorder: RecorderThread instance for frame capture control.
        encoders: Per-camera encoder instances.
        writer: DatasetWriter for persisting episodes.
        audio: AudioFeedback for audible cues (optional).
        fps: Recording FPS (used for trim calculation).
        start_episode: Episode index to start from (for resume).
    """

    def __init__(
        self,
        recorder,  # RecorderThread — forward ref to avoid circular import
        encoders: dict[str, NvencEncoder],
        writer,    # DatasetWriter — forward ref
        audio=None,  # AudioFeedback | None
        fps: int = 60,
        start_episode: int = 0,
    ):
        self._recorder = recorder
        self._encoders = encoders
        self._writer = writer
        self._audio = audio
        self._fps = fps
        self.episode_index: int = start_episode
        self.countdown = CountdownTimer()

    @property
    def task(self) -> str:
        return self._writer.task

    @task.setter
    def task(self, value: str):
        self._writer.task = value
        self._writer._write_tasks_parquet()
        self._writer._write_info_json()
        logger.info("Task updated: %s", value)

    # -- Episode lifecycle --------------------------------------------------

    def prepare(self):
        """Pre-roll: open encoder containers, start cameras rolling, GC.

        Call during countdown (at T=2) to move expensive work out of the
        recording hot path.
        """
        self._recorder.prepare_episode(self.episode_index)

    def begin(self):
        """Mark episode boundary and start recording.

        Encoders transition from pre-roll to real episode frames.
        Scalar buffer starts accumulating.
        """
        self._recorder.begin_episode(self.episode_index)
        logger.info("Recording started: episode %d", self.episode_index)

    def end(self, trigger: StopTrigger) -> bool:
        """End current episode: stop recording, trim, save.

        Args:
            trigger: How the episode was stopped (determines trim amount).

        Returns:
            True if the episode was saved successfully, False if empty
            or save failed.
        """
        t0 = time.perf_counter()

        # Compute trim BEFORE stopping (need frame_index and gesture info)
        trim = compute_trim(
            trigger,
            self._fps,
            frame_index=self._recorder.frame_index,
            gesture_first_close_frame=self._recorder.gesture_first_close_frame,
        )

        if trigger in (StopTrigger.REST_POSE, StopTrigger.REST_POSE_AUTO_HOME):
            logger.info(
                "Auto-end: rest pose detected at frame %d (trim=%d)",
                self._recorder.frame_index, trim,
            )

        # Stop recorder — returns buffered scalar frames + signals encoders
        scalar_frames = self._recorder.end_episode()

        # Trim trailing frames
        if trim > 0 and len(scalar_frames) > trim:
            original_len = len(scalar_frames)
            scalar_frames = scalar_frames[:-trim]
            logger.info(
                "Episode %d: trimmed %d trailing frames (%d -> %d)",
                self.episode_index, trim, original_len, len(scalar_frames),
            )

        if not scalar_frames:
            logger.warning("Episode %d has 0 frames, skipping save",
                           self.episode_index)
            self._drain_encoder_results()
            return False

        # Wait for encoder results
        video_results = self._collect_encoder_results()

        # Write dataset files
        try:
            self._writer.save_episode(
                self.episode_index, scalar_frames, video_results,
            )
        except Exception:
            logger.exception(
                "FAILED to save episode %d (%d frames LOST). "
                "Check disk space and permissions: %s",
                self.episode_index, len(scalar_frames),
                self._writer.dataset_dir,
            )
            if self._audio is not None:
                self._audio.error(f"Save failed for episode {self.episode_index}")
            return False

        dt_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "Episode %d saved in %.0f ms (%d frames)",
            self.episode_index, dt_ms, len(scalar_frames),
        )

        if self._audio is not None:
            self._audio.episode_end(self.episode_index)

        self.episode_index += 1
        return True

    def discard(self):
        """Discard current episode: drain encoders and delete orphan MP4s."""
        logger.info("Discarding episode %d", self.episode_index)
        self._recorder.end_episode()  # drain buffer, signal encoders

        # Wait for encoder results and delete orphan MP4 files
        for enc in self._encoders.values():
            try:
                result = enc.result_queue.get(timeout=5.0)
                if result.mp4_path and result.mp4_path.exists():
                    result.mp4_path.unlink()
                    logger.debug("Deleted orphan %s", result.mp4_path)
            except queue.Empty:
                pass

        if self._audio is not None:
            self._audio.episode_discarded(self.episode_index)

    # -- Internal -----------------------------------------------------------

    def _collect_encoder_results(self) -> dict[str, EncoderResult]:
        """Wait for all encoder results with timeout."""
        video_results: dict[str, EncoderResult] = {}
        for cam_key, encoder in self._encoders.items():
            try:
                result = encoder.result_queue.get(timeout=5.0)
                video_results[cam_key] = result
            except queue.Empty:
                logger.warning(
                    "Encoder %s timed out for episode %d",
                    cam_key, self.episode_index,
                )
                video_results[cam_key] = EncoderResult(
                    episode_index=self.episode_index,
                    mp4_path=Path(),
                    frame_count=0,
                    stats={},
                )
        return video_results

    def _drain_encoder_results(self):
        """Drain encoder results without saving (for empty/failed episodes)."""
        for cam_key, encoder in self._encoders.items():
            try:
                encoder.result_queue.get(timeout=3.0)
            except queue.Empty:
                pass
