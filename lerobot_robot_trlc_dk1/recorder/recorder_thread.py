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

"""Recorder thread — captures observations at recording FPS and dispatches
camera frames to encoder queues + scalar data to an in-memory buffer.

Runs in its own thread, decoupled from the high-rate teleop thread.
Camera async_read() may block up to ~16ms (at 60 Hz cameras), but this
does NOT affect the teleop thread.
"""

from __future__ import annotations

import gc
import logging
import queue
import threading
import time

import numpy as np

from lerobot.utils.robot_utils import precise_sleep

from lerobot_robot_trlc_dk1.recorder.nvenc_encoder import (
    EndEpisode,
    NvencEncoder,
    PrepareEpisode,
    StartEpisode,
    VideoFrame,
)
from lerobot_robot_trlc_dk1.recorder.teleop_thread import TeleopThread

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Feature packing constants
# ---------------------------------------------------------------------------

# Full observation keys grouped by signal type — 40 elements.
# Layout: 14 pos [0:14] | 12 vel [14:26] | 14 torque [26:40]
# pos[:14] matches ACTION_KEYS ordering exactly.
# Filtering for dataset storage happens at the output boundary.
OBS_STATE_KEYS = [
    # -- positions (14) -- joints + grippers, same order as ACTION_KEYS
    "left_joint_1.pos", "left_joint_2.pos", "left_joint_3.pos",
    "left_joint_4.pos", "left_joint_5.pos", "left_joint_6.pos",
    "left_gripper.pos",
    "right_joint_1.pos", "right_joint_2.pos", "right_joint_3.pos",
    "right_joint_4.pos", "right_joint_5.pos", "right_joint_6.pos",
    "right_gripper.pos",
    # -- velocities (12) -- joints only (no gripper velocity)
    "left_joint_1.vel", "left_joint_2.vel", "left_joint_3.vel",
    "left_joint_4.vel", "left_joint_5.vel", "left_joint_6.vel",
    "right_joint_1.vel", "right_joint_2.vel", "right_joint_3.vel",
    "right_joint_4.vel", "right_joint_5.vel", "right_joint_6.vel",
    # -- torques (14) -- joints + grippers
    "left_joint_1.torque", "left_joint_2.torque", "left_joint_3.torque",
    "left_joint_4.torque", "left_joint_5.torque", "left_joint_6.torque",
    "left_gripper.torque",
    "right_joint_1.torque", "right_joint_2.torque", "right_joint_3.torque",
    "right_joint_4.torque", "right_joint_5.torque", "right_joint_6.torque",
    "right_gripper.torque",
]  # 40 elements

ACTION_KEYS = [
    "left_joint_1.pos", "left_joint_2.pos", "left_joint_3.pos",
    "left_joint_4.pos", "left_joint_5.pos", "left_joint_6.pos",
    "left_gripper.pos",
    "right_joint_1.pos", "right_joint_2.pos", "right_joint_3.pos",
    "right_joint_4.pos", "right_joint_5.pos", "right_joint_6.pos",
    "right_gripper.pos",
]  # 14 elements


def build_obs_state_keys(signals: list[str]) -> list[str]:
    """Build observation state keys filtered by signal types.

    Args:
        signals: List of signal types to include, e.g. ["pos"], ["pos", "vel"],
                 or ["pos", "vel", "torque"] (full).

    Returns:
        Filtered list of observation state keys.
    """
    return [k for k in OBS_STATE_KEYS if k.rsplit(".", 1)[1] in signals]


def pack_observation_state(obs: dict[str, float]) -> np.ndarray:
    """Pack observation dict into full float32[40] vector in documented order."""
    return np.array([obs[k] for k in OBS_STATE_KEYS], dtype=np.float32)


def pack_action(action: dict[str, float]) -> np.ndarray:
    """Pack action dict into float32[14] vector in documented order."""
    return np.array([action[k] for k in ACTION_KEYS], dtype=np.float32)


# ---------------------------------------------------------------------------
# RecorderThread
# ---------------------------------------------------------------------------

class RecorderThread:
    """Captures observations at recording FPS and dispatches to encoders.

    The recorder thread runs independently from the teleop thread. It:
    1. Reads observations from the follower (seqlock + cameras)
    2. Snapshots the latest action from the teleop thread
    3. Packs obs→float32[40] (full state) and action→float32[14]
    4. Dispatches camera frames to per-camera encoder queues (non-blocking)
    5. Buffers scalar frames in memory for the dataset writer

    The ``recording`` event controls whether frames are captured. When
    cleared, the thread idles with minimal CPU usage.
    """

    def __init__(
        self,
        follower,
        teleop: TeleopThread,
        encoders: dict[str, NvencEncoder],
        camera_keys: list[str],
        fps: int = 60,
        rerun_enabled: bool = False,
        rerun_obs_keys: list[str] | None = None,
    ):
        self.follower = follower
        self.teleop = teleop
        self.encoders = encoders
        self.camera_keys = camera_keys
        self.fps = fps
        self.rerun_enabled = rerun_enabled

        # Precompute (index, name) pairs for Rerun obs logging.
        # Only the signals selected by --obs-signals are sent to Rerun.
        rr_keys = rerun_obs_keys if rerun_obs_keys is not None else OBS_STATE_KEYS
        self._rerun_obs: list[tuple[int, str]] = [
            (i, k) for i, k in enumerate(OBS_STATE_KEYS) if k in set(rr_keys)
        ]

        # Recording state (controlled from main thread)
        self.recording = threading.Event()
        self.episode_index: int = 0
        self.frame_index: int = 0
        self.episode_buffer: list[dict] = []

        # Rest pose auto-end detection (signaled to main thread)
        self.rest_pose_triggered = threading.Event()
        self._rest_pose_detector = None
        try:
            from lerobot_robot_trlc_dk1.recorder.rest_pose_detector import RestPoseDetector
            self._rest_pose_detector = RestPoseDetector(fps=fps)
        except ImportError:
            pass

        # Gesture detection at recording rate (60 Hz, not 20 Hz event loop)
        self.gesture_triggered = threading.Event()
        self.gesture_first_close_frame: int = 0  # frame_index at first close
        self._gesture_left = None
        self._gesture_right = None
        try:
            from lerobot_robot_trlc_dk1.recorder.gesture_detector import GripperGestureDetector
            self._gesture_left = GripperGestureDetector()
            self._gesture_right = GripperGestureDetector()
        except ImportError:
            pass

        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._actual_fps: float = 0.0
        self._drop_count: int = 0
        self._rerun_global_frame: int = 0
        self._pre_rolling: bool = False  # cameras rolling before episode start
        self._auto_home_settle_count: int = 0

        # Guards encoder queue puts at episode boundaries so that
        # StartEpisode/EndEpisode cannot interleave with frame dispatches.
        # Only held for fast queue operations, never during camera reads.
        self._episode_lock = threading.Lock()

        # Per-motor staleness monitoring. The C++ RT loop tracks
        # ``motor_stale[7]`` (per arm) — flips true if a motor hasn't
        # sent a state reply in ``per_motor_stale_threshold`` cycles.
        # Bus / cabling problems (e.g. a gripper that still receives
        # commands but stops emitting state) are otherwise invisible
        # to the recorder: the producer keeps republishing the last
        # cached value forever, so the parquet looks live but is
        # frozen at one float for thousands of rows. Polling
        # ``get_health()`` once per second and logging on transition
        # surfaces the failure live so you can stop the session
        # instead of finding garbage data afterwards.
        self._motor_names = ["joint_1", "joint_2", "joint_3", "joint_4",
                             "joint_5", "joint_6", "gripper"]
        # (side, motor_idx) -> first frame_index where stale was seen
        self._stale_first_seen: dict[tuple[str, int], int] = {}
        # (side, motor_idx) -> total frames where stale was observed
        self._stale_frames: dict[tuple[str, int], int] = {}
        # previous-poll stale state so we only log on transitions
        self._stale_prev: dict[tuple[str, int], bool] = {}
        self._health_poll_period = max(1, int(self.fps))  # ~1 Hz
        self._health_poll_counter = 0

    @property
    def actual_fps(self) -> float:
        return self._actual_fps

    @property
    def drop_count(self) -> int:
        return self._drop_count

    # -- Lifecycle ----------------------------------------------------------

    def start(self):
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="recorder"
        )
        self._thread.start()
        logger.info("Recorder thread started (target %d fps)", self.fps)

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        logger.info("Recorder thread stopped")

    # -- Episode control (called from main thread) --------------------------

    def prepare_episode(self, episode_index: int):
        """Pre-open encoder containers, start cameras rolling, and run GC.

        Call during the countdown to move expensive work (av.open,
        NVENC session setup, GC) out of the recording hot path.
        Cameras start rolling immediately — encoders accept frames
        from this point. StartEpisode later marks the real episode boundary.
        """
        # Collect garbage now so it doesn't trigger during recording
        gc.collect()

        # Reset per-episode motor-staleness tracking so health_summary()
        # at end_episode reports staleness scoped to THIS episode.
        self._stale_first_seen = {}
        self._stale_frames = {}
        self._stale_prev = {}
        self._health_poll_counter = 0

        # Tell encoders to open containers and start accepting frames
        for enc in self.encoders.values():
            enc.frame_queue.put(PrepareEpisode(episode_index))

        # Start dispatching camera frames to encoders (pre-roll)
        self._pre_rolling = True

        # Identify encoder types so the log is honest — NVENC-specific
        # message was misleading in e.g. jpeg_offline mode.
        enc_kinds = sorted({type(e).__name__ for e in self.encoders.values()})
        enc_desc = "+".join(enc_kinds) if enc_kinds else "none"
        logger.info("Episode %d: pre-init (GC + %s) — cameras rolling",
                    episode_index, enc_desc)

    def begin_episode(self, episode_index: int):
        """Signal encoders and start recording frames.

        If prepare_episode() was called earlier, encoders are already rolling
        and this just marks the episode boundary. Otherwise, encoders open
        containers on the fly (slower, ~500ms NVENC init on first frame).
        """
        self.episode_index = episode_index
        self.frame_index = 0
        self.episode_buffer = []
        self._drop_count = 0
        self.rest_pose_triggered.clear()
        self.gesture_triggered.clear()

        # Reset rest pose detector (will capture pose on first frame)
        if self._rest_pose_detector is not None:
            self._rest_pose_detector.reset()

        # Mark episode boundary in encoders (they may already be rolling).
        # Hold _episode_lock so an in-flight _dispatch_pre_roll() finishes
        # its queue puts before StartEpisode is enqueued.
        with self._episode_lock:
            self._pre_rolling = False
            for enc in self.encoders.values():
                enc.frame_queue.put(StartEpisode(episode_index))

        self.recording.set()
        logger.info("Recording started: episode %d", episode_index)

    def end_episode(self) -> list[dict]:
        """Stop recording and return buffered scalar frames.

        Also sends EndEpisode to all encoder queues so they finalize
        their MP4 files and post results.
        """
        self.recording.clear()

        # Hold _episode_lock so an in-flight _capture_and_dispatch() finishes
        # its queue puts + scalar append before EndEpisode is enqueued.
        # The lock also ensures the buffer snapshot is consistent.
        with self._episode_lock:
            for enc in self.encoders.values():
                enc.frame_queue.put(EndEpisode())
            buf = self.episode_buffer
            self.episode_buffer = []

        logger.info(
            "Recording stopped: episode %d, %d frames, %d drops",
            self.episode_index, len(buf), self._drop_count,
        )
        return buf

    # -- Main loop ----------------------------------------------------------

    def _run(self):
        period = 1.0 / self.fps
        fps_filter = 0.0

        while not self._stop_event.is_set():
            if not self.recording.is_set():
                if self._pre_rolling:
                    # Cameras rolling during countdown — dispatch to encoders only
                    self._dispatch_pre_roll()
                # Idle — still poll gestures at ~30 Hz for start detection
                self._poll_gestures()
                time.sleep(1.0 / self.fps)
                continue

            t0 = time.perf_counter()
            self._capture_and_dispatch()
            elapsed = time.perf_counter() - t0

            sleep_time = period - elapsed
            if sleep_time > 0:
                precise_sleep(sleep_time)
            else:
                logger.debug(
                    "Recorder overrun: %.1f ms (budget %.1f ms)",
                    elapsed * 1000, period * 1000,
                )

            dt = time.perf_counter() - t0
            hz = 1.0 / dt if dt > 0 else 0
            fps_filter = 0.95 * fps_filter + 0.05 * hz
            self._actual_fps = fps_filter

    def _capture_and_dispatch(self):
        """Read one frame of observation + action and dispatch."""
        t_start = time.perf_counter()

        # 1. Read joint state (seqlock — should be ~1µs per arm)
        t0 = time.perf_counter()
        obs: dict = {}
        try:
            if hasattr(self.follower, 'left_arm'):
                # BiDK1Follower: read each arm separately
                left_obs = self.follower.left_arm._get_observation_impedance()
                obs.update({f"left_{k}": v for k, v in left_obs.items()})
                right_obs = self.follower.right_arm._get_observation_impedance()
                obs.update({f"right_{k}": v for k, v in right_obs.items()})
            elif hasattr(self.follower, '_get_observation_impedance'):
                # Single DK1Follower
                obs = self.follower._get_observation_impedance()
            else:
                # Generic fallback (includes cameras — slower)
                obs = self.follower.get_observation()
                # Cameras already read, skip step 2
        except Exception:
            logger.exception("Failed to read joint state")
            return
        t_joints = time.perf_counter()

        # 1b. Per-motor staleness check (cheap seqlock read on the C++
        # health snapshot). Polled at ~1 Hz so it doesn't add overhead
        # to the recording hot path.
        self._health_poll_counter += 1
        if self._health_poll_counter >= self._health_poll_period:
            self._health_poll_counter = 0
            self._poll_motor_health()

        # 2. Read cameras one by one with timing.
        #    async_read() should timeout after 200ms, but we add an outer
        #    guard to detect if it hangs longer (GIL starvation, etc).
        for cam_key in self.camera_keys:
            tc0 = time.perf_counter()
            try:
                cam = self.follower.cameras.get(cam_key)
                if cam is not None:
                    obs[cam_key] = cam.async_read()
            except TimeoutError:
                logger.warning(
                    "Frame %d camera %s: async_read timeout (%.0f ms)",
                    self.frame_index, cam_key,
                    (time.perf_counter() - tc0) * 1000,
                )
                continue
            except Exception:
                logger.exception("Camera %s: read error", cam_key)
                continue
            tc1 = time.perf_counter()
            dt_cam_ms = (tc1 - tc0) * 1000
            if dt_cam_ms > 50:
                logger.warning(
                    "Frame %d camera %s: slow read %.0f ms",
                    self.frame_index, cam_key, dt_cam_ms,
                )
            elif self.frame_index < 5 or self.frame_index % 100 == 0:
                logger.debug(
                    "Frame %d camera %s: %.1f ms",
                    self.frame_index, cam_key, dt_cam_ms,
                )
        t_cams = time.perf_counter()

        # 3. Snapshot latest action from teleop thread (atomic read)
        action = self.teleop.latest_action
        if action is None:
            return  # Teleop not started yet

        # 3b. Poll gesture detectors at recording rate (60 Hz)
        self._poll_gestures()

        # 4. Pack scalar data (outside lock — no shared-state mutation yet)
        obs_state = pack_observation_state(obs)
        action_vec = pack_action(action)
        timestamp = np.float32(self.frame_index / self.fps)

        # 5. Dispatch video + buffer scalar under _episode_lock.
        # This guarantees EndEpisode (from end_episode) cannot slip between
        # our VideoFrame puts and the scalar append.
        with self._episode_lock:
            if not self.recording.is_set():
                return  # episode ended while we were reading cameras
            for cam_key in self.camera_keys:
                image = obs.get(cam_key)
                if image is None:
                    continue
                encoder = self.encoders.get(cam_key)
                if encoder is None:
                    continue
                try:
                    encoder.frame_queue.put_nowait(
                        VideoFrame(self.frame_index, image)
                    )
                except queue.Full:
                    self._drop_count += 1
            self.episode_buffer.append({
                "observation.state": obs_state,
                "action": action_vec,
                "timestamp": timestamp,
                "frame_index": self.frame_index,
                "episode_index": self.episode_index,
                "task_index": 0,
            })
            self.frame_index += 1
        t_dispatch = time.perf_counter()
        t_pack = t_dispatch  # packing moved before dispatch

        # 6. Rest pose auto-end detection
        #    Two modes: when auto-home ramp is complete, check against
        #    absolute zero (the auto-home target) with tight tolerance
        #    and short settle (~100ms). Otherwise use the standard detector
        #    which checks against the captured start pose.
        if self._rest_pose_detector is not None:
            if self.frame_index == 1:
                # Capture rest pose from first frame
                self._rest_pose_detector.capture_rest_pose(obs_state)
                self._auto_home_settle_count = 0
                logger.info(
                    "Rest pose detector armed (departure_threshold=%.2f rad). "
                    "Joint pos sample: [%.2f, %.2f, %.2f, ...]",
                    self._rest_pose_detector.departure_threshold_rad,
                    obs_state[0], obs_state[3], obs_state[6],
                )
            elif self.teleop.auto_home_at_rest:
                # Auto-home ramp done — check joints against absolute zero,
                # restricted to arms that actually ramped (undeparted arms
                # are merely within 0.5 rad hysteresis, not 0.1 rad zero).
                from lerobot_robot_trlc_dk1.recorder.rest_pose_detector import (
                    _PER_ARM_JOINT_POS_INDICES, _PER_ARM_VEL_INDICES,
                )
                settled_arms = self.teleop.auto_home_settled_arms
                pos_idx = [i for arm in settled_arms
                           for i in _PER_ARM_JOINT_POS_INDICES[arm]]
                vel_idx = [i for arm in settled_arms
                           for i in _PER_ARM_VEL_INDICES[arm]]
                joint_pos = obs_state[pos_idx]
                joint_vel = obs_state[vel_idx]
                pos_err = float(np.max(np.abs(joint_pos)))
                max_vel = float(np.max(np.abs(joint_vel)))
                pos_ok = pos_err < 0.1
                vel_ok = max_vel < 0.1
                if pos_ok and vel_ok:
                    self._auto_home_settle_count += 1
                else:
                    self._auto_home_settle_count = 0

                # Heartbeat (~1 Hz): when we enter this branch we EXPECT
                # termination, so emit the blocking condition at INFO so
                # a stuck episode self-reports which check is failing.
                if self.frame_index % max(1, self.fps) == 0:
                    worst_pos_key = OBS_STATE_KEYS[
                        pos_idx[int(np.argmax(np.abs(joint_pos)))]]
                    worst_vel_key = OBS_STATE_KEYS[
                        vel_idx[int(np.argmax(np.abs(joint_vel)))]]
                    logger.info(
                        "Auto-home settle frame %d [arms=%s]: pos_err=%.3f (%s) "
                        "vel=%.3f (%s) | %s/%s | settle=%d/%d",
                        self.frame_index, ",".join(settled_arms),
                        pos_err, worst_pos_key,
                        max_vel, worst_vel_key,
                        "pos_ok" if pos_ok else "POS_FAIL",
                        "vel_ok" if vel_ok else "VEL_FAIL",
                        self._auto_home_settle_count,
                        max(1, self.fps // 10),
                    )

                # ~100ms settle at recording fps
                if self._auto_home_settle_count >= max(1, self.fps // 10):
                    logger.info(
                        "Auto-home settled at frame %d "
                        "(pos_err=%.3f rad, max_vel=%.3f rad/s, %d frames)",
                        self.frame_index, pos_err, max_vel,
                        self._auto_home_settle_count,
                    )
                    self.rest_pose_triggered.set()
            elif self.teleop.auto_home_ramping:
                # Ramp in progress — suppress standard detector
                self._rest_pose_detector._settle_count = 0
                self._auto_home_settle_count = 0
            else:
                if self._rest_pose_detector.update(obs_state, self.frame_index):
                    self.rest_pose_triggered.set()
        elif self.frame_index == 1:
            logger.warning("Rest pose detector not available")
        t_rest = time.perf_counter()

        # 7. Log to Rerun (if enabled)
        if self.rerun_enabled:
            self._log_rerun(obs, obs_state, action_vec)
        t_rerun = time.perf_counter()

        # Log timing for first few frames and periodically
        if self.frame_index <= 5 or self.frame_index % 100 == 0:
            total_ms = (time.perf_counter() - t_start) * 1000
            logger.debug(
                "Frame %d: joints=%.1fms cams=%.1fms dispatch=%.1fms "
                "pack=%.1fms rest=%.1fms rerun=%.1fms total=%.1fms",
                self.frame_index - 1,
                (t_joints - t0) * 1000,
                (t_cams - t_joints) * 1000,
                (t_dispatch - t_cams) * 1000,
                (t_pack - t_dispatch) * 1000,
                (t_rest - t_pack) * 1000,
                (t_rerun - t_rest) * 1000,
                total_ms,
            )

    def _dispatch_pre_roll(self):
        """Dispatch camera frames to encoders during pre-roll (countdown).

        Only sends video frames — no scalar data is buffered. This keeps the
        NVENC pipeline warm so the first real episode frame encodes instantly.

        Camera reads happen outside the lock (they release the GIL and can
        take ~16ms each). Queue puts happen under _episode_lock so they
        cannot interleave with StartEpisode from begin_episode().
        """
        # Read all cameras first (slow, GIL-releasing — no lock needed)
        images: dict[str, np.ndarray] = {}
        for cam_key in self.camera_keys:
            try:
                cam = self.follower.cameras.get(cam_key)
                if cam is not None:
                    images[cam_key] = cam.async_read()
            except (TimeoutError, Exception):
                pass

        # Dispatch under lock — if begin_episode() already ran,
        # _pre_rolling is False and we skip the stale frames.
        with self._episode_lock:
            if not self._pre_rolling:
                return
            for cam_key, image in images.items():
                encoder = self.encoders.get(cam_key)
                if encoder is not None:
                    try:
                        encoder.frame_queue.put_nowait(VideoFrame(0, image))
                    except queue.Full:
                        pass

    def _poll_motor_health(self) -> None:
        """Poll the C++ RT loop's per-motor staleness flags for both arms.

        Logs (once) when any motor flips stale, and again when it
        recovers. Counts cumulative stale frames per (side, motor) pair
        for the post-episode summary so a downstream filter can flag
        rows recorded while a motor was silent.
        """
        for side in ("left", "right"):
            arm = getattr(self.follower, f"{side}_arm", None)
            if arm is None or getattr(arm, "_robot", None) is None:
                continue
            try:
                health = arm._robot.get_health()
            except Exception:
                logger.debug("get_health failed for %s arm", side, exc_info=True)
                continue
            if health is None:
                continue
            try:
                stale = list(health.motor_stale)
            except Exception:
                continue
            for i, is_stale in enumerate(stale):
                key = (side, i)
                prev = self._stale_prev.get(key, False)
                if is_stale and not prev:
                    logger.warning(
                        "%s_%s: motor STALE at frame %d (no state reply for "
                        "%d cycles; reads will be cached values)",
                        side, self._motor_names[i], self.frame_index,
                        getattr(health, "loop_count", 0)
                            - int(health.motor_last_seen_cycle[i]),
                    )
                    self._stale_first_seen.setdefault(key, self.frame_index)
                elif prev and not is_stale:
                    logger.warning(
                        "%s_%s: motor RECOVERED at frame %d (state replies "
                        "resumed)",
                        side, self._motor_names[i], self.frame_index,
                    )
                if is_stale:
                    self._stale_frames[key] = (
                        self._stale_frames.get(key, 0) + self._health_poll_period
                    )
                self._stale_prev[key] = is_stale

    def health_summary(self) -> dict[str, list[dict]]:
        """Return per-side per-motor staleness summary collected since
        the last ``prepare_episode``. Empty list when nothing went stale.

        Shape:
          {"left":  [{"motor": "gripper", "first_frame": 154,
                      "stale_frames": 7900}, ...],
           "right": [...]}
        """
        out: dict[str, list[dict]] = {"left": [], "right": []}
        for (side, i), first in self._stale_first_seen.items():
            out[side].append({
                "motor": self._motor_names[i],
                "first_frame": first,
                "stale_frames": self._stale_frames.get((side, i), 0),
            })
        for side in out:
            out[side].sort(key=lambda d: d["first_frame"])
        return out

    def _poll_gestures(self):
        """Check gripper gesture detectors using latest teleop action."""
        if self._gesture_left is None:
            return
        action = self.teleop.latest_action
        if action is None:
            return
        left = action.get("left_gripper.pos", 0)
        right = action.get("right_gripper.pos", 0)
        left_result = self._gesture_left.update(left)
        right_result = self._gesture_right.update(right)
        first_close_time = left_result or right_result
        if first_close_time:
            # Convert first-close wall time to frame index.
            # Each frame is 1/fps apart; frame_index is current (about to
            # be incremented). Estimate how many frames ago the first close
            # happened based on elapsed time.
            elapsed = time.monotonic() - first_close_time
            frames_ago = int(elapsed * self.fps + 0.5)
            self.gesture_first_close_frame = max(0, self.frame_index - frames_ago)
            self.gesture_triggered.set()

    # 14 distinct colors for joints: left arm = warm, right arm = cool.
    # Each has a dark variant (follower .pos) and light variant (leader .pos,
    # follower .vel/.torque).
    #                          dark (R,G,B)      light (R,G,B)
    _JOINT_COLORS: list[tuple[tuple[int,int,int], tuple[int,int,int]]] = [
        # Left arm (warm): joint 1-6, gripper
        ((192, 48, 48),  (232, 144, 144)),  # red
        ((192, 80, 32),  (232, 160, 128)),  # vermilion
        ((192, 112, 16), (232, 184, 112)),  # orange
        ((160, 144, 16), (208, 200, 96)),   # amber
        ((96, 160, 32),  (168, 208, 112)),  # lime
        ((32, 160, 64),  (128, 208, 152)),  # green
        ((32, 160, 128), (120, 208, 184)),  # teal (gripper)
        # Right arm (cool): joint 1-6, gripper
        ((32, 144, 160), (120, 200, 216)),  # cyan
        ((32, 112, 192), (128, 176, 232)),  # azure
        ((48, 80, 192),  (144, 152, 224)),  # blue
        ((80, 48, 192),  (168, 144, 224)),  # indigo
        ((128, 32, 176), (192, 128, 216)),  # purple
        ((176, 32, 144), (216, 128, 192)),  # magenta
        ((192, 32, 96),  (224, 128, 152)),  # rose (gripper)
    ]

    def _init_rerun_styles(self):
        """Log static SeriesLines styles for all joint signals (called once)."""
        import rerun as rr

        for i, name in enumerate(ACTION_KEYS):
            _, light = self._JOINT_COLORS[i]
            # Leader: light color, thin line
            rr.log(f"leader/{name}", rr.SeriesLines(
                colors=[light], widths=[1.0],
            ), static=True)

        for i, name in self._rerun_obs:
            # Map obs key to joint index (0-13) via its .pos counterpart
            base = name.rsplit(".", 1)[0]  # e.g. "left_joint_1"
            sig = name.rsplit(".", 1)[1]   # "pos", "vel", or "torque"
            joint_idx = next(
                (j for j, ak in enumerate(ACTION_KEYS)
                 if ak.rsplit(".", 1)[0] == base), 0
            )
            dark, light = self._JOINT_COLORS[joint_idx]
            if sig == "pos":
                # Follower pos: dark color, thick line
                rr.log(f"follower/{name}", rr.SeriesLines(
                    colors=[dark], widths=[2.0],
                ), static=True)
            else:
                # Follower vel/torque: light color, normal line
                rr.log(f"follower/{name}", rr.SeriesLines(
                    colors=[light], widths=[1.5],
                ), static=True)

    def init_rerun_styles(self):
        """Log static SeriesLines styles (call once at startup, not during recording)."""
        if not self.rerun_enabled:
            return
        self._init_rerun_styles()
        logger.info("Rerun styles initialized (%d obs + %d action series)",
                     len(self._rerun_obs), len(ACTION_KEYS))

    def _log_rerun(self, obs: dict, obs_state: np.ndarray, action_vec: np.ndarray):
        """Log current frame to Rerun viewer."""
        try:
            import rerun as rr

            rr.set_time("frame", sequence=self._rerun_global_frame)
            self._rerun_global_frame += 1

            # Camera images — static=True so only latest frame is kept in memory.
            for cam_key in self.camera_keys:
                image = obs.get(cam_key)
                if image is not None:
                    rr.log(f"cameras/{cam_key}", rr.Image(image), static=True)

            # Follower actual state (filtered to --obs-signals)
            for i, name in self._rerun_obs:
                rr.log(f"follower/{name}", rr.Scalars([obs_state[i]]))

            # Leader commanded positions
            for i, name in enumerate(ACTION_KEYS):
                rr.log(f"leader/{name}", rr.Scalars([action_vec[i]]))

        except Exception:
            if self.frame_index <= 2:
                logger.exception("Rerun logging error (frame %d)", self.frame_index)
