#!/usr/bin/env python3
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

"""DK1 Recorder — main entry point and RecorderApp orchestrator.

High-performance bimanual teleop recording with:
- Decoupled teleop (~250 Hz) and recording (configurable fps)
- NVENC streaming H.264 encoding (per-episode MP4)
- LeRobot v3 compatible dataset output
- Terminal-first UI with keyboard + gripper gesture controls
- Table-driven state machine for clean episode flow
"""

from __future__ import annotations

import argparse
import atexit
import json
import logging
import os
import queue
import select
import signal
import sys
import termios
import time
from pathlib import Path

from lerobot.cameras.opencv.camera_opencv import OpenCVCamera, OpenCVCameraConfig

from lerobot_robot_trlc_dk1.bi_follower import BiDK1Follower, BiDK1FollowerConfig
from lerobot_robot_trlc_dk1.bi_leader import BiDK1Leader, BiDK1LeaderConfig
from lerobot_robot_trlc_dk1.recorder.auto_home_controller import AutoHomeController
from lerobot_robot_trlc_dk1.recorder.dataset_writer import (
    DatasetWriter,
    build_features_schema,
)
from lerobot_robot_trlc_dk1.recorder.episode_manager import EpisodeManager
from lerobot_robot_trlc_dk1.recorder.nvenc_encoder import (
    NvencEncoder,
    detect_codec,
)
from lerobot_robot_trlc_dk1.recorder.recorder_thread import (
    RecorderThread,
    build_obs_state_keys,
)
from lerobot_robot_trlc_dk1.recorder.state_machine import (
    InputEvent,
    State,
    StateMachine,
    build_transition_table,
)
from lerobot_robot_trlc_dk1.recorder.teleop_thread import TeleopThread
from lerobot_robot_trlc_dk1.recorder.trim_policy import StopTrigger

logger = logging.getLogger(__name__)

# Camera configuration defaults
DEFAULT_CAMERA_WIDTH = 640
DEFAULT_CAMERA_HEIGHT = 360
DEFAULT_CAMERA_FPS = 60
CAMERA_KEYS = ["head", "left_wrist", "right_wrist"]

# Event loop constants
GESTURE_COOLDOWN_S = 1.5   # Ignore gesture signals for this long after state changes
MIN_FRAMES_PER_EPISODE = 10
GRIPPER_OPEN_THRESHOLD = 0.3


# ---------------------------------------------------------------------------
# RecorderApp — top-level orchestrator
# ---------------------------------------------------------------------------

class RecorderApp:
    """Top-level orchestrator: owns the event loop and state machine.

    Replaces the previous ``_run_event_loop()`` free function with a
    clean, table-driven state machine and focused action methods.
    """

    def __init__(
        self,
        recorder: RecorderThread,
        teleop: TeleopThread,
        episodes: EpisodeManager,
        auto_home: AutoHomeController | None,
        ui,
        audio,
        writer: DatasetWriter,
        fps: int,
    ):
        self._sm = StateMachine(State.IDLE, build_transition_table())
        self._recorder = recorder
        self._teleop = teleop
        self._episodes = episodes
        self._auto_home = auto_home
        self._ui = ui
        self._audio = audio
        self._writer = writer
        self._fps = fps
        self._shutdown = False
        self._signal_count = 0

        # Heartbeat monitoring
        self._loop_count = 0
        self._last_heartbeat = time.monotonic()

    @property
    def shutdown_requested(self) -> bool:
        return self._shutdown

    def request_shutdown(self):
        """Request graceful shutdown (called from signal handler)."""
        self._signal_count += 1
        self._shutdown = True
        if self._signal_count >= 3:
            logger.warning("Forced exit (3rd signal)")
            os._exit(1)
        elif self._signal_count >= 2:
            logger.warning("Second signal received — will force-exit on next")

    def run(self):
        """Main event loop at ~20 Hz."""
        while not self._shutdown:
            time.sleep(0.05)

            for event in self._collect_events():
                self._handle(event)

            # Auto-home: only active during recording
            if self._auto_home is not None:
                self._auto_home.enabled = (self._sm.state == State.RECORDING)

            self._update_ui()
            self._heartbeat()

        if self._audio is not None:
            self._audio.recording_done()

    # -- Event collection ---------------------------------------------------

    def _collect_events(self) -> list[InputEvent]:
        """Poll all input sources, return pending events in priority order."""
        events: list[InputEvent] = []

        # Keyboard — handle special keys (QUIT, TASK) immediately,
        # only add state-machine-relevant keys to the events list.
        key = self._poll_keyboard()
        if key == InputEvent.QUIT:
            if self._sm.state == State.RECORDING:
                self._episodes.end(StopTrigger.QUIT)
                if self._auto_home is not None:
                    self._auto_home.release()
            self._shutdown = True
            return []
        elif key == InputEvent.TASK:
            if self._sm.state in (State.IDLE, State.WAITING):
                self._handle_task_change()
        elif key is not None:
            events.append(key)

        # Gesture (with cooldown)
        cooldown_active = (
            time.monotonic() - self._sm.transition_time
        ) < GESTURE_COOLDOWN_S
        if not cooldown_active and self._recorder.gesture_triggered.is_set():
            self._recorder.gesture_triggered.clear()
            # Beep only for stop gestures (during recording)
            if self._audio is not None and self._sm.state == State.RECORDING:
                self._audio.gesture_detected()
            events.append(InputEvent.GESTURE)

        # Grippers open (only relevant in STARTING state)
        if self._sm.state == State.STARTING and self._check_grippers_open():
            events.append(InputEvent.GRIPPERS_OPEN)

        # Rest pose (from recorder thread). Gate on MIN_FRAMES_PER_EPISODE
        # so we don't consume-and-drop the flag during the first ~10 frames
        # — the auto-home settle path can fire arbitrarily early, and the
        # recorder thread will keep re-setting the flag as long as the
        # settle conditions hold.
        if (self._recorder.rest_pose_triggered.is_set()
                and self._sm.state == State.RECORDING
                and self._recorder.frame_index >= MIN_FRAMES_PER_EPISODE):
            self._recorder.rest_pose_triggered.clear()
            events.append(InputEvent.REST_POSE)

        # Countdown tick (internally generated)
        if self._sm.state == State.COUNTDOWN:
            count, should_beep, done = self._episodes.countdown.tick()
            if done:
                events.append(InputEvent.COUNTDOWN_DONE)
            elif should_beep:
                self._on_countdown_tick(count)

        return events

    def _poll_keyboard(self) -> InputEvent | None:
        """Poll keyboard input from UI thread or stdin fallback."""
        if self._ui is not None:
            try:
                key_str = self._ui.key_queue.get_nowait()
            except queue.Empty:
                return None
        else:
            key_str = _poll_stdin_nonblocking()
            if key_str is None:
                return None

        # Map key string to InputEvent
        return {
            "space": InputEvent.SPACE,
            "rerecord": InputEvent.RERECORD,
            "quit": InputEvent.QUIT,
            "task": InputEvent.TASK,
        }.get(key_str)

    def _check_grippers_open(self) -> bool:
        """Check if both grippers are fully open."""
        action = self._teleop.latest_action
        if action is None:
            return False
        left_grip = action.get("left_gripper.pos", 1.0)
        right_grip = action.get("right_gripper.pos", 1.0)
        return (left_grip <= GRIPPER_OPEN_THRESHOLD
                and right_grip <= GRIPPER_OPEN_THRESHOLD)

    # -- Event dispatch -----------------------------------------------------

    def _handle(self, event: InputEvent):
        """Dispatch event through state machine, call transition action."""
        transition = self._sm.transition(event)
        if transition is not None:
            method = getattr(self, transition.action)
            method()

    # -- Transition actions -------------------------------------------------

    def start_countdown(self):
        """Start 3-2-1-GO countdown (unified — replaces 3 copy-pasted blocks)."""
        self._episodes.countdown.start()
        if self._audio is not None:
            self._audio.countdown_tick(3)
        if self._ui is not None:
            self._ui.countdown = 3
        else:
            _print_state(State.COUNTDOWN, self._episodes.episode_index, countdown=3)

    def on_start_gesture(self):
        """Gesture detected — waiting for grippers to open."""
        logger.info("Start gesture detected — waiting for grippers to open")
        if self._ui is None:
            _print_state(State.STARTING, self._episodes.episode_index)

    def on_cancel(self):
        """Cancel from STARTING state."""
        if self._ui is None:
            _print_state(State.IDLE, self._episodes.episode_index)

    def on_cancel_countdown(self):
        """Cancel countdown — go to IDLE or WAITING depending on context."""
        self._episodes.countdown.reset()
        # Dynamic target: IDLE if first episode, WAITING if we've recorded before
        if self._episodes.episode_index > 0:
            self._sm.state = State.WAITING
        logger.info("Countdown cancelled")
        if self._ui is None:
            _print_state(self._sm.state, self._episodes.episode_index)

    def begin_recording(self):
        """Countdown complete — start recording."""
        if self._audio is not None:
            self._audio.countdown_go()
        self._episodes.begin()
        # Reset auto-home: must depart before ramp can activate
        if self._auto_home is not None:
            self._auto_home.reset_departure()
        logger.info("Countdown complete — recording started")
        if self._ui is None:
            _print_state(State.RECORDING, self._episodes.episode_index)

    def end_episode_keyboard(self):
        """Space pressed during recording — immediate stop, no trim."""
        self._episodes.end(StopTrigger.KEYBOARD)
        self._post_episode()

    def end_episode_gesture(self):
        """Gesture during recording — stop with trim, reject if too short."""
        if self._recorder.frame_index < MIN_FRAMES_PER_EPISODE:
            logger.warning(
                "Episode too short (%d frames < %d), ignoring",
                self._recorder.frame_index, MIN_FRAMES_PER_EPISODE,
            )
            # Reject: revert state to RECORDING
            self._sm.state = State.RECORDING
            return
        self._episodes.end(StopTrigger.GESTURE)
        self._post_episode()

    def end_episode_rest_pose(self):
        """Rest pose detected — auto-end with appropriate trim.

        ``_collect_events`` gates this on ``frame_index >=
        MIN_FRAMES_PER_EPISODE``, so a too-short episode cannot reach here.
        """
        trigger = (
            StopTrigger.REST_POSE_AUTO_HOME
            if self._auto_home is not None and self._auto_home.ramping
            else StopTrigger.REST_POSE
        )
        if self._audio is not None:
            self._audio.gesture_detected()  # audible confirmation
        self._episodes.end(trigger)
        self._post_episode()

    def discard_episode(self):
        """Discard current episode (re-record)."""
        self._episodes.discard()
        self._post_episode()

    def _post_episode(self):
        """Common post-episode cleanup: release auto-home, update state."""
        if self._auto_home is not None:
            self._auto_home.release()
        self._sm.state = State.WAITING
        _print_state(State.WAITING, self._episodes.episode_index)

    # -- Countdown ticks ----------------------------------------------------

    def _on_countdown_tick(self, count: int):
        """Handle countdown tick (beep + pre-roll at T=2)."""
        if self._audio is not None:
            self._audio.countdown_tick(count)
        if count == 2:
            # Pre-open encoder containers + GC during countdown
            self._episodes.prepare()
        if self._ui is not None:
            self._ui.countdown = count
        else:
            _print_state(State.COUNTDOWN, self._episodes.episode_index,
                         countdown=count)

    # -- Task change --------------------------------------------------------

    def _handle_task_change(self):
        """Prompt for new task description."""
        new_task = None
        if self._ui is not None:
            new_task = self._ui.prompt_text("New task description",
                                            self._episodes.task)
        else:
            try:
                sys.stdout.write(
                    f"\r\033[K  New task [{self._episodes.task}]: ")
                sys.stdout.flush()
                text = input().strip()
                new_task = text if text else self._episodes.task
            except (EOFError, KeyboardInterrupt):
                pass
        if new_task and new_task != self._episodes.task:
            self._episodes.task = new_task

    # -- UI updates ---------------------------------------------------------

    def _update_ui(self):
        """Push current state to the terminal UI."""
        if self._ui is None:
            return
        self._ui.state = self._sm.state.value
        self._ui.episode = self._episodes.episode_index
        self._ui.fps_actual = self._recorder.actual_fps
        self._ui.teleop_hz = self._teleop.actual_hz
        self._ui.frame_count = self._recorder.frame_index
        self._ui.encoder_drops = self._recorder.drop_count

    def _heartbeat(self):
        """Log heartbeat during recording to detect stuck main thread."""
        self._loop_count += 1
        now = time.monotonic()
        if (self._sm.state == State.RECORDING
                and now - self._last_heartbeat > 2.0):
            logger.warning(
                "Main loop heartbeat: tick=%d state=%s rec_frames=%d "
                "rec_fps=%.0f teleop_hz=%.0f",
                self._loop_count, self._sm.state.value,
                self._recorder.frame_index,
                self._recorder.actual_fps, self._teleop.actual_hz,
            )
            self._last_heartbeat = now


# ---------------------------------------------------------------------------
# Console output helpers
# ---------------------------------------------------------------------------

_G = "\033[92m\033[1m"  # green bold
_Y = "\033[93m\033[1m"  # yellow bold
_D = "\033[2m"          # dim
_R = "\033[0m"          # reset


def _print_state(state: State, episode_index: int, countdown: int = 0):
    """Simple colored console status (used when terminal_ui is not available)."""
    if state == State.COUNTDOWN:
        if countdown > 0:
            print(f"\r  {_Y}▸ {countdown}...{_R}  episode {episode_index} (Bksp=cancel)              ", end="", flush=True)
        else:
            print(f"\r  {_G}▸ GO!{_R}  episode {episode_index}                                          ", end="", flush=True)
    elif state == State.RECORDING:
        print(f"\r  {_G}● RECORDING{_R} episode {episode_index} (Space=stop, R=re-record, Q=quit)", end="", flush=True)
    elif state == State.STARTING:
        print(f"\r  {_Y}* OPEN GRIPPERS{_R} to start episode {episode_index} (Space=force start) ", end="", flush=True)
    elif state == State.WAITING:
        print(f"\r  {_Y}○ WAITING{_R} for reset (Space=start episode {episode_index}, Q=quit)   ", end="", flush=True)
    elif state == State.IDLE:
        print(f"\r  {_D}  IDLE{_R} (Space=start recording)                                       ", end="", flush=True)


def _poll_stdin_nonblocking() -> str | None:
    """Non-blocking stdin read (fallback when terminal_ui is not available)."""
    try:
        rlist, _, _ = select.select([sys.stdin], [], [], 0)
    except (ValueError, OSError):
        return None
    if not rlist:
        return None
    try:
        ch = sys.stdin.read(1)
    except (EOFError, OSError):
        return None
    if ch == " ":
        return "space"
    elif ch.lower() == "r" or ch == "\x7f":  # R or Backspace
        return "rerecord"
    elif ch.lower() == "q":
        return "quit"
    elif ch.lower() == "t":
        return "task"
    # Ignore ESC and escape sequences (cursor keys etc.)
    return None


# ---------------------------------------------------------------------------
# Existing dataset handling
# ---------------------------------------------------------------------------

def handle_existing_dataset(dataset_dir: Path, resume: bool) -> tuple[int, int]:
    """Handle existing dataset directory. Returns (start_episode, start_frame)."""
    info_path = dataset_dir / "meta" / "info.json"

    if not dataset_dir.exists():
        return 0, 0

    if not info_path.exists():
        return 0, 0

    info = json.loads(info_path.read_text())
    total_episodes = info.get("total_episodes", 0)
    total_frames = info.get("total_frames", 0)

    if total_episodes == 0:
        return 0, 0

    if resume:
        print(f"Resuming dataset: {dataset_dir}")
        print(f"  Existing: {total_episodes} episodes, {total_frames} frames")
        return total_episodes, total_frames

    # Interactive prompt
    print(f"\nDataset directory already exists: {dataset_dir}")
    print(f"  Contains {total_episodes} episodes ({total_frames} frames)\n")
    print("  [R] Resume recording (start at episode %d)" % total_episodes)
    print("  [O] Overwrite (delete existing data)")
    print("  [Q] Quit\n")

    while True:
        try:
            choice = input("  Choice: ").strip().upper()
        except (EOFError, KeyboardInterrupt):
            sys.exit(0)

        if choice == "R":
            return total_episodes, total_frames
        elif choice == "O":
            import shutil
            shutil.rmtree(dataset_dir)
            return 0, 0
        elif choice == "Q":
            sys.exit(0)
        else:
            print("  Invalid choice. Enter R, O, or Q.")


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_env_config(env_file: Path) -> dict[str, str]:
    """Load shell-style export VAR=value from an env file."""
    config = {}
    if not env_file.exists():
        return config
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if line.startswith("#") or not line:
            continue
        if line.startswith("export "):
            line = line[7:]
        if "=" in line:
            key, _, value = line.partition("=")
            config[key.strip()] = value.strip().strip('"').strip("'")
    return config


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="DK1 Recorder — high-performance bimanual teleop recording"
    )
    p.add_argument(
        "--dataset-dir", type=Path, required=True,
        help="Output directory for the LeRobot v3 dataset",
    )
    p.add_argument(
        "--task", type=str, default="Perform a bimanual manipulation task.",
        help="Task description string stored in the dataset",
    )
    p.add_argument(
        "--fps", type=int, default=60,
        help="Recording frames per second (default: 60)",
    )
    p.add_argument(
        "--teleop-hz", type=float, default=250.0,
        help="Teleop loop target frequency in Hz (default: 250)",
    )
    p.add_argument(
        "--codec", type=str, default="h264_nvenc",
        help="Video codec: h264_nvenc, hevc_nvenc, av1_nvenc (default: h264_nvenc, fallback: libx264)",
    )
    p.add_argument(
        "--resume", action="store_true",
        help="Resume recording into an existing dataset without prompting",
    )
    p.add_argument(
        "--port-config", type=Path, default=Path("port_config.env"),
        help="Port config env file (default: port_config.env)",
    )
    p.add_argument(
        "--camera-width", type=int, default=DEFAULT_CAMERA_WIDTH,
    )
    p.add_argument(
        "--camera-height", type=int, default=DEFAULT_CAMERA_HEIGHT,
    )
    p.add_argument(
        "--camera-fps", type=int, default=DEFAULT_CAMERA_FPS,
    )
    p.add_argument(
        "--obs-signals", type=str, default="pos,vel,torque",
        help="Comma-separated observation signals to record: pos, vel, torque "
             "(default: pos,vel,torque — full 40-element observation)",
    )
    p.add_argument(
        "--auto-home", type=float, nargs="?", const=0.3, default=0.0,
        metavar="RAD",
        help="Auto-return to zero home pose between episodes when leader "
             "joints are within RAD of zero (default threshold: 0.3 rad ≈ 17°, "
             "0 = disabled). Uses hysteresis to prevent immediate snap.",
    )
    p.add_argument(
        "--visualize", action="store_true",
        help="Enable Rerun visualization (opt-in)",
    )
    p.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable debug logging",
    )
    return p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = build_argparser().parse_args()

    # Save terminal settings BEFORE anything touches the tty.
    _saved_termios = None
    if sys.stdin.isatty():
        try:
            _saved_termios = termios.tcgetattr(sys.stdin)
            def _restore():
                try:
                    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, _saved_termios)
                except (termios.error, ValueError, OSError):
                    pass
            atexit.register(_restore)
        except termios.error:
            pass

    # Logging — use rich for colored, compact output
    level = logging.DEBUG if args.verbose else logging.INFO
    try:
        from rich.logging import RichHandler
        logging.basicConfig(
            level=level,
            format="%(message)s",
            datefmt="[%H:%M:%S]",
            handlers=[RichHandler(
                rich_tracebacks=True,
                tracebacks_show_locals=args.verbose,
                show_path=False,
            )],
        )
    except ImportError:
        logging.basicConfig(
            level=level,
            format="%(asctime)s %(name)s %(levelname)s: %(message)s",
            datefmt="%H:%M:%S",
        )

    # Load port/camera config from env file
    cfg = load_env_config(args.port_config)

    left_follower_port = cfg.get("LEFT_FOLLOWER", os.environ.get("LEFT_FOLLOWER", ""))
    right_follower_port = cfg.get("RIGHT_FOLLOWER", os.environ.get("RIGHT_FOLLOWER", ""))
    left_leader_port = cfg.get("LEFT_LEADER", os.environ.get("LEFT_LEADER", ""))
    right_leader_port = cfg.get("RIGHT_LEADER", os.environ.get("RIGHT_LEADER", ""))

    head_cam_path = cfg.get("CONTEXT_CAM", os.environ.get("CONTEXT_CAM", ""))
    left_wrist_path = cfg.get("WRIST_LEFT", os.environ.get("WRIST_LEFT", ""))
    right_wrist_path = cfg.get("WRIST_RIGHT", os.environ.get("WRIST_RIGHT", ""))

    if not all([left_follower_port, right_follower_port, left_leader_port, right_leader_port]):
        logger.error("Missing port config. Run: python examples/identify_ports.py -m bimanual")
        sys.exit(1)
    if not all([head_cam_path, left_wrist_path, right_wrist_path]):
        logger.error("Missing camera config. Check port_config.env")
        sys.exit(1)

    # Detect codec
    codec = detect_codec(args.codec)
    video_codec = "h264" if "h264" in codec else "av1" if "av1" in codec else "hevc"

    # Parse --obs-signals
    obs_signals = [s.strip() for s in args.obs_signals.split(",")]
    valid_signals = {"pos", "vel", "torque"}
    if not set(obs_signals).issubset(valid_signals):
        logger.error("Invalid --obs-signals: %s (valid: pos, vel, torque)", args.obs_signals)
        sys.exit(1)
    obs_state_keys = build_obs_state_keys(obs_signals)
    logger.info("Observation signals: %s (%d elements)", args.obs_signals, len(obs_state_keys))

    # Handle existing dataset
    start_episode, start_frame = handle_existing_dataset(args.dataset_dir, args.resume)

    # Build feature schema
    features = build_features_schema(
        camera_keys=CAMERA_KEYS,
        camera_height=args.camera_height,
        camera_width=args.camera_width,
        fps=args.fps,
        video_codec=video_codec,
        obs_state_keys=obs_state_keys,
    )

    # -- Warmup NVENC BEFORE hardware (RT loop's mlockall starves CUDA) ----

    videos_dir = args.dataset_dir / "videos"
    encoders: dict[str, NvencEncoder] = {}
    logger.info("Warming up video encoders (before hardware connect)...")
    for cam_key in CAMERA_KEYS:
        enc = NvencEncoder(
            cam_key=cam_key,
            width=args.camera_width,
            height=args.camera_height,
            fps=args.fps,
            codec=codec,
            videos_dir=videos_dir,
        )
        enc.warmup()
        encoders[cam_key] = enc
    actual_codecs = {k: e.codec for k, e in encoders.items()}
    logger.info("Encoder codecs: %s", actual_codecs)
    if any("x264" in c for c in actual_codecs.values()):
        video_codec = "h264"
        features = build_features_schema(
            camera_keys=CAMERA_KEYS,
            camera_height=args.camera_height,
            camera_width=args.camera_width,
            fps=args.fps,
            video_codec=video_codec,
            obs_state_keys=obs_state_keys,
        )

    # -- Initialize hardware ------------------------------------------------

    logger.info("Connecting hardware...")

    camera_configs = {
        "head": OpenCVCameraConfig(
            index_or_path=head_cam_path,
            width=args.camera_width,
            height=args.camera_height,
            fps=args.camera_fps,
            fourcc="MJPG",
        ),
        "left_wrist": OpenCVCameraConfig(
            index_or_path=left_wrist_path,
            width=args.camera_width,
            height=args.camera_height,
            fps=args.camera_fps,
            rotation=180,
            fourcc="MJPG",
        ),
        "right_wrist": OpenCVCameraConfig(
            index_or_path=right_wrist_path,
            width=args.camera_width,
            height=args.camera_height,
            fps=args.camera_fps,
            rotation=180,
            fourcc="MJPG",
        ),
    }

    leader_config = BiDK1LeaderConfig(
        left_arm_port=left_leader_port,
        right_arm_port=right_leader_port,
    )
    leader = BiDK1Leader(leader_config)
    logger.info("Connecting leader arms...")
    leader.connect()
    logger.info("Leader arms connected")

    follower_config = BiDK1FollowerConfig(
        left_arm_port=left_follower_port,
        right_arm_port=right_follower_port,
        control_mode="rt_impedance",
        cameras={},
    )
    follower = BiDK1Follower(follower_config)
    logger.info("Connecting follower arms (ensure E-Stop is released)...")
    follower.connect()
    logger.info("Follower arms connected")

    logger.info("Connecting cameras...")
    cameras = {}
    for cam_key, cam_cfg in camera_configs.items():
        cam = OpenCVCamera(cam_cfg)
        cam.connect()
        cameras[cam_key] = cam
        logger.info("  %s connected", cam_key)
    follower.cameras = cameras
    logger.info("All hardware connected")

    # -- Initialize components ---------------------------------------------

    writer = DatasetWriter(
        dataset_dir=args.dataset_dir,
        fps=args.fps,
        features=features,
        robot_type="bi_dk1_follower",
        task=args.task,
        start_episode=start_episode,
        start_frame=start_frame,
        obs_state_keys=obs_state_keys,
    )

    # Auto-home controller (shared by teleop + recorder)
    auto_home = None
    if args.auto_home > 0:
        auto_home = AutoHomeController(threshold=args.auto_home)

    teleop = TeleopThread(
        follower=follower,
        leader=leader,
        target_hz=args.teleop_hz,
        auto_home=auto_home,
    )

    # Rerun (opt-in)
    rerun_enabled = args.visualize
    if rerun_enabled:
        try:
            import rerun as rr
            import rerun.blueprint as rrb

            _left_joints = [f"left_joint_{i}" for i in range(1, 7)] + ["left_gripper"]
            _right_joints = [f"right_joint_{i}" for i in range(1, 7)] + ["right_gripper"]

            def _arm_tabs(joints, side):
                tabs = [
                    rrb.TimeSeriesView(
                        name=f"{side} Positions",
                        origin="/",
                        contents=[
                            f"+ follower/{j}.pos" for j in joints
                        ] + [
                            f"+ leader/{j}.pos" for j in joints
                        ],
                    ),
                ]
                if "vel" in obs_signals:
                    tabs.append(rrb.TimeSeriesView(
                        name=f"{side} Velocities",
                        origin="/",
                        contents=[
                            f"+ follower/{j}.vel"
                            for j in joints if "gripper" not in j
                        ],
                    ))
                if "torque" in obs_signals:
                    tabs.append(rrb.TimeSeriesView(
                        name=f"{side} Torques",
                        origin="/",
                        contents=[
                            f"+ follower/{j}.torque" for j in joints
                        ],
                    ))
                return rrb.Tabs(*tabs) if len(tabs) > 1 else tabs[0]

            blueprint = rrb.Blueprint(
                rrb.Vertical(
                    rrb.Horizontal(
                        rrb.Spatial2DView(name="Head", origin="cameras/head"),
                        rrb.Spatial2DView(name="Left Wrist", origin="cameras/left_wrist"),
                        rrb.Spatial2DView(name="Right Wrist", origin="cameras/right_wrist"),
                    ),
                    rrb.Horizontal(
                        _arm_tabs(_left_joints, "Left"),
                        _arm_tabs(_right_joints, "Right"),
                    ),
                    row_shares=[3, 2],
                ),
                collapse_panels=True,
            )

            rr.init("dk1-recorder")
            rr.spawn(
                memory_limit=os.environ.get("LEROBOT_RERUN_MEMORY_LIMIT", "10%")
            )
            rr.send_blueprint(blueprint)
        except ImportError:
            logger.warning("rerun-sdk not installed, disabling visualization")
            rerun_enabled = False

    recorder = RecorderThread(
        follower=follower,
        teleop=teleop,
        encoders=encoders,
        camera_keys=CAMERA_KEYS,
        auto_home=auto_home,
        fps=args.fps,
        rerun_enabled=rerun_enabled,
        rerun_obs_keys=obs_state_keys,
    )
    recorder.init_rerun_styles()

    # Audio feedback
    audio = None
    try:
        from lerobot_robot_trlc_dk1.recorder.audio_feedback import AudioFeedback
        audio = AudioFeedback()
    except ImportError:
        pass

    # Episode manager
    episodes = EpisodeManager(
        recorder=recorder,
        encoders=encoders,
        writer=writer,
        audio=audio,
        fps=args.fps,
        start_episode=start_episode,
    )

    # Terminal UI
    ui = None
    try:
        from lerobot_robot_trlc_dk1.recorder.terminal_ui import TerminalUI, StatusLineLogHandler
        ui = TerminalUI()
        ui.start()
        root = logging.getLogger()
        for h in root.handlers[:]:
            root.removeHandler(h)
        handler = StatusLineLogHandler()
        handler.setLevel(level)
        root.addHandler(handler)
    except ImportError:
        pass

    # -- SIGUSR1 thread dump ------------------------------------------------

    def _dump_threads(sig, frame):
        import traceback as _tb
        import threading as _thr
        lines = ["\n===== THREAD DUMP (SIGUSR1) ====="]
        for tid, tframe in sys._current_frames().items():
            tname = "?"
            for t in _thr.enumerate():
                if t.ident == tid:
                    tname = t.name
                    break
            lines.append(f"\n--- Thread {tname} (id={tid}) ---")
            lines.extend(_tb.format_stack(tframe))
        lines.append("===== END THREAD DUMP =====\n")
        sys.stderr.write("\n".join(lines))
        sys.stderr.flush()

    signal.signal(signal.SIGUSR1, _dump_threads)

    # -- Build and run app --------------------------------------------------

    app = RecorderApp(
        recorder=recorder,
        teleop=teleop,
        episodes=episodes,
        auto_home=auto_home,
        ui=ui,
        audio=audio,
        writer=writer,
        fps=args.fps,
    )

    # Signal handlers
    signal.signal(signal.SIGINT, lambda s, f: app.request_shutdown())
    signal.signal(signal.SIGTERM, lambda s, f: app.request_shutdown())

    # -- Start threads ------------------------------------------------------

    for enc in encoders.values():
        enc.start()
    teleop.start()
    recorder.start()

    _B = "\033[1m"
    _N = "\033[0m"
    _C = "\033[96m"
    print(f"""
  {_B}DK1 Recorder{_N} — ready
  Dataset: {args.dataset_dir}
  Task:    {args.task}
  Codec:   {next(iter(actual_codecs.values()))}
  Video:   {args.camera_width}x{args.camera_height} @ {args.camera_fps}fps capture, {args.fps}fps recording
  Obs:     {args.obs_signals} ({len(obs_state_keys)} elements)
  Home:    {"auto @ %.2f rad" % args.auto_home if args.auto_home > 0 else "manual"}
  Teleop:  {args.teleop_hz:.0f} Hz

  {_B}Keyboard controls:{_N}
    {_C}Space{_N}     Start recording / end episode
    {_C}R / Bksp{_N}  Discard current episode and re-record
    {_C}T{_N}         Change task description (idle/waiting)
    {_C}Q{_N}         Stop recording and save dataset

  {_B}Gripper gesture:{_N}
    Double-close either gripper (close → open → close within 0.8s)
    Start: waits for grippers to fully open before recording
    Stop:  trims gesture frames from episode tail (250ms before first close)

  {_B}Auto-end:{_N}
    Episode ends automatically when both arms return to their
    starting position with grippers open and joints settled (~1s).
""")

    try:
        app.run()
    finally:
        # -- Cleanup (with timeouts to avoid hangs) -------------------------
        logger.info("Shutting down...")

        try:
            recorder.stop()
        except Exception:
            logger.exception("Error stopping recorder")

        for enc in encoders.values():
            try:
                enc.stop()
            except Exception:
                logger.exception("Error stopping encoder %s", enc.cam_key)

        try:
            teleop.stop()
        except Exception:
            logger.exception("Error stopping teleop")

        if ui is not None:
            try:
                ui.stop()
            except Exception:
                pass

        if audio is not None:
            try:
                audio.stop()
            except Exception:
                pass

        try:
            writer.finalize()
        except Exception:
            logger.exception("Error finalizing dataset")

        try:
            follower.disconnect()
        except Exception:
            logger.exception("Error disconnecting follower")
        try:
            leader.disconnect()
        except Exception:
            logger.exception("Error disconnecting leader")

        logger.info("Done.")


if __name__ == "__main__":
    main()
