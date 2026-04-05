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

"""DK1 Recorder — main entry point and state machine orchestrator.

High-performance bimanual teleop recording with:
- Decoupled teleop (~200 Hz) and recording (configurable fps)
- NVENC streaming H.264 encoding (per-episode MP4)
- LeRobot v3 compatible dataset output
- Terminal-first UI with keyboard + gripper gesture controls
"""

from __future__ import annotations

import argparse
import atexit
import json
import logging
import os
import queue
import signal
import sys
import termios
import time
from pathlib import Path

from lerobot.cameras.opencv.camera_opencv import OpenCVCamera, OpenCVCameraConfig

from lerobot_robot_trlc_dk1.bi_follower import BiDK1Follower, BiDK1FollowerConfig
from lerobot_robot_trlc_dk1.bi_leader import BiDK1Leader, BiDK1LeaderConfig
from lerobot_robot_trlc_dk1.recorder.dataset_writer import (
    DatasetWriter,
    build_features_schema,
)
from lerobot_robot_trlc_dk1.recorder.nvenc_encoder import (
    EncoderResult,
    NvencEncoder,
    detect_codec,
)
from lerobot_robot_trlc_dk1.recorder.recorder_thread import (
    RecorderThread,
    build_obs_state_keys,
)
from lerobot_robot_trlc_dk1.recorder.teleop_thread import TeleopThread

logger = logging.getLogger(__name__)

# Camera configuration defaults
DEFAULT_CAMERA_WIDTH = 640
DEFAULT_CAMERA_HEIGHT = 360
DEFAULT_CAMERA_FPS = 60
CAMERA_KEYS = ["head", "left_wrist", "right_wrist"]


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------

class RecorderState:
    IDLE = "idle"
    STARTING = "starting"   # gesture detected, waiting for grippers to open
    COUNTDOWN = "countdown"  # 3-2-1-GO countdown before recording
    RECORDING = "recording"
    SAVING = "saving"
    WAITING = "waiting"


# ---------------------------------------------------------------------------
# Existing dataset handling
# ---------------------------------------------------------------------------

def handle_existing_dataset(dataset_dir: Path, resume: bool) -> tuple[int, int]:
    """Handle existing dataset directory. Returns (start_episode, start_frame).

    If --resume is set, reads info.json to get current episode/frame counts.
    Otherwise, prompts the user interactively.
    """
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
# Main
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
        "--fps", type=int, default=30,
        help="Recording frames per second (default: 30)",
    )
    p.add_argument(
        "--teleop-hz", type=float, default=250.0,
        help="Teleop loop target frequency in Hz (default: 250)",
    )
    p.add_argument(
        "--codec", type=str, default="h264_nvenc",
        help="Video codec (default: h264_nvenc, fallback: libx264)",
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


def main():
    args = build_argparser().parse_args()

    # Save terminal settings BEFORE anything touches the tty.
    # atexit restores them even if the process crashes or is killed (SIGTERM).
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
    # For the feature schema, strip encoder suffix (h264_nvenc → h264)
    video_codec = "h264" if "h264" in codec else "hevc"

    # Parse --obs-signals to determine which signals to store in the dataset.
    # Internally the recorder always captures the full 40-element state vector;
    # filtering happens at the output boundary (dataset parquet + Rerun views).
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
    # Update video_codec in case we fell back to libx264
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

    # Connect leader arms first (lightweight Dynamixel, fast) to verify
    # USB serial is healthy before the heavier follower init.
    leader_config = BiDK1LeaderConfig(
        left_arm_port=left_leader_port,
        right_arm_port=right_leader_port,
    )
    leader = BiDK1Leader(leader_config)
    logger.info("Connecting leader arms...")
    leader.connect()
    logger.info("Leader arms connected")

    # Connect follower arms WITHOUT cameras — cameras are opened separately
    # so that a stuck arm serial bus doesn't prevent camera init.
    follower_config = BiDK1FollowerConfig(
        left_arm_port=left_follower_port,
        right_arm_port=right_follower_port,
        control_mode="rt_impedance",
        cameras={},  # cameras connected separately below
    )
    follower = BiDK1Follower(follower_config)
    logger.info("Connecting follower arms (ensure E-Stop is released)...")
    follower.connect()
    logger.info("Follower arms connected")

    # Connect cameras separately
    logger.info("Connecting cameras...")
    cameras = {}
    for cam_key, cam_cfg in camera_configs.items():
        from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
        cam = OpenCVCamera(cam_cfg)
        cam.connect()
        cameras[cam_key] = cam
        logger.info("  %s connected", cam_key)
    # Attach cameras to follower so get_observation() includes them
    follower.cameras = cameras
    logger.info("All hardware connected")

    # -- Initialize components (encoders already created above) ---------------

    # Dataset writer
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

    # Teleop thread
    teleop = TeleopThread(
        follower=follower,
        leader=leader,
        target_hz=args.teleop_hz,
        auto_home_threshold=args.auto_home,
    )

    # Rerun (opt-in) — init before recorder so it can log frames
    rerun_enabled = args.visualize
    if rerun_enabled:
        try:
            import rerun as rr
            import rerun.blueprint as rrb

            # Build explicit content lists (globs don't work reliably)
            _left_joints = [f"left_joint_{i}" for i in range(1, 7)] + ["left_gripper"]
            _right_joints = [f"right_joint_{i}" for i in range(1, 7)] + ["right_gripper"]

            def _arm_tabs(joints, side):
                """Build Pos/Vel/Torque tabs for one arm (respects --obs-signals)."""
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
                    # Top row: 3 camera feeds side by side
                    rrb.Horizontal(
                        rrb.Spatial2DView(name="Head", origin="cameras/head"),
                        rrb.Spatial2DView(name="Left Wrist", origin="cameras/left_wrist"),
                        rrb.Spatial2DView(name="Right Wrist", origin="cameras/right_wrist"),
                    ),
                    # Bottom row: left arm | right arm, each with Pos/Vel/Torque tabs
                    rrb.Horizontal(
                        _arm_tabs(_left_joints, "Left"),
                        _arm_tabs(_right_joints, "Right"),
                    ),
                    row_shares=[3, 2],  # cameras 60%, plots 40%
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

    # Recorder thread
    recorder = RecorderThread(
        follower=follower,
        teleop=teleop,
        encoders=encoders,
        camera_keys=CAMERA_KEYS,
        fps=args.fps,
        rerun_enabled=rerun_enabled,
        rerun_obs_keys=obs_state_keys,
    )

    # Initialize Rerun styles at startup (not during first frame recording)
    recorder.init_rerun_styles()

    # -- SIGUSR1 thread dump (kill -USR1 <pid> to diagnose hangs) ----------

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

    # -- Start threads ------------------------------------------------------

    for enc in encoders.values():
        enc.start()
    teleop.start()
    recorder.start()

    # -- State machine ------------------------------------------------------

    state = RecorderState.IDLE
    episode_index = start_episode
    shutdown_requested = False
    _signal_count = 0

    def signal_handler(sig, frame):
        nonlocal shutdown_requested, _signal_count
        _signal_count += 1
        shutdown_requested = True
        if _signal_count >= 3:
            # Third signal: force-exit immediately (skip cleanup)
            logger.warning("Forced exit (3rd signal)")
            os._exit(1)
        elif _signal_count >= 2:
            logger.warning("Second signal received — will force-exit on next")

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Try to import Phase 2 modules (may not exist yet)
    try:
        from lerobot_robot_trlc_dk1.recorder.terminal_ui import TerminalUI, StatusLineLogHandler
        ui = TerminalUI()
        ui.start()
        # Replace default log handlers with status-line-aware handler
        # so log messages don't overwrite the pinned status line.
        root = logging.getLogger()
        for h in root.handlers[:]:
            root.removeHandler(h)
        handler = StatusLineLogHandler()
        handler.setLevel(level)
        root.addHandler(handler)
    except ImportError:
        ui = None

    try:
        from lerobot_robot_trlc_dk1.recorder.gesture_detector import GripperGestureDetector
        gesture_left = GripperGestureDetector()
        gesture_right = GripperGestureDetector()
    except ImportError:
        gesture_left = gesture_right = None

    try:
        from lerobot_robot_trlc_dk1.recorder.audio_feedback import AudioFeedback
        audio = AudioFeedback()
    except ImportError:
        audio = None

    _B = "\033[1m"  # bold
    _N = "\033[0m"  # reset
    _C = "\033[96m" # cyan
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
        _run_event_loop(
            state=state,
            episode_index=episode_index,
            recorder=recorder,
            encoders=encoders,
            writer=writer,
            teleop=teleop,
            ui=ui,
            audio=audio,
            gesture_left=gesture_left,
            gesture_right=gesture_right,
            rerun_enabled=rerun_enabled,
            shutdown_requested_ref=lambda: shutdown_requested,
            fps=args.fps,
        )
    finally:
        # -- Cleanup (with timeouts to avoid hangs) -------------------------
        logger.info("Shutting down...")

        # Stop recorder first (stops dispatching to encoder queues)
        try:
            recorder.stop()
        except Exception:
            logger.exception("Error stopping recorder")

        # Stop encoders (sends None sentinel, joins with timeout)
        for enc in encoders.values():
            try:
                enc.stop()
            except Exception:
                logger.exception("Error stopping encoder %s", enc.cam_key)

        # Stop teleop (joins with timeout)
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

        # Finalize dataset (write stats.json) — skip if no data
        try:
            writer.finalize()
        except Exception:
            logger.exception("Error finalizing dataset")

        # Disconnect hardware (serial ports, cameras)
        try:
            follower.disconnect()
        except Exception:
            logger.exception("Error disconnecting follower")
        try:
            leader.disconnect()
        except Exception:
            logger.exception("Error disconnecting leader")

        logger.info("Done.")


# ---------------------------------------------------------------------------
# Event loop
# ---------------------------------------------------------------------------

def _run_event_loop(
    *,
    state: str,
    episode_index: int,
    recorder: RecorderThread,
    encoders: dict[str, NvencEncoder],
    writer: DatasetWriter,
    teleop: TeleopThread,
    ui,
    audio,
    gesture_left,
    gesture_right,
    rerun_enabled: bool,
    shutdown_requested_ref,
    fps: int,
):
    """Main event loop: polls for key events and gesture signals,
    drives state transitions."""

    # Cooldown: ignore gesture signals for this long after any state transition.
    # Prevents one gripper's double-close from triggering IDLE→RECORDING and
    # the other gripper immediately triggering RECORDING→SAVING.
    GESTURE_COOLDOWN_S = 1.5
    MIN_FRAMES_PER_EPISODE = 10  # don't save episodes shorter than this

    # Extra margin (in frames) to trim before the first close of the
    # double-close gesture, so we don't include the start of the grab.
    STOP_TRIM_MARGIN = int(0.25 * fps)  # 250ms before first close

    # Gripper open threshold for STARTING state (both grippers must be below this)
    GRIPPER_OPEN_THRESHOLD = 0.3

    _loop_count = 0
    _last_heartbeat = time.monotonic()

    last_transition_time = 0.0  # time.monotonic() of last state change
    countdown_start = 0.0       # time.monotonic() when countdown began
    countdown_beeped: set = set()  # which counts have been beeped

    while not shutdown_requested_ref():
        time.sleep(0.05)  # 20 Hz event loop

        # -- Poll keyboard (from UI thread or simple stdin) -----------------
        key_event = None
        if ui is not None:
            try:
                key_event = ui.key_queue.get_nowait()
            except queue.Empty:
                pass
        else:
            # Fallback: non-blocking stdin read (works without terminal_ui)
            key_event = _poll_stdin_nonblocking()

        # -- Poll gesture detectors (fed at 30 Hz from recorder thread) ------
        gesture_triggered = False
        cooldown_active = (time.monotonic() - last_transition_time) < GESTURE_COOLDOWN_S
        if not cooldown_active and recorder.gesture_triggered.is_set():
            gesture_triggered = True
            recorder.gesture_triggered.clear()
            # Only beep for stop gestures (during recording).
            # Start gestures get the countdown beeps instead.
            if audio is not None and state == RecorderState.RECORDING:
                audio.gesture_detected()

        # -- Check if both grippers are open (for STARTING state) -----------
        grippers_open = False
        action = teleop.latest_action
        if action is not None:
            left_grip = action.get("left_gripper.pos", 1.0)
            right_grip = action.get("right_gripper.pos", 1.0)
            grippers_open = (
                left_grip <= GRIPPER_OPEN_THRESHOLD
                and right_grip <= GRIPPER_OPEN_THRESHOLD
            )

        # -- State transitions ----------------------------------------------

        if state == RecorderState.IDLE:
            if key_event == "space":
                # Space = start countdown (no gripper wait)
                countdown_start = time.monotonic()
                countdown_beeped = set()
                state = RecorderState.COUNTDOWN
                last_transition_time = time.monotonic()
                if audio is not None:
                    audio.countdown_tick(3)
                countdown_beeped.add(3)
                if ui is not None:
                    ui.countdown = 3
                else:
                    _print_state(state, episode_index, countdown=3)
            elif gesture_triggered:
                # Gesture = wait for grippers to open first
                state = RecorderState.STARTING
                last_transition_time = time.monotonic()
                logger.info("Start gesture detected — waiting for grippers to open")
                if ui is None:
                    _print_state(state, episode_index)

        elif state == RecorderState.STARTING:
            # Wait for both grippers to be fully open before countdown
            if grippers_open:
                countdown_start = time.monotonic()
                countdown_beeped = set()
                state = RecorderState.COUNTDOWN
                last_transition_time = time.monotonic()
                if audio is not None:
                    audio.countdown_tick(3)
                countdown_beeped.add(3)
                logger.info("Grippers open — countdown started")
                if ui is not None:
                    ui.countdown = 3
                else:
                    _print_state(state, episode_index, countdown=3)
            elif key_event == "space":
                # Space = skip gripper wait, go to countdown
                countdown_start = time.monotonic()
                countdown_beeped = set()
                state = RecorderState.COUNTDOWN
                last_transition_time = time.monotonic()
                if audio is not None:
                    audio.countdown_tick(3)
                countdown_beeped.add(3)
                if ui is not None:
                    ui.countdown = 3
                else:
                    _print_state(state, episode_index, countdown=3)
            elif key_event in ("rerecord", "quit"):
                # Cancel back to idle
                state = RecorderState.IDLE
                last_transition_time = time.monotonic()
                if ui is None:
                    _print_state(state, episode_index)

        elif state == RecorderState.COUNTDOWN:
            # 3-2-1-GO countdown (cancelable)
            if key_event in ("rerecord", "quit"):
                # Cancel countdown
                state = RecorderState.IDLE if episode_index == 0 else RecorderState.WAITING
                last_transition_time = time.monotonic()
                logger.info("Countdown cancelled")
                if ui is None:
                    _print_state(state, episode_index)
            else:
                elapsed = time.monotonic() - countdown_start
                if elapsed < 1.0:
                    count = 3
                elif elapsed < 2.0:
                    count = 2
                    if 2 not in countdown_beeped:
                        countdown_beeped.add(2)
                        if audio is not None:
                            audio.countdown_tick(2)
                        # Pre-open encoder containers + GC during countdown
                        recorder.prepare_episode(episode_index)
                elif elapsed < 3.0:
                    count = 1
                    if 1 not in countdown_beeped:
                        countdown_beeped.add(1)
                        if audio is not None:
                            audio.countdown_tick(1)
                else:
                    # GO!
                    if audio is not None:
                        audio.countdown_go()
                    recorder.begin_episode(episode_index)
                    # Reset auto-home: must depart before ramp can activate
                    teleop._auto_home_departed = False
                    teleop._auto_home_ramping = False
                    state = RecorderState.RECORDING
                    last_transition_time = time.monotonic()
                    logger.info("Countdown complete — recording started")
                    if ui is None:
                        _print_state(state, episode_index)

                # Update UI countdown value (UI thread handles display)
                if ui is not None and state == RecorderState.COUNTDOWN:
                    ui.countdown = count
                elif ui is None and state == RecorderState.COUNTDOWN:
                    _print_state(state, episode_index, countdown=count)

        elif state == RecorderState.RECORDING:
            too_short = recorder.frame_index < MIN_FRAMES_PER_EPISODE

            # Check rest pose auto-end (signaled from recorder thread)
            rest_pose_end = recorder.rest_pose_triggered.is_set()

            if key_event == "space" or (gesture_triggered and not too_short) or rest_pose_end:
                # End current episode
                if too_short and not key_event == "space":
                    logger.warning(
                        "Episode too short (%d frames < %d), ignoring",
                        recorder.frame_index, MIN_FRAMES_PER_EPISODE,
                    )
                else:
                    # Determine trim: gesture-triggered stops trim trailing
                    # frames to remove the double-close motion from data.
                    # Rest pose detection already waited for settle, so trim
                    # just the settle window (those frames are at rest, not task).
                    if gesture_triggered:
                        # Trim relative to first close of the double-close gesture
                        frames_since_first_close = (
                            recorder.frame_index - recorder.gesture_first_close_frame
                        )
                        trim = frames_since_first_close + STOP_TRIM_MARGIN
                    elif rest_pose_end:
                        # Trim the settle period (robot was stationary at rest)
                        trim = fps  # ~1s settle window
                        if audio is not None:
                            audio.gesture_detected()  # audible confirmation
                        logger.info(
                            "Auto-end: rest pose detected at frame %d",
                            recorder.frame_index,
                        )
                    else:
                        trim = 0
                    state = RecorderState.SAVING
                    _save_episode(
                        recorder, encoders, writer, episode_index, audio,
                        trim_tail_frames=trim,
                    )
                    episode_index += 1
                    state = RecorderState.WAITING
                    last_transition_time = time.monotonic()
                    # Smooth ramp from zero back to leader positions
                    teleop.release_auto_home()
                    _print_state(state, episode_index)

            elif key_event == "rerecord":
                # Discard current episode
                logger.info("Discarding episode %d", episode_index)
                recorder.end_episode()  # drain buffer, signal encoders
                # Wait for encoder results and delete orphan MP4 files
                for enc in encoders.values():
                    try:
                        result = enc.result_queue.get(timeout=5.0)
                        if result.mp4_path and result.mp4_path.exists():
                            result.mp4_path.unlink()
                            logger.debug("Deleted orphan %s", result.mp4_path)
                    except queue.Empty:
                        pass
                if audio is not None:
                    audio.episode_discarded(episode_index)
                state = RecorderState.WAITING
                last_transition_time = time.monotonic()
                teleop.release_auto_home()
                _print_state(state, episode_index)

        elif state == RecorderState.WAITING:
            if key_event == "space":
                # Space = start countdown
                countdown_start = time.monotonic()
                countdown_beeped = set()
                state = RecorderState.COUNTDOWN
                last_transition_time = time.monotonic()
                if audio is not None:
                    audio.countdown_tick(3)
                countdown_beeped.add(3)
                _print_state(state, episode_index, countdown=3)
            elif gesture_triggered:
                # Gesture = wait for grippers to open first
                state = RecorderState.STARTING
                last_transition_time = time.monotonic()
                logger.info("Start gesture detected — waiting for grippers to open")
                _print_state(state, episode_index)

        # Change task description (only in IDLE or WAITING)
        if key_event == "task" and state in (RecorderState.IDLE, RecorderState.WAITING):
            new_task = None
            if ui is not None:
                new_task = ui.prompt_text("New task description", writer.task)
            else:
                # Fallback without terminal UI
                try:
                    sys.stdout.write(f"\r\033[K  New task [{writer.task}]: ")
                    sys.stdout.flush()
                    text = input().strip()
                    new_task = text if text else writer.task
                except (EOFError, KeyboardInterrupt):
                    pass
            if new_task and new_task != writer.task:
                writer.task = new_task
                writer._write_tasks_parquet()
                writer._write_info_json()
                logger.info("Task updated: %s", new_task)

        # Quit from any state
        if key_event == "quit":
            if state == RecorderState.RECORDING:
                # Save current episode before quitting
                state = RecorderState.SAVING
                _save_episode(
                    recorder, encoders, writer, episode_index, audio
                )
            break

        # -- Auto-home: only active during recording (return-to-home phase).
        # Between episodes, release_auto_home() handles the smooth ramp back.
        teleop.auto_home_active = state == RecorderState.RECORDING

        # -- Update UI ------------------------------------------------------
        if ui is not None:
            ui.state = state
            ui.episode = episode_index
            ui.fps_actual = recorder.actual_fps
            ui.teleop_hz = teleop.actual_hz
            ui.frame_count = recorder.frame_index
            ui.encoder_drops = recorder.drop_count

        # -- Heartbeat (detect main thread stuck) --------------------------
        _loop_count += 1
        now = time.monotonic()
        if state == RecorderState.RECORDING and now - _last_heartbeat > 2.0:
            logger.warning(
                "Main loop heartbeat: tick=%d state=%s rec_frames=%d "
                "rec_fps=%.0f teleop_hz=%.0f",
                _loop_count, state, recorder.frame_index,
                recorder.actual_fps, teleop.actual_hz,
            )
            _last_heartbeat = now

    if audio is not None:
        audio.recording_done()


def _save_episode(
    recorder: RecorderThread,
    encoders: dict[str, NvencEncoder],
    writer: DatasetWriter,
    episode_index: int,
    audio,
    trim_tail_frames: int = 0,
):
    """Handle RECORDING → SAVING → WAITING transition.

    Args:
        trim_tail_frames: Number of frames to drop from the end of the
            scalar buffer. Used to remove the stop gesture motion from
            the training data. The MP4 files keep the extra frames but
            the episode metadata (length, to_timestamp, parquet data)
            reflects the trimmed count, so training/visualization never
            sees the gesture frames.
    """
    t0 = time.perf_counter()

    # 1. Stop recorder — returns buffered scalar frames + signals encoders
    scalar_frames = recorder.end_episode()

    # 2. Trim trailing gesture frames from scalar buffer
    if trim_tail_frames > 0 and len(scalar_frames) > trim_tail_frames:
        original_len = len(scalar_frames)
        scalar_frames = scalar_frames[:-trim_tail_frames]
        logger.info(
            "Episode %d: trimmed %d trailing gesture frames (%d → %d)",
            episode_index, trim_tail_frames, original_len, len(scalar_frames),
        )

    if not scalar_frames:
        logger.warning("Episode %d has 0 frames, skipping save", episode_index)
        # Still need to drain encoder results (they got EndEpisode)
        for cam_key, encoder in encoders.items():
            try:
                encoder.result_queue.get(timeout=3.0)
            except queue.Empty:
                pass
        return

    # 2. Wait for all encoders to finish (blocking, typically 10-50ms)
    #    Use shorter timeout — if an encoder is wedged, don't block forever.
    video_results: dict[str, EncoderResult] = {}
    for cam_key, encoder in encoders.items():
        try:
            result = encoder.result_queue.get(timeout=5.0)
            video_results[cam_key] = result
        except queue.Empty:
            logger.warning("Encoder %s timed out for episode %d", cam_key, episode_index)
            video_results[cam_key] = EncoderResult(
                episode_index=episode_index,
                mp4_path=Path(),
                frame_count=0,
                stats={},
            )

    # 3. Write dataset files
    try:
        writer.save_episode(episode_index, scalar_frames, video_results)
    except Exception:
        logger.exception(
            "FAILED to save episode %d (%d frames LOST). "
            "Check disk space and permissions: %s",
            episode_index, len(scalar_frames), writer.dataset_dir,
        )
        if audio is not None:
            audio.error(f"Save failed for episode {episode_index}")
        return

    dt_ms = (time.perf_counter() - t0) * 1000
    logger.info(
        "Episode %d saved in %.0f ms (%d frames)",
        episode_index, dt_ms, len(scalar_frames),
    )

    if audio is not None:
        audio.episode_end(episode_index)


_G = "\033[92m\033[1m"  # green bold
_Y = "\033[93m\033[1m"  # yellow bold
_D = "\033[2m"          # dim
_R = "\033[0m"          # reset


def _print_state(state: str, episode_index: int, countdown: int = 0):
    """Simple colored console status (used when terminal_ui is not available)."""
    if state == RecorderState.COUNTDOWN:
        if countdown > 0:
            print(f"\r  {_Y}▸ {countdown}...{_R}  episode {episode_index} (Bksp=cancel)              ", end="", flush=True)
        else:
            print(f"\r  {_G}▸ GO!{_R}  episode {episode_index}                                          ", end="", flush=True)
    elif state == RecorderState.RECORDING:
        print(f"\r  {_G}● RECORDING{_R} episode {episode_index} (Space=stop, R=re-record, Q=quit)", end="", flush=True)
    elif state == RecorderState.STARTING:
        print(f"\r  {_Y}* OPEN GRIPPERS{_R} to start episode {episode_index} (Space=force start) ", end="", flush=True)
    elif state == RecorderState.WAITING:
        print(f"\r  {_Y}○ WAITING{_R} for reset (Space=start episode {episode_index}, Q=quit)   ", end="", flush=True)
    elif state == RecorderState.IDLE:
        print(f"\r  {_D}  IDLE{_R} (Space=start recording)                                       ", end="", flush=True)


def _poll_stdin_nonblocking() -> str | None:
    """Non-blocking stdin read (fallback when terminal_ui is not available)."""
    import select
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
    # Ignore ESC and escape sequences (cursor keys etc.)
    return None


if __name__ == "__main__":
    main()
