# DK1 Recorder — Implementation Plan (v2)

> Date: 2026-03-24 (revised)
> Platform: NVIDIA Jetson AGX Thor (aarch64, L4T 38.4.0)
> Robot: Bimanual DK1 (2× DM motor arms, 7 motors each + 3× USB cameras 1280×720@60fps)
> Output format: LeRobot v3.0 dataset (loadable by `LeRobotDataset()`)
> Prerequisites: [lerobot-record-analysis.md](lerobot-record-analysis.md)

---

## 1. Design Goals

1. **Low-latency teleop** — follower tracks leader at ~200 Hz, never stops
2. **Zero-downtime episode transitions** — <100ms pause, teleop uninterrupted
3. **NVENC streaming encoding** — H.264 GPU encoding at ~1.4ms/frame per camera
4. **Per-episode MP4 files** — one file per episode per camera, clean training access
5. **User-signaled boundaries** — gripper gesture or keyboard, no fixed timers
6. **Terminal-first UI** — no GUI dependency, Rerun opt-in
7. **LeRobot v3 compatible output** — standard `video_path` template, loadable by `LeRobotDataset()`

---

## 2. Architecture Overview

### Key insight: decouple teleop rate from recording rate

The C++ RT control loop runs at 250 Hz (SCHED_FIFO). The Python teleop
thread feeds it position commands as fast as the Dynamixel leader bus allows
(~200 Hz). Recording happens at a separate, lower rate (default 30 fps).

Coupling teleop to 30 Hz (as lerobot-record does) adds up to 33ms of
leader-to-follower latency. At 200 Hz the worst case is ~9ms.

```
┌──────────────────────────────────────────────────────────────────────────┐
│                           Main Process                                    │
│                                                                          │
│  ┌──────────────────────────┐    ┌──────────────────────────────────┐    │
│  │  Teleop Thread            │    │  Terminal UI Thread               │    │
│  │  (always running)         │    │  (status line + keyboard input)   │    │
│  │                           │    │                                    │    │
│  │  loop @ ~200 Hz:          │    │  - renders pinned status line     │    │
│  │    leader.get_action()    │    │  - reads stdin (cbreak mode)      │    │
│  │    follower.send_action() │    │  - forwards key events to main    │    │
│  │    store latest_action    │    │                                    │    │
│  └──────────────────────────┘    └──────────────────────────────────┘    │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐    │
│  │  Recorder Thread (30 Hz, only active during RECORDING state)      │    │
│  │                                                                    │    │
│  │  follower.get_observation()  → joint state (seqlock) + cameras     │    │
│  │  snapshot teleop.latest_action                                     │    │
│  │  pack observation.state [40] and action [14] vectors               │    │
│  │  dispatch:  camera frames → encoder queues (put_nowait)            │    │
│  │             scalar data   → dataset writer queue                   │    │
│  └──────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐    │
│  │  Encoder Threads (1 per camera, 3 total)                          │    │
│  │                                                                    │    │
│  │  h264_nvenc via PyAV (max_b_frames=0, preset=p4, tune=ull)        │    │
│  │  typed messages: StartEpisode / VideoFrame / EndEpisode / None     │    │
│  │  writes MP4 per episode per camera                                 │    │
│  │  computes per-episode RunningQuantileStats (image: per-channel)    │    │
│  │  posts EncoderResult to result_queue on EndEpisode                 │    │
│  └──────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐    │
│  │  Dataset Writer (called on episode boundary, not a long-lived       │    │
│  │  thread — just a class with methods called from the main thread)    │    │
│  │                                                                    │    │
│  │  - buffers scalar frames in memory during recording                │    │
│  │  - on save_episode(): writes data parquet + episode metadata       │    │
│  │  - on finalize(): writes stats.json, updates info.json totals      │    │
│  └──────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  Camera background threads (3×, from LeRobot OpenCV camera class)        │
│  RT control loop (C++ SCHED_FIFO @ 250 Hz, 2× per follower arm)         │
└──────────────────────────────────────────────────────────────────────────┘
```

### Thread inventory

| Thread | Priority | Rate | Purpose |
|---|---|---|---|
| RT control (C++, 2×) | SCHED_FIFO 80 | 250 Hz | Impedance control + gravity comp |
| Camera background (3×) | Normal | 60 Hz | Continuous `cv2.read()` capture |
| **Teleop** | Elevated (nice -5) | ~200 Hz | Leader read → follower command |
| **Recorder** | Normal | 30 Hz | Obs + action capture, frame dispatch |
| **Encoder** (3×) | Normal | On frame arrival | NVENC H.264 encoding + stats |
| **Terminal UI** | Normal | 10 Hz | Status display + key input |

### Timing budget

| Operation | Latency | Notes |
|---|---|---|
| `leader.get_action()` (2 arms) | ~1.0ms | Dynamixel sync_read @1Mbaud, 7 motors×2 |
| `follower.send_action()` | ~0.1µs | Seqlock write to RT thread |
| `follower._get_observation_impedance()` | ~1µs | Seqlock read |
| `cam.async_read()` (×3) | 0.1–16ms | Near-instant if frame ready, blocks up to 1/60s |
| NVENC `stream.encode()` | ~1.4ms | Per frame, releases GIL |
| `VideoFrame.from_ndarray()` | ~0.2ms | RGB24 → YUV420P conversion |
| `RunningQuantileStats.update()` (image) | ~1ms | 921K pixels → (921K, 3) |
| Parquet write (episode) | ~5–20ms | Small: ~100–3000 rows |

Teleop iteration (no cameras): ~1.1ms → achievable rate **~500 Hz**, target 200 Hz.
Recording iteration (with cameras): ~1ms teleop + 1–16ms cameras + 0.1ms dispatch.
The recorder thread runs independently, so camera blocking doesn't affect teleop.

---

## 3. Module Breakdown

### 3.1 `dk1_recorder.py` — Main entry point and orchestrator

Responsibilities:
- Parse CLI arguments
- Initialize robot (BiDK1Follower), teleop (BiDK1Leader), cameras
- Handle existing dataset directory (resume / rename / overwrite prompt)
- Create and wire all components (teleop, recorder, encoders, dataset writer, UI, audio)
- Run the main event loop: poll UI for key events, poll gesture detector, drive state transitions
- Finalize dataset on exit (Ctrl-C or Escape)

```python
# CLI interface
dk1-record \
    --dataset-dir ./data/fold_towels \
    --task "Pick up a towel and fold it." \
    --fps 30 \
    --teleop-hz 200 \
    --codec h264_nvenc \       # or libx264 for CPU fallback
    --visualize                # opt-in Rerun
    --resume                   # resume existing dataset
```

Config loaded from `port_config.env` and `cam_config.env` (already exist).
No draccus/hydra — simple argparse.

### 3.2 `teleop_thread.py` — High-rate leader→follower

The fastest loop in the system. Never touches cameras, queues, or I/O.

```python
class TeleopThread:
    """Always-on teleop: reads leader arms, commands follower arms.

    Runs at target_hz (~200 Hz), limited by Dynamixel sync_read latency.
    Stores latest action for the recorder thread to snapshot.
    """

    def __init__(self, follower: BiDK1Follower, leader: BiDK1Leader,
                 target_hz: float = 200.0):
        self.follower = follower
        self.leader = leader
        self.target_hz = target_hz

        # Latest action — read by recorder thread.
        # Safe without lock: dict reference assignment is atomic under CPython GIL,
        # and leader.get_action() returns a fresh dict each call.
        self._latest_action: dict[str, float] | None = None

        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._actual_hz: float = 0.0  # for UI display

    @property
    def latest_action(self) -> dict[str, float] | None:
        return self._latest_action

    @property
    def actual_hz(self) -> float:
        return self._actual_hz

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="teleop")
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def _run(self):
        period = 1.0 / self.target_hz
        hz_filter = 0.0  # exponential moving average
        while not self._stop_event.is_set():
            t0 = time.perf_counter()

            action = self.leader.get_action()
            self.follower.send_action(action)
            self._latest_action = action  # atomic reference swap

            elapsed = time.perf_counter() - t0
            precise_sleep(max(0, period - elapsed))

            dt = time.perf_counter() - t0
            hz = 1.0 / dt if dt > 0 else 0
            hz_filter = 0.95 * hz_filter + 0.05 * hz
            self._actual_hz = hz_filter
```

Key properties:
- **Never stops**: runs from connect to disconnect, in all recorder states
- **No recording dependency**: knows nothing about episodes, queues, or datasets
- **Atomic action sharing**: `self._latest_action = action` is a single pointer swap

### 3.3 `recorder_thread.py` — Observation capture + frame dispatch

Runs at recording FPS. Only active during RECORDING state. Reads cameras
(which may block up to 16ms waiting for a new frame at 60 Hz), but this
does NOT affect the teleop thread.

```python
# Feature packing constants
OBS_STATE_KEYS = [
    "left_joint_1.pos", "left_joint_1.vel", "left_joint_1.torque",
    "left_joint_2.pos", "left_joint_2.vel", "left_joint_2.torque",
    "left_joint_3.pos", "left_joint_3.vel", "left_joint_3.torque",
    "left_joint_4.pos", "left_joint_4.vel", "left_joint_4.torque",
    "left_joint_5.pos", "left_joint_5.vel", "left_joint_5.torque",
    "left_joint_6.pos", "left_joint_6.vel", "left_joint_6.torque",
    "left_gripper.pos", "left_gripper.torque",
    "right_joint_1.pos", "right_joint_1.vel", "right_joint_1.torque",
    "right_joint_2.pos", "right_joint_2.vel", "right_joint_2.torque",
    "right_joint_3.pos", "right_joint_3.vel", "right_joint_3.torque",
    "right_joint_4.pos", "right_joint_4.vel", "right_joint_4.torque",
    "right_joint_5.pos", "right_joint_5.vel", "right_joint_5.torque",
    "right_joint_6.pos", "right_joint_6.vel", "right_joint_6.torque",
    "right_gripper.pos", "right_gripper.torque",
]  # 40 elements

ACTION_KEYS = [
    "left_joint_1.pos", "left_joint_2.pos", "left_joint_3.pos",
    "left_joint_4.pos", "left_joint_5.pos", "left_joint_6.pos",
    "left_gripper.pos",
    "right_joint_1.pos", "right_joint_2.pos", "right_joint_3.pos",
    "right_joint_4.pos", "right_joint_5.pos", "right_joint_6.pos",
    "right_gripper.pos",
]  # 14 elements

CAMERA_KEYS = ["head", "left_wrist", "right_wrist"]


def pack_observation_state(obs: dict[str, float]) -> np.ndarray:
    return np.array([obs[k] for k in OBS_STATE_KEYS], dtype=np.float32)


def pack_action(action: dict[str, float]) -> np.ndarray:
    return np.array([action[k] for k in ACTION_KEYS], dtype=np.float32)


class RecorderThread:
    """Captures observations at recording FPS and dispatches to encoders/writer."""

    def __init__(self, follower: BiDK1Follower, teleop: TeleopThread,
                 encoders: dict[str, NvencEncoder],
                 fps: int = 30):
        self.follower = follower
        self.teleop = teleop
        self.encoders = encoders
        self.fps = fps

        self.recording = threading.Event()  # set when RECORDING
        self.episode_index = 0
        self.frame_index = 0
        self.episode_buffer: list[dict] = []  # scalar frames for current episode

        self._stop_event = threading.Event()
        self._actual_fps: float = 0.0

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="recorder")
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        self._thread.join(timeout=2.0)

    def begin_episode(self, episode_index: int):
        """Called from main thread to start recording a new episode."""
        self.episode_index = episode_index
        self.frame_index = 0
        self.episode_buffer = []
        for enc in self.encoders.values():
            enc.frame_queue.put(StartEpisode(episode_index))
        self.recording.set()

    def end_episode(self) -> list[dict]:
        """Called from main thread. Stops recording, returns buffered scalar frames."""
        self.recording.clear()
        # Signal encoders to finalize current episode
        for enc in self.encoders.values():
            enc.frame_queue.put(EndEpisode())
        buf = self.episode_buffer
        self.episode_buffer = []
        return buf

    def _run(self):
        period = 1.0 / self.fps
        fps_filter = 0.0
        while not self._stop_event.is_set():
            if not self.recording.is_set():
                time.sleep(0.01)  # idle — check every 10ms
                continue

            t0 = time.perf_counter()
            self._capture_and_dispatch()
            elapsed = time.perf_counter() - t0

            sleep_time = period - elapsed
            if sleep_time > 0:
                precise_sleep(sleep_time)
            # else: overrun — proceed immediately, log warning

            dt = time.perf_counter() - t0
            hz = 1.0 / dt if dt > 0 else 0
            fps_filter = 0.95 * fps_filter + 0.05 * hz
            self._actual_fps = fps_filter

    def _capture_and_dispatch(self):
        # 1. Read observation (seqlock + cameras — may block briefly for camera frames)
        obs = self.follower.get_observation()

        # 2. Snapshot latest action from teleop thread (atomic read)
        action = self.teleop.latest_action
        if action is None:
            return

        # 3. Dispatch camera frames to encoder queues (non-blocking)
        for cam_key in CAMERA_KEYS:
            image = obs[cam_key]  # numpy HWC uint8
            try:
                self.encoders[cam_key].frame_queue.put_nowait(
                    VideoFrame(self.frame_index, image))
            except queue.Full:
                pass  # drop frame — encoder can't keep up

        # 4. Pack and buffer scalar data
        obs_state = pack_observation_state(obs)
        action_vec = pack_action(action)
        timestamp = self.frame_index / self.fps

        self.episode_buffer.append({
            "observation.state": obs_state,
            "action": action_vec,
            "timestamp": np.float32(timestamp),
            "frame_index": self.frame_index,
            "episode_index": self.episode_index,
            "task_index": 0,
        })
        self.frame_index += 1
```

Key design decisions:
- **Teleop independence**: recorder thread is separate; camera blocking doesn't affect teleop
- **Packing at capture time**: obs dict → `float32[40]` vector, action dict → `float32[14]` vector
- **Non-blocking dispatch**: `put_nowait()` drops frames rather than blocking
- **Buffer in memory**: scalar frames buffered as list of dicts, flushed on episode boundary

### 3.4 `nvenc_encoder.py` — Per-camera NVENC encoder thread

Uses typed messages instead of string sentinels:

```python
from dataclasses import dataclass

@dataclass
class StartEpisode:
    episode_index: int

@dataclass
class VideoFrame:
    frame_index: int
    image: np.ndarray  # HWC uint8

@dataclass
class EndEpisode:
    pass

@dataclass
class EncoderResult:
    episode_index: int
    mp4_path: Path
    frame_count: int
    stats: dict[str, np.ndarray]  # min, max, mean, std, count, q01-q99


class NvencEncoder:
    """Encodes camera frames to per-episode MP4 files using NVENC."""

    def __init__(self, cam_key: str, width: int, height: int, fps: int,
                 codec: str = "h264_nvenc",
                 videos_dir: Path = Path(),
                 chunks_size: int = 1000,
                 queue_maxsize: int = 60):
        self.cam_key = cam_key
        self.video_key = f"observation.images.{cam_key}"
        self.width = width
        self.height = height
        self.fps = fps
        self.codec = codec
        self.videos_dir = videos_dir
        self.chunks_size = chunks_size

        self.frame_queue: queue.Queue = queue.Queue(maxsize=queue_maxsize)
        self.result_queue: queue.Queue = queue.Queue()  # EncoderResult items

        self.codec_options = {
            "preset": "p4",
            "tune": "ull",
        }
        if "nvenc" in codec:
            self.codec_options["max_b_frames"] = "0"

        self._thread: threading.Thread | None = None
        self._stop = False

    def _episode_path(self, ep_index: int) -> Path:
        """Standard LeRobot v3 video path: chunk-NNN/file-NNN.mp4"""
        chunk = ep_index // self.chunks_size
        file_idx = ep_index % self.chunks_size
        return (self.videos_dir / self.video_key /
                f"chunk-{chunk:03d}" / f"file-{file_idx:03d}.mp4")

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name=f"encoder-{self.cam_key}")
        self._thread.start()

    def stop(self):
        self._stop = True
        self.frame_queue.put(None)  # wake up blocked get()
        if self._thread:
            self._thread.join(timeout=5.0)

    def _run(self):
        """Main loop — waits for StartEpisode, encodes until EndEpisode."""
        while not self._stop:
            msg = self.frame_queue.get()
            if msg is None:
                break
            if isinstance(msg, StartEpisode):
                self._encode_episode(msg.episode_index)

    def _encode_episode(self, ep_index: int):
        """Encode frames for one episode into a single MP4."""
        mp4_path = self._episode_path(ep_index)
        mp4_path.parent.mkdir(parents=True, exist_ok=True)

        container = av.open(str(mp4_path), mode="w")
        stream = container.add_stream(self.codec, rate=self.fps)
        stream.width = self.width
        stream.height = self.height
        stream.pix_fmt = "yuv420p"
        stream.options = self.codec_options

        # Per-channel image stats: shape (H*W, 3) per frame
        stats = RunningQuantileStats()
        frame_count = 0

        while True:
            msg = self.frame_queue.get()
            if msg is None:
                break  # shutdown
            if isinstance(msg, EndEpisode):
                break
            if not isinstance(msg, VideoFrame):
                continue  # skip unexpected messages

            video_frame = av.VideoFrame.from_ndarray(msg.image, format="rgb24")
            video_frame.pts = frame_count
            for packet in stream.encode(video_frame):
                container.mux(packet)

            # Update per-channel stats: reshape (H, W, 3) → (H*W, 3)
            stats.update(msg.image.reshape(-1, 3).astype(np.float32))
            frame_count += 1

        # Flush encoder
        for packet in stream.encode():
            container.mux(packet)
        container.close()

        # Post result
        result = EncoderResult(
            episode_index=ep_index,
            mp4_path=mp4_path,
            frame_count=frame_count,
            stats=stats.get_statistics() if frame_count >= 2 else {},
        )
        self.result_queue.put(result)
```

Key design decisions:
- **Typed messages**: `StartEpisode`, `VideoFrame`, `EndEpisode` dataclasses — no string sentinels
- **Standard LeRobot paths**: `chunk-{chunk:03d}/file-{file:03d}.mp4` matching the v3 template
- **`RunningQuantileStats`**: from `lerobot.datasets.compute_stats` — exact format compatibility
- **Queue maxsize=60**: ~2 seconds at 30fps, backpressure via frame drops in recorder
- **Result queue**: `EncoderResult` posted when episode is complete, waited on by main thread

### 3.5 `dataset_writer.py` — LeRobot v3 parquet + metadata writer

Not a thread — a class with methods called from the main thread during
episode boundary handling. All blocking happens on the main thread, which
is acceptable because teleop runs independently.

```python
class DatasetWriter:
    """Writes LeRobot v3 compatible dataset files."""

    def __init__(self, dataset_dir: Path, fps: int, features: dict,
                 robot_type: str, task: str, chunks_size: int = 1000,
                 start_episode: int = 0):
        self.dataset_dir = dataset_dir
        self.fps = fps
        self.features = features
        self.robot_type = robot_type
        self.task = task
        self.chunks_size = chunks_size

        self.total_episodes = start_episode
        self.global_frame_index = 0  # across all episodes

        # Aggregate stats (across all episodes) for stats.json
        self._agg_stats: dict[str, RunningQuantileStats] = {}

        self._init_dataset_dir()

    def _init_dataset_dir(self):
        """Create directory structure and initial metadata."""
        (self.dataset_dir / "meta" / "episodes" / "chunk-000").mkdir(
            parents=True, exist_ok=True)
        (self.dataset_dir / "data" / "chunk-000").mkdir(
            parents=True, exist_ok=True)
        self._write_info_json()
        self._write_tasks_parquet()

    def save_episode(self, ep_index: int, scalar_frames: list[dict],
                     video_results: dict[str, EncoderResult]):
        """
        Finalize one episode. Called from main thread after encoders
        have posted their results.

        Args:
            ep_index: episode number
            scalar_frames: list of scalar frame dicts from recorder
            video_results: cam_key → EncoderResult from encoders
        """
        if not scalar_frames:
            return

        n_frames = len(scalar_frames)
        from_index = self.global_frame_index
        to_index = from_index + n_frames

        # 1. Write data parquet (scalar features)
        self._write_data_parquet(ep_index, scalar_frames, from_index)

        # 2. Write episode metadata parquet
        self._write_episode_metadata(
            ep_index, n_frames, from_index, to_index, video_results)

        # 3. Update aggregate stats
        self._update_aggregate_stats(scalar_frames, video_results)

        # 4. Update info.json totals
        self.global_frame_index = to_index
        self.total_episodes = ep_index + 1
        self._write_info_json()

    def _write_data_parquet(self, ep_index: int, frames: list[dict],
                            from_index: int):
        """Write one parquet file per episode.

        Path: data/chunk-{chunk}/file-{file}.parquet
        Columns: index, frame_index, episode_index, timestamp, task_index,
                 observation.state (40 floats as list), action (14 floats as list)
        """
        chunk = ep_index // self.chunks_size
        file_idx = ep_index % self.chunks_size
        path = (self.dataset_dir / "data" /
                f"chunk-{chunk:03d}" / f"file-{file_idx:03d}.parquet")
        path.parent.mkdir(parents=True, exist_ok=True)

        rows = []
        for i, f in enumerate(frames):
            rows.append({
                "index": from_index + i,
                "frame_index": f["frame_index"],
                "episode_index": f["episode_index"],
                "timestamp": float(f["timestamp"]),
                "task_index": f["task_index"],
                "observation.state": f["observation.state"].tolist(),
                "action": f["action"].tolist(),
            })

        table = pa.Table.from_pylist(rows)
        pq.write_table(table, path)

    def _write_episode_metadata(self, ep_index, n_frames, from_index,
                                to_index, video_results):
        """Write/append to episode metadata parquet.

        Path: meta/episodes/chunk-000/file-000.parquet
        One row per episode, accumulates over the session.
        """
        chunk = ep_index // self.chunks_size
        file_idx = ep_index % self.chunks_size

        row = {
            "episode_index": ep_index,
            "tasks": [self.task],
            "length": n_frames,
            "meta/episodes/chunk_index": 0,
            "meta/episodes/file_index": 0,
            "data/chunk_index": chunk,
            "data/file_index": file_idx,
            "dataset_from_index": from_index,
            "dataset_to_index": to_index,
        }

        # Per-video metadata
        for cam_key, result in video_results.items():
            vk = f"observation.images.{cam_key}"
            v_chunk = ep_index // self.chunks_size
            v_file = ep_index % self.chunks_size
            row[f"videos/{vk}/chunk_index"] = v_chunk
            row[f"videos/{vk}/file_index"] = v_file
            row[f"videos/{vk}/from_timestamp"] = 0.0
            row[f"videos/{vk}/to_timestamp"] = result.frame_count / self.fps

            # Per-episode per-feature stats
            for stat_key, stat_val in result.stats.items():
                row[f"stats/{vk}/{stat_key}"] = stat_val.tolist()

        # Read existing metadata, append new row, rewrite
        meta_path = self.dataset_dir / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
        if meta_path.exists():
            existing = pq.read_table(meta_path).to_pylist()
        else:
            existing = []
        existing.append(row)
        pq.write_table(pa.Table.from_pylist(existing), meta_path)

    def _update_aggregate_stats(self, scalar_frames, video_results):
        """Update global RunningQuantileStats for stats.json."""
        # Scalar features
        if scalar_frames:
            obs_batch = np.stack([f["observation.state"] for f in scalar_frames])
            act_batch = np.stack([f["action"] for f in scalar_frames])

            if "observation.state" not in self._agg_stats:
                self._agg_stats["observation.state"] = RunningQuantileStats()
                self._agg_stats["action"] = RunningQuantileStats()

            self._agg_stats["observation.state"].update(obs_batch)
            self._agg_stats["action"].update(act_batch)

        # Video features — feed aggregate from per-episode stats
        # Note: video stats are already computed in encoder threads.
        # For the global aggregate, we would need raw pixel data or
        # a merge strategy. Simplest: maintain a separate global
        # RunningQuantileStats fed with one summary row per episode
        # using the episode's mean/min/max. However, this is an
        # approximation. For exact global stats, the encoder threads
        # should also feed the global stat object.
        #
        # Practical approach: feed the global stat with the per-episode
        # mean as a single-row batch. This gives correct global mean
        # and reasonable quantile estimates for normalization.
        for cam_key, result in video_results.items():
            vk = f"observation.images.{cam_key}"
            if vk not in self._agg_stats:
                self._agg_stats[vk] = RunningQuantileStats()
            if "mean" in result.stats:
                # Approximate: feed episode mean as a data point
                self._agg_stats[vk].update(result.stats["mean"].reshape(1, -1))

    def _write_info_json(self):
        """Write meta/info.json."""
        info = {
            "codebase_version": "v3.0",
            "robot_type": self.robot_type,
            "total_episodes": self.total_episodes,
            "total_frames": self.global_frame_index,
            "total_tasks": 1,
            "chunks_size": self.chunks_size,
            "data_files_size_in_mb": 100,
            "video_files_size_in_mb": 999999,  # prevent concatenation
            "fps": self.fps,
            "splits": {"train": f"0:{self.total_episodes}"},
            "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
            "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
            "features": self.features,
        }
        path = self.dataset_dir / "meta" / "info.json"
        path.write_text(json.dumps(info, indent=2))

    def _write_tasks_parquet(self):
        """Write meta/tasks.parquet with task_index and task columns."""
        table = pa.Table.from_pylist([
            {"task_index": 0, "task": self.task},
        ])
        pq.write_table(table, self.dataset_dir / "meta" / "tasks.parquet")

    def finalize(self):
        """Write stats.json and final info.json. Called once at end of session."""
        # Write aggregate stats
        stats_dict = {}
        for key, rqs in self._agg_stats.items():
            try:
                stats_dict[key] = {
                    k: v.tolist() for k, v in rqs.get_statistics().items()
                }
            except ValueError:
                pass  # fewer than 2 samples

        stats_path = self.dataset_dir / "meta" / "stats.json"
        stats_path.write_text(json.dumps(stats_dict, indent=2))

        # Final info.json update
        self._write_info_json()
```

### 3.6 `gesture_detector.py` — Gripper double-close detection

Operates on **leader** gripper positions (what the operator squeezes),
not follower positions (which lag behind).

```python
class GripperGestureDetector:
    """Detect double-close gripper gesture for episode boundary signaling.

    Gripper position: 0 = fully open, 1 = fully closed.
    Double-close = close → open → close within window.
    """

    def __init__(self, threshold_close: float = 0.85,
                 threshold_open: float = 0.3,
                 double_close_window_s: float = 0.8):
        self.threshold_close = threshold_close
        self.threshold_open = threshold_open
        self.double_close_window_s = double_close_window_s
        self._reset()

    def _reset(self):
        self._was_closed = False
        self._last_close_time: float | None = None
        self._close_count = 0

    def update(self, gripper_pos: float) -> bool:
        """Feed gripper position. Returns True on double-close."""
        is_closed = gripper_pos >= self.threshold_close
        now = time.monotonic()

        # Detect rising edge: gripper just closed
        if is_closed and not self._was_closed:
            if (self._last_close_time is not None and
                    now - self._last_close_time < self.double_close_window_s):
                self._close_count += 1
                if self._close_count >= 2:
                    self._reset()
                    return True  # Double-close!
            else:
                self._close_count = 1
            self._last_close_time = now

        # Track open/closed state
        if gripper_pos <= self.threshold_open:
            self._was_closed = False
        if is_closed:
            self._was_closed = True

        # Expire stale single-close
        if (self._last_close_time is not None and
                now - self._last_close_time > self.double_close_window_s * 1.5):
            self._close_count = 0

        return False
```

Usage: one detector per leader gripper. Either left or right can signal.
The main thread feeds `action["left_gripper.pos"]` and `action["right_gripper.pos"]`
from the teleop thread's `latest_action` on every UI tick (~10 Hz is fine for gesture
detection since the double-close window is 800ms).

### 3.7 `terminal_ui.py` — Terminal status display and keyboard input

Uses cbreak mode (not raw mode) to preserve Ctrl-C signal handling.
Renders a pinned status line at the bottom. Log messages print above.

```python
class TerminalUI:
    """Terminal UI: pinned status line + keyboard input in cbreak mode.

    Log output goes above the status line using ANSI cursor save/restore.
    Key events are posted to a thread-safe queue for the main thread.
    """

    def __init__(self):
        self.state = "idle"
        self.episode = 0
        self.fps_actual = 0.0
        self.teleop_hz = 0.0
        self.frame_count = 0
        self.camera_ok = {}    # cam_key → bool
        self.encoder_drops = 0

        self.key_queue: queue.Queue[str] = queue.Queue()
        self._stop_event = threading.Event()

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="terminal-ui")
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        self._thread.join(timeout=1.0)

    def _run(self):
        old_settings = termios.tcgetattr(sys.stdin)
        try:
            tty.setcbreak(sys.stdin.fileno())  # cbreak preserves Ctrl-C
            while not self._stop_event.is_set():
                self._render_status()
                self._poll_keyboard(timeout=0.1)
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
            # Clear status line on exit
            sys.stdout.write("\r" + " " * 80 + "\r")
            sys.stdout.flush()

    def _render_status(self):
        """Render pinned status line using ANSI escapes."""
        cam_str = " ".join(
            f"{k}:ok" if ok else f"{k}:STALE"
            for k, ok in self.camera_ok.items()
        )
        drop_str = f" drops:{self.encoder_drops}" if self.encoder_drops > 0 else ""

        if self.fps_actual > 0 and self.state == "recording":
            elapsed_s = self.frame_count / max(self.fps_actual, 1)
            time_str = f"{int(elapsed_s // 60):02d}:{int(elapsed_s % 60):02d}"
        else:
            time_str = "--:--"

        line = (f"  Ep {self.episode} | {self.state:9s} | "
                f"rec:{self.fps_actual:4.0f}Hz teleop:{self.teleop_hz:4.0f}Hz | "
                f"{cam_str}{drop_str} | {time_str}")

        # Save cursor, move to bottom, write status, restore cursor
        sys.stdout.write(f"\033[s\033[999B\r{line}\033[K\033[u")
        sys.stdout.flush()

    def _poll_keyboard(self, timeout: float):
        """Non-blocking stdin read (cbreak mode, select-based)."""
        import select
        rlist, _, _ = select.select([sys.stdin], [], [], timeout)
        if rlist:
            ch = sys.stdin.read(1)
            if ch == " ":
                self.key_queue.put("space")
            elif ch.lower() == "r":
                self.key_queue.put("rerecord")
            elif ch.lower() == "q" or ch == "\x1b":  # q or Escape
                self.key_queue.put("quit")

    def log(self, message: str):
        """Print a message above the status line."""
        sys.stdout.write(f"\033[s\033[999B\033[1A\r{message}\033[K\n\033[u")
        sys.stdout.flush()
```

Key bindings (terminal-scoped, NOT global):
- **Space** — start recording / end episode
- **R** — discard current episode and re-record
- **Escape / Q** — stop and finalize dataset

### 3.8 `audio_feedback.py` — Audio cues

Pre-generates beep samples at init time. Uses subprocess for playback
but avoids spawning Python interpreters.

```python
class AudioFeedback:
    """Non-blocking audio feedback for state transitions."""

    def __init__(self, enabled: bool = True, tts_engine: str = "espeak"):
        self.enabled = enabled
        self.tts_engine = tts_engine  # "piper", "espeak", or "none"

        # Pre-generate beep WAV data
        self._beep_files: dict[str, Path] = {}
        if enabled:
            self._generate_beeps()

    def _generate_beeps(self):
        """Pre-generate beep WAV files in /tmp for instant playback."""
        import wave
        for name, freq, dur_ms in [
            ("episode_end", 800, 200),
            ("gesture", 1200, 100),
            ("error", 400, 500),
            ("start", 600, 150),
        ]:
            path = Path(f"/tmp/dk1_beep_{name}.wav")
            samples = np.sin(
                2 * np.pi * freq * np.arange(int(22050 * dur_ms / 1000)) / 22050
            )
            data = (samples * 32767).astype("<i2")
            with wave.open(str(path), "w") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(22050)
                wf.writeframes(data.tobytes())
            self._beep_files[name] = path

    def _play_beep(self, name: str):
        if self.enabled and name in self._beep_files:
            subprocess.Popen(["aplay", "-q", str(self._beep_files[name])])

    def episode_start(self, episode_num: int):
        self._play_beep("start")
        self._speak(f"Recording episode {episode_num}")

    def episode_end(self, episode_num: int):
        self._play_beep("episode_end")

    def gesture_detected(self):
        self._play_beep("gesture")

    def error(self, message: str):
        self._play_beep("error")
        self._speak(message)

    def recording_done(self):
        self._speak("Recording complete.")

    def _speak(self, text: str):
        if not self.enabled or self.tts_engine == "none":
            return
        if self.tts_engine == "piper":
            subprocess.Popen(
                f'echo "{text}" | piper --model en_US-lessac-medium '
                f'--output-raw | aplay -r 22050 -f S16_LE -c 1 -q',
                shell=True)
        elif self.tts_engine == "espeak":
            subprocess.Popen(["spd-say", text])
```

---

## 4. State Machine

```
                ┌─────────┐
                │  IDLE   │◄─────────────────────────────────┐
                └────┬────┘                                   │
                     │ Space / double-close                   │
                     ▼                                        │
                ┌──────────┐                                  │
          ┌────►│RECORDING │                                  │
          │     └────┬─────┘                                  │
          │          │ double-close / Space                    │
          │          ▼                                         │
          │     ┌──────────┐  (brief: flush encoders,         │
          │     │ SAVING   │   write parquet, ~10-50ms)       │
          │     └────┬─────┘                                  │
          │          │ save complete                           │
          │          ▼                                         │
          │     ┌──────────┐                                  │
          │     │ WAITING  │  (user resets environment,       │
          │     │          │   teleop still active)            │
          │     └────┬─────┘                                  │
          │          │ Space / double-close                    │
          └──────────┘                                        │
                                                              │
                Escape / Q ───────────────────────────────────┘
                R (during RECORDING) → discard, back to WAITING
```

**Invariant**: teleop thread runs in ALL states at ~200 Hz.

### Episode boundary sequence (RECORDING → SAVING → WAITING)

This is the critical path. Detailed synchronization:

```python
# In dk1_recorder.py main event loop:

def _handle_episode_boundary(self):
    """Transition from RECORDING through SAVING to WAITING."""
    self.state = "saving"

    # 1. Stop recorder — returns buffered scalar frames
    scalar_frames = self.recorder.end_episode()
    #    end_episode() also puts EndEpisode on each encoder queue

    # 2. Wait for ALL encoders to finish (blocking with timeout)
    #    This is the only blocking step — typically 10-50ms for NVENC flush.
    video_results = {}
    for cam_key, encoder in self.encoders.items():
        try:
            result: EncoderResult = encoder.result_queue.get(timeout=10.0)
            video_results[cam_key] = result
        except queue.Empty:
            self.ui.log(f"WARNING: encoder {cam_key} timed out")
            # Create empty result for robustness
            video_results[cam_key] = EncoderResult(
                episode_index=self.episode_index,
                mp4_path=Path(),
                frame_count=0,
                stats={},
            )

    # 3. Write dataset files (parquet + metadata, ~5-20ms)
    self.writer.save_episode(self.episode_index, scalar_frames, video_results)

    # 4. Audio + UI feedback
    self.audio.episode_end(self.episode_index)
    self.ui.log(f"Episode {self.episode_index} saved "
                f"({len(scalar_frames)} frames)")

    # 5. Advance state
    self.episode_index += 1
    self.state = "waiting"
```

**Total blocking time**: ~15-70ms (encoder flush + parquet write).
**Teleop during this time**: runs uninterrupted at ~200 Hz.
**Compare with lerobot-record**: 10-60 seconds of teleop freeze.

---

## 5. Data Flow: One Recording Timestep

```
Teleop thread (continuous, ~200 Hz):
  1. action = leader.get_action()         # ~1ms, Dynamixel sync_read
  2. follower.send_action(action)         # ~0.1µs, seqlock write
  3. self._latest_action = action         # atomic reference swap

Recorder thread (30 Hz, when RECORDING):
  4. obs = follower.get_observation()     # ~1µs seqlock + 1-16ms cameras
  5. action = teleop.latest_action        # atomic reference read
  6. obs_state = pack_observation_state(obs)   # dict → float32[40]
  7. action_vec = pack_action(action)          # dict → float32[14]
  8. For each camera:
       encoder.frame_queue.put_nowait(VideoFrame(idx, image))
  9. episode_buffer.append({obs_state, action_vec, timestamp, ...})

Encoder threads (per camera, on frame arrival):
  10. video_frame = av.VideoFrame.from_ndarray(image)   # ~0.2ms
  11. packets = stream.encode(video_frame)               # ~1.4ms, GPU
  12. container.mux(packets)
  13. stats.update(image.reshape(-1, 3))                 # ~1ms
```

Teleop-to-follower latency: ~5ms (one teleop period) + ~4ms (RT loop period) = ~9ms worst case.
Action-observation desync: ≤5ms (one teleop period). Negligible for imitation learning
where human temporal precision is ~50-100ms.

---

## 6. LeRobot v3 Output Format

### Directory structure

```
<dataset_dir>/
├── meta/
│   ├── info.json
│   ├── stats.json
│   ├── tasks.parquet              # {task_index: 0, task: "..."}
│   └── episodes/
│       └── chunk-000/
│           └── file-000.parquet   # all episode metadata rows
├── data/
│   └── chunk-000/
│       ├── file-000.parquet       # episode 0 scalar data
│       ├── file-001.parquet       # episode 1 scalar data
│       └── ...
└── videos/
    ├── observation.images.head/
    │   └── chunk-000/
    │       ├── file-000.mp4       # episode 0
    │       ├── file-001.mp4       # episode 1
    │       └── ...
    ├── observation.images.left_wrist/
    │   └── chunk-000/
    │       └── ...
    └── observation.images.right_wrist/
        └── chunk-000/
            └── ...
```

### Path templates (standard LeRobot v3)

```
data_path:  "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
video_path: "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
```

Mapping: `chunk_index = episode_index // chunks_size`, `file_index = episode_index % chunks_size`.
With `chunks_size=1000`, the first 1000 episodes all live in `chunk-000/`.

Each video file contains exactly ONE episode (from_timestamp=0.0, to_timestamp=length/fps).
This is achieved by streaming encoding + high `video_files_size_in_mb` (prevents concatenation).

### Feature schema (info.json `features` field)

```json
{
  "observation.images.head": {
    "dtype": "video",
    "shape": [720, 1280, 3],
    "names": ["height", "width", "channels"],
    "info": {
      "video.height": 720,
      "video.width": 1280,
      "video.codec": "h264",
      "video.pix_fmt": "yuv420p",
      "video.fps": 30,
      "video.channels": 3,
      "video.is_depth_map": false,
      "has_audio": false
    }
  },
  "observation.images.left_wrist":  { "...same..." },
  "observation.images.right_wrist": { "...same..." },

  "observation.state": {
    "dtype": "float32",
    "shape": [40],
    "names": [
      "left_joint_1.pos", "left_joint_1.vel", "left_joint_1.torque",
      "left_joint_2.pos", "left_joint_2.vel", "left_joint_2.torque",
      "left_joint_3.pos", "left_joint_3.vel", "left_joint_3.torque",
      "left_joint_4.pos", "left_joint_4.vel", "left_joint_4.torque",
      "left_joint_5.pos", "left_joint_5.vel", "left_joint_5.torque",
      "left_joint_6.pos", "left_joint_6.vel", "left_joint_6.torque",
      "left_gripper.pos", "left_gripper.torque",
      "right_joint_1.pos", "right_joint_1.vel", "right_joint_1.torque",
      "right_joint_2.pos", "right_joint_2.vel", "right_joint_2.torque",
      "right_joint_3.pos", "right_joint_3.vel", "right_joint_3.torque",
      "right_joint_4.pos", "right_joint_4.vel", "right_joint_4.torque",
      "right_joint_5.pos", "right_joint_5.vel", "right_joint_5.torque",
      "right_joint_6.pos", "right_joint_6.vel", "right_joint_6.torque",
      "right_gripper.pos", "right_gripper.torque"
    ]
  },

  "action": {
    "dtype": "float32",
    "shape": [14],
    "names": [
      "left_joint_1.pos", "left_joint_2.pos", "left_joint_3.pos",
      "left_joint_4.pos", "left_joint_5.pos", "left_joint_6.pos",
      "left_gripper.pos",
      "right_joint_1.pos", "right_joint_2.pos", "right_joint_3.pos",
      "right_joint_4.pos", "right_joint_5.pos", "right_joint_6.pos",
      "right_gripper.pos"
    ]
  },

  "timestamp":     { "dtype": "float32", "shape": [1] },
  "frame_index":   { "dtype": "int64",   "shape": [1] },
  "episode_index": { "dtype": "int64",   "shape": [1] },
  "index":         { "dtype": "int64",   "shape": [1] },
  "task_index":    { "dtype": "int64",   "shape": [1] }
}
```

### stats.json format

Aggregated across all episodes. Produced by `DatasetWriter.finalize()`.

```json
{
  "observation.state": {
    "min": [40 floats], "max": [40 floats],
    "mean": [40 floats], "std": [40 floats],
    "count": [N],
    "q01": [40 floats], "q10": [40 floats],
    "q50": [40 floats], "q90": [40 floats], "q99": [40 floats]
  },
  "action": {
    "min": [14 floats], "max": [14 floats], "...same keys..."
  },
  "observation.images.head": {
    "min": [3 floats], "max": [3 floats], "...per-channel stats..."
  },
  "observation.images.left_wrist":  { "..." },
  "observation.images.right_wrist": { "..." }
}
```

Uses `lerobot.datasets.compute_stats.RunningQuantileStats` for exact format
compatibility (histogram-based quantile estimation, same bins/precision).

### tasks.parquet

```
task_index (int64) | task (string)
0                  | "Pick up a towel and fold it."
```

### Compatibility verification

Phase 1 integration test MUST verify:
```python
ds = LeRobotDataset(root=dataset_dir)
assert len(ds) == total_frames
frame = ds[0]
assert frame["observation.state"].shape == (40,)
assert frame["action"].shape == (14,)
assert frame["observation.images.head"].shape == (3, 720, 1280)  # CHW after decode
```

---

## 7. Existing Dataset Handling

When `--dataset-dir` points to an existing directory:

```
Dataset directory already exists: ./data/fold_towels
  Contains 5 episodes (150 frames)

  [R] Resume recording (start at episode 5)
  [N] New name (enter name or press Enter for auto: fold_towels_2)
  [O] Overwrite (delete existing data)
  [Q] Quit

  Choice:
```

With `--resume` flag, skip the prompt and resume automatically.

Resume loads existing info.json to recover:
- `start_episode = info["total_episodes"]`
- `global_frame_index = info["total_frames"]`

---

## 8. Rerun Integration (opt-in)

When `--visualize` is passed:

```python
if args.visualize:
    import rerun as rr
    rr.init("dk1-recorder")
    rr.spawn(memory_limit=os.environ.get("LEROBOT_RERUN_MEMORY_LIMIT", "10%"))

    # In recorder thread, after dispatch:
    for cam_key in CAMERA_KEYS:
        rr.log(f"cameras/{cam_key}", rr.Image(obs[cam_key]), static=True)
    for i, name in enumerate(OBS_STATE_KEYS):
        rr.log(f"observation/{name}", rr.Scalars(obs_state[i]))
```

Static image mode (proven, low overhead). Timeline integration deferred until
Rerun VideoStream API stabilizes (issue #10422).

---

## 9. Implementation Order

### Phase 1: Core recording (MVP)

| Step | Module | Description | Validation |
|---|---|---|---|
| 1.1 | `nvenc_encoder.py` | Encoder thread with typed messages, per-episode MP4, stats | Unit test: encode 100 frames, verify MP4 + stats |
| 1.2 | `dataset_writer.py` | Parquet + metadata + info.json + stats.json | Unit test: write 3 episodes, load with `LeRobotDataset()` |
| 1.3 | `teleop_thread.py` | Always-on high-rate teleop | Manual test: verify ~200 Hz on hardware |
| 1.4 | `recorder_thread.py` | Obs/action packing, frame dispatch | Unit test: pack/unpack round-trip |
| 1.5 | `dk1_recorder.py` | Wiring + state machine + basic Space key control | **Integration test: record 3 episodes, load with `LeRobotDataset()`** |

### Phase 2: User experience

| Step | Module | Description |
|---|---|---|
| 2.1 | `terminal_ui.py` | Pinned status line, cbreak key input, ANSI log output |
| 2.2 | `gesture_detector.py` | Double-close gripper detection |
| 2.3 | `audio_feedback.py` | Pre-generated beeps + TTS (espeak initially, piper later) |
| 2.4 | Existing dataset handling | Resume / rename / overwrite prompt |

### Phase 3: Polish

| Step | Module | Description |
|---|---|---|
| 3.1 | Rerun opt-in | `--visualize` flag with static image mode |
| 3.2 | NVENC fallback | Auto-detect NVENC, fall back to `libx264 preset=ultrafast` |
| 3.3 | Error recovery | Handle camera disconnect, serial errors gracefully |

---

## 10. Dependencies

### Already available
- `av` (PyAV 15.1.0) — with h264_nvenc confirmed
- `lerobot` — for `LeRobotDataset` loading/validation + `RunningQuantileStats`
- `opencv-python` — camera capture (LeRobot camera class)
- `pyarrow` — parquet writing
- `numpy`

### New (optional)
- `piper-tts` — high-quality local neural TTS (Phase 2.3)

### Not needed
- `pynput` — replaced by cbreak terminal input
- `draccus` — simple argparse instead
- `rerun-sdk` — only if `--visualize` used

---

## 11. File Layout

```
lerobot_robot_trlc_dk1/
├── __init__.py              # existing
├── bi_follower.py           # existing
├── bi_leader.py             # existing
├── follower.py              # existing
├── leader.py                # existing
├── recorder/                # NEW
│   ├── __init__.py
│   ├── dk1_recorder.py      # main entry point + state machine
│   ├── teleop_thread.py     # high-rate leader→follower (~200 Hz)
│   ├── recorder_thread.py   # observation capture + packing (30 Hz)
│   ├── nvenc_encoder.py     # per-camera NVENC encoder thread
│   ├── dataset_writer.py    # LeRobot v3 parquet + metadata writer
│   ├── gesture_detector.py  # gripper double-close detection
│   ├── terminal_ui.py       # pinned status line + cbreak input
│   └── audio_feedback.py    # pre-generated beeps + TTS
```

Entry point in `pyproject.toml`:
```toml
[project.scripts]
dk1-record = "lerobot_robot_trlc_dk1.recorder.dk1_recorder:main"
```

---

## 12. Resolved Issues from Review

| # | Issue | Resolution |
|---|---|---|
| C1 | Teleop at 30 Hz | Decoupled: teleop ~200 Hz, recording 30 Hz |
| C2 | video_path breaks loading | Standard template: `chunk-{chunk_index:03d}/file-{file_index:03d}.mp4` |
| C3 | Data parquet format | Standard template, one-per-episode. Integration test verifies round-trip. |
| C4 | observation.state packing | Explicit `pack_observation_state()` with documented key order |
| C5 | Episode boundary race | Blocking `result_queue.get(timeout=10)` per encoder with timeout |
| C6 | String sentinels | Typed dataclasses: `StartEpisode`, `VideoFrame`, `EndEpisode`, `EncoderResult` |
| C7 | Video stats underspecified | Uses `RunningQuantileStats` from lerobot for exact format compatibility |
| C8 | splits never updated | `_write_info_json()` sets `"train": f"0:{total_episodes}"` on every update |
| C9 | tasks.parquet missing | `_write_tasks_parquet()` writes `{task_index, task}` at init |
| C10 | _beep() spawns Python | Pre-generated WAV files at init, playback via `aplay` |
| C11 | Raw terminal vs logging | cbreak mode (preserves Ctrl-C), ANSI pinned status line, logs scroll above |
