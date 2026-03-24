# LeRobot Record Script — Architectural Analysis

> Analysis date: 2026-03-24
> Source: `lerobot/scripts/lerobot_record.py` (installed via uv from HuggingFace lerobot)

---

## 1. High-Level Architecture

The `lerobot-record` command orchestrates a multi-threaded recording pipeline.
The outer `record()` function runs an episode loop; the inner `record_loop()`
runs the real-time control/capture loop at target FPS.

```
┌─────────────────────────────────────────────────────────────────┐
│  record() — outer loop (episodes)                               │
│                                                                 │
│  for each episode:                                              │
│    1. log_say("Recording episode N")                            │
│    2. record_loop(control_time_s=episode_time_s, dataset=ds)    │
│    3. log_say("Reset the environment")                          │
│    4. record_loop(control_time_s=reset_time_s, dataset=None)    │
│    5. dataset.save_episode()   ← BLOCKS for encoding            │
│                                                                 │
│  finally: dataset.finalize(), robot.disconnect()                │
└─────────────────────────────────────────────────────────────────┘
```

Key files:
- `lerobot/scripts/lerobot_record.py` — main script
- `lerobot/datasets/lerobot_dataset.py` — dataset with `add_frame()`, `save_episode()`, `finalize()`
- `lerobot/datasets/image_writer.py` — async PNG writing (thread/process pool)
- `lerobot/datasets/video_utils.py` — `StreamingVideoEncoder`, `VideoEncodingManager`
- `lerobot/cameras/opencv/camera_opencv.py` — camera capture with background threads
- `lerobot/utils/control_utils.py` — keyboard listener (pynput)
- `lerobot/utils/utils.py` — `log_say()` TTS
- `lerobot/utils/robot_utils.py` — `precise_sleep()`

---

## 2. Thread Map

| Thread | Purpose | Lifetime |
|---|---|---|
| **Main thread** | `record_loop()` — reads sensors, gets teleop action, sends to robot, calls `add_frame()` | Per-episode |
| **Camera background thread** (1 per camera) | Continuously `cv2.read()` → stores `latest_frame` + `latest_timestamp` behind a lock | Robot connect → disconnect |
| **AsyncImageWriter threads** (N threads, default 4 per camera) | Pull PNG write jobs from a `queue.Queue` | Dataset create → finalize |
| **StreamingVideoEncoder thread** (1 per camera, if `streaming_encoding=True`) | Pulls frames from per-camera queue, encodes to MP4 via PyAV (releases GIL) | Per-episode |
| **pynput keyboard Listener** (daemon thread) | Global X11 key monitoring → sets `events["exit_early"]` etc. | `init_keyboard_listener()` → `listener.stop()` |

---

## 3. The Control Loop (`record_loop`)

Each iteration at target FPS (default 30 Hz):

```
start_loop_t = perf_counter()

1. obs = robot.get_observation()         # motor state + sequential async_read() per camera
2. obs_processed = observation_processor(obs)
3. act = teleop.get_action()             # reads leader arm positions
4. act_processed = teleop_action_processor((act, obs))
5. robot_action = robot_action_processor((act_processed, obs))
6. robot.send_action(robot_action)       # sends to follower
7. dataset.add_frame(frame)              # queues images async, feeds streaming encoder
8. precise_sleep(1/fps - elapsed)        # time.sleep on Linux, busy-spin on macOS/Win

timestamp = perf_counter() - start_episode_t
```

### Timing enforcement

- `precise_sleep()` on Linux is just `time.sleep(seconds)` (kernel timer resolution is adequate).
- On macOS/Windows it uses a hybrid sleep+spin approach (spin for last ~10ms).
- If the loop overshoots target period, a warning is logged but **no frame dropping
  or compensation occurs** — the loop simply runs the next iteration immediately.

---

## 4. How Video Recording and Teleoperation Work Together

### Data flow per timestep

1. `robot.get_observation()` returns motor state + camera frames (dict of numpy arrays)
2. `teleop.get_action()` returns leader arm joint positions
3. Both are combined into a single `frame` dict
4. `dataset.add_frame(frame)` stores the frame:
   - Appends scalar data (joint positions, task label) to an in-memory buffer (later written to parquet)
   - Routes camera images to either the PNG writer or the streaming encoder

### Two encoding paths

#### Path A — Batch/Immediate (default, `streaming_encoding=False`)

1. `add_frame()` queues each camera image to `AsyncImageWriter` → written as PNG files on disk
   by a thread pool (queue-based, non-blocking for the main thread).
2. `save_episode()` calls:
   - `_wait_image_writer()` → `queue.join()` — **blocks** until all PNGs are written
   - Launches `ProcessPoolExecutor` to encode PNGs → MP4 per camera — **blocks**
   - Computes episode stats
3. **This is the blocking path that freezes teleop between episodes.**

#### Path B — Streaming (`streaming_encoding=True`)

1. `add_frame()` feeds raw numpy frames to `StreamingVideoEncoder.feed_frame()`
   via per-camera `queue.Queue(maxsize=30)` (~1 second buffer at 30fps).
2. Each encoder thread encodes frames in real-time using PyAV (`output_stream.encode()`
   releases the GIL, so encoding runs truly in parallel with the main thread).
3. Backpressure: if the encoder queue is full, `feed_frame()` drops the frame with a
   warning (0.1s timeout) instead of blocking the control loop.
4. `save_episode()` calls `finish_episode()`:
   - Sends `None` sentinels to all frame queues
   - Waits for encoder threads to flush (up to 120s timeout)
   - Collects stats from result queues
5. **Much faster but still blocks briefly at episode boundary while encoders flush.**

### `VideoEncodingManager` context manager

Wraps the entire recording session. On exit (normal or exception):
- Streaming mode: cancels in-progress episode if exception, calls `close()`
- Batch mode: encodes any remaining queued episodes
- Always: calls `dataset.finalize()` to close parquet writers

---

## 5. Timestep Alignment and Jitter

### No inter-camera synchronization

- Each camera runs its own background thread (`_read_loop`) with its own
  `latest_frame` and `latest_timestamp` (captured via `time.perf_counter()`).
- `robot.get_observation()` reads cameras **sequentially** in a for-loop:
  ```python
  for cam_key, cam in self.cameras.items():
      obs[cam_key] = cam.async_read()
  ```
- Timestamps are **not exposed** in the observation dict — the dataset has no
  per-camera timestamp column.
- Frame N for camera A and frame N for camera B may have been captured
  milliseconds apart.

### What happens when a camera drops below target FPS

Example: target 30 Hz, one camera temporarily at 28 FPS.

1. `async_read()` waits up to 200ms for a `new_frame_event` from the background thread.
   If no new frame arrives within the timeout, it raises `TimeoutError`.
2. `read_latest()` (used by some robots) checks frame age vs `max_age_ms` (default 500ms).
   If the latest frame is too old, it raises `TimeoutError`.
3. **No interpolation, duplication detection, or frame-drop markers** — if the camera
   is slow, the same stale frame may be silently read twice on consecutive control loop
   iterations.
4. The control loop logs a warning if it runs below target FPS:
   ```
   Record loop is running slower (28.0 Hz) than the target FPS (30 Hz).
   ```
   but takes **no corrective action**.
5. There is no frame-drop counter or jitter metric stored in the dataset metadata.

### Resolution configuration and mismatch

- `_validate_width_and_height()` sets the requested resolution via OpenCV's
  `CAP_PROP_FRAME_WIDTH/HEIGHT`, then reads back the actual value.
- If the camera driver can't match the request: **hard `RuntimeError`**.
- There is **no resize, crop, or scaling pipeline** in the camera layer.
- No auto-negotiation to the nearest supported resolution.

---

## 6. Keyboard Listener

```python
# control_utils.py
listener = keyboard.Listener(on_press=on_press)
listener.start()  # daemon thread
```

**Events tracked** (shared dict with boolean flags):

| Key | Events set |
|---|---|
| Right Arrow | `exit_early = True` |
| Left Arrow | `exit_early = True`, `rerecord_episode = True` |
| Escape | `exit_early = True`, `stop_recording = True` |

The listener uses `pynput.keyboard.Listener` — a **global X11 keyboard hook** that
intercepts keys system-wide (not scoped to the recording terminal).

---

## 7. Audio Feedback (`log_say`)

```python
# Linux implementation
cmd = ["spd-say", text]          # non-blocking (subprocess.Popen)
cmd = ["spd-say", "--wait", text]  # blocking mode
```

Uses Speech Dispatcher (`spd-say`) which typically backends to espeak — a
robotic-sounding synthesizer. Messages are minimal: "Recording episode 3",
"Reset the environment", "Stop recording".

---

## 8. Identified Shortcomings

### S1: Teleop loop abruptly interrupted during video encoding

**Severity: Critical (safety hazard)**

In the `record()` outer loop (line 581), `dataset.save_episode()` is called
**synchronously** between episodes. With batch encoding (`streaming_encoding=False`),
this:

1. Waits for all pending PNG writes (`_wait_image_writer()` → `queue.join()`)
2. Encodes all camera videos via `ProcessPoolExecutor`
3. Computes episode stats

During this entire time, **no `record_loop()` is running** — the teleop loop is
completely dead. The leader arm can move freely but the follower doesn't track it.
When recording resumes, the follower snaps to the leader's current position —
causing a **dangerous, abrupt jump**.

Even with `streaming_encoding=True`, there's a gap: `finish_episode()` blocks
waiting for encoder threads to flush, and no teleop bridging happens in between.

**Required fix**: Teleop must run continuously, independent of episode boundaries
and encoding. A separate "always-on" teleop thread should keep the follower
tracking the leader at all times.

### S2: Unacceptable encoding wait between episodes

**Severity: High**

With default settings (`streaming_encoding=False`, `video_encoding_batch_size=1`),
every episode triggers a full encode cycle. For 3 cameras at 640×480@30fps over
60s episodes, this can take **10–60+ seconds per episode** on CPU.

The `video_encoding_batch_size` parameter exists to defer encoding, and
`streaming_encoding=True` makes `save_episode()` near-instant, but:
- Streaming encoding still blocks briefly at episode boundaries
- Batch encoding eventually blocks when the batch is full
- Neither approach achieves zero-downtime transitions

**Required fix**: Either:
- Fully pipelined encoding that never blocks the teleop/recording thread
- "Record everything first, encode later" workflow (encode only on explicit
  user request or after all episodes are done)
- Ideally both options available

### S3: Fixed episode and reset times

**Severity: High (workflow friction)**

```python
episode_time_s: int | float = 60   # fixed recording duration
reset_time_s: int | float = 60     # fixed reset duration
```

Both `record_loop()` calls use `while timestamp < control_time_s`. The only way
to end early is pressing Right Arrow (`exit_early`). There is no support for:

- User-signaled episode completion (e.g. gripper gesture like double-close)
- Variable-length episodes without keyboard interaction
- Automatic detection that a task is complete
- Countdown or progress indication

**Required fix**: Episodes and resets should end when the user signals completion,
not after a fixed timer. Gesture-based signals (e.g. double-close gripper = "done")
would keep hands on the robot. Keyboard should remain as fallback.

### S4: System-wide keyboard hooks block other applications

**Severity: Medium**

`pynput.keyboard.Listener` registers a **global X11 keyboard hook**:

- Arrow keys and Escape are intercepted **in any window** — pressing Right Arrow
  in a terminal or editor triggers `exit_early`
- On Linux/X11 this may require elevated permissions
- On Wayland, pynput often doesn't work at all (falls back to headless mode)
- No way to scope the listener to just the recording terminal

**Required fix**: Replace global keyboard hooks with terminal-scoped input
(e.g. reading stdin in raw mode, or using a focused GUI window). Alternatively,
use non-keyboard signals entirely (gripper gestures, foot pedal, voice).

### S5: Hard error on existing output directory

**Severity: Medium (workflow friction)**

`LeRobotDataset.create()` fails if the repo directory already exists (unless
`--resume=True` is explicitly passed). There's no interactive prompt.

**Required fix**: On directory conflict, interactively ask:
- Resume existing dataset?
- Choose a new name? (with auto-suggestion, e.g. append `-2`)
- Just press Enter for auto-naming

### S6: Poor voice audio output

**Severity: Medium**

On Linux, TTS uses `spd-say` → espeak — robotic-sounding, hard to understand in
noisy environments. The messages are minimal strings with no richness.

Missing capabilities:
- No configurable TTS engine (e.g. piper, Coqui, or cloud TTS)
- No voice input for labeling or commands
- No rich audio feedback (countdown beeps, success/failure sounds)
- No subgoal labeling via voice during recording

**Required fix**:
- Replace with a better local TTS model (e.g. piper-tts for low-latency offline)
- Add distinctive audio cues (beeps, tones) for state transitions
- Plan voice input as extension: speech-to-text for dense subgoal labeling,
  voice commands for episode control

---

## 9. Additional Shortcomings Discovered During Analysis

### S7: Resolution fails hard instead of adapting

Requesting a non-native camera resolution crashes at connect time with
`RuntimeError`. There is no auto-negotiation to the nearest supported resolution
and no software rescaling fallback.

### S8: Sequential multi-camera reads

Cameras are read one-by-one in `get_observation()`, adding latency proportional
to camera count. With background threads this is usually fast (~1-5ms per camera
if a frame is ready), but under load the sequential pattern can compound delays
and increase inter-camera timestamp skew.

### S9: Concatenated multi-episode video files hurt training performance

LeRobot concatenates multiple episodes into a single MP4 per camera, splitting
only when the file reaches a size threshold (default 200 MB). The file structure
looks like:

```
videos/observation.images.laptop/chunk-000/file-000.mp4   ← episodes 0,1,2,...
videos/observation.images.laptop/chunk-000/file-001.mp4   ← continues after 200MB
```

Episode boundaries are tracked via `from_timestamp` / `to_timestamp` in parquet
metadata. During training, loading an episode requires seeking into the
concatenated MP4 at the correct offset:

```python
from_timestamp = ep[f"videos/{vid_key}/from_timestamp"]
shifted_query_ts = [from_timestamp + ts for ts in query_ts]
frames = decode_video_frames(video_path, shifted_query_ts, ...)
```

Problems with this approach:

1. **Random access overhead** — seeking into the middle of a large MP4 requires
   parsing the container index. With many short episodes in a 200 MB file, there's
   repeated seeking across the file.
2. **No parallel episode loading** — two DataLoader workers loading episodes from
   the same MP4 contend on the same file descriptor / OS page cache region.
3. **Corruption blast radius** — a corrupted file loses all episodes in that file,
   not just one.
4. **No trivial episode deletion** — removing an episode requires re-encoding or
   rewriting the concatenated file.
5. **Concatenation cost** — `concatenate_video_files()` is an extra remuxing step
   during recording.

One file per episode per camera (e.g. `videos/observation.images.laptop/episode_000042.mp4`)
would be simpler and better for randomized training access. Modern filesystems
handle thousands of small files without issue, and the training access pattern
strongly favors independent files.

---

## 10. NVENC Hardware Encoding Benchmark (Jetson AGX Thor)

> Tested 2026-03-24 on NVIDIA Jetson AGX Thor (aarch64, Tegra, L4T 38.4.0).
> PyAV 15.1.0 with `h264_nvenc` and `hevc_nvenc` codecs available.

### Single stream performance

| Codec | Throughput | Per-frame latency (median) | Output bitrate |
|---|---|---|---|
| **h264_nvenc** (GPU) | 380 FPS | 1.4 ms | 8.5 Mbps |
| **hevc_nvenc** (GPU) | 254 FPS | — | 4.7 Mbps |
| libx264 ultrafast (CPU) | 143 FPS | — | 35 Mbps |

Settings: 640×480 @30fps, `preset=p4`, `tune=ull` (ultra-low-latency), `max_b_frames=0`.

### Concurrent multi-camera streams (pre-generated frames, no CPU overhead)

| Scenario | Effective throughput | Wall clock (300 frames) |
|---|---|---|
| 3× h264_nvenc concurrent | **711 FPS total** (~237 each) | 1.27s |
| 3× hevc_nvenc concurrent | ~710 FPS total | ~1.3s |

The GPU encoder handles 3 concurrent 640×480 streams with massive headroom.

### Per-frame encode latency (h264_nvenc, single stream)

| Percentile | Latency |
|---|---|
| Median | 1.39 ms |
| P95 | 1.65 ms |
| P99 | 3.58 ms |
| Max | 163 ms (first-frame init) |
| Budget at 30 Hz | 33.3 ms |

### Bottleneck analysis

The observed slowdown in concurrent encoding with live frame generation (~15 FPS
per stream) was caused by **synthetic numpy noise generation on CPU** (~8.4ms/frame),
not the GPU encoder. In a real recording scenario:
- Camera frames arrive from background threads (no numpy generation)
- `VideoFrame.from_ndarray()` conversion is ~0.2ms per frame
- NVENC encoding is ~1.4ms per frame
- Total per-frame overhead: ~1.6ms — well within 33ms budget at 30 Hz

### Recommendation: NVENC streaming encoding as default

NVENC H.264 should be the **default encoding path** in the replacement recording
script. The architecture:

```
Camera background thread
    │
    ▼
Control loop (main thread)
    ├── robot.send_action()
    ├── dataset.add_frame()  ──→  encoder_queue (per camera)
    │                                    │
    │                              Encoder thread (per camera)
    │                                    │
    │                              h264_nvenc via PyAV
    │                                    │
    │                              ├──→  MP4 file on disk (per episode)
    │                              └──→  rr.VideoStream (optional, for Rerun live view)
    │
    └── precise_sleep()
```

Key design points:
- **Always-on streaming**: frames are encoded in real-time, never buffered as PNGs
- **Per-episode MP4 files**: no concatenation, clean separation (see S9)
- **Zero-blocking episode save**: episode boundary just closes the MP4 container
  and opens a new one — no post-hoc encoding step
- **No B-frames** (`max_b_frames=0`): required for streaming compatibility
- **Fallback**: if NVENC is not available (non-NVIDIA platform), fall back to
  `libx264` with `preset=ultrafast, tune=zerolatency` which still achieves 143 FPS
  on this machine — adequate for 3 cameras at 30 Hz

### Visualization strategy: terminal-first, Rerun optional

The operator's hands are on the robot during recording — a GUI is not the primary
interface. The default should be a **terminal-only UI** with compact status output:

```
Episode 3/25 | recording | 29.8 Hz | cam0 ✓ cam1 ✓ cam2 ✓ | 00:42
```

**Rerun is opt-in** (`--visualize` flag) for debugging sessions (camera feed issues,
timing problems, joint drift). When enabled, use the current proven `static=True`
image logging approach — low overhead, no timeline complexity.

**Rerun `VideoStream` dual-pipe is deferred**: feeding NVENC packets to both MP4
and `rr.VideoStream` is architecturally appealing but adds complexity and depends
on Rerun's unstable `VideoStream` API (no B-frame support, `VideoFrameReference`
not yet working — issue #10422). Not worth blocking on for v1.

---

## 11. Summary: What a Replacement Must Address

| # | Issue | Priority |
|---|---|---|
| S1 | Continuous teleop (never freeze follower) | **Critical** |
| S2 | Non-blocking encoding (zero teleop downtime) | **High** |
| S3 | User-signaled episode/reset boundaries | **High** |
| S4 | Scoped input (no global key hooks) | **Medium** |
| S5 | Graceful handling of existing datasets | **Medium** |
| S6 | Better TTS + voice input planning | **Medium** |
| S7 | Graceful resolution fallback | **Low** |
| S8 | Parallel multi-camera reads | **Low** |
| S9 | Per-episode video files for training performance | **Medium** |
