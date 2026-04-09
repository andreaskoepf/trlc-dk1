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

"""Per-camera NVENC H.264 encoder thread with per-episode MP4 output.

Each NvencEncoder runs in its own thread, consuming VideoFrame messages from
a queue and encoding them to MP4 via PyAV's h264_nvenc backend. Episode
boundaries are signaled via StartEpisode / EndEpisode typed messages.
"""

from __future__ import annotations

import logging
import queue
import threading
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

import av
import numpy as np

from lerobot.datasets.compute_stats import RunningQuantileStats

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Typed messages for the encoder queue
# ---------------------------------------------------------------------------

@dataclass
class PrepareEpisode:
    """Pre-open MP4 container and start encoding (cameras rolling)."""
    episode_index: int


@dataclass
class StartEpisode:
    """Mark the start of the real episode data in an already-rolling encoder."""
    episode_index: int


@dataclass
class VideoFrame:
    """A single camera frame to encode."""
    frame_index: int
    image: np.ndarray  # HWC uint8 (RGB)


@dataclass
class EndEpisode:
    """Signal the encoder to finalize the current episode MP4."""
    pass


@dataclass
class EncoderResult:
    """Result posted by encoder after finishing an episode."""
    episode_index: int
    mp4_path: Path
    frame_count: int
    pts_offset: int = 0  # number of priming frames before real data
    stats: dict[str, np.ndarray] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Codec detection
# ---------------------------------------------------------------------------

def detect_codec(preferred: str = "h264_nvenc") -> str:
    """Return *preferred* if available, otherwise fall back to libx264."""
    try:
        codec = av.codec.Codec(preferred, "w")
        if codec is not None:
            return preferred
    except Exception:
        pass
    logger.warning("Codec %s not available, falling back to libx264", preferred)
    return "libx264"


# ---------------------------------------------------------------------------
# NvencEncoder
# ---------------------------------------------------------------------------

class NvencEncoder:
    """Encodes camera frames to per-episode MP4 files.

    One encoder instance per camera. Runs in its own daemon thread.
    Frames are received via ``frame_queue``; results are posted to
    ``result_queue`` after each episode is finalized.
    """

    def __init__(
        self,
        cam_key: str,
        width: int,
        height: int,
        fps: int,
        codec: str = "h264_nvenc",
        videos_dir: Path = Path(),
        chunks_size: int = 1000,
        queue_maxsize: int = 60,
    ):
        self.cam_key = cam_key
        self.video_key = f"observation.images.{cam_key}"
        self.width = width
        self.height = height
        self.fps = fps
        self.codec = codec
        self.videos_dir = videos_dir
        self.chunks_size = chunks_size
        # Image stats on every Nth frame to avoid GIL saturation.
        # RunningQuantileStats.update on a 1280x720 image takes ~70ms of
        # GIL-holding numpy work.  With 3 encoders, every-frame stats would
        # consume ~200ms of GIL time per 33ms recording period — starving
        # the teleop, recorder, and UI threads.
        self.stats_every_n_frames = fps  # ~1 stats update per second

        self.frame_queue: queue.Queue = queue.Queue(maxsize=queue_maxsize)
        self.result_queue: queue.Queue[EncoderResult] = queue.Queue()

        # Build codec options
        self.codec_options: dict[str, str] = {}
        if "nvenc" in codec:
            self.codec_options = {
                "preset": "p4",
                "tune": "ull",
                "max_b_frames": "0",
            }
        elif codec == "libx264":
            self.codec_options = {
                "preset": "ultrafast",
                "tune": "zerolatency",
            }

        self._thread: threading.Thread | None = None
        self._stop = False

    # -- Path helpers -------------------------------------------------------

    def _episode_path(self, ep_index: int) -> Path:
        """Standard LeRobot v3 video path: chunk-NNN/file-NNN.mp4"""
        chunk = ep_index // self.chunks_size
        file_idx = ep_index % self.chunks_size
        return (
            self.videos_dir
            / self.video_key
            / f"chunk-{chunk:03d}"
            / f"file-{file_idx:03d}.mp4"
        )

    # -- Lifecycle ----------------------------------------------------------

    def warmup(self) -> bool:
        """Encode a single dummy frame to trigger CUDA/NVENC initialization.

        Call this on the main thread BEFORE starting the encoder thread.
        If NVENC init fails (cuInit error), falls back to libx264 and
        returns False. Returns True if the configured codec works.
        """
        test_path = self.videos_dir / f".warmup_{self.cam_key}.mp4"
        test_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            container = av.open(str(test_path), mode="w")
            stream = container.add_stream(self.codec, rate=self.fps)
            stream.width = self.width
            stream.height = self.height
            stream.pix_fmt = "yuv420p"
            stream.time_base = Fraction(1, self.fps)
            stream.options = self.codec_options

            dummy = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            vf = av.VideoFrame.from_ndarray(dummy, format="rgb24")
            vf.pts = 0
            for pkt in stream.encode(vf):
                container.mux(pkt)
            for pkt in stream.encode():
                container.mux(pkt)
            container.close()
            test_path.unlink(missing_ok=True)
            logger.info("Encoder %s: %s warmup OK", self.cam_key, self.codec)
            return True
        except Exception as e:
            logger.warning(
                "Encoder %s: %s warmup FAILED (%s), falling back to libx264",
                self.cam_key, self.codec, e,
            )
            test_path.unlink(missing_ok=True)
            self.codec = "libx264"
            self.codec_options = {"preset": "ultrafast", "tune": "zerolatency"}
            return False

    def start(self):
        self._thread = threading.Thread(
            target=self._run, daemon=True, name=f"encoder-{self.cam_key}"
        )
        self._thread.start()

    def stop(self):
        self._stop = True
        # Wake up a potentially blocked get()
        try:
            self.frame_queue.put_nowait(None)
        except queue.Full:
            pass
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    # -- Main loop ----------------------------------------------------------

    def _run(self):
        """Main encoder loop — handles PrepareEpisode/StartEpisode/EndEpisode lifecycle."""
        logger.info("Encoder thread started: %s (codec=%s)", self.cam_key, self.codec)
        while not self._stop:
            try:
                msg = self.frame_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            if msg is None:
                break
            if isinstance(msg, PrepareEpisode):
                self._encode_episode(msg.episode_index, rolling=True)
            elif isinstance(msg, StartEpisode):
                # Fallback: no PrepareEpisode was sent — open + encode directly
                self._encode_episode(msg.episode_index, rolling=False)
        logger.info("Encoder thread stopped: %s", self.cam_key)

    def _encode_episode(self, ep_index: int, rolling: bool = False):
        """Encode frames for one episode into a single MP4.

        If rolling=True (PrepareEpisode path), the encoder opens the container
        and immediately starts encoding frames from the queue (cameras rolling
        during countdown). StartEpisode marks where the real episode begins.

        If rolling=False (direct StartEpisode fallback), encoding starts
        immediately with no pre-roll.

        The MP4 contains: [pre-roll frames] [episode frames] [gesture frames]
        Episode metadata from_timestamp/to_timestamp clips to the real data.
        """
        mp4_path = self._episode_path(ep_index)
        mp4_path.parent.mkdir(parents=True, exist_ok=True)

        container = av.open(str(mp4_path), mode="w")
        stream = container.add_stream(self.codec, rate=self.fps)
        stream.width = self.width
        stream.height = self.height
        stream.pix_fmt = "yuv420p"
        # Explicit time_base ensures pts=N maps to time N/fps regardless of
        # codec (h264_nvenc, hevc_nvenc, av1_nvenc all use the same base).
        stream.time_base = Fraction(1, self.fps)
        stream.options = self.codec_options

        total_frames = 0      # all frames in MP4 (pre-roll + episode)
        episode_frames = 0    # frames after StartEpisode
        pre_roll_frames = 0   # frames before StartEpisode
        recording = not rolling  # if not rolling, we're recording immediately
        stats_frames: list[np.ndarray] = []

        logger.debug("Encoder %s: episode %d %s → %s",
                      self.cam_key, ep_index,
                      "rolling" if rolling else "recording", mp4_path)

        while True:
            try:
                msg = self.frame_queue.get(timeout=5.0)
            except queue.Empty:
                if self._stop:
                    break
                continue

            if msg is None:
                break

            if isinstance(msg, StartEpisode):
                # Mark the boundary: everything before is pre-roll
                pre_roll_frames = total_frames
                recording = True
                logger.debug("Encoder %s: episode start at MP4 frame %d",
                             self.cam_key, total_frames)
                continue

            if isinstance(msg, EndEpisode):
                break

            if not isinstance(msg, VideoFrame):
                continue

            # Encode frame
            try:
                video_frame = av.VideoFrame.from_ndarray(msg.image, format="rgb24")
                video_frame.pts = total_frames
                for packet in stream.encode(video_frame):
                    container.mux(packet)
            except Exception:
                if total_frames == 0:
                    logger.exception(
                        "Encoder %s: first frame encode failed (CUDA/NVENC init error?)",
                        self.cam_key,
                    )
                    break
                else:
                    logger.exception("Encoder %s: encode error at frame %d",
                                     self.cam_key, total_frames)
                    total_frames += 1
                    continue

            # Stash every Nth frame for deferred stats (only during episode)
            if recording and episode_frames % self.stats_every_n_frames == 0:
                stats_frames.append(msg.image)

            total_frames += 1
            if recording:
                episode_frames += 1

        # Flush encoder
        for packet in stream.encode():
            container.mux(packet)
        container.close()

        # Compute image stats on deferred frames (outside recording hot path)
        ep_stats: dict[str, np.ndarray] = {}
        if stats_frames:
            stats = RunningQuantileStats()
            for img in stats_frames:
                stats.update(img.reshape(-1, 3).astype(np.float32))
            stats_frames.clear()
            try:
                ep_stats = stats.get_statistics()
            except ValueError:
                pass

        result = EncoderResult(
            episode_index=ep_index,
            mp4_path=mp4_path,
            frame_count=episode_frames,
            pts_offset=pre_roll_frames,
            stats=ep_stats,
        )
        self.result_queue.put(result)

        logger.debug(
            "Encoder %s: episode %d done (%d pre-roll + %d episode frames, %.1f MB)",
            self.cam_key, ep_index, pre_roll_frames, episode_frames,
            mp4_path.stat().st_size / 1e6 if mp4_path.exists() else 0,
        )
