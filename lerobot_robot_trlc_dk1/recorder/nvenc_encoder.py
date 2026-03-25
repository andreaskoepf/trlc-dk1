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
    """Pre-open MP4 container during countdown (before recording starts)."""
    episode_index: int


@dataclass
class StartEpisode:
    """Signal the encoder to begin accepting frames."""
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

    # Compute image stats on every Nth frame to avoid GIL saturation.
    # RunningQuantileStats.update on a 1280x720 image takes ~70ms of
    # GIL-holding numpy work.  With 3 encoders, every-frame stats would
    # consume ~200ms of GIL time per 33ms recording period — starving
    # the teleop, recorder, and UI threads.
    STATS_EVERY_N_FRAMES = 30  # ~1 stats update per second at 30fps

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
                # Pre-open container during countdown, then wait for StartEpisode
                self._encode_episode(msg.episode_index, wait_for_start=True)
            elif isinstance(msg, StartEpisode):
                # No PrepareEpisode was sent — open + encode immediately (fallback)
                self._encode_episode(msg.episode_index, wait_for_start=False)
        logger.info("Encoder thread stopped: %s", self.cam_key)

    def _open_container(self, ep_index: int, prime_encoder: bool = False):
        """Open MP4 container and NVENC stream (expensive, do during countdown).

        If prime_encoder=True, encode and discard a dummy frame to force
        NVENC lazy initialization (session setup, GPU buffer alloc) so the
        first real frame encodes without a ~400ms GIL stall.
        """
        mp4_path = self._episode_path(ep_index)
        mp4_path.parent.mkdir(parents=True, exist_ok=True)

        container = av.open(str(mp4_path), mode="w")
        stream = container.add_stream(self.codec, rate=self.fps)
        stream.width = self.width
        stream.height = self.height
        stream.pix_fmt = "yuv420p"
        stream.options = self.codec_options

        pts_offset = 0
        if prime_encoder:
            # Force NVENC lazy init by encoding+muxing a dummy black frame.
            # This frame will be overwritten by the real first frame at pts=0,
            # so we use pts=0 for the dummy and start real frames at pts=1.
            dummy = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            vf = av.VideoFrame.from_ndarray(dummy, format="rgb24")
            vf.pts = 0
            for pkt in stream.encode(vf):
                container.mux(pkt)
            pts_offset = 1

        logger.debug("Encoder %s: container pre-opened → %s", self.cam_key, mp4_path)
        return container, stream, mp4_path, pts_offset

    def _encode_episode(self, ep_index: int, wait_for_start: bool = True):
        """Encode frames for one episode into a single MP4.

        If wait_for_start=True (PrepareEpisode path), the container is opened
        and NVENC is primed with a dummy frame during countdown.
        If wait_for_start=False (direct StartEpisode), encoding starts immediately.
        """
        container, stream, mp4_path, pts_offset = self._open_container(
            ep_index, prime_encoder=wait_for_start,
        )

        if wait_for_start:
            # Wait for StartEpisode (container is ready, just waiting for GO)
            while not self._stop:
                try:
                    msg = self.frame_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                if msg is None:
                    container.close()
                    mp4_path.unlink(missing_ok=True)
                    return
                if isinstance(msg, StartEpisode):
                    break
                # Ignore other messages while waiting

        frame_count = 0
        stats_frames: list[np.ndarray] = []

        logger.debug("Encoder %s: episode %d recording", self.cam_key, ep_index)

        while True:
            try:
                msg = self.frame_queue.get(timeout=5.0)
            except queue.Empty:
                if self._stop:
                    break
                continue

            if msg is None:
                break

            if isinstance(msg, EndEpisode):
                break

            if not isinstance(msg, VideoFrame):
                continue

            # Encode frame (PyAV releases GIL during NVENC encode)
            try:
                video_frame = av.VideoFrame.from_ndarray(msg.image, format="rgb24")
                video_frame.pts = frame_count + pts_offset
                for packet in stream.encode(video_frame):
                    container.mux(packet)
            except Exception:
                if frame_count == 0:
                    logger.exception(
                        "Encoder %s: first frame encode failed (CUDA/NVENC init error?)",
                        self.cam_key,
                    )
                    break
                else:
                    logger.exception("Encoder %s: encode error at frame %d", self.cam_key, frame_count)
                    frame_count += 1
                    continue

            # Stash every Nth frame for deferred stats computation.
            if frame_count % self.STATS_EVERY_N_FRAMES == 0:
                stats_frames.append(msg.image)

            frame_count += 1

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
            frame_count=frame_count,
            stats=ep_stats,
        )
        self.result_queue.put(result)

        logger.debug(
            "Encoder %s: episode %d done (%d frames, %.1f MB)",
            self.cam_key,
            ep_index,
            frame_count,
            mp4_path.stat().st_size / 1e6 if mp4_path.exists() else 0,
        )
