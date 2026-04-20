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

"""Deferred-encoding alternative to :class:`NvencEncoder`.

Writes one JPEG per frame to disk during recording; MP4s are produced
offline by ``scripts/finalize_jpeg_recordings.py`` (which can use NVENC
when no inference is running, so the GPU is fully available).

Use this when:

- Other heavy GPU work is running concurrently with recording. On
  Jetson Thor under the fastwam inference stack at 30-60 Hz dispatch,
  live NVENC adds ~240 ms/cycle of extra latency and ~33 % per-camera
  frame-drop rate because the encoder's frame queue backs up. Deferred
  JPEG writing is disk-I/O-bound and doesn't touch the CUDA scheduler.
- No pre-roll is needed (JPEG writes are independent per frame).
  Contrast with NVENC, which drops the first ~1 s of frames until its
  session is hot.
- Episodes must stay recoverable if the run crashes — JPEGs are
  fsync-able per-frame; the offline script can pick up where things
  stopped.

Trade-offs vs. :class:`NvencEncoder`:

- ~1 GB/min of disk per 3-camera 60 fps recording (JPEG q=90).
- Offline encoding step required before training can consume the
  dataset (otherwise `videos/.../<file>.mp4` is missing).

Interface-compatible with :class:`NvencEncoder`: same ``frame_queue``
message protocol (PrepareEpisode / StartEpisode / VideoFrame /
EndEpisode / None), same ``result_queue`` with :class:`EncoderResult`.
Consumers can swap encoders without other code changes.

**Frame-accurate sync guarantee.** Each JPEG filename uses the
recorder's authoritative ``frame_index`` (the same integer that
indexes scalar parquet rows), NOT an encoder-internal counter.
That way, if a rare drop occurs, the scalar parquet's row
``frame_index=N`` ALWAYS corresponds to the JPEG
``frame_{N:06d}.jpg`` (or to a gap if that frame dropped).
``finalize_jpeg_recordings.py`` detects gaps and duplicates the
preceding frame so the resulting MP4 stays 1:1 with the parquet.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from lerobot_robot_trlc_dk1.recorder.nvenc_encoder import (
    EncoderResult,
    EndEpisode,
    PrepareEpisode,
    StartEpisode,
    VideoFrame,
)

logger = logging.getLogger(__name__)


class JpegOfflineEncoder:
    """Writes per-frame JPEGs to disk; offline script produces MP4s later.

    Output layout per episode (chunks_size=1000 gives one chunk dir per
    1000 episodes, matching NvencEncoder's path scheme)::

        {videos_dir}/observation.images.{cam_key}/chunk-NNN/file-NNN.d/
            frame_000000.jpg
            frame_000001.jpg
            ...

    The ``.d`` suffix marks the directory as a frame-sequence awaiting
    offline encoding. The final MP4 target is the same path without
    ``.d``: ``file-NNN.mp4``. :class:`DatasetWriter` does not need the
    MP4 to exist at save time — episode metadata uses paths derived
    from (ep_index, chunk_index), not from :attr:`EncoderResult.mp4_path`.
    """

    # RGB → BGR for cv2.imencode, which expects BGR.
    _IMENCODE_PARAMS = [cv2.IMWRITE_JPEG_QUALITY, 90,
                         cv2.IMWRITE_JPEG_OPTIMIZE, 0]

    # Marker suffix on the per-episode frame directory. Matched by the
    # offline encoding script.
    DIR_SUFFIX = ".d"

    def __init__(
        self,
        cam_key: str,
        width: int,
        height: int,
        fps: int,
        codec: str = "jpeg_offline",  # accepted + ignored
        videos_dir: Path = Path(),
        chunks_size: int = 1000,
        queue_maxsize: int = 600,  # 10 s of headroom at 60 Hz
        jpeg_quality: int = 90,
    ):
        self.cam_key = cam_key
        self.video_key = f"observation.images.{cam_key}"
        self.width = width
        self.height = height
        self.fps = fps
        # Surface a stable string so logs / dataset info.json report
        # something sensible. Consumers checking `"nvenc" in codec` will
        # get False, which is what we want (different downstream path).
        self.codec = "jpeg_offline"
        self.videos_dir = videos_dir
        self.chunks_size = chunks_size
        self.jpeg_quality = int(jpeg_quality)

        self.frame_queue: queue.Queue = queue.Queue(maxsize=queue_maxsize)
        self.result_queue: queue.Queue[EncoderResult] = queue.Queue()

        self._thread: threading.Thread | None = None
        self._stop = False

    # -- Lifecycle ---------------------------------------------------------

    def warmup(self) -> bool:
        """No-op. JPEG writing has no hardware session to warm up.
        Return True so callers that check for NVENC fallback don't
        downgrade the codec."""
        return True

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._run, daemon=True,
            name=f"jpeg-encoder-{self.cam_key}",
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop = True
        # Wake up a potentially blocked get()
        try:
            self.frame_queue.put_nowait(None)
        except queue.Full:
            pass
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    # -- Path helpers ------------------------------------------------------

    def _episode_dir(self, ep_index: int) -> Path:
        chunk = ep_index // self.chunks_size
        file_idx = ep_index % self.chunks_size
        return (
            self.videos_dir
            / self.video_key
            / f"chunk-{chunk:03d}"
            / f"file-{file_idx:03d}{self.DIR_SUFFIX}"
        )

    def _target_mp4_path(self, ep_index: int) -> Path:
        """Where the offline script will place the encoded MP4."""
        chunk = ep_index // self.chunks_size
        file_idx = ep_index % self.chunks_size
        return (
            self.videos_dir
            / self.video_key
            / f"chunk-{chunk:03d}"
            / f"file-{file_idx:03d}.mp4"
        )

    # -- Main loop ---------------------------------------------------------

    def _run(self) -> None:
        logger.info("JPEG encoder thread started: %s (quality=%d)",
                    self.cam_key, self.jpeg_quality)
        while not self._stop:
            try:
                msg = self.frame_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            if msg is None:
                break
            if isinstance(msg, PrepareEpisode):
                # RecorderThread sends PrepareEpisode for NVENC's
                # session warmup; JPEG has no such session, so we just
                # pre-create the output directory so writes below hit
                # an existing dir. Actual recording waits for
                # StartEpisode. Skipping this branch (not entering
                # _handle_episode) also means VideoFrames arriving
                # between PrepareEpisode and StartEpisode are dropped
                # — same semantics as NVENC's "pre-roll frames" that
                # JPEG mode intentionally doesn't record (per docstring).
                self._episode_dir(msg.episode_index).mkdir(
                    parents=True, exist_ok=True)
            elif isinstance(msg, StartEpisode):
                # Real episode start — drain frames until EndEpisode.
                self._handle_episode(msg.episode_index)

    def _handle_episode(self, ep_index: int) -> None:
        ep_dir = self._episode_dir(ep_index)
        ep_dir.mkdir(parents=True, exist_ok=True)

        t_start = time.perf_counter()
        # frames_written counts JPEGs written; max_frame_index tracks
        # the highest frame_index we received. Reported frame_count
        # in EncoderResult is max_frame_index + 1 so the dataset's
        # video span matches the scalar parquet span even when a few
        # frames dropped (the offline script fills gaps).
        frames_written = 0
        max_frame_index = -1
        bytes_written = 0
        while not self._stop:
            try:
                msg = self.frame_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            if msg is None:
                logger.warning("JPEG encoder %s: shutdown mid-episode %d "
                               "(%d JPEGs written, last idx=%d)",
                               self.cam_key, ep_index, frames_written,
                               max_frame_index)
                return
            if isinstance(msg, VideoFrame):
                # msg.image is HWC uint8 RGB per VideoFrame contract
                # (matches NvencEncoder's input). OpenCV expects BGR,
                # so swap channels before imencode.
                bgr = cv2.cvtColor(msg.image, cv2.COLOR_RGB2BGR)
                ok, jpeg_bytes = cv2.imencode(
                    ".jpg", bgr,
                    [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality],
                )
                if not ok:
                    logger.warning(
                        "JPEG encoder %s: imencode failed for frame %d",
                        self.cam_key, msg.frame_index)
                    continue
                # CRITICAL: filename uses msg.frame_index (authoritative
                # from RecorderThread), NOT a local counter. The scalar
                # parquet's `frame_index` column uses the same values,
                # so frame N in the MP4 (after offline encoding) will
                # always correspond to scalar row N. On dropped frames
                # this produces a filename gap that the offline script
                # repairs before ffmpeg encoding.
                frame_path = ep_dir / f"frame_{msg.frame_index:06d}.jpg"
                data = jpeg_bytes.tobytes()
                frame_path.write_bytes(data)
                bytes_written += len(data)
                frames_written += 1
                if msg.frame_index > max_frame_index:
                    max_frame_index = msg.frame_index
            elif isinstance(msg, PrepareEpisode):
                # NVENC parity — the recorder pre-announces the NEXT
                # episode via PrepareEpisode before the current one's
                # EndEpisode arrives. For JPEG mode we just pre-create
                # the target directory and carry on capturing the
                # current episode.
                self._episode_dir(msg.episode_index).mkdir(
                    parents=True, exist_ok=True)
            elif isinstance(msg, StartEpisode):
                # Real new-episode boundary without intervening
                # EndEpisode — shouldn't happen from a well-behaved
                # recorder, but handle defensively: finalize the
                # current one and recurse.
                logger.warning(
                    "JPEG encoder %s: StartEpisode %d received before "
                    "EndEpisode on %d; finalizing implicitly",
                    self.cam_key, msg.episode_index, ep_index)
                self._finalize_episode(
                    ep_index, frames_written, max_frame_index,
                    bytes_written, t_start)
                self._handle_episode(msg.episode_index)
                return
            elif isinstance(msg, EndEpisode):
                break

        self._finalize_episode(
            ep_index, frames_written, max_frame_index,
            bytes_written, t_start)

    def _finalize_episode(self, ep_index: int, frames_written: int,
                           max_frame_index: int,
                           bytes_written: int, t_start: float) -> None:
        dt = time.perf_counter() - t_start
        mp4_path = self._target_mp4_path(ep_index)
        # Report the FULL span of frame_index values we saw. The
        # scalar parquet also covers that span (row 0 .. max_frame_index).
        # A few dropped JPEGs within the span are repaired offline.
        # `frame_count` is the number of valid indices to span, not the
        # number of physical JPEGs on disk.
        reported_count = max(frames_written, max_frame_index + 1)
        result = EncoderResult(
            episode_index=ep_index,
            mp4_path=mp4_path,  # target path; file doesn't exist yet
            frame_count=reported_count,
            pts_offset=0,       # no pre-roll in JPEG mode
            stats={},           # image stats deferred to offline step
        )
        self.result_queue.put(result)
        if frames_written == reported_count:
            drop_note = ""
        else:
            drop_note = (f", {reported_count - frames_written} dropped "
                         "(offline script will pad)")
        logger.info(
            "JPEG encoder %s: episode %d done (%d JPEGs, %.1f MB, "
            "%.1f s wall; span 0..%d%s; target %s)",
            self.cam_key, ep_index, frames_written,
            bytes_written / 1e6, dt, max_frame_index, drop_note, mp4_path,
        )
