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

"""LeRobot v3 compatible dataset writer.

Writes parquet data files, episode metadata, info.json, tasks.parquet, and
stats.json in the standard LeRobot v3 directory layout. Not a thread — methods
are called from the main thread during episode boundary handling.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from lerobot.datasets.compute_stats import RunningQuantileStats

from lerobot_robot_trlc_dk1.recorder.nvenc_encoder import EncoderResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Feature schema builder
# ---------------------------------------------------------------------------

def _ordered_dict(*pairs) -> dict:
    """Create a dict preserving insertion order (for JSON key order)."""
    return dict(pairs)


def build_features_schema(
    camera_keys: list[str],
    camera_height: int,
    camera_width: int,
    fps: int,
    video_codec: str = "h264",
    obs_state_keys: list[str] | None = None,
) -> dict:
    """Build the ``features`` dict for info.json.

    Feature order matches LeRobot reference datasets:
    action → observation.state → video features → metadata columns.
    Key order within features: dtype, names, shape (matching reference).
    """
    # Default: full 40-element observation
    if obs_state_keys is None:
        from lerobot_robot_trlc_dk1.recorder.recorder_thread import _ALL_OBS_STATE_KEYS
        obs_state_keys = _ALL_OBS_STATE_KEYS

    features: dict = {}

    # Action FIRST (matching reference dataset order)
    features["action"] = _ordered_dict(
        ("dtype", "float32"),
        ("names", [
            "left_joint_1.pos", "left_joint_2.pos", "left_joint_3.pos",
            "left_joint_4.pos", "left_joint_5.pos", "left_joint_6.pos",
            "left_gripper.pos",
            "right_joint_1.pos", "right_joint_2.pos", "right_joint_3.pos",
            "right_joint_4.pos", "right_joint_5.pos", "right_joint_6.pos",
            "right_gripper.pos",
        ]),
        ("shape", [14]),
    )

    # Observation state SECOND (dynamic based on --obs-signals)
    features["observation.state"] = _ordered_dict(
        ("dtype", "float32"),
        ("names", list(obs_state_keys)),
        ("shape", [len(obs_state_keys)]),
    )

    # Video features THIRD
    for cam_key in camera_keys:
        features[f"observation.images.{cam_key}"] = _ordered_dict(
            ("dtype", "video"),
            ("shape", [camera_height, camera_width, 3]),
            ("names", ["height", "width", "channels"]),
            ("info", {
                "video.height": camera_height,
                "video.width": camera_width,
                "video.codec": video_codec,
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "video.fps": fps,
                "video.channels": 3,
                "has_audio": False,
            }),
        )

    # Scalar metadata columns LAST (names: null required by LeRobot reference format)
    features["timestamp"] = _ordered_dict(("dtype", "float32"), ("shape", [1]), ("names", None))
    features["frame_index"] = _ordered_dict(("dtype", "int64"), ("shape", [1]), ("names", None))
    features["episode_index"] = _ordered_dict(("dtype", "int64"), ("shape", [1]), ("names", None))
    features["index"] = _ordered_dict(("dtype", "int64"), ("shape", [1]), ("names", None))
    features["task_index"] = _ordered_dict(("dtype", "int64"), ("shape", [1]), ("names", None))

    return features


# ---------------------------------------------------------------------------
# DatasetWriter
# ---------------------------------------------------------------------------

class DatasetWriter:
    """Writes LeRobot v3 compatible dataset files.

    Not a thread — methods are called from the main thread during episode
    boundary handling. Teleop runs in a separate thread and is unaffected.
    """

    def __init__(
        self,
        dataset_dir: Path,
        fps: int,
        features: dict,
        robot_type: str,
        task: str,
        chunks_size: int = 1000,
        start_episode: int = 0,
        start_frame: int = 0,
    ):
        self.dataset_dir = Path(dataset_dir)
        self.fps = fps
        self.features = features
        self.robot_type = robot_type
        self.task = task
        self.chunks_size = chunks_size

        self.total_episodes = start_episode
        self.global_frame_index = start_frame

        # Aggregate stats across all episodes for stats.json
        self._agg_stats: dict[str, RunningQuantileStats] = {}

        # Accumulated episode metadata rows (rewritten each save)
        self._episode_rows: list[dict] = []

        self._init_dataset_dir()

    # -- Initialization -----------------------------------------------------

    def _init_dataset_dir(self):
        """Create directory structure and write initial metadata."""
        (self.dataset_dir / "meta" / "episodes" / "chunk-000").mkdir(
            parents=True, exist_ok=True
        )
        (self.dataset_dir / "data" / "chunk-000").mkdir(
            parents=True, exist_ok=True
        )
        self._write_info_json()
        self._write_tasks_parquet()

    def _chunk_file(self, ep_index: int) -> tuple[int, int]:
        """Map episode index to chunk/file indices."""
        return ep_index // self.chunks_size, ep_index % self.chunks_size

    # -- Episode save -------------------------------------------------------

    def save_episode(
        self,
        ep_index: int,
        scalar_frames: list[dict],
        video_results: dict[str, EncoderResult],
    ):
        """Finalize one episode: write data parquet, episode metadata, update info.json.

        Args:
            ep_index: Episode number.
            scalar_frames: List of scalar frame dicts from the recorder thread.
                Each dict has keys: observation.state (ndarray), action (ndarray[14]),
                timestamp (float32), frame_index (int), episode_index (int), task_index (int).
            video_results: cam_key → EncoderResult from encoder threads.
        """
        if not scalar_frames:
            logger.warning("save_episode(%d): no frames, skipping", ep_index)
            return

        n_frames = len(scalar_frames)
        from_index = self.global_frame_index
        to_index = from_index + n_frames

        # 1. Compute per-episode scalar stats (for episode metadata)
        scalar_stats = self._compute_scalar_episode_stats(
            scalar_frames, from_index
        )

        # 2. Write data parquet
        self._write_data_parquet(ep_index, scalar_frames, from_index)

        # 3. Append episode metadata row and rewrite metadata parquet
        self._append_episode_metadata(
            ep_index, n_frames, from_index, to_index, video_results,
            scalar_stats,
        )

        # 3. Update aggregate stats
        self._update_aggregate_stats(scalar_frames, video_results)

        # 4. Update totals
        self.global_frame_index = to_index
        self.total_episodes = ep_index + 1
        self._write_info_json()

        logger.info(
            "Episode %d saved: %d frames, global_index %d→%d",
            ep_index, n_frames, from_index, to_index,
        )

    # -- Data parquet -------------------------------------------------------

    def _write_data_parquet(
        self, ep_index: int, frames: list[dict], from_index: int
    ):
        """Write one parquet file per episode with scalar + vector features.

        Path: data/chunk-{chunk}/file-{file}.parquet

        Vector features (observation.state, action) are stored as list columns
        (Arrow list<float>) matching HuggingFace Sequence format.
        Scalar features (timestamp, frame_index, etc.) are stored as plain values.
        """
        chunk, file_idx = self._chunk_file(ep_index)
        path = (
            self.dataset_dir / "data"
            / f"chunk-{chunk:03d}" / f"file-{file_idx:03d}.parquet"
        )
        path.parent.mkdir(parents=True, exist_ok=True)

        # Build columnar arrays
        n = len(frames)
        indices = list(range(from_index, from_index + n))
        frame_indices = [f["frame_index"] for f in frames]
        episode_indices = [f["episode_index"] for f in frames]
        timestamps = [float(f["timestamp"]) for f in frames]
        task_indices = [f["task_index"] for f in frames]

        # Vector features as variable-length lists of float32.
        # hyparquet (JS reader used by HF visualizer) cannot read
        # fixed_size_list — it requires plain list<float>.
        obs_states = [f["observation.state"].tolist() for f in frames]
        actions = [f["action"].tolist() for f in frames]
        obs_dim = len(obs_states[0])
        act_dim = len(actions[0])

        # Column order matches LeRobot reference datasets:
        # action, observation.state first, then metadata columns
        table = pa.table({
            "action": pa.array(actions, type=pa.list_(pa.float32())),
            "observation.state": pa.array(obs_states, type=pa.list_(pa.float32())),
            "timestamp": pa.array(timestamps, type=pa.float32()),
            "frame_index": pa.array(frame_indices, type=pa.int64()),
            "episode_index": pa.array(episode_indices, type=pa.int64()),
            "index": pa.array(indices, type=pa.int64()),
            "task_index": pa.array(task_indices, type=pa.int64()),
        })

        # Embed HuggingFace feature metadata in parquet schema.
        # hyparquet (the JS parquet reader used by the HF dataset visualizer)
        # requires this to correctly deserialize fixed_size_list columns.
        hf_meta = {
            "info": {
                "features": {
                    "action": {"feature": {"dtype": "float32", "_type": "Value"}, "length": act_dim, "_type": "List"},
                    "observation.state": {"feature": {"dtype": "float32", "_type": "Value"}, "length": obs_dim, "_type": "List"},
                    "timestamp": {"dtype": "float32", "_type": "Value"},
                    "frame_index": {"dtype": "int64", "_type": "Value"},
                    "episode_index": {"dtype": "int64", "_type": "Value"},
                    "index": {"dtype": "int64", "_type": "Value"},
                    "task_index": {"dtype": "int64", "_type": "Value"},
                }
            }
        }
        table = table.replace_schema_metadata({
            b"huggingface": json.dumps(hf_meta).encode(),
            **(table.schema.metadata or {}),
        })

        pq.write_table(table, path, compression="snappy")

    # -- Episode metadata ---------------------------------------------------

    def _compute_scalar_episode_stats(
        self, scalar_frames: list[dict], from_index: int,
    ) -> dict[str, dict[str, list]]:
        """Compute per-episode stats for all scalar features.

        Returns dict of feature_name → {stat_name: [values]}.
        """
        stats_out: dict[str, dict[str, list]] = {}
        n = len(scalar_frames)

        # observation.state and action
        for feat_key in ("observation.state", "action"):
            batch = np.stack([f[feat_key] for f in scalar_frames])
            rqs = RunningQuantileStats()
            rqs.update(batch)
            try:
                raw = rqs.get_statistics()
                stats_out[feat_key] = {k: v.tolist() for k, v in raw.items()}
            except ValueError:
                pass

        # Scalar metadata features: timestamp, frame_index, episode_index, index, task_index
        for feat_key in ("timestamp", "frame_index", "episode_index", "task_index"):
            if feat_key == "timestamp":
                vals = np.array([float(f[feat_key]) for f in scalar_frames], dtype=np.float32)
            elif feat_key == "frame_index":
                vals = np.array([f[feat_key] for f in scalar_frames], dtype=np.float64)
            elif feat_key == "episode_index":
                vals = np.array([f[feat_key] for f in scalar_frames], dtype=np.float64)
            elif feat_key == "task_index":
                vals = np.array([f[feat_key] for f in scalar_frames], dtype=np.float64)
            else:
                continue
            rqs = RunningQuantileStats()
            rqs.update(vals.reshape(-1, 1))
            try:
                raw = rqs.get_statistics()
                stats_out[feat_key] = {k: v.tolist() for k, v in raw.items()}
            except ValueError:
                pass

        # index (global)
        indices = np.arange(from_index, from_index + n, dtype=np.float64)
        rqs = RunningQuantileStats()
        rqs.update(indices.reshape(-1, 1))
        try:
            raw = rqs.get_statistics()
            stats_out["index"] = {k: v.tolist() for k, v in raw.items()}
        except ValueError:
            pass

        return stats_out

    def _append_episode_metadata(
        self,
        ep_index: int,
        n_frames: int,
        from_index: int,
        to_index: int,
        video_results: dict[str, EncoderResult],
        scalar_stats: dict[str, dict[str, list]] | None = None,
    ):
        """Append episode metadata row and rewrite the metadata parquet."""
        chunk, file_idx = self._chunk_file(ep_index)

        row: dict = {
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

        # Per-video metadata + stats
        # to_timestamp uses n_frames (trimmed scalar count), NOT result.frame_count
        # (full MP4 count) — the MP4 may contain extra trailing gesture frames
        # that were trimmed from the scalar data.
        for cam_key, result in video_results.items():
            vk = f"observation.images.{cam_key}"
            v_chunk, v_file = self._chunk_file(ep_index)
            row[f"videos/{vk}/chunk_index"] = v_chunk
            row[f"videos/{vk}/file_index"] = v_file
            row[f"videos/{vk}/from_timestamp"] = 0.0
            row[f"videos/{vk}/to_timestamp"] = n_frames / self.fps

            for stat_key, stat_val in result.stats.items():
                row[f"stats/{vk}/{stat_key}"] = stat_val.tolist()

        # Per-episode scalar stats (observation.state, action, timestamp, etc.)
        if scalar_stats:
            for feat_key, feat_stats in scalar_stats.items():
                for stat_key, stat_val in feat_stats.items():
                    row[f"stats/{feat_key}/{stat_key}"] = stat_val

        self._episode_rows.append(row)
        self._write_episodes_parquet()

    def _write_episodes_parquet(self):
        """Rewrite the episode metadata parquet with all accumulated rows."""
        path = (
            self.dataset_dir / "meta" / "episodes"
            / "chunk-000" / "file-000.parquet"
        )

        # Convert list of dicts to a pyarrow table
        # Each row may have different columns (different cameras), so we
        # need to handle the union of all keys.
        if not self._episode_rows:
            return

        all_keys = set()
        for row in self._episode_rows:
            all_keys.update(row.keys())

        columns: dict[str, list] = {k: [] for k in sorted(all_keys)}
        for row in self._episode_rows:
            for k in columns:
                columns[k].append(row.get(k))

        table = pa.Table.from_pydict(columns)
        pq.write_table(table, path, compression="snappy")

    # -- Aggregate stats ----------------------------------------------------

    def _update_aggregate_stats(
        self, scalar_frames: list[dict], video_results: dict[str, EncoderResult]
    ):
        """Update global RunningQuantileStats for stats.json."""
        if not scalar_frames:
            return

        # Observation state
        obs_batch = np.stack([f["observation.state"] for f in scalar_frames])
        if "observation.state" not in self._agg_stats:
            self._agg_stats["observation.state"] = RunningQuantileStats()
        self._agg_stats["observation.state"].update(obs_batch)

        # Action
        act_batch = np.stack([f["action"] for f in scalar_frames])
        if "action" not in self._agg_stats:
            self._agg_stats["action"] = RunningQuantileStats()
        self._agg_stats["action"].update(act_batch)

        # Video features: per-episode stats are already computed in encoder.
        # For global aggregate, feed the per-episode mean as a sample.
        # This gives correct global mean and reasonable quantile estimates.
        for cam_key, result in video_results.items():
            vk = f"observation.images.{cam_key}"
            if vk not in self._agg_stats:
                self._agg_stats[vk] = RunningQuantileStats()
            if "mean" in result.stats:
                self._agg_stats[vk].update(
                    result.stats["mean"].reshape(1, -1)
                )

    # -- info.json ----------------------------------------------------------

    def _compute_dir_size_mb(self, dir_path: Path, pattern: str) -> int:
        """Compute total size of files matching pattern in directory, in MB."""
        total = sum(f.stat().st_size for f in dir_path.rglob(pattern)) if dir_path.exists() else 0
        return round(total / (1024 * 1024))

    def _write_info_json(self):
        """Write meta/info.json with current totals."""
        info = {
            "codebase_version": "v3.0",
            "robot_type": self.robot_type,
            "total_episodes": self.total_episodes,
            "total_frames": self.global_frame_index,
            "total_tasks": 1,
            "chunks_size": self.chunks_size,
            "data_files_size_in_mb": self._compute_dir_size_mb(self.dataset_dir / "data", "*.parquet"),
            "video_files_size_in_mb": self._compute_dir_size_mb(self.dataset_dir / "videos", "*.mp4"),
            "fps": self.fps,
            "splits": {"train": f"0:{self.total_episodes}"},
            "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
            "video_path": (
                "videos/{video_key}/chunk-{chunk_index:03d}/"
                "file-{file_index:03d}.mp4"
            ),
            "features": self.features,
        }
        path = self.dataset_dir / "meta" / "info.json"
        path.write_text(json.dumps(info, indent=2))

    # -- tasks.parquet ------------------------------------------------------

    def _write_tasks_parquet(self):
        """Write meta/tasks.parquet."""
        table = pa.table({
            "task_index": pa.array([0], type=pa.int64()),
            "task": pa.array([self.task]),
        })
        pq.write_table(
            table, self.dataset_dir / "meta" / "tasks.parquet",
            compression="snappy",
        )

    # -- Finalize -----------------------------------------------------------

    def finalize(self):
        """Write stats.json and final info.json. Called once at end of session."""
        # Write aggregate stats
        stats_dict: dict = {}
        for key, rqs in self._agg_stats.items():
            try:
                raw = rqs.get_statistics()
                stats_dict[key] = {k: v.tolist() for k, v in raw.items()}
            except ValueError:
                logger.warning("Could not compute stats for %s (not enough data)", key)

        stats_path = self.dataset_dir / "meta" / "stats.json"
        stats_path.write_text(json.dumps(stats_dict, indent=2))

        # Final info.json
        self._write_info_json()

        logger.info(
            "Dataset finalized: %d episodes, %d frames → %s",
            self.total_episodes, self.global_frame_index, self.dataset_dir,
        )
