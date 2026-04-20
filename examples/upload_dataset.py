#!/usr/bin/env python3
"""Upload a recorded dk1-record dataset to HuggingFace Hub.

Generates a LeRobot-compatible README.md and uploads all dataset files.

Usage:
    # First login to HuggingFace:
    huggingface-cli login

    # Upload dataset:
    python examples/upload_dataset.py \
        --dataset-dir ./data/fold_towels \
        --repo-id your-username/dk1_fold_towels \
        --description "Bimanual towel folding demonstrations with DK1 robot"

    # Upload as private dataset:
    python examples/upload_dataset.py \
        --dataset-dir ./data/fold_towels \
        --repo-id your-username/dk1_fold_towels \
        --private

    # Just generate README without uploading:
    python examples/upload_dataset.py \
        --dataset-dir ./data/fold_towels \
        --repo-id your-username/dk1_fold_towels \
        --readme-only
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def generate_readme(
    repo_id: str,
    info: dict,
    task: str,
    description: str = "",
    license: str = "apache-2.0",
) -> str:
    """Generate a LeRobot-compatible README.md for the dataset."""

    total_episodes = info.get("total_episodes", 0)
    total_frames = info.get("total_frames", 0)
    fps = info.get("fps", 30)
    robot_type = info.get("robot_type", "unknown")
    features = info.get("features", {})

    # Calculate total duration
    total_seconds = total_frames / fps if fps > 0 else 0
    total_minutes = total_seconds / 60
    avg_episode_s = total_seconds / total_episodes if total_episodes > 0 else 0

    # Camera info
    cameras = []
    for key, feat in features.items():
        if feat.get("dtype") == "video":
            cam_info = feat.get("info", {})
            cameras.append({
                "key": key,
                "resolution": f"{cam_info.get('video.width', '?')}x{cam_info.get('video.height', '?')}",
                "codec": cam_info.get("video.codec", "?"),
                "fps": cam_info.get("video.fps", "?"),
            })

    # Observation/action info
    obs_state = features.get("observation.state", {})
    action = features.get("action", {})

    info_json_str = json.dumps(info, indent=4)

    camera_table = ""
    if cameras:
        camera_table = "| Camera | Resolution | Codec | FPS |\n|---|---|---|---|\n"
        for c in cameras:
            camera_table += f"| `{c['key']}` | {c['resolution']} | {c['codec']} | {c['fps']} |\n"

    readme = f"""---
license: {license}
task_categories:
- robotics
tags:
- LeRobot
configs:
- config_name: default
  data_files: data/*/*.parquet
---

This dataset was created using [LeRobot](https://github.com/huggingface/lerobot).

<a class="flex" href="https://huggingface.co/spaces/lerobot/visualize_dataset?path={repo_id}">
<img class="block dark:hidden" src="https://huggingface.co/datasets/huggingface/badges/resolve/main/visualize-this-dataset-xl.svg"/>
<img class="hidden dark:block" src="https://huggingface.co/datasets/huggingface/badges/resolve/main/visualize-this-dataset-xl-dark.svg"/>
</a>

## Dataset Description

{description}

- **Robot:** TRLC DK1 bimanual (`{robot_type}`) — 2× 6-DOF arms with grippers
- **Task:** {task}
- **Episodes:** {total_episodes}
- **Total frames:** {total_frames:,} ({total_minutes:.1f} min @ {fps} fps)
- **Avg episode length:** {avg_episode_s:.1f}s
- **License:** {license}

### Cameras

{camera_table}
### Observation Space

- **`observation.state`**: float32[{obs_state.get('shape', ['?'])[0]}] — {', '.join(obs_state.get('names', [])[:6])}... (joint positions, velocities, torques for both arms + grippers)
- **`observation.images.*`**: {len(cameras)} camera streams

### Action Space

- **`action`**: float32[{action.get('shape', ['?'])[0]}] — {', '.join(action.get('names', [])[:6])}... (joint position targets for both arms + grippers)

## Loading the Dataset

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

dataset = LeRobotDataset("{repo_id}")

# Access a frame
frame = dataset[0]
print(frame["observation.state"].shape)  # torch.Size([{obs_state.get('shape', ['?'])[0]}])
print(frame["action"].shape)             # torch.Size([{action.get('shape', ['?'])[0]}])
print(frame["observation.images.head"].shape)  # torch.Size([3, 720, 1280])
```

## Dataset Structure

[meta/info.json](meta/info.json):
```json
{info_json_str}
```

## Citation

**BibTeX:**

```bibtex
@misc{{{repo_id.replace('/', '_').replace('-', '_')},
  title = {{{task}}},
  author = {{The Robot Learning Company}},
  year = {{2026}},
  publisher = {{HuggingFace}},
  url = {{https://huggingface.co/datasets/{repo_id}}}
}}
```
"""
    return readme


def _build_hf_parquet_metadata(table: pa.Table) -> dict:
    """Build HuggingFace schema metadata for a parquet table.

    The HF datasets library uses this metadata to correctly interpret
    list columns as fixed-length sequences.
    """
    features = {}
    for field in table.schema:
        if "fixed_size_list" in str(field.type) or "list" in str(field.type):
            # Extract list size
            if hasattr(field.type, 'list_size'):
                length = field.type.list_size
            else:
                # Variable-length list — try to get size from first row
                col = table.column(field.name)
                length = len(col[0].as_py()) if len(col) > 0 else 0
            features[field.name] = {
                "feature": {"dtype": "float32", "_type": "Value"},
                "length": length,
                "_type": "List",
            }
        elif "int" in str(field.type):
            features[field.name] = {"dtype": "int64", "_type": "Value"}
        else:
            features[field.name] = {"dtype": "float32", "_type": "Value"}
    return {b"huggingface": json.dumps({"info": {"features": features}}).encode()}


def _compute_episode_scalar_stats(data_table: pa.Table) -> dict[str, dict]:
    """Compute per-episode stats for scalar features in a data parquet."""
    from lerobot.datasets.compute_stats import RunningQuantileStats
    import numpy as np

    stats = {}
    for col_name in data_table.column_names:
        col = data_table.column(col_name)
        col_type = str(data_table.schema.field(col_name).type)

        if "list" in col_type:
            # Vector feature (observation.state, action)
            pydata = col.to_pylist()
            if not pydata or pydata[0] is None:
                continue
            batch = np.array(pydata, dtype=np.float32)
        elif "int" in col_type or "float" in col_type:
            # Scalar feature
            batch = np.array(col.to_pylist(), dtype=np.float64).reshape(-1, 1)
        else:
            continue

        if len(batch) < 2:
            continue
        rqs = RunningQuantileStats()
        rqs.update(batch)
        try:
            raw = rqs.get_statistics()
            stats[col_name] = {k: v.tolist() for k, v in raw.items()}
        except ValueError:
            pass
    return stats


def _repair_episode_metadata(dataset_dir: Path):
    """Rebuild episode metadata with full per-episode stats from data parquet files.

    Reads each data parquet, computes stats for ALL features, and rewrites
    the episode metadata parquet with the complete stats columns.
    """
    meta_path = dataset_dir / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    if not meta_path.exists():
        return

    meta = pq.read_table(meta_path)
    rows = meta.to_pydict()
    n = len(rows["episode_index"])

    info = json.loads((dataset_dir / "meta" / "info.json").read_text())
    chunks_size = info.get("chunks_size", 1000)

    print(f"  Repairing episode metadata stats for {n} episodes...")

    for i in range(n):
        ep_idx = rows["episode_index"][i]
        chunk = ep_idx // chunks_size
        file_idx = ep_idx % chunks_size
        data_path = dataset_dir / "data" / f"chunk-{chunk:03d}" / f"file-{file_idx:03d}.parquet"

        if not data_path.exists():
            continue

        data_table = pq.read_table(data_path)
        ep_stats = _compute_episode_scalar_stats(data_table)

        for feat_key, feat_stats in ep_stats.items():
            for stat_key, stat_val in feat_stats.items():
                col_name = f"stats/{feat_key}/{stat_key}"
                if col_name not in rows:
                    rows[col_name] = [None] * n
                rows[col_name][i] = stat_val

    pq.write_table(pa.Table.from_pydict(rows), meta_path, compression="snappy")

    # Count stats columns
    stats_cols = [c for c in rows if c.startswith("stats/")]
    print(f"  Episode metadata now has {len(stats_cols)} stats columns")


def _reorder_info_features(dataset_dir: Path):
    """Reorder info.json features: action, observation.state first, then videos, then metadata.

    The HF dataset visualizer iterates features in order and expects
    action/observation.state before video features.
    """
    info_path = dataset_dir / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    features = info.get("features", {})

    def _reorder_keys(feat: dict) -> dict:
        """Reorder keys within a feature dict to match reference: dtype, names, shape, ..."""
        out = {}
        for k in ["dtype", "names", "shape", "info"]:
            if k in feat:
                out[k] = feat[k]
        for k in feat:
            if k not in out:
                out[k] = feat[k]
        return out

    # Desired order: action, observation.state, video features, scalar metadata
    ordered = {}
    for key in ["action", "observation.state"]:
        if key in features:
            ordered[key] = _reorder_keys(features[key])
    for key in features:
        if key.startswith("observation.images."):
            ordered[key] = _reorder_keys(features[key])
    for key in ["timestamp", "frame_index", "episode_index", "index", "task_index"]:
        if key in features:
            ordered[key] = _reorder_keys(features[key])
    for key in features:
        if key not in ordered:
            ordered[key] = _reorder_keys(features[key])

    # Ensure scalar features have "names": null (required by LeRobot reference format)
    for key in ("timestamp", "frame_index", "episode_index", "index", "task_index"):
        if key in ordered and "names" not in ordered[key]:
            ordered[key]["names"] = None

    if ordered != features:
        info["features"] = ordered
        info_path.write_text(json.dumps(info, indent=2))
        print("  Fixed info.json features (order / missing names)")
        return True
    return False


def _mp4_frame_count(path: Path) -> int | None:
    """Return the packet count of the first video stream in `path`, or None
    if ffprobe fails or the file is missing.

    Used by `_verify_video_frame_counts` as the authoritative video length
    — `nb_read_packets` counts every packet that ffmpeg decodes without
    needing to actually decode, so it's fast even for long episodes.
    """
    import subprocess
    if not path.exists():
        return None
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-count_packets", "-show_entries", "stream=nb_read_packets",
             "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, check=True).stdout.strip()
        return int(out) if out else None
    except (subprocess.CalledProcessError, ValueError, FileNotFoundError):
        return None


def _verify_video_frame_counts(dataset_dir: Path, meta_dict: dict,
                                chunks_size: int) -> list[str]:
    """Sanity-check that every camera's MP4 covers its episode's
    parquet length.

    LeRobot's dataloader slices videos by timestamp derived from the
    parquet row count. If an MP4 has FEWER frames than the parquet has
    rows (e.g. frame drops during recording left the video short while
    scalar rows kept being appended), the dataloader will read past the
    MP4's EOF on later frames and return garbage — which is why the
    plain `to_timestamp = n_rows/fps` fix is not enough. We bail out
    here so the bad dataset never reaches training or the Hub.

    Returns a list of human-readable problem descriptions (empty =
    all episodes pass). Also tolerates MP4s slightly LONGER than
    parquet (trailing gesture frames, pre-roll tail) — those are
    already handled correctly by the to_timestamp fix.
    """
    problems: list[str] = []
    # Camera keys are inferred from videos/<key>/ subdirs, not hard-coded,
    # so this works for any camera layout.
    video_root = dataset_dir / "videos"
    if not video_root.exists():
        return problems
    video_keys = sorted(d.name for d in video_root.iterdir()
                        if d.is_dir() and d.name.startswith("observation.images."))
    if not video_keys:
        return problems

    n_eps = len(meta_dict.get("episode_index", []))
    for i in range(n_eps):
        ep_idx = meta_dict["episode_index"][i]
        n_rows = (meta_dict["dataset_to_index"][i]
                  - meta_dict["dataset_from_index"][i])
        chunk = ep_idx // chunks_size
        file_idx = ep_idx % chunks_size
        for vk in video_keys:
            mp4 = (video_root / vk / f"chunk-{chunk:03d}"
                   / f"file-{file_idx:03d}.mp4")
            n_video = _mp4_frame_count(mp4)
            if n_video is None:
                problems.append(
                    f"episode {ep_idx}: {vk} video missing or unreadable "
                    f"({mp4})")
                continue
            if n_video < n_rows:
                problems.append(
                    f"episode {ep_idx}: {vk} has {n_video} video frames "
                    f"but parquet has {n_rows} rows "
                    f"(short by {n_rows - n_video} frames / "
                    f"{(n_rows - n_video) / 60:.1f}s @60fps) — dataloader "
                    f"will read past EOF. Drop the episode via "
                    f"examples/delete_episode.py or re-record.")
    return problems


def cleanup_dataset(dataset_dir: Path):
    """Remove orphan files, re-index episodes contiguously, fix global index.

    The dk1-recorder can leave orphan data/video files from discarded episodes
    that aren't in the episode metadata. This also re-indexes non-contiguous
    episode indices (e.g. 2,3,4... → 0,1,2...) and ensures the global `index`
    column starts at 0 and is contiguous across episodes.
    """
    meta_path = dataset_dir / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    info_path = dataset_dir / "meta" / "info.json"

    info_changed = _reorder_info_features(dataset_dir)

    if not meta_path.exists():
        print("  No episode metadata found, skipping cleanup")
        return

    meta = pq.read_table(meta_path)
    old_indices = meta.column("episode_index").to_pylist()
    info = json.loads(info_path.read_text())
    chunks_size = info.get("chunks_size", 1000)
    n = len(old_indices)

    # Build old→new episode index mapping
    need_reindex = old_indices != list(range(n))
    index_map = {old: new for new, old in enumerate(sorted(old_indices))}

    if need_reindex:
        print(f"  Re-indexing {n} episodes: {old_indices[0]}..{old_indices[-1]} → 0..{n-1}")

        # Rename data parquet files and update episode_index inside
        for old_idx, new_idx in index_map.items():
            if old_idx == new_idx:
                continue
            old_chunk, old_file = old_idx // chunks_size, old_idx % chunks_size
            new_chunk, new_file = new_idx // chunks_size, new_idx % chunks_size
            old_path = dataset_dir / "data" / f"chunk-{old_chunk:03d}" / f"file-{old_file:03d}.parquet"
            new_path = dataset_dir / "data" / f"chunk-{new_chunk:03d}" / f"file-{new_file:03d}.parquet"
            if old_path.exists():
                t = pq.read_table(old_path)
                cols = t.to_pydict()
                cols["episode_index"] = [new_idx] * len(cols["episode_index"])
                new_path.parent.mkdir(parents=True, exist_ok=True)
                pq.write_table(pa.Table.from_pydict(cols), new_path, compression="snappy")
                if old_path != new_path:
                    old_path.unlink()

        # Rename video files
        if (dataset_dir / "videos").exists():
            for cam_dir in (dataset_dir / "videos").iterdir():
                if not cam_dir.is_dir():
                    continue
                for old_idx, new_idx in index_map.items():
                    if old_idx == new_idx:
                        continue
                    old_chunk, old_file = old_idx // chunks_size, old_idx % chunks_size
                    new_chunk, new_file = new_idx // chunks_size, new_idx % chunks_size
                    old_path = cam_dir / f"chunk-{old_chunk:03d}" / f"file-{old_file:03d}.mp4"
                    new_path = cam_dir / f"chunk-{new_chunk:03d}" / f"file-{new_file:03d}.mp4"
                    if old_path.exists():
                        new_path.parent.mkdir(parents=True, exist_ok=True)
                        old_path.rename(new_path)

    # Remove orphan files
    _remove_orphan_files(dataset_dir, set(range(n)), chunks_size)

    # Check parquet schemas and global index column
    meta_dict = meta.to_pydict()
    if need_reindex:
        meta_dict["episode_index"] = list(range(n))
    global_idx = 0
    parquets_fixed = 0
    meta_changed = need_reindex
    for i in range(n):
        ep_idx = i if need_reindex else meta_dict["episode_index"][i]
        chunk = ep_idx // chunks_size
        file_idx = ep_idx % chunks_size
        data_path = dataset_dir / "data" / f"chunk-{chunk:03d}" / f"file-{file_idx:03d}.parquet"
        if not data_path.exists():
            print(f"  WARNING: missing {data_path}")
            continue

        t = pq.read_table(data_path)
        n_rows = t.num_rows
        needs_write = False

        # Fix global index column
        old_start = t.column("index")[0].as_py() if n_rows > 0 else global_idx
        if old_start != global_idx:
            needs_write = True

        # Fix parquet schema (ensure list<float>, HF metadata, column order)
        needs_schema_fix = _needs_parquet_fix(t)
        if needs_schema_fix:
            needs_write = True

        if needs_write:
            cols = {}
            col_order = ["action", "observation.state", "timestamp",
                         "frame_index", "episode_index", "index", "task_index"]
            all_cols = [c for c in col_order if c in t.column_names]
            all_cols += [c for c in t.column_names if c not in col_order]

            for name in all_cols:
                col = t.column(name)
                if name == "index":
                    cols[name] = pa.array(list(range(global_idx, global_idx + n_rows)), type=pa.int64())
                elif name in ("observation.state", "action") and needs_schema_fix:
                    cols[name] = pa.array(col.to_pylist(), type=pa.list_(pa.float32()))
                elif ("float" in str(t.schema.field(name).type) or "double" in str(t.schema.field(name).type)) and "list" not in str(t.schema.field(name).type):
                    cols[name] = pa.array(col.to_pylist(), type=pa.float32())
                else:
                    cols[name] = col

            new_table = pa.table(cols)
            new_table = new_table.replace_schema_metadata({
                **_build_hf_parquet_metadata(new_table),
                **(new_table.schema.metadata or {}),
            })
            pq.write_table(new_table, data_path, compression="snappy")
            parquets_fixed += 1

        # Update episode metadata indices if changed
        if need_reindex:
            new_chunk, new_file = i // chunks_size, i % chunks_size
            meta_dict["data/chunk_index"][i] = new_chunk
            meta_dict["data/file_index"][i] = new_file
            for col in meta_dict:
                if col.startswith("videos/") and col.endswith("/chunk_index"):
                    meta_dict[col][i] = new_chunk
                elif col.startswith("videos/") and col.endswith("/file_index"):
                    meta_dict[col][i] = new_file

        old_from = meta_dict["dataset_from_index"][i]
        old_to = meta_dict["dataset_to_index"][i]
        if old_from != global_idx or old_to != global_idx + n_rows:
            meta_changed = True
        meta_dict["dataset_from_index"][i] = global_idx
        meta_dict["dataset_to_index"][i] = global_idx + n_rows

        # Fix video to_timestamp: must match scalar frame count, not MP4 frame count.
        # The MP4 may contain trailing gesture frames that were trimmed from parquet.
        fps = info.get("fps", 30)
        expected_to_ts = n_rows / fps
        for col in meta_dict:
            if col.startswith("videos/") and col.endswith("/to_timestamp"):
                if abs(meta_dict[col][i] - expected_to_ts) > 0.001:
                    meta_changed = True
                meta_dict[col][i] = expected_to_ts

        global_idx += n_rows

    if parquets_fixed:
        print(f"  Fixed {parquets_fixed} parquet files (schema/index)")

    # Check if episode stats need repair
    stats_cols = [c for c in meta_dict if c.startswith("stats/")]
    needs_stats_repair = len(stats_cols) == 0

    if meta_changed:
        pq.write_table(pa.Table.from_pydict(meta_dict), meta_path, compression="snappy")

    if needs_stats_repair:
        _repair_episode_metadata(dataset_dir)

    # Check if info.json needs updating
    info_needs_update = False
    data_dir = dataset_dir / "data"
    video_dir = dataset_dir / "videos"
    new_data_mb = round(sum(f.stat().st_size for f in data_dir.rglob("*.parquet")) / (1024 * 1024))
    new_video_mb = round(sum(f.stat().st_size for f in video_dir.rglob("*.mp4")) / (1024 * 1024)) if video_dir.exists() else 0

    if (info["total_episodes"] != n or info["total_frames"] != global_idx
            or info.get("data_files_size_in_mb") != new_data_mb
            or info.get("video_files_size_in_mb") != new_video_mb):
        info["total_episodes"] = n
        info["total_frames"] = global_idx
        info["splits"] = {"train": f"0:{n}"}
        info["data_files_size_in_mb"] = new_data_mb
        info["video_files_size_in_mb"] = new_video_mb
        info_path.write_text(json.dumps(info, indent=2))
        info_needs_update = True

    if not (info_changed or need_reindex or parquets_fixed or meta_changed
            or needs_stats_repair or info_needs_update):
        print(f"  Dataset OK: {n} episodes, {global_idx} frames (nothing to fix)")
    else:
        print(f"  Done: {n} episodes, {global_idx} frames")

    # Verify every camera's MP4 has at least as many frames as its
    # episode's parquet. Runs AFTER the to_timestamp fix above because
    # an episode with a short video would otherwise look fine in the
    # metadata but blow up at training time.
    print("  Verifying video frame counts match parquet row counts ...")
    video_problems = _verify_video_frame_counts(
        dataset_dir, meta_dict, chunks_size)
    if video_problems:
        print(f"  [!] Found {len(video_problems)} video/parquet mismatch(es):")
        for p in video_problems:
            print(f"      {p}")
        # Attach so callers (main()) can refuse to upload without
        # re-running ffprobe on every episode.
        cleanup_dataset._last_video_problems = video_problems
    else:
        cleanup_dataset._last_video_problems = []
        print(f"  Video frame counts OK across all episodes")


def _needs_parquet_fix(table: pa.Table) -> bool:
    """Check if a parquet table needs schema fixes."""
    for field in table.schema:
        if field.name in ("observation.state", "action"):
            ftype = str(field.type)
            if "fixed_size_list" in ftype or "double" in ftype:
                return True
    if not (table.schema.metadata and b"huggingface" in table.schema.metadata):
        return True
    # Check column order
    col_order = ["action", "observation.state", "timestamp",
                 "frame_index", "episode_index", "index", "task_index"]
    existing = [c for c in col_order if c in table.column_names]
    if existing != table.column_names[:len(existing)]:
        return True
    return False


def _remove_orphan_files(dataset_dir: Path, valid_indices: set[int], chunks_size: int = 1000):
    """Remove data/video files that aren't in the valid episode set."""
    removed = 0
    for data_file in sorted((dataset_dir / "data").rglob("*.parquet")):
        try:
            idx = int(data_file.stem.split("-")[-1])
            chunk = int(data_file.parent.name.split("-")[-1])
            ep_idx = chunk * chunks_size + idx
            if ep_idx not in valid_indices:
                data_file.unlink()
                removed += 1
        except (ValueError, IndexError):
            pass

    for vid_file in sorted((dataset_dir / "videos").rglob("*.mp4")):
        try:
            idx = int(vid_file.stem.split("-")[-1])
            chunk = int(vid_file.parent.name.split("-")[-1])
            ep_idx = chunk * chunks_size + idx
            if ep_idx not in valid_indices:
                vid_file.unlink()
                removed += 1
        except (ValueError, IndexError):
            pass

    if removed:
        print(f"  Removed {removed} orphan files")


def main():
    p = argparse.ArgumentParser(description="Upload dk1-record dataset to HuggingFace Hub")
    p.add_argument("--dataset-dir", type=Path, required=True, help="Local dataset directory")
    p.add_argument("--repo-id", type=str, required=True, help="HuggingFace repo ID (e.g. username/dataset_name)")
    p.add_argument("--description", type=str, default="", help="Dataset description for the README")
    p.add_argument("--license", type=str, default="apache-2.0", help="License identifier")
    p.add_argument("--private", action="store_true", help="Create private dataset")
    p.add_argument("--readme-only", action="store_true", help="Only generate README, don't upload")
    p.add_argument("--num-workers", type=int, default=2, help="Concurrent upload workers (default: 2)")
    p.add_argument("--ignore-video-mismatch", action="store_true",
                   help="Upload even when a per-camera MP4 has fewer "
                        "frames than its episode's parquet has rows. "
                        "DANGEROUS: the training dataloader will read "
                        "past the video's EOF on late frames. Only set "
                        "this if you've truncated parquet rows elsewhere "
                        "to match the short video.")
    args = p.parse_args()

    dataset_dir = args.dataset_dir
    info_path = dataset_dir / "meta" / "info.json"

    if not info_path.exists():
        print(f"Error: {info_path} not found. Is this a valid dataset directory?", file=sys.stderr)
        sys.exit(1)

    # Cleanup: re-index episodes and remove orphans
    print("Cleaning up dataset...")
    cleanup_dataset(dataset_dir)

    video_problems = getattr(cleanup_dataset, "_last_video_problems", [])
    if video_problems and not args.ignore_video_mismatch:
        print(f"\nRefusing to upload: {len(video_problems)} video/parquet "
              f"mismatch(es) detected. Fix via "
              f"examples/delete_episode.py or re-record those episodes, "
              f"or pass --ignore-video-mismatch to upload anyway "
              f"(dangerous — training will read past video EOF).",
              file=sys.stderr)
        sys.exit(1)

    info = json.loads(info_path.read_text())

    # Get task from tasks.parquet
    task = "Bimanual manipulation task"
    tasks_path = dataset_dir / "meta" / "tasks.parquet"
    if tasks_path.exists():
        try:
            table = pq.read_table(tasks_path)
            tasks = table.to_pydict().get("task", [])
            if tasks:
                task = tasks[0]
        except Exception:
            pass

    # Generate README
    readme = generate_readme(
        repo_id=args.repo_id,
        info=info,
        task=task,
        description=args.description,
        license=args.license,
    )

    # Write README to dataset dir
    readme_path = dataset_dir / "README.md"
    readme_path.write_text(readme)
    print(f"Generated: {readme_path}")

    if args.readme_only:
        print("\n--- README preview ---")
        print(readme[:2000])
        if len(readme) > 2000:
            print(f"... ({len(readme)} chars total)")
        return

    # Upload to HuggingFace
    print(f"\nUploading to https://huggingface.co/datasets/{args.repo_id} ...")

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("Error: huggingface_hub not installed. Run: pip install huggingface_hub", file=sys.stderr)
        sys.exit(1)

    api = HfApi()

    # Create repo
    api.create_repo(
        repo_id=args.repo_id,
        private=args.private,
        repo_type="dataset",
        exist_ok=True,
    )

    # Upload all files (upload_large_folder handles batching and retries)
    print("Uploading files...")
    api.upload_large_folder(
        folder_path=str(dataset_dir),
        repo_id=args.repo_id,
        repo_type="dataset",
        ignore_patterns=["images/"],
        num_workers=args.num_workers,
    )

    print(f"\nDone! Dataset available at:")
    print(f"  https://huggingface.co/datasets/{args.repo_id}")
    print(f"\nVisualize at:")
    print(f"  https://huggingface.co/spaces/lerobot/visualize_dataset?path={args.repo_id}")


if __name__ == "__main__":
    main()
