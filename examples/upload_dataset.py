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
- bimanual
- teleoperation
- DK1
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

    hyparquet (the JS parquet reader) uses this to deserialize
    fixed_size_list columns into JavaScript arrays.
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


def _fix_parquet_schema(path: Path):
    """Re-encode a data parquet to use fixed_size_list<float> instead of list<double>.

    The LeRobot visualizer requires fixed_size_list columns for observation.state
    and action features.
    """
    t = pq.read_table(path)
    needs_fix = False
    for field in t.schema:
        if field.name in ("observation.state", "action"):
            if "fixed_size_list" not in str(field.type):
                needs_fix = True
                break

    if not needs_fix:
        # Still reorder columns and ensure HF metadata exists
        col_order = ["action", "observation.state", "timestamp",
                     "frame_index", "episode_index", "index", "task_index"]
        existing = [c for c in col_order if c in t.column_names]
        extra = [c for c in t.column_names if c not in col_order]
        needs_reorder = (existing + extra != t.column_names)
        has_hf_meta = t.schema.metadata and b"huggingface" in t.schema.metadata
        if needs_reorder or not has_hf_meta:
            if needs_reorder:
                t = t.select(existing + extra)
            if not has_hf_meta:
                t = t.replace_schema_metadata({
                    **_build_hf_parquet_metadata(t),
                    **(t.schema.metadata or {}),
                })
            pq.write_table(t, path, compression="snappy")
        return

    cols = {}
    # Process in reference column order
    col_order = ["action", "observation.state", "timestamp",
                 "frame_index", "episode_index", "index", "task_index"]
    all_cols = [c for c in col_order if c in t.column_names]
    all_cols += [c for c in t.column_names if c not in col_order]

    for name in all_cols:
        col = t.column(name)
        if name in ("observation.state", "action"):
            pydata = col.to_pylist()
            dim = len(pydata[0])
            flat = [float(v) for row in pydata for v in row]
            cols[name] = pa.FixedSizeListArray.from_arrays(
                pa.array(flat, type=pa.float32()), list_size=dim
            )
        elif "float" in str(t.schema.field(name).type) or "double" in str(t.schema.field(name).type):
            cols[name] = pa.array(col.to_pylist(), type=pa.float32())
        else:
            cols[name] = col

    new_table = pa.table(cols)
    new_table = new_table.replace_schema_metadata({
        **_build_hf_parquet_metadata(new_table),
        **(new_table.schema.metadata or {}),
    })
    pq.write_table(new_table, path, compression="snappy")


def cleanup_dataset(dataset_dir: Path):
    """Remove orphan files and re-index episodes to be contiguous from 0.

    The dk1-recorder can leave orphan data/video files from discarded episodes
    that aren't in the episode metadata. This also re-indexes non-contiguous
    episode indices (e.g. 2,3,4... → 0,1,2...) which is required by the
    LeRobot visualizer.
    """
    meta_path = dataset_dir / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    info_path = dataset_dir / "meta" / "info.json"

    if not meta_path.exists():
        print("  No episode metadata found, skipping cleanup")
        return

    meta = pq.read_table(meta_path)
    old_indices = meta.column("episode_index").to_pylist()

    if old_indices == list(range(len(old_indices))):
        print(f"  Episodes already contiguous (0..{len(old_indices)-1})")
        _remove_orphan_files(dataset_dir, set(old_indices))
        # Still fix parquet schemas
        print("  Fixing parquet schemas...")
        for parquet_file in sorted((dataset_dir / "data").rglob("*.parquet")):
            _fix_parquet_schema(parquet_file)
        return

    print(f"  Re-indexing {len(old_indices)} episodes: {old_indices[0]}..{old_indices[-1]} → 0..{len(old_indices)-1}")

    info = json.loads(info_path.read_text())
    chunks_size = info.get("chunks_size", 1000)

    # Build old→new index mapping
    index_map = {old: new for new, old in enumerate(sorted(old_indices))}

    # 1. Rename data parquet files and update episode_index inside
    for old_idx, new_idx in index_map.items():
        old_chunk, old_file = old_idx // chunks_size, old_idx % chunks_size
        new_chunk, new_file = new_idx // chunks_size, new_idx % chunks_size

        old_path = dataset_dir / "data" / f"chunk-{old_chunk:03d}" / f"file-{old_file:03d}.parquet"
        new_path = dataset_dir / "data" / f"chunk-{new_chunk:03d}" / f"file-{new_file:03d}.parquet"

        if old_path.exists() and old_idx != new_idx:
            # Read, update episode_index, write to new path
            t = pq.read_table(old_path)
            cols = t.to_pydict()
            cols["episode_index"] = [new_idx] * len(cols["episode_index"])
            new_table = pa.Table.from_pydict(cols)
            new_path.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(new_table, new_path, compression="snappy")
            if old_path != new_path:
                old_path.unlink()

    # 2. Rename video files
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

    # 3. Rewrite episode metadata with new indices
    rows = meta.to_pydict()
    n = len(old_indices)
    rows["episode_index"] = list(range(n))
    for i, old_idx in enumerate(old_indices):
        new_idx = index_map[old_idx]
        new_chunk, new_file = new_idx // chunks_size, new_idx % chunks_size
        rows["data/chunk_index"][i] = new_chunk
        rows["data/file_index"][i] = new_file
        # Update video indices
        for col in rows:
            if col.startswith("videos/") and col.endswith("/chunk_index"):
                rows[col][i] = new_chunk
            elif col.startswith("videos/") and col.endswith("/file_index"):
                rows[col][i] = new_file

    # Recompute dataset_from_index / dataset_to_index
    running_idx = 0
    for i in range(n):
        length = rows["length"][i]
        rows["dataset_from_index"][i] = running_idx
        rows["dataset_to_index"][i] = running_idx + length
        running_idx += length

    pq.write_table(pa.Table.from_pydict(rows), meta_path, compression="snappy")

    # 4. Update info.json
    info["total_episodes"] = n
    info["total_frames"] = running_idx
    info["splits"] = {"train": f"0:{n}"}
    info_path.write_text(json.dumps(info, indent=2))

    # 5. Remove orphan files
    valid_indices = set(range(n))
    _remove_orphan_files(dataset_dir, valid_indices)

    # 6. Fix parquet schema (list<double> → fixed_size_list<float>)
    print("  Fixing parquet schemas...")
    for parquet_file in sorted((dataset_dir / "data").rglob("*.parquet")):
        _fix_parquet_schema(parquet_file)

    print(f"  Done: {n} episodes, {running_idx} frames")


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
    args = p.parse_args()

    dataset_dir = args.dataset_dir
    info_path = dataset_dir / "meta" / "info.json"

    if not info_path.exists():
        print(f"Error: {info_path} not found. Is this a valid dataset directory?", file=sys.stderr)
        sys.exit(1)

    # Cleanup: re-index episodes and remove orphans
    print("Cleaning up dataset...")
    cleanup_dataset(dataset_dir)

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

    # Upload all files
    print("Uploading files (this may take a while for large datasets)...")
    api.upload_folder(
        folder_path=str(dataset_dir),
        repo_id=args.repo_id,
        repo_type="dataset",
        ignore_patterns=["images/"],  # skip raw images if any
    )

    print(f"\nDone! Dataset available at:")
    print(f"  https://huggingface.co/datasets/{args.repo_id}")
    print(f"\nVisualize at:")
    print(f"  https://huggingface.co/spaces/lerobot/visualize_dataset?path={args.repo_id}")


if __name__ == "__main__":
    main()
