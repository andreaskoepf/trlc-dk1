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

    info = json.loads(info_path.read_text())

    # Get task from tasks.parquet
    task = "Bimanual manipulation task"
    tasks_path = dataset_dir / "meta" / "tasks.parquet"
    if tasks_path.exists():
        try:
            import pyarrow.parquet as pq
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
