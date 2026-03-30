#!/usr/bin/env python3
"""Delete one or more episodes from a LeRobot v3 dataset.

Removes the episode data (parquet), videos, and episode metadata, then
re-indexes all subsequent episodes so there are no gaps.  Updates
info.json, stats.json, and the meta/episodes parquet.

Usage:
    # Delete a single episode:
    python examples/delete_episode.py --dataset-dir ./data/my_dataset --episode 45

    # Delete multiple episodes:
    python examples/delete_episode.py --dataset-dir ./data/my_dataset --episode 3 17 45

    # Dry-run (show what would be deleted without modifying anything):
    python examples/delete_episode.py --dataset-dir ./data/my_dataset --episode 45 --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def ep_chunk_file(ep_idx: int, chunks_size: int) -> tuple[int, int]:
    """Return (chunk_index, file_index) for an episode."""
    return ep_idx // chunks_size, ep_idx % chunks_size


def data_path(dataset_dir: Path, ep_idx: int, chunks_size: int) -> Path:
    chunk, file = ep_chunk_file(ep_idx, chunks_size)
    return dataset_dir / "data" / f"chunk-{chunk:03d}" / f"file-{file:03d}.parquet"


def video_path(dataset_dir: Path, video_key: str, ep_idx: int, chunks_size: int) -> Path:
    chunk, file = ep_chunk_file(ep_idx, chunks_size)
    return dataset_dir / "videos" / video_key / f"chunk-{chunk:03d}" / f"file-{file:03d}.mp4"


def discover_video_keys(dataset_dir: Path) -> list[str]:
    """Return video key names from the videos/ directory."""
    vid_dir = dataset_dir / "videos"
    if not vid_dir.exists():
        return []
    return sorted(d.name for d in vid_dir.iterdir() if d.is_dir())


def discover_total_episodes(dataset_dir: Path, chunks_size: int) -> int:
    """Count episodes from data parquet files on disk."""
    total = 0
    data_dir = dataset_dir / "data"
    for chunk_dir in sorted(data_dir.iterdir()):
        if chunk_dir.is_dir() and chunk_dir.name.startswith("chunk-"):
            total += len(list(chunk_dir.glob("file-*.parquet")))
    return total


def compute_dataset_stats(dataset_dir: Path, total_eps: int, chunks_size: int) -> dict:
    """Recompute dataset-level stats from all episode parquets."""
    all_dfs = []
    for ep in range(total_eps):
        path = data_path(dataset_dir, ep, chunks_size)
        all_dfs.append(pq.read_table(path).to_pandas())
    full = pd.concat(all_dfs, ignore_index=True)

    stats_path = dataset_dir / "meta" / "stats.json"
    old_stats = {}
    if stats_path.exists():
        old_stats = json.loads(stats_path.read_text())

    new_stats = {}
    for key in old_stats:
        if key in full.columns:
            vals = np.stack(full[key].values).astype(float)
            if vals.ndim == 1:
                vals = vals.reshape(-1, 1)
            new_stats[key] = {
                "min": vals.min(axis=0).tolist(),
                "max": vals.max(axis=0).tolist(),
                "mean": vals.mean(axis=0).tolist(),
                "std": vals.std(axis=0).tolist(),
                "count": [len(vals)] * vals.shape[1],
                "q01": np.quantile(vals, 0.01, axis=0).tolist(),
                "q10": np.quantile(vals, 0.10, axis=0).tolist(),
                "q50": np.quantile(vals, 0.50, axis=0).tolist(),
                "q90": np.quantile(vals, 0.90, axis=0).tolist(),
                "q99": np.quantile(vals, 0.99, axis=0).tolist(),
            }
        else:
            new_stats[key] = old_stats[key]

    return new_stats


def validate_dataset(dataset_dir: Path, chunks_size: int, video_keys: list[str]) -> list[str]:
    """Validate dataset integrity. Returns a list of errors (empty = OK)."""
    errors = []

    info_path = dataset_dir / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    info_total_eps = info.get("total_episodes", 0)
    info_total_frames = info.get("total_frames", 0)

    # Count data files on disk
    disk_eps = discover_total_episodes(dataset_dir, chunks_size)

    # Load episodes metadata
    meta_path = dataset_dir / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    meta_eps = 0
    meta_ep_indices = set()
    meta_frames = {}  # ep_index -> length
    if meta_path.exists():
        meta_df = pq.read_table(meta_path).to_pandas()
        meta_eps = len(meta_df)
        meta_ep_indices = set(meta_df["episode_index"].tolist())
        if "length" in meta_df.columns:
            meta_frames = dict(zip(meta_df["episode_index"], meta_df["length"]))
    else:
        errors.append("meta/episodes parquet not found")

    # Cross-reference counts
    if info_total_eps != disk_eps:
        errors.append(
            f"info.json says {info_total_eps} episodes but {disk_eps} data files on disk"
        )
    if meta_eps != disk_eps:
        errors.append(
            f"episodes metadata has {meta_eps} rows but {disk_eps} data files on disk"
        )

    # Check each episode has data, metadata, and videos
    expected_eps = set(range(disk_eps))
    orphan_files = expected_eps - meta_ep_indices
    missing_files = meta_ep_indices - expected_eps
    if orphan_files:
        errors.append(
            f"Data files without metadata (orphans): episodes {sorted(orphan_files)}"
        )
    if missing_files:
        errors.append(
            f"Metadata without data files: episodes {sorted(missing_files)}"
        )

    # Verify contiguous global indices and frame counts
    prev_end = -1
    total_frames = 0
    for ep in range(disk_eps):
        path = data_path(dataset_dir, ep, chunks_size)
        if not path.exists():
            errors.append(f"Episode {ep}: data file missing")
            continue
        df = pq.read_table(path).to_pandas()
        n_frames = len(df)
        total_frames += n_frames

        if list(df["episode_index"].unique()) != [ep]:
            errors.append(
                f"Episode {ep}: wrong episode_index in data "
                f"(found {df['episode_index'].unique().tolist()})"
            )
        if len(df) > 0:
            if df["index"].iloc[0] != prev_end + 1:
                errors.append(
                    f"Episode {ep}: global index gap "
                    f"(expected {prev_end + 1}, got {df['index'].iloc[0]})"
                )
            prev_end = df["index"].iloc[-1]

        # Check metadata frame count matches actual data
        if ep in meta_frames and meta_frames[ep] != n_frames:
            errors.append(
                f"Episode {ep}: metadata says {meta_frames[ep]} frames "
                f"but data has {n_frames}"
            )

        # Check videos exist
        for vk in video_keys:
            vp = video_path(dataset_dir, vk, ep, chunks_size)
            if not vp.exists():
                errors.append(f"Episode {ep}: missing video {vk}")

    if total_frames != info_total_frames:
        errors.append(
            f"info.json says {info_total_frames} total frames "
            f"but data has {total_frames}"
        )

    return errors


def delete_episodes(dataset_dir: Path, episodes: list[int], *, dry_run: bool = False):
    """Delete episodes and re-index the dataset."""
    info_path = dataset_dir / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    chunks_size = info.get("chunks_size", 1000)
    total_eps = discover_total_episodes(dataset_dir, chunks_size)
    video_keys = discover_video_keys(dataset_dir)
    episodes = sorted(set(episodes))

    # --- Validate dataset integrity before making changes ---
    print("Validating dataset integrity...")
    validation_errors = validate_dataset(dataset_dir, chunks_size, video_keys)
    if validation_errors:
        print("ERROR: Dataset has integrity issues that must be fixed first:",
              file=sys.stderr)
        for e in validation_errors:
            print(f"  - {e}", file=sys.stderr)
        print("\nRefusing to delete episodes from a broken dataset.", file=sys.stderr)
        sys.exit(1)
    print("  Dataset OK\n")

    # --- Validate requested episodes ---
    for ep in episodes:
        if ep < 0 or ep >= total_eps:
            print(f"Error: episode {ep} out of range [0, {total_eps - 1}]", file=sys.stderr)
            sys.exit(1)
        path = data_path(dataset_dir, ep, chunks_size)
        if not path.exists():
            print(f"Error: {path} not found", file=sys.stderr)
            sys.exit(1)

    # --- Summarise ---
    frame_counts = {}
    for ep in episodes:
        t = pq.read_table(data_path(dataset_dir, ep, chunks_size))
        frame_counts[ep] = t.num_rows

    total_frames_deleted = sum(frame_counts.values())
    files_to_delete = len(episodes) * (1 + len(video_keys))

    print(f"Dataset:  {dataset_dir}")
    print(f"Chunks:   size={chunks_size}")
    print(f"Episodes: {total_eps} total, deleting {len(episodes)}: {episodes}")
    print(f"Frames:   {total_frames_deleted} frames across {len(episodes)} episodes")
    print(f"Files:    {files_to_delete} files to delete ({len(episodes)} parquet + "
          f"{len(episodes) * len(video_keys)} videos)")
    print(f"Videos:   {', '.join(video_keys) if video_keys else 'none'}")
    print()

    if dry_run:
        print("[dry-run] No changes made.")
        return

    # --- Step 1: Delete episode files ---
    for ep in episodes:
        data_path(dataset_dir, ep, chunks_size).unlink()
        for vk in video_keys:
            vp = video_path(dataset_dir, vk, ep, chunks_size)
            if vp.exists():
                vp.unlink()

    print(f"Deleted {files_to_delete} files for episodes {episodes}")

    # --- Step 2: Build old→new mapping, collect frame counts ---
    deleted_set = set(episodes)
    keep = [i for i in range(total_eps) if i not in deleted_set]
    new_total = len(keep)
    old_to_new = {old: new for new, old in enumerate(keep)}

    ep_frames = {}
    for old_ep in keep:
        t = pq.read_table(data_path(dataset_dir, old_ep, chunks_size))
        ep_frames[old_ep] = t.num_rows

    # --- Step 3: Rename and re-index data + video files ---
    # Process in ascending order.  Since new_ep <= old_ep, writing new
    # before reading the next old is safe (no collisions).
    global_idx = 0
    for new_ep, old_ep in enumerate(keep):
        old_data = data_path(dataset_dir, old_ep, chunks_size)
        new_data = data_path(dataset_dir, new_ep, chunks_size)

        n_frames = ep_frames[old_ep]
        needs_move = (old_data != new_data)
        t = pq.read_table(old_data)
        df = t.to_pandas()
        old_global_start = df["index"].iloc[0] if len(df) > 0 else None

        if needs_move or old_global_start != global_idx:
            df["episode_index"] = new_ep
            df["index"] = range(global_idx, global_idx + n_frames)
            new_data.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(pa.Table.from_pandas(df, preserve_index=False), new_data)
            if needs_move and old_data.exists():
                old_data.unlink()

        if needs_move:
            for vk in video_keys:
                old_vid = video_path(dataset_dir, vk, old_ep, chunks_size)
                new_vid = video_path(dataset_dir, vk, new_ep, chunks_size)
                if old_vid.exists():
                    new_vid.parent.mkdir(parents=True, exist_ok=True)
                    old_vid.rename(new_vid)

        global_idx += n_frames

    new_total_frames = global_idx

    # Remove empty chunk directories
    for subdir in ("data", "videos"):
        base = dataset_dir / subdir
        if not base.exists():
            continue
        for d in sorted(base.rglob("chunk-*"), reverse=True):
            if d.is_dir() and not any(d.iterdir()):
                d.rmdir()

    print(f"Re-indexed {new_total} episodes, {new_total_frames} frames")

    # --- Step 4: Update meta/episodes parquet ---
    meta_dir = dataset_dir / "meta" / "episodes"
    if meta_dir.exists():
        for meta_chunk_dir in sorted(meta_dir.iterdir()):
            if not meta_chunk_dir.is_dir():
                continue
            for meta_file in sorted(meta_chunk_dir.glob("file-*.parquet")):
                meta_df = pq.read_table(meta_file).to_pandas()

                # Drop deleted episodes
                meta_df = meta_df[~meta_df["episode_index"].isin(deleted_set)].copy()
                if len(meta_df) == 0:
                    meta_file.unlink()
                    continue

                # Remap episode indices
                meta_df["episode_index"] = meta_df["episode_index"].map(old_to_new)
                meta_df["data/file_index"] = meta_df["episode_index"].apply(
                    lambda e: ep_chunk_file(e, chunks_size)[1]
                )
                meta_df["data/chunk_index"] = meta_df["episode_index"].apply(
                    lambda e: ep_chunk_file(e, chunks_size)[0]
                )
                for vk in video_keys:
                    fi_col = f"videos/{vk}/file_index"
                    ci_col = f"videos/{vk}/chunk_index"
                    if fi_col in meta_df.columns:
                        meta_df[fi_col] = meta_df["data/file_index"]
                    if ci_col in meta_df.columns:
                        meta_df[ci_col] = meta_df["data/chunk_index"]

                # Recompute dataset_from/to_index
                # Build cumulative frame sums for new episode ordering
                new_ep_frames = {old_to_new[old]: ep_frames[old] for old in keep}
                cum = {}
                s = 0
                for e in range(new_total):
                    cum[e] = s
                    s += new_ep_frames[e]

                meta_df = meta_df.sort_values("episode_index").reset_index(drop=True)
                meta_df["dataset_from_index"] = meta_df["episode_index"].map(cum)
                meta_df["dataset_to_index"] = meta_df["episode_index"].apply(
                    lambda e: cum[e] + new_ep_frames[e]
                )

                # Update per-episode stats for episode_index
                for stat in ["min", "max", "mean", "q01", "q10", "q50", "q90", "q99"]:
                    col = f"stats/episode_index/{stat}"
                    if col in meta_df.columns:
                        meta_df[col] = meta_df["episode_index"].apply(lambda x: [float(x)])

                # Rewrite to correct chunk location based on new episode range
                # For simplicity, collect all rows and rewrite as single file
                pq.write_table(
                    pa.Table.from_pandas(meta_df, preserve_index=False),
                    meta_file,
                )

        # Clean up empty dirs
        for d in sorted(meta_dir.rglob("chunk-*"), reverse=True):
            if d.is_dir() and not any(d.iterdir()):
                d.rmdir()

        print("Updated meta/episodes parquet")

    # --- Step 5: Update info.json ---
    info["total_episodes"] = new_total
    info["total_frames"] = new_total_frames
    info["splits"]["train"] = f"0:{new_total}"

    data_dir = dataset_dir / "data"
    vid_dir = dataset_dir / "videos"
    info["data_files_size_in_mb"] = round(
        sum(f.stat().st_size for f in data_dir.rglob("*.parquet")) / (1024 * 1024)
    )
    if vid_dir.exists():
        info["video_files_size_in_mb"] = round(
            sum(f.stat().st_size for f in vid_dir.rglob("*.mp4")) / (1024 * 1024)
        )

    info_path.write_text(json.dumps(info, indent=2) + "\n")
    print(f"Updated info.json: {new_total} episodes, {new_total_frames} frames")

    # --- Step 6: Recompute stats.json ---
    print("Recomputing dataset stats...")
    new_stats = compute_dataset_stats(dataset_dir, new_total, chunks_size)
    stats_path = dataset_dir / "meta" / "stats.json"
    stats_path.write_text(json.dumps(new_stats, indent=2) + "\n")
    print("Updated stats.json")

    # --- Step 7: Verify ---
    print("\nVerifying...")
    prev_end = -1
    errors = []
    for ep in range(new_total):
        path = data_path(dataset_dir, ep, chunks_size)
        if not path.exists():
            errors.append(f"Missing data file for episode {ep}")
            continue
        df = pq.read_table(path).to_pandas()
        if list(df["episode_index"].unique()) != [ep]:
            errors.append(f"Episode {ep}: wrong episode_index {df['episode_index'].unique()}")
        if df["index"].iloc[0] != prev_end + 1:
            errors.append(f"Episode {ep}: index gap (expected {prev_end + 1}, "
                          f"got {df['index'].iloc[0]})")
        if df["frame_index"].iloc[0] != 0:
            errors.append(f"Episode {ep}: frame_index doesn't start at 0")
        prev_end = df["index"].iloc[-1]

        for vk in video_keys:
            vp = video_path(dataset_dir, vk, ep, chunks_size)
            if not vp.exists():
                errors.append(f"Episode {ep}: missing video {vk}")

    if prev_end + 1 != new_total_frames:
        errors.append(f"Total frames mismatch: {prev_end + 1} vs {new_total_frames}")

    if errors:
        print("ERRORS found:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    else:
        print(f"OK: {new_total} episodes, {new_total_frames} frames, "
              f"indices 0-{prev_end} contiguous")


def main():
    p = argparse.ArgumentParser(
        description="Delete episodes from a LeRobot v3 dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--dataset-dir", type=Path, required=True,
                    help="Path to the dataset directory")
    p.add_argument("--episode", "-e", type=int, nargs="+", required=True,
                    help="Episode index(es) to delete")
    p.add_argument("--dry-run", "-n", action="store_true",
                    help="Show what would be deleted without modifying anything")
    args = p.parse_args()

    if not (args.dataset_dir / "meta" / "info.json").exists():
        print(f"Error: {args.dataset_dir} is not a valid dataset directory", file=sys.stderr)
        sys.exit(1)

    delete_episodes(args.dataset_dir, args.episode, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
