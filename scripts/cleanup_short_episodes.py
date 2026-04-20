#!/usr/bin/env python3
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

"""Remove short or broken episodes from a LeRobot v3 dataset.

Two cleanup passes:

1. **Orphan `.d` directories**: folders left by the JPEG encoder when
   a recording aborted after `prepare_episode` but before `save_episode`
   landed a parquet row (e.g., cold-start safety halt, crash, or SIGKILL
   during init). These have no corresponding data parquet and would
   otherwise confuse `finalize_jpeg_recordings.py` (the "no frames
   found" log line).

2. **Short episodes**: episodes where either the data parquet has
   fewer than `--threshold` rows OR ANY camera's MP4 has fewer than
   `--threshold` frames. Such episodes are typically "the dispatch
   loop aborted after a handful of ticks" and carry no usable training
   signal — yet they'd still be pulled into training batches and waste
   compute.

Uses `examples/delete_episode.py`'s `delete_episodes()` for the actual
removal + re-indexing, so the post-cleanup dataset is contiguous
(0..N-1) and its `info.json` / episode metadata / stats.json are
rebuilt consistently.

Usage::

    # See what would be removed:
    python scripts/cleanup_short_episodes.py \\
        --dataset-dir /path/to/dataset --dry-run

    # Default threshold is 100 frames (~1.7s at 60fps):
    python scripts/cleanup_short_episodes.py \\
        --dataset-dir /path/to/dataset

    # Stricter — drop anything under 5 seconds at 60fps:
    python scripts/cleanup_short_episodes.py \\
        --dataset-dir /path/to/dataset --threshold 300
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
# `delete_episodes()` and its helpers live in examples/delete_episode.py;
# vendor it via sys.path rather than importing via the package to keep
# this script single-file runnable without an install.
sys.path.insert(0, str(_ROOT / "examples"))

from delete_episode import (  # noqa: E402
    delete_episodes,
    discover_total_episodes,
    discover_video_keys,
    data_path,
    video_path,
)
import json  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402


def _parquet_row_count(path: Path) -> int | None:
    if not path.exists():
        return None
    try:
        return pq.read_table(str(path)).num_rows
    except Exception:
        return None


def _mp4_frame_count(path: Path) -> int | None:
    if not path.exists():
        return None
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-count_packets", "-show_entries", "stream=nb_read_packets",
             "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        return int(out) if out else None
    except (subprocess.CalledProcessError, ValueError, FileNotFoundError):
        return None


def find_orphan_jpeg_dirs(dataset_dir: Path) -> list[Path]:
    """`.d` JPEG directories with no corresponding data parquet — the
    episode's `save_episode` never ran. Safe to remove unconditionally."""
    videos_dir = dataset_dir / "videos"
    data_dir = dataset_dir / "data"
    orphans: list[Path] = []
    for d in sorted(videos_dir.rglob("file-*.d")):
        # videos/<key>/chunk-NNN/file-NNN.d → data/chunk-NNN/file-NNN.parquet
        parquet = data_dir / d.parent.name / (d.stem + ".parquet")
        if not parquet.exists():
            orphans.append(d)
    return orphans


def find_short_episodes(
    dataset_dir: Path, threshold: int, chunks_size: int,
    video_keys: list[str],
) -> list[tuple[int, str]]:
    """Episode indices whose parquet OR any camera MP4 has fewer than
    `threshold` frames. Returns (episode_index, reason) tuples."""
    total_eps = discover_total_episodes(dataset_dir, chunks_size)
    shorts: list[tuple[int, str]] = []
    for ep in range(total_eps):
        pq_path = data_path(dataset_dir, ep, chunks_size)
        n_rows = _parquet_row_count(pq_path)
        if n_rows is None:
            # Covered by the orphan pass if .d exists; otherwise the
            # episode is referenced by meta/episodes but has no
            # parquet — definitely broken, deserves removal too.
            shorts.append((ep, "parquet missing"))
            continue
        if n_rows < threshold:
            shorts.append((ep, f"parquet has only {n_rows} rows"))
            continue
        worst: tuple[str, int] | None = None
        for vk in video_keys:
            mp4 = video_path(dataset_dir, vk, ep, chunks_size)
            n_frames = _mp4_frame_count(mp4)
            if n_frames is None:
                worst = (vk, 0)
                break
            if worst is None or n_frames < worst[1]:
                worst = (vk, n_frames)
        if worst is not None and worst[1] < threshold:
            shorts.append((ep, f"{worst[0]} has only {worst[1]} frames"))
    return shorts


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-dir", required=True, type=Path,
                    help="Dataset directory (LeRobot v3 layout).")
    ap.add_argument("--threshold", type=int, default=100,
                    help="Minimum frame count. Episodes below this in "
                         "parquet OR in any camera's MP4 are removed. "
                         "Default 100 = ~1.7s at 60fps.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print what would be removed; make no changes.")
    ap.add_argument("--orphans-only", action="store_true",
                    help="Only remove orphan .d directories (skip the "
                         "short-episode pass). Useful before a "
                         "subsequent finalize_jpeg_recordings.py run.")
    args = ap.parse_args()

    if not args.dataset_dir.is_dir():
        print(f"[!] {args.dataset_dir} is not a directory", file=sys.stderr)
        return 2

    info_path = args.dataset_dir / "meta" / "info.json"
    if not info_path.exists():
        print(f"[!] {info_path} missing — not a LeRobot dataset?",
              file=sys.stderr)
        return 2

    # --- Pass 1: orphan .d directories ---
    orphans = find_orphan_jpeg_dirs(args.dataset_dir)
    if orphans:
        print(f"[+] Orphan JPEG directories (no matching parquet): "
              f"{len(orphans)}")
        for d in orphans:
            n = len(list(d.glob("frame_*.jpg")))
            rel = d.relative_to(args.dataset_dir)
            print(f"    {rel}  ({n} JPEGs)")
        if not args.dry_run:
            for d in orphans:
                shutil.rmtree(d)
            print(f"[+] Removed {len(orphans)} orphan .d directories")
    else:
        print("[+] No orphan .d directories")

    if args.orphans_only:
        return 0

    # --- Pass 2: short episodes ---
    info = json.loads(info_path.read_text())
    chunks_size = info.get("chunks_size", 1000)
    video_keys = discover_video_keys(args.dataset_dir)

    shorts = find_short_episodes(
        args.dataset_dir, args.threshold, chunks_size, video_keys)
    if not shorts:
        print(f"[+] No episodes below threshold {args.threshold}")
        return 0

    print(f"[+] Episodes below threshold {args.threshold}: {len(shorts)}")
    for ep, reason in shorts:
        print(f"    episode {ep}: {reason}")

    if args.dry_run:
        print("[+] --dry-run: no changes made")
        return 0

    episodes_to_delete = [ep for ep, _ in shorts]
    print(f"\n[*] Delegating to delete_episodes(...) for "
          f"{len(episodes_to_delete)} episode(s)...")
    delete_episodes(args.dataset_dir, episodes_to_delete, dry_run=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
