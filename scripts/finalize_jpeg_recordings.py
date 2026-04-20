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

"""Encode the per-frame JPEGs produced by :class:`JpegOfflineEncoder`
into per-episode MP4 files usable by the LeRobot training pipeline.

Intended to run **after** an auto-rollout recording session, when the
GPU is idle. The on-robot script writes JPEGs to:

    {dataset}/videos/observation.images.{cam}/chunk-NNN/file-NNN.d/
        frame_000000.jpg
        ...

This script walks the ``videos/`` tree, finds every ``*.d`` directory,
and runs ``ffmpeg`` to produce ``file-NNN.mp4`` next to it. The
resulting MP4s match the GOP / keyframe parameters NvencEncoder uses
(2 keyframes/s, no B-frames, h264_nvenc preferred with libx264
fallback) so the dataset is indistinguishable from one recorded via
streaming NVENC.

Idempotent — re-running skips episodes whose MP4 already exists and
whose file size is non-zero. Pass ``--delete-jpegs`` to remove the
source ``.d`` directories after a successful encode (saves ~1 GB per
minute of recording per camera).

Usage::

    python scripts/finalize_jpeg_recordings.py \\
        --dataset /path/to/dk1_duplo_auto_2026-04-20

    # with cleanup:
    python scripts/finalize_jpeg_recordings.py \\
        --dataset /path/to/dk1_duplo_auto_2026-04-20 --delete-jpegs

    # change default encoder / fps (matches NvencEncoder defaults):
    python scripts/finalize_jpeg_recordings.py \\
        --dataset /path/to/dk1_duplo_auto_2026-04-20 \\
        --codec libx264 --fps 30
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


def detect_ffmpeg_codec(preferred: str) -> str:
    """Probe ffmpeg for *preferred*; fall back to ``libx264`` if missing.

    Matches the fallback logic in :func:`nvenc_encoder.detect_codec` so
    an offline-encoded MP4 looks identical to one produced live on a
    machine without NVENC support.
    """
    try:
        out = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True, text=True, check=True,
        )
        if f" {preferred} " in out.stdout:
            return preferred
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass
    print(f"[!] ffmpeg does not advertise {preferred!r}; falling back "
          f"to libx264", file=sys.stderr)
    return "libx264"


def find_jpeg_episode_dirs(videos_dir: Path) -> list[Path]:
    """Return every `*.d/` directory under videos_dir, sorted."""
    return sorted(videos_dir.rglob("file-*.d"))


def encode_one(
    ep_dir: Path,
    target_mp4: Path,
    codec: str,
    fps: int,
    overwrite: bool,
    threads: int = 2,
) -> tuple[bool, str]:
    """Encode one JPEG sequence to MP4. Returns (ok, message).

    Skips if the target MP4 already exists with non-zero size unless
    ``overwrite=True``.
    """
    if target_mp4.exists() and target_mp4.stat().st_size > 0 and not overwrite:
        return True, f"skipped (exists, {target_mp4.stat().st_size / 1e6:.1f} MB)"

    # Enumerate JPEG frames + detect gaps. Filenames are
    # frame_NNNNNN.jpg with the authoritative frame_index (same as
    # scalar parquet row). A gap at index K means the recorder
    # dropped that tick's frame (queue full, imencode failure, ...)
    # but still logged the scalar row. To keep video and parquet in
    # 1:1 sync we duplicate the preceding frame into the gap BEFORE
    # running ffmpeg.
    frames = sorted(ep_dir.glob("frame_*.jpg"))
    if not frames:
        return False, "no frames found"

    indices = []
    for f in frames:
        try:
            indices.append(int(f.stem.split("_")[1]))
        except (IndexError, ValueError):
            pass
    indices.sort()
    max_idx = indices[-1]
    expected = set(range(max_idx + 1))
    missing = sorted(expected - set(indices))
    gaps_filled = 0
    if missing:
        # Fill each gap with a copy of the nearest preceding frame.
        present = set(indices)
        for idx in missing:
            prev = idx - 1
            while prev not in present and prev > 0:
                prev -= 1
            if prev not in present:
                return False, (f"missing frame {idx} and no preceding "
                                f"frame to duplicate from")
            src = ep_dir / f"frame_{prev:06d}.jpg"
            dst = ep_dir / f"frame_{idx:06d}.jpg"
            # Hard link if possible (saves disk); fall back to copy.
            try:
                dst.hardlink_to(src)
            except OSError:
                shutil.copyfile(src, dst)
            present.add(idx)
            gaps_filled += 1
        print(f"    [{ep_dir.name}] filled {gaps_filled} gap(s); "
              f"span now 0..{max_idx}")

    # GOP = fps // 2 matches NvencEncoder's default for seekability.
    gop = max(1, fps // 2)
    codec_opts: list[str] = []
    if "nvenc" in codec:
        codec_opts = [
            "-preset", "p4", "-tune", "hq",
            "-g", str(gop), "-bf", "0",
        ]
    elif codec == "libx264":
        codec_opts = [
            "-preset", "veryfast",
            "-g", str(gop), "-keyint_min", str(gop),
            "-sc_threshold", "0", "-bf", "0",
        ]

    target_mp4.parent.mkdir(parents=True, exist_ok=True)
    # Write to a sibling .partial.mp4 first, then rename atomically —
    # so a crash mid-encode doesn't leave a half-written MP4 that
    # would be mistaken for complete on a re-run. Keep the `.mp4`
    # suffix so ffmpeg picks the mp4 muxer from the extension
    # (ffmpeg would choke on `.mp4.tmp`: "Unable to choose output
    # format for '*.mp4.tmp'"). The `.partial.mp4` pattern preserves
    # both that hint and the "not done yet" marker.
    tmp_out = target_mp4.with_name(target_mp4.stem + ".partial.mp4")
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "warning", "-y",
        "-threads", str(threads),
        "-framerate", str(fps),
        "-i", str(ep_dir / "frame_%06d.jpg"),
        "-c:v", codec, *codec_opts,
        "-pix_fmt", "yuv420p",
        "-f", "mp4",
        str(tmp_out),
    ]
    try:
        # Capture stderr so real errors (bad JPEG, codec failure,
        # disk-full) surface in our output — ffmpeg sends diagnostics
        # to stderr, and a silent non-zero exit is unhelpful.
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            if tmp_out.exists():
                tmp_out.unlink()
            err = (result.stderr or result.stdout or "").strip()
            # Keep the message one line so the summary stays readable.
            err_head = err.splitlines()[-1] if err else "(no stderr)"
            return False, (f"ffmpeg rc={result.returncode}: {err_head}")
    except FileNotFoundError:
        return False, "ffmpeg not on PATH"
    tmp_out.rename(target_mp4)
    return True, (f"encoded {len(frames)} frames → "
                  f"{target_mp4.stat().st_size / 1e6:.1f} MB")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True, type=Path,
                    help="Dataset directory (e.g., "
                         "~/.../dk1_duplo_auto_2026-04-20).")
    ap.add_argument("--codec", default="h264_nvenc",
                    help="ffmpeg codec. Defaults to h264_nvenc; "
                         "falls back to libx264 if unavailable.")
    ap.add_argument("--fps", type=int, default=None,
                    help="Output fps. Default: read from dataset "
                         "meta/info.json.")
    ap.add_argument("--overwrite", action="store_true",
                    help="Re-encode even if target MP4 already exists.")
    ap.add_argument("--delete-jpegs", action="store_true",
                    help="Remove source .d directories after a "
                         "successful encode. Saves ~1 GB per minute of "
                         "recording per camera. Idempotent (re-runs "
                         "after delete are no-ops).")
    ap.add_argument("--parallel", type=int, default=3,
                    help="Number of episodes to encode concurrently. "
                         "Each ffmpeg job is mostly bottlenecked on "
                         "CPU JPEG decode (NVENC itself is idle ~90%% "
                         "of the time when fed a single stream). "
                         "Running 3 jobs in parallel saturates NVMe "
                         "read + CPU decode + NVENC without oversub "
                         "on Thor. Set 1 to disable parallelism for "
                         "predictable ordering.")
    ap.add_argument("--threads-per-job", type=int, default=2,
                    help="ffmpeg -threads value, for JPEG decode "
                         "parallelism WITHIN a single encode job. "
                         "Default 2; bump if --parallel is low and "
                         "CPU still idle.")
    args = ap.parse_args()

    if not args.dataset.is_dir():
        print(f"[!] {args.dataset} is not a directory", file=sys.stderr)
        return 2
    videos_dir = args.dataset / "videos"
    if not videos_dir.is_dir():
        print(f"[!] {videos_dir} not found; was --record-encoding set?",
              file=sys.stderr)
        return 2

    fps = args.fps
    if fps is None:
        info_json = args.dataset / "meta" / "info.json"
        if info_json.exists():
            try:
                fps = int(json.loads(info_json.read_text()).get("fps", 60))
            except Exception:
                fps = 60
        else:
            fps = 60
    print(f"[+] fps={fps}")

    codec = detect_ffmpeg_codec(args.codec)
    print(f"[+] codec={codec}")

    ep_dirs = find_jpeg_episode_dirs(videos_dir)
    print(f"[+] found {len(ep_dirs)} JPEG episode dirs under {videos_dir}")
    parallel = max(1, args.parallel)
    print(f"[+] parallel={parallel}, threads-per-job={args.threads_per_job}")

    def _target_for(ep_dir: Path) -> Path:
        # .../file-NNN.d → .../file-NNN.mp4
        target = ep_dir.with_suffix(".mp4")
        if target.suffix == ".mp4" and target.name.endswith(".d.mp4"):
            # Path.with_suffix on a directory like file-NNN.d is
            # actually file-NNN.mp4 (trailing dir suffix is treated
            # like an extension). Be defensive anyway.
            target = ep_dir.parent / (ep_dir.name[:-2] + ".mp4")
        return target

    def _encode_and_maybe_cleanup(ep_dir: Path):
        """Wrapper so the pool maps one future per episode. Returns
        (ep_dir, target, ok, msg) so the outer loop can log + decide
        on cleanup ordering deterministically."""
        target = _target_for(ep_dir)
        ok, msg = encode_one(
            ep_dir, target, codec, fps,
            overwrite=args.overwrite,
            threads=args.threads_per_job,
        )
        return ep_dir, target, ok, msg

    n_ok = n_skip = n_fail = 0

    def _handle_result(ep_dir, target, ok, msg):
        nonlocal n_ok, n_skip, n_fail
        rel = ep_dir.relative_to(videos_dir)
        if ok:
            if "skipped" in msg:
                n_skip += 1
            else:
                n_ok += 1
            print(f"  [{rel}] {msg}")
            if args.delete_jpegs and target.exists():
                try:
                    shutil.rmtree(ep_dir)
                except OSError as e:
                    print(f"  [{rel}] could not remove JPEG dir: {e}",
                          file=sys.stderr)
        else:
            n_fail += 1
            print(f"  [{rel}] FAILED: {msg}", file=sys.stderr)

    if parallel == 1:
        # Sequential path: deterministic ordering, no ThreadPoolExecutor
        # overhead. Useful when debugging a specific episode's encode.
        try:
            for d in ep_dirs:
                _handle_result(*_encode_and_maybe_cleanup(d))
        except KeyboardInterrupt:
            print("\n[!] Interrupted — partial .partial.mp4 files remain; "
                  "re-run to resume.", file=sys.stderr)
    else:
        # ThreadPoolExecutor: ffmpeg runs in a subprocess (releases
        # the GIL during encode). Each NVENC encode uses one hardware
        # session; Thor supports several concurrently.
        #
        # Ctrl+C path: raises KeyboardInterrupt in the main thread.
        # We then cancel PENDING futures and ask the executor to
        # shut down without waiting. Currently-running ffmpeg
        # subprocesses share our process group so they receive SIGINT
        # from the shell and exit; `encode_one` removes their
        # .partial.mp4 on non-zero exit, so the dataset stays in a
        # resumable state (re-running the script encodes anything
        # still missing).
        import concurrent.futures as _cf
        with _cf.ThreadPoolExecutor(
            max_workers=parallel, thread_name_prefix="jpeg_enc",
        ) as ex:
            futures = {ex.submit(_encode_and_maybe_cleanup, d): d
                       for d in ep_dirs}
            try:
                for f in _cf.as_completed(futures):
                    _handle_result(*f.result())
            except KeyboardInterrupt:
                print("\n[!] Interrupted — cancelling pending jobs; "
                      "running ffmpeg subprocesses will stop shortly "
                      "(they share our process group). Re-run the "
                      "script to resume; already-finalized MP4s are "
                      "skipped, in-flight .partial.mp4 files are "
                      "cleaned up by encode_one on ffmpeg's non-zero "
                      "exit.", file=sys.stderr)
                for fut in futures:
                    fut.cancel()
                # shutdown(wait=False, cancel_futures=True) is
                # effectively what `with:` does once we exit — we
                # just let the context manager handle it.
                raise

    print(f"[+] encoded={n_ok}, skipped={n_skip}, failed={n_fail}")

    # Post-pass verification: cross-check every camera's MP4 frame
    # count against the episode's parquet row count. Catches cases
    # where a camera dropped frames mid-episode (producing a short
    # MP4 while the scalar parquet kept growing) — those episodes
    # are unusable for training because the dataloader seeks past
    # MP4 EOF. Reported here rather than at upload time so the
    # operator can re-record or drop the episode before it ships.
    print("[+] Verifying MP4 frame counts match parquet row counts ...")
    import subprocess as _sp
    try:
        import pyarrow.parquet as _pq
    except ImportError:
        print("[!] pyarrow not installed — skipping parquet/video check")
        return 0 if n_fail == 0 else 1

    data_dir = args.dataset / "data"
    video_root = args.dataset / "videos"
    mismatches: list[str] = []

    if data_dir.exists() and video_root.exists():
        cam_dirs = sorted(d for d in video_root.iterdir()
                          if d.is_dir() and d.name.startswith("observation.images."))
        for pq in sorted(data_dir.rglob("file-*.parquet")):
            try:
                n_rows = _pq.read_table(str(pq)).num_rows
            except Exception as e:
                print(f"    [!] could not read {pq.name}: {e}")
                continue
            # "chunk-000/file-007.parquet" → ("000", "007")
            chunk = pq.parent.name
            file_stem = pq.stem
            for cam_dir in cam_dirs:
                mp4 = cam_dir / chunk / f"{file_stem}.mp4"
                if not mp4.exists():
                    mismatches.append(
                        f"{cam_dir.name}/{chunk}/{file_stem}.mp4 missing "
                        f"(parquet has {n_rows} rows)")
                    continue
                try:
                    out = _sp.run(
                        ["ffprobe", "-v", "error", "-select_streams", "v:0",
                         "-count_packets", "-show_entries",
                         "stream=nb_read_packets", "-of", "csv=p=0",
                         str(mp4)],
                        capture_output=True, text=True, check=True,
                    ).stdout.strip()
                    n_video = int(out) if out else 0
                except Exception as e:
                    mismatches.append(
                        f"{cam_dir.name}/{chunk}/{file_stem}.mp4 ffprobe "
                        f"failed: {e}")
                    continue
                if n_video < n_rows:
                    mismatches.append(
                        f"{cam_dir.name}/{chunk}/{file_stem}.mp4 has "
                        f"{n_video} frames but parquet has {n_rows} rows "
                        f"(short by {n_rows - n_video} / "
                        f"{(n_rows - n_video) / fps:.1f}s)")
    if mismatches:
        print(f"[!] {len(mismatches)} video/parquet mismatch(es):")
        for m in mismatches:
            print(f"    {m}")
        print("    Training will read past MP4 EOF on affected episodes. "
              "Drop them via examples/delete_episode.py before upload.")
    else:
        print("[+] All videos OK (≥ parquet row count)")

    return 0 if (n_fail == 0 and not mismatches) else 1


if __name__ == "__main__":
    sys.exit(main())
