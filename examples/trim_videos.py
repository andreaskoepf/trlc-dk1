#!/usr/bin/env python3
"""Trim pre-roll / post-roll from a LeRobot v3 dataset's episode videos.

Each recorded MP4 is laid out as ``[pre-roll][episode][tail]``: the cameras
roll during the countdown (pre-roll) and, for gesture-ended episodes, keep
recording the stop motion (tail). The scalar parquet is already trimmed to
just the episode; only the videos carry the extra frames, and the LeRobot
viewer ignores the ``from_timestamp``/``to_timestamp`` window for per-episode
video files. This tool physically cuts each video down to exactly the episode
so begin/end align, and rewrites the metadata (from_timestamp -> 0,
to_timestamp -> length/fps, video_files_size_in_mb).

Cutting strategy (per video), maximising quality retention:

  * Lossless stream-copy when the episode starts on a keyframe and ends at
    EOF (no tail). Bit-identical, no re-encode. New recordings force an IDR at
    the episode start, so their pre-roll trims losslessly.
  * Smart-cut otherwise: re-encode only the partial GOP at each cut boundary
    (<= one keyframe interval, ~0.5s) and stream-copy the bit-identical middle.
    Used for legacy datasets (mid-GOP start) and gesture-ended tails. The
    re-encoded and copied halves are stitched through an MPEG-TS intermediate
    (parameter sets carried in-band) so HEVC VPS/SPS/PPS survive the concat.
  * Full re-encode fallback for clips too short to split (< 2 keyframes).

Every output is validated (frame count == episode length, clean decode) before
it is committed; in --in-place mode a failing video is left untouched.

Usage:
    # Write a trimmed copy to a new directory (source untouched):
    python examples/trim_videos.py --dataset-dir ./data/foo --output ./data/foo_trimmed

    # Overwrite videos in place (temp + atomic rename):
    python examples/trim_videos.py --dataset-dir ./data/foo --in-place

    # Preview without writing anything:
    python examples/trim_videos.py --dataset-dir ./data/foo --output ./data/foo_trimmed --dry-run
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

# MP4 codec tag string -> (libavcodec encoder, annexb bitstream filter)
_CODEC_MAP = {
    "hevc": ("libx265", "hevc_mp4toannexb"),
    "h264": ("libx264", "h264_mp4toannexb"),
}


# ---------------------------------------------------------------------------
# ffprobe / ffmpeg helpers
# ---------------------------------------------------------------------------

def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True)


def probe_codec(path: Path) -> str:
    return _run([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=codec_name", "-of", "csv=p=0", str(path),
    ]).stdout.strip()


def probe_frame_count(path: Path) -> int:
    out = _run([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
        "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", str(path),
    ]).stdout.strip()
    return int(out) if out else 0


def probe_keyframes(path: Path, fps: int) -> list[int]:
    """Return sorted keyframe positions as integer frame indices."""
    out = _run([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-skip_frame", "nokey",
        "-show_entries", "frame=pts_time", "-of", "csv=p=0", str(path),
    ]).stdout.split()
    return sorted({int(round(float(x) * fps)) for x in out if x})


# ---------------------------------------------------------------------------
# Cut planning
# ---------------------------------------------------------------------------

@dataclass
class Segment:
    op: str       # "copy" | "reenc"
    start: int    # inclusive frame
    end: int      # exclusive frame


@dataclass
class Plan:
    kind: str             # "copyfile" | "copy" | "smartcut" | "reenc"
    segments: list[Segment] = field(default_factory=list)
    n_out: int = 0        # expected output frame count
    reenc_frames: int = 0 # how many frames get re-encoded
    warnings: list[str] = field(default_factory=list)


def plan_cut(S: int, n: int, total: int, kf: list[int]) -> Plan:
    """Decide how to cut a video down to episode frames [S, S+n).

    S      pre-roll frames (episode start within the MP4)
    n      episode length (scalar frame count)
    total  frames actually in the MP4
    kf     keyframe frame indices
    """
    warns: list[str] = []
    E = S + n
    if E > total:
        # Video is shorter than the scalar data (a recording frame-drop).
        # Keep every real frame we have; can't fabricate the missing ones.
        warns.append(f"video short by {E - total} frame(s); trimming to available {total}")
        E = total
    n_out = E - S
    kfset = set(kf)
    head_clean = S in kfset          # S==0 is always a keyframe
    tail_clean = E >= total          # cutting at EOF needs no tail re-encode

    if head_clean and tail_clean:
        kind = "copyfile" if (S == 0 and E == total) else "copy"
        return Plan(kind, [Segment("copy", S, E)], n_out, 0, warns)

    ks_ge = [k for k in kf if k >= S]
    K_head = S if head_clean else (ks_ge[0] if ks_ge else None)
    ks_le = [k for k in kf if k <= E - 1]
    K_tail = ks_le[-1] if ks_le else None

    body_start = K_head
    body_end = E if tail_clean else K_tail

    # Not enough keyframes to carve a clean copyable middle -> re-encode it all.
    if (body_start is None or body_start >= E
            or (not tail_clean and (K_tail is None or K_tail <= body_start))):
        return Plan("reenc", [Segment("reenc", S, E)], n_out, n_out, warns)

    segs: list[Segment] = []
    reenc = 0
    if not head_clean:
        segs.append(Segment("reenc", S, K_head))
        reenc += K_head - S
    segs.append(Segment("copy", body_start, body_end))
    if not tail_clean:
        segs.append(Segment("reenc", K_tail, E))
        reenc += E - K_tail
    return Plan("smartcut", segs, n_out, reenc, warns)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _render_segment(src: Path, seg: Segment, total: int, fps: int,
                    encoder: str, bsf: str, crf: int, dst_ts: Path) -> None:
    if seg.op == "copy":
        cmd = ["ffmpeg", "-v", "error", "-y",
               "-ss", f"{(seg.start + 0.5) / fps}", "-i", str(src),
               "-c", "copy", "-bsf:v", bsf, "-an", "-f", "mpegts", str(dst_ts)]
        if seg.end < total:  # interior copy: bound the frame count
            cmd[-3:-3] = ["-frames:v", str(seg.end - seg.start)]
        _check(cmd, src, seg)
    else:
        cmd = ["ffmpeg", "-v", "error", "-y", "-i", str(src),
               "-vf", f"trim=start_frame={seg.start}:end_frame={seg.end},setpts=PTS-STARTPTS",
               "-c:v", encoder, "-crf", str(crf), "-pix_fmt", "yuv420p",
               "-an", "-f", "mpegts", str(dst_ts)]
        if encoder == "libx265":
            cmd[cmd.index("-crf"):cmd.index("-crf")] = ["-x265-params", "log-level=none"]
        _check(cmd, src, seg)


def _check(cmd: list[str], src: Path, seg: Segment) -> None:
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg failed for {src.name} {seg}: {r.stderr[-300:]}")


def render(src: Path, plan: Plan, fps: int, crf: int, dst: Path,
           tag_hvc1: bool) -> None:
    """Produce the trimmed video at *dst* per *plan*."""
    if plan.kind == "copyfile":
        shutil.copy2(src, dst)
        return

    codec = probe_codec(src)
    encoder, bsf = _CODEC_MAP.get(codec, ("libx264", "h264_mp4toannexb"))
    total = probe_frame_count(src)

    # All non-trivial kinds (lossless "copy", "smartcut", "reenc") are rendered
    # as segments stitched through an MPEG-TS intermediate. A direct
    # "-ss -c copy" into MP4 is NOT used: it leaves a leading edit-list (the
    # dropped GOP stays in the file, presentation just skips it) which makes the
    # frame count wrong and seeks fragile. TS carries parameter sets in-band and
    # has no edit lists, so the cut is exact.
    with tempfile.TemporaryDirectory() as td:
        parts = []
        for i, seg in enumerate(plan.segments):
            ts = Path(td) / f"seg{i}.ts"
            _render_segment(src, seg, total, fps, encoder, bsf, crf, ts)
            parts.append(str(ts))
        cmd = ["ffmpeg", "-v", "error", "-y", "-i", "concat:" + "|".join(parts),
               "-c", "copy"]
        if tag_hvc1 and codec == "hevc":
            cmd += ["-tag:v", "hvc1"]
        cmd += ["-movflags", "+faststart", str(dst)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"concat failed for {src.name}: {r.stderr[-300:]}")


def validate(dst: Path, expected: int) -> None:
    """Assert the trimmed video has the expected frame count and decodes clean."""
    got = probe_frame_count(dst)
    if got != expected:
        raise RuntimeError(f"frame count {got} != expected {expected}")
    r = _run(["ffmpeg", "-v", "error", "-i", str(dst), "-f", "null", "-"])
    if r.stderr.strip():
        raise RuntimeError(f"decode errors: {r.stderr[-300:]}")


# ---------------------------------------------------------------------------
# Dataset driver
# ---------------------------------------------------------------------------

@dataclass
class VideoJob:
    ep: int
    vk: str
    src: Path
    dst: Path
    S: int
    n: int


def video_rel_path(template: str, vk: str, chunk: int, file: int) -> str:
    return template.format(video_key=vk, chunk_index=chunk, file_index=file)


def build_jobs(meta: dict, info: dict, src_root: Path, dst_root: Path,
               video_keys: list[str]) -> list[VideoJob]:
    fps = info["fps"]
    template = info["video_path"]
    jobs = []
    for i, ep in enumerate(meta["episode_index"]):
        n = meta["length"][i]
        for vk in video_keys:
            S = round(meta[f"videos/{vk}/from_timestamp"][i] * fps)
            ch = meta[f"videos/{vk}/chunk_index"][i]
            fi = meta[f"videos/{vk}/file_index"][i]
            rel = video_rel_path(template, vk, ch, fi)
            jobs.append(VideoJob(ep, vk, src_root / rel, dst_root / rel, S, n))
    return jobs


def process(dataset_dir: Path, dst_dir: Path, in_place: bool, crf: int,
            jobs_n: int, tag_hvc1: bool, dry_run: bool) -> int:
    info = json.loads((dataset_dir / "meta" / "info.json").read_text())
    fps = info["fps"]
    meta_path = dataset_dir / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    meta = pq.read_table(meta_path).to_pydict()
    video_keys = sorted(d.name for d in (dataset_dir / "videos").iterdir()
                        if d.is_dir() and d.name.startswith("observation.images."))

    out_root = dataset_dir if in_place else dst_dir
    jobs = build_jobs(meta, info, dataset_dir, out_root, video_keys)
    print(f"{len(meta['episode_index'])} episodes x {len(video_keys)} cameras "
          f"= {len(jobs)} videos | fps={fps} | mode={'in-place' if in_place else 'copy'}")

    # actual_n[(ep, vk)] = output frame count (may be < length for short videos)
    actual_n: dict[tuple[int, str], int] = {}
    plans: dict[tuple[int, str], Plan] = {}

    # 1. Plan every video (cheap; ffprobe only).
    for j in jobs:
        if not j.src.exists():
            print(f"  [!] ep{j.ep} {j.vk}: missing {j.src}")
            continue
        total = probe_frame_count(j.src)
        kf = probe_keyframes(j.src, fps)
        p = plan_cut(j.S, j.n, total, kf)
        plans[(j.ep, j.vk)] = p
        actual_n[(j.ep, j.vk)] = p.n_out
        for w in p.warnings:
            print(f"  [!] ep{j.ep} {j.vk}: {w}")

    kinds = {}
    re = 0
    for p in plans.values():
        kinds[p.kind] = kinds.get(p.kind, 0) + 1
        re += p.reenc_frames
    print(f"  plan: {kinds} | total frames re-encoded: {re}")

    if dry_run:
        print("  dry-run: no files written")
        return 0

    # 2. Prepare output tree (copy data + meta; videos written fresh).
    if not in_place:
        dst_dir.mkdir(parents=True, exist_ok=True)
        for sub in ("data", "meta"):
            if (dataset_dir / sub).exists():
                shutil.copytree(dataset_dir / sub, dst_dir / sub, dirs_exist_ok=True)
        for j in jobs:
            j.dst.parent.mkdir(parents=True, exist_ok=True)

    # 3. Render videos (parallel).
    failures = 0

    def work(j: VideoJob):
        p = plans.get((j.ep, j.vk))
        if p is None:
            return j, "missing", None
        target = j.dst
        tmp = target.with_suffix(".trim.tmp.mp4") if in_place else target
        if in_place:
            target.parent.mkdir(parents=True, exist_ok=True)
        try:
            render(j.src, p, fps, crf, tmp, tag_hvc1)
            validate(tmp, p.n_out)
        except Exception as e:  # noqa: BLE001
            if in_place and tmp.exists():
                tmp.unlink()
            return j, "FAIL", str(e)
        if in_place:
            tmp.replace(target)
        return j, p.kind, None

    done = 0
    with ThreadPoolExecutor(max_workers=jobs_n) as ex:
        futs = [ex.submit(work, j) for j in jobs if (j.ep, j.vk) in plans]
        for fut in as_completed(futs):
            j, status, err = fut.result()
            done += 1
            if status == "FAIL":
                failures += 1
                print(f"  [FAIL] ep{j.ep} {j.vk}: {err}")
            if done % 50 == 0 or done == len(futs):
                print(f"  rendered {done}/{len(futs)}")

    # 4. Rewrite metadata (from_timestamp=0, to_timestamp=n/fps).
    for i, ep in enumerate(meta["episode_index"]):
        for vk in video_keys:
            nout = actual_n.get((ep, vk), meta["length"][i])
            meta[f"videos/{vk}/from_timestamp"][i] = 0.0
            meta[f"videos/{vk}/to_timestamp"][i] = nout / fps
    out_meta = (out_root / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    pq.write_table(pa.Table.from_pydict(meta), out_meta, compression="snappy")

    out_info = json.loads((out_root / "meta" / "info.json").read_text())
    vid_dir = out_root / "videos"
    out_info["video_files_size_in_mb"] = round(
        sum(f.stat().st_size for f in vid_dir.rglob("*.mp4")) / (1024 * 1024))
    (out_root / "meta" / "info.json").write_text(json.dumps(out_info, indent=2))

    print(f"Done. {len(jobs) - failures}/{len(jobs)} videos trimmed"
          + (f", {failures} FAILED" if failures else "")
          + f" -> {out_root}")
    return 1 if failures else 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-dir", type=Path, required=True, help="Source dataset")
    dest = p.add_mutually_exclusive_group(required=True)
    dest.add_argument("--in-place", action="store_true",
                      help="Overwrite videos in the source (temp + atomic rename)")
    dest.add_argument("--output", type=Path, help="Write a trimmed copy to this dir")
    p.add_argument("--crf", type=int, default=12,
                   help="CRF for re-encoded boundary GOPs (default: 12, near-lossless)")
    p.add_argument("--jobs", type=int, default=4, help="Parallel ffmpeg workers")
    p.add_argument("--no-hvc1-tag", action="store_true",
                   help="Do not retag HEVC output as hvc1 (browser-compatible)")
    p.add_argument("--dry-run", action="store_true",
                   help="Plan only; write nothing")
    args = p.parse_args()

    if not (args.dataset_dir / "meta" / "info.json").exists():
        print(f"No dataset at {args.dataset_dir}")
        return 2
    if args.output and args.output.resolve() == args.dataset_dir.resolve():
        print("--output must differ from --dataset-dir (use --in-place)")
        return 2

    return process(args.dataset_dir, args.output, args.in_place, args.crf,
                   args.jobs, not args.no_hvc1_tag, args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
