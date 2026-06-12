#!/usr/bin/env python3
"""Verify a trimmed dataset against its source.

Checks, across ALL episodes x cameras:
  - every trimmed video's frame count == metadata length
  - from_timestamp == 0, to_timestamp == frame_count / fps
  - episode indices contiguous; info.total_frames == sum(length)
And a random pixel-alignment sample (copied body bit-identical to source[S+i]).

Exit 0 if everything passes, 1 otherwise.

Usage: verify_trim.py <source_dataset_dir> <trimmed_dataset_dir>
"""
import sys, json, subprocess, hashlib, random
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import pyarrow.parquet as pq


def count(p):
    if not p.exists():
        return None
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
         "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", str(p)],
        capture_output=True, text=True).stdout
    return int(out) if out.strip() else None


def frame_md5(video, idx):
    out = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(video), "-vf",
         f"select=eq(n\\,{idx})", "-vframes", "1", "-f", "rawvideo",
         "-pix_fmt", "rgb24", "-"], capture_output=True).stdout
    return hashlib.md5(out).hexdigest()


def main():
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    info = json.loads((dst / "meta" / "info.json").read_text())
    fps = info["fps"]
    sm = pq.read_table(list((src / "meta/episodes").rglob("*.parquet"))[0]).to_pydict()
    dm = pq.read_table(list((dst / "meta/episodes").rglob("*.parquet"))[0]).to_pydict()
    vks = sorted({c.split("/")[1] for c in dm if c.startswith("videos/")})
    n = len(dm["episode_index"])
    print(f"verify: {n} episodes x {len(vks)} cameras (fps={fps})")

    def vpath(meta, i, vk):
        ch = meta[f"videos/{vk}/chunk_index"][i]; fi = meta[f"videos/{vk}/file_index"][i]
        return f"videos/{vk}/chunk-{ch:03d}/file-{fi:03d}.mp4"

    def check(i):
        probs = []
        length = dm["length"][i]
        for vk in vks:
            fts = dm[f"videos/{vk}/from_timestamp"][i]
            tts = dm[f"videos/{vk}/to_timestamp"][i]
            c = count(dst / vpath(dm, i, vk))
            if c is None:
                probs.append(f"ep{i} {vk}: MISSING"); continue
            if fts != 0.0:
                probs.append(f"ep{i} {vk}: from_ts={fts}!=0")
            if c != length:
                probs.append(f"ep{i} {vk}: frames={c}!=length={length}")
            if abs(tts - c / fps) > 1e-6:
                probs.append(f"ep{i} {vk}: to_ts={tts}!=frames/fps")
        return probs

    problems = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        for r in ex.map(check, range(n)):
            problems += r

    if dm["episode_index"] != list(range(n)):
        problems.append("episode indices not contiguous")
    if info["total_frames"] != sum(dm["length"]):
        problems.append(f"total_frames {info['total_frames']} != sum(length) {sum(dm['length'])}")

    # pixel alignment on a random sample (copied body == source[S+mid])
    random.seed(0)
    sample = random.sample(range(n), min(8, n))
    for i in sample:
        for vk in vks:
            S = round(sm[f"videos/{vk}/from_timestamp"][i] * fps)
            mid = dm["length"][i] // 2
            if frame_md5(dst / vpath(dm, i, vk), mid) != frame_md5(src / vpath(sm, i, vk), S + mid):
                problems.append(f"ep{i} {vk}: MISALIGNED body frame")

    print(f"  full frame-count/metadata check: {n * len(vks)} videos")
    print(f"  pixel-alignment sample: {len(sample)} episodes x {len(vks)} cameras")
    if problems:
        print(f"FAILED: {len(problems)} problem(s)")
        for p in problems[:30]:
            print("  " + p)
        return 1
    print("PASS: all frame counts == length, from_ts=0, to_ts=frames/fps, body bit-identical")
    return 0


if __name__ == "__main__":
    sys.exit(main())
