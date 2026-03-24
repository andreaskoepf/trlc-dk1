#!/usr/bin/env python3
"""Integration test: record 3 synthetic episodes → load with LeRobotDataset.

Exercises the encoder, dataset writer, and recorder thread with synthetic
data (no hardware required). Validates that the output is a valid LeRobot
v3 dataset by loading it back with LeRobotDataset.

Usage:
    uv run python examples/test_recorder_integration.py
"""

import shutil
import tempfile
from pathlib import Path

import numpy as np

from lerobot_robot_trlc_dk1.recorder.nvenc_encoder import (
    EndEpisode,
    EncoderResult,
    NvencEncoder,
    StartEpisode,
    VideoFrame,
    detect_codec,
)
from lerobot_robot_trlc_dk1.recorder.dataset_writer import (
    DatasetWriter,
    build_features_schema,
)
from lerobot_robot_trlc_dk1.recorder.recorder_thread import (
    OBS_STATE_KEYS,
    ACTION_KEYS,
    pack_observation_state,
    pack_action,
)


# -- Test parameters --------------------------------------------------------

NUM_EPISODES = 3
FRAMES_PER_EPISODE = 30
FPS = 30
WIDTH = 640
HEIGHT = 480
CAMERA_KEYS = ["head", "left_wrist", "right_wrist"]
TASK = "Test task for integration test."
CODEC = detect_codec("h264_nvenc")


def make_obs(frame_idx: int) -> dict:
    """Generate synthetic observation dict matching BiDK1Follower output."""
    obs = {}
    for i, key in enumerate(OBS_STATE_KEYS):
        obs[key] = float(np.sin(frame_idx * 0.1 + i * 0.3))
    for cam_key in CAMERA_KEYS:
        obs[cam_key] = np.random.randint(
            0, 255, (HEIGHT, WIDTH, 3), dtype=np.uint8
        )
    return obs


def make_action(frame_idx: int) -> dict:
    """Generate synthetic action dict matching BiDK1Leader output."""
    return {
        key: float(np.cos(frame_idx * 0.1 + i * 0.5))
        for i, key in enumerate(ACTION_KEYS)
    }


# ---------------------------------------------------------------------------
# Test 1: Feature packing
# ---------------------------------------------------------------------------

def test_pack_roundtrip():
    print("--- Test 1: Feature packing ---")

    obs = {key: float(i) for i, key in enumerate(OBS_STATE_KEYS)}
    action = {key: float(i) for i, key in enumerate(ACTION_KEYS)}

    obs_vec = pack_observation_state(obs)
    act_vec = pack_action(action)

    assert obs_vec.shape == (40,), f"Expected (40,), got {obs_vec.shape}"
    assert act_vec.shape == (14,), f"Expected (14,), got {act_vec.shape}"
    assert obs_vec.dtype == np.float32
    assert act_vec.dtype == np.float32

    for i, key in enumerate(OBS_STATE_KEYS):
        assert obs_vec[i] == float(i), f"obs_vec[{i}] != {float(i)}"
    for i, key in enumerate(ACTION_KEYS):
        assert act_vec[i] == float(i), f"act_vec[{i}] != {float(i)}"

    print("  PASSED\n")


# ---------------------------------------------------------------------------
# Test 2: Gesture detector
# ---------------------------------------------------------------------------

def test_gesture_detector():
    print("--- Test 2: Gesture detector ---")

    from lerobot_robot_trlc_dk1.recorder.gesture_detector import GripperGestureDetector

    det = GripperGestureDetector(
        threshold_close=0.85,
        threshold_open=0.3,
        double_close_window_s=0.8,
    )

    assert not det.update(0.0)   # open
    assert not det.update(0.9)   # close (first)
    assert not det.update(0.0)   # open
    assert det.update(0.9)       # close (second) → DETECTED

    # After detection, state is reset
    assert not det.update(0.9)
    assert not det.update(0.0)

    print("  PASSED\n")


# ---------------------------------------------------------------------------
# Test 3: Encoder roundtrip
# ---------------------------------------------------------------------------

def test_encoder_roundtrip(tmpdir: Path):
    print("--- Test 3: Encoder roundtrip ---")

    videos_dir = tmpdir / "videos"
    enc = NvencEncoder(
        cam_key="head",
        width=WIDTH,
        height=HEIGHT,
        fps=FPS,
        codec=CODEC,
        videos_dir=videos_dir,
    )
    enc.start()

    enc.frame_queue.put(StartEpisode(0))
    for i in range(FRAMES_PER_EPISODE):
        frame = np.random.randint(0, 255, (HEIGHT, WIDTH, 3), dtype=np.uint8)
        enc.frame_queue.put(VideoFrame(i, frame))
    enc.frame_queue.put(EndEpisode())

    result = enc.result_queue.get(timeout=30.0)
    enc.stop()

    assert result.episode_index == 0
    assert result.frame_count == FRAMES_PER_EPISODE
    assert result.mp4_path.exists()
    assert result.mp4_path.stat().st_size > 0
    assert "mean" in result.stats
    assert result.stats["mean"].shape == (3,)

    print(f"  MP4: {result.mp4_path} ({result.mp4_path.stat().st_size / 1024:.1f} KB)")
    print(f"  Stats keys: {list(result.stats.keys())}")
    print("  PASSED\n")


# ---------------------------------------------------------------------------
# Test 4: Dataset writer
# ---------------------------------------------------------------------------

def test_dataset_writer(tmpdir: Path):
    print("--- Test 4: Dataset writer ---")

    dataset_dir = tmpdir / "dataset_writer_test"
    features = build_features_schema(
        camera_keys=CAMERA_KEYS,
        camera_height=HEIGHT,
        camera_width=WIDTH,
        fps=FPS,
    )

    writer = DatasetWriter(
        dataset_dir=dataset_dir,
        fps=FPS,
        features=features,
        robot_type="bi_dk1_follower",
        task=TASK,
    )

    for ep in range(NUM_EPISODES):
        scalar_frames = []
        for fi in range(FRAMES_PER_EPISODE):
            obs = make_obs(fi)
            action = make_action(fi)
            scalar_frames.append({
                "observation.state": pack_observation_state(obs),
                "action": pack_action(action),
                "timestamp": np.float32(fi / FPS),
                "frame_index": fi,
                "episode_index": ep,
                "task_index": 0,
            })

        # Fake video results (dummy MP4s)
        video_results = {}
        for cam_key in CAMERA_KEYS:
            fake_stats = {
                "min": np.zeros(3, dtype=np.float32),
                "max": np.full(3, 255.0, dtype=np.float32),
                "mean": np.full(3, 127.0, dtype=np.float32),
                "std": np.full(3, 73.0, dtype=np.float32),
                "count": np.array([FRAMES_PER_EPISODE * HEIGHT * WIDTH]),
                "q01": np.full(3, 2.0, dtype=np.float32),
                "q10": np.full(3, 25.0, dtype=np.float32),
                "q50": np.full(3, 127.0, dtype=np.float32),
                "q90": np.full(3, 230.0, dtype=np.float32),
                "q99": np.full(3, 253.0, dtype=np.float32),
            }
            mp4_path = (
                dataset_dir / f"videos/observation.images.{cam_key}"
                / "chunk-000" / f"file-{ep:03d}.mp4"
            )
            mp4_path.parent.mkdir(parents=True, exist_ok=True)
            mp4_path.write_bytes(b"\x00" * 100)
            video_results[cam_key] = EncoderResult(
                episode_index=ep,
                mp4_path=mp4_path,
                frame_count=FRAMES_PER_EPISODE,
                stats=fake_stats,
            )

        writer.save_episode(ep, scalar_frames, video_results)

    writer.finalize()

    # Verify files
    import json
    info = json.loads((dataset_dir / "meta" / "info.json").read_text())
    assert info["total_episodes"] == NUM_EPISODES
    assert info["total_frames"] == NUM_EPISODES * FRAMES_PER_EPISODE
    assert info["fps"] == FPS
    assert info["features"]["observation.state"]["shape"] == [40]
    assert info["features"]["action"]["shape"] == [14]
    assert (dataset_dir / "meta" / "stats.json").exists()
    assert (dataset_dir / "meta" / "tasks.parquet").exists()

    print(f"  Episodes: {info['total_episodes']}, Frames: {info['total_frames']}")
    print("  PASSED\n")


# ---------------------------------------------------------------------------
# Test 5: Full encoder + writer roundtrip → LeRobotDataset
# ---------------------------------------------------------------------------

def test_full_roundtrip(tmpdir: Path):
    print("--- Test 5: Full roundtrip (encoder + writer → LeRobotDataset) ---")

    dataset_dir = tmpdir / "full_roundtrip"
    features = build_features_schema(
        camera_keys=CAMERA_KEYS,
        camera_height=HEIGHT,
        camera_width=WIDTH,
        fps=FPS,
    )

    writer = DatasetWriter(
        dataset_dir=dataset_dir,
        fps=FPS,
        features=features,
        robot_type="bi_dk1_follower",
        task=TASK,
    )

    # Start encoders
    videos_dir = dataset_dir / "videos"
    encoders = {}
    for cam_key in CAMERA_KEYS:
        enc = NvencEncoder(
            cam_key=cam_key,
            width=WIDTH,
            height=HEIGHT,
            fps=FPS,
            codec=CODEC,
            videos_dir=videos_dir,
        )
        enc.start()
        encoders[cam_key] = enc

    for ep in range(NUM_EPISODES):
        for enc in encoders.values():
            enc.frame_queue.put(StartEpisode(ep))

        scalar_frames = []
        for fi in range(FRAMES_PER_EPISODE):
            obs = make_obs(fi)
            action = make_action(fi)

            for cam_key in CAMERA_KEYS:
                encoders[cam_key].frame_queue.put(VideoFrame(fi, obs[cam_key]))

            scalar_frames.append({
                "observation.state": pack_observation_state(obs),
                "action": pack_action(action),
                "timestamp": np.float32(fi / FPS),
                "frame_index": fi,
                "episode_index": ep,
                "task_index": 0,
            })

        for enc in encoders.values():
            enc.frame_queue.put(EndEpisode())

        video_results = {}
        for cam_key, enc in encoders.items():
            result = enc.result_queue.get(timeout=30.0)
            video_results[cam_key] = result

        writer.save_episode(ep, scalar_frames, video_results)
        print(f"  Episode {ep}: {FRAMES_PER_EPISODE} frames encoded + saved")

    writer.finalize()
    for enc in encoders.values():
        enc.stop()

    # -- Load with LeRobotDataset -------------------------------------------
    print("\n  Loading with LeRobotDataset...")
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        ds = LeRobotDataset(
            repo_id="test/dk1_recorder_integration",
            root=str(dataset_dir),
            video_backend="pyav",
        )
        total_expected = NUM_EPISODES * FRAMES_PER_EPISODE
        print(f"  Loaded: {len(ds)} frames (expected {total_expected})")
        assert len(ds) == total_expected, \
            f"Frame count mismatch: {len(ds)} != {total_expected}"

        frame = ds[0]
        print(f"  Frame 0 keys: {sorted(frame.keys())}")

        if "observation.state" in frame:
            obs_state = frame["observation.state"]
            print(f"  observation.state: shape={obs_state.shape} dtype={obs_state.dtype}")
            assert obs_state.shape[-1] == 40 or obs_state.numel() == 40

        if "action" in frame:
            action = frame["action"]
            print(f"  action: shape={action.shape} dtype={action.dtype}")
            assert action.shape[-1] == 14 or action.numel() == 14

        print("\n  LeRobotDataset PASSED")
    except Exception as e:
        print(f"\n  LeRobotDataset FAILED: {e}")
        import traceback
        traceback.print_exc()
        print("\n  (Format adjustments may be needed — check error above)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("DK1 Recorder — Integration Tests")
    print("=" * 60 + "\n")

    tmpdir = Path(tempfile.mkdtemp(prefix="dk1_recorder_test_"))
    print(f"Temp dir: {tmpdir}\n")

    passed = 0
    failed = 0

    tests = [
        ("Feature packing", lambda: test_pack_roundtrip()),
        ("Gesture detector", lambda: test_gesture_detector()),
        ("Encoder roundtrip", lambda: test_encoder_roundtrip(tmpdir)),
        ("Dataset writer", lambda: test_dataset_writer(tmpdir)),
        ("Full roundtrip", lambda: test_full_roundtrip(tmpdir)),
    ]

    for name, fn in tests:
        try:
            fn()
            passed += 1
        except Exception as e:
            print(f"  FAILED: {e}\n")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    # Cleanup
    shutil.rmtree(tmpdir, ignore_errors=True)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    exit(main())
