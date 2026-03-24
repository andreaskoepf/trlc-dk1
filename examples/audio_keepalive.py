#!/usr/bin/env python3
"""Audio keepalive — loop a sound file to prevent HDMI audio spin-up delay.

Monitors with HDMI audio (e.g. ASUS) have a ~1s delay before audio starts
playing when the link has been idle. Running this script in a separate
terminal keeps the audio pipeline active so beeps and TTS play instantly.

Usage:
    # Loop built-in quiet tone (default):
    python examples/audio_keepalive.py

    # Loop your own music file at 20% volume:
    python examples/audio_keepalive.py --file ~/music/lofi.mp3 --volume 20

    # List available audio devices:
    python examples/audio_keepalive.py --list-devices

    # Use a specific ALSA device:
    python examples/audio_keepalive.py --device hw:0,3
"""

from __future__ import annotations

import argparse
import shutil
import signal
import subprocess
import sys
import time
import wave
from pathlib import Path

import numpy as np


def generate_ambient_tone(path: Path, duration_s: int = 300):
    """Generate a 5-minute ambient WAV: gentle chord with slow fade waves."""
    sr = 22050
    n = sr * duration_s
    t = np.arange(n, dtype=np.float64) / sr

    # Warm ambient chord: C3 + E3 + G3 with slow amplitude modulation
    freqs = [130.81, 164.81, 196.00]  # C3, E3, G3
    signal = np.zeros(n)
    for i, f in enumerate(freqs):
        # Each note has a slightly different tremolo rate for richness
        tremolo = 0.5 + 0.5 * np.sin(2 * np.pi * (0.05 + i * 0.02) * t)
        signal += np.sin(2 * np.pi * f * t) * tremolo

    # Normalize and apply very low amplitude
    signal = signal / np.max(np.abs(signal)) * 0.02  # ~2% amplitude
    data = (signal * 32767).astype("<i2")

    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(data.tobytes())


def list_devices():
    """List ALSA playback devices."""
    subprocess.run(["aplay", "-l"], check=False)


def main():
    p = argparse.ArgumentParser(
        description="Audio keepalive — loop a sound file to prevent HDMI audio delay"
    )
    p.add_argument(
        "--file", "-f", type=Path, default=None,
        help="Audio file to loop (WAV, MP3, OGG, etc). Default: built-in ambient tone",
    )
    p.add_argument(
        "--volume", "-v", type=int, default=5,
        help="Volume percentage 1-100 (default: 5)",
    )
    p.add_argument(
        "--device", "-d", type=str, default=None,
        help="ALSA device (e.g. hw:0,3). Default: system default",
    )
    p.add_argument(
        "--list-devices", action="store_true",
        help="List available audio devices and exit",
    )
    args = p.parse_args()

    if args.list_devices:
        list_devices()
        return

    # Determine playback tool
    has_ffplay = shutil.which("ffplay") is not None
    has_mpv = shutil.which("mpv") is not None
    has_aplay = shutil.which("aplay") is not None

    # Determine audio file
    audio_file = args.file
    if audio_file is None:
        audio_file = Path("/tmp/dk1_keepalive_ambient.wav")
        if not audio_file.exists():
            print("Generating ambient tone...", end=" ", flush=True)
            generate_ambient_tone(audio_file)
            print("done")

    if not audio_file.exists():
        print(f"Error: file not found: {audio_file}", file=sys.stderr)
        sys.exit(1)

    suffix = audio_file.suffix.lower()
    vol = max(1, min(100, args.volume))

    # Build playback command
    if suffix == ".wav" and has_aplay:
        # aplay can only play WAV, no volume control — we pre-scale the file
        if args.file is None and vol != 100:
            # Re-generate with requested volume for the built-in tone
            scaled_path = Path(f"/tmp/dk1_keepalive_v{vol}.wav")
            if not scaled_path.exists():
                generate_ambient_tone_scaled(scaled_path, vol)
            audio_file = scaled_path
        device_args = ["-D", args.device] if args.device else []
        cmd = ["aplay", "-q"] + device_args + [str(audio_file)]
        player = "aplay"
    elif has_ffplay:
        cmd = [
            "ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet",
            "-volume", str(vol),
            str(audio_file),
        ]
        player = "ffplay"
    elif has_mpv:
        cmd = [
            "mpv", "--no-video", "--really-quiet",
            f"--volume={vol}",
            str(audio_file),
        ]
        player = "mpv"
    elif has_aplay and suffix == ".wav":
        device_args = ["-D", args.device] if args.device else []
        cmd = ["aplay", "-q"] + device_args + [str(audio_file)]
        player = "aplay"
    else:
        print("Error: no audio player found. Install ffplay, mpv, or use a .wav file with aplay.",
              file=sys.stderr)
        sys.exit(1)

    # Signal handling for clean exit
    stop = False
    def handler(sig, frame):
        nonlocal stop
        stop = True
    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)

    print(f"Audio keepalive: looping {audio_file.name} at {vol}% volume ({player})")
    print("Press Ctrl+C to stop\n")

    loop_count = 0
    while not stop:
        loop_count += 1
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            while proc.poll() is None and not stop:
                time.sleep(0.2)
            if stop:
                proc.terminate()
                proc.wait(timeout=2)
        except Exception as e:
            print(f"Playback error: {e}", file=sys.stderr)
            time.sleep(1)

    print("\nStopped.")


def generate_ambient_tone_scaled(path: Path, volume_pct: int, duration_s: int = 300):
    """Generate ambient tone with a specific volume level."""
    sr = 22050
    n = sr * duration_s
    t = np.arange(n, dtype=np.float64) / sr

    freqs = [130.81, 164.81, 196.00]
    sig = np.zeros(n)
    for i, f in enumerate(freqs):
        tremolo = 0.5 + 0.5 * np.sin(2 * np.pi * (0.05 + i * 0.02) * t)
        sig += np.sin(2 * np.pi * f * t) * tremolo

    amplitude = (volume_pct / 100.0) * 0.4  # scale from the base
    sig = sig / np.max(np.abs(sig)) * amplitude
    sig = np.clip(sig, -1.0, 1.0)
    data = (sig * 32767).astype("<i2")

    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(data.tobytes())


if __name__ == "__main__":
    main()
