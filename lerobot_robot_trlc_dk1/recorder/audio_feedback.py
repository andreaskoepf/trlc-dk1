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

"""Audio feedback — pre-generated beep sounds + TTS for state transitions.

Beep WAV files are generated once at init time and played via ``aplay``
(non-blocking subprocess). TTS uses ``spd-say`` (espeak) by default,
with optional ``piper-tts`` for higher quality.

To keep HDMI audio active (avoiding spin-up delay on monitors like ASUS),
run the separate ``audio_keepalive.py`` script in another terminal before
starting the recorder.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import wave
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class AudioFeedback:
    """Non-blocking audio feedback for recording state transitions.

    Args:
        enabled: Master enable/disable for all audio.
        tts_engine: TTS backend — "espeak" (default), "piper", or "none".
    """

    def __init__(self, enabled: bool = True, tts_engine: str = "espeak"):
        self.enabled = enabled
        self.tts_engine = tts_engine
        self._beep_files: dict[str, Path] = {}

        # Check if aplay is available
        self._has_aplay = shutil.which("aplay") is not None
        if not self._has_aplay:
            logger.warning("aplay not found, beep sounds disabled")

        if enabled and self._has_aplay:
            self._generate_beeps()

    def _generate_beeps(self):
        """Pre-generate beep WAV files in /tmp for instant playback."""
        beep_defs = [
            ("start", 600, 150),
            ("episode_end", 800, 200),
            ("gesture", 1200, 100),
            ("error", 400, 500),
            ("done", 1000, 300),
        ]
        for name, freq, dur_ms in beep_defs:
            path = Path(f"/tmp/dk1_beep_{name}.wav")
            n_samples = int(22050 * dur_ms / 1000)
            t = np.arange(n_samples) / 22050.0
            samples = np.sin(2 * np.pi * freq * t)
            # Apply fade-in/out to avoid clicks (5ms)
            fade = min(int(22050 * 0.005), n_samples // 2)
            if fade > 0:
                samples[:fade] *= np.linspace(0, 1, fade)
                samples[-fade:] *= np.linspace(1, 0, fade)
            data = (samples * 32767).astype("<i2")
            try:
                with wave.open(str(path), "w") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(22050)
                    wf.writeframes(data.tobytes())
                self._beep_files[name] = path
            except OSError:
                logger.warning("Failed to create beep file: %s", path)

    def stop(self):
        """Cleanup (no-op now, kept for API compatibility)."""
        pass

    def _play_beep(self, name: str):
        """Play a pre-generated beep (non-blocking)."""
        if not self.enabled or not self._has_aplay:
            return
        path = self._beep_files.get(name)
        if path is None or not path.exists():
            return
        try:
            subprocess.Popen(
                ["aplay", "-q", str(path)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except OSError:
            pass

    def _speak(self, text: str):
        """Speak text via TTS (non-blocking)."""
        if not self.enabled or self.tts_engine == "none":
            return
        try:
            if self.tts_engine == "piper" and shutil.which("piper") is not None:
                subprocess.Popen(
                    f'echo "{text}" | piper --model en_US-lessac-medium '
                    f"--output-raw | aplay -r 22050 -f S16_LE -c 1 -q",
                    shell=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            elif shutil.which("spd-say") is not None:
                subprocess.Popen(
                    ["spd-say", text],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
        except OSError:
            pass

    # -- Public API ---------------------------------------------------------

    def episode_start(self, episode_num: int):
        self._play_beep("start")
        self._speak(f"Recording episode {episode_num}")

    def episode_end(self, episode_num: int):
        self._play_beep("episode_end")
        self._speak(f"Episode {episode_num} saved")

    def gesture_detected(self):
        self._play_beep("gesture")

    def error(self, message: str):
        self._play_beep("error")
        self._speak(message)

    def episode_discarded(self, episode_num: int):
        self._play_beep("error")
        self._speak(f"Episode {episode_num} discarded")

    def recording_done(self):
        self._play_beep("done")
        self._speak("Recording complete.")
