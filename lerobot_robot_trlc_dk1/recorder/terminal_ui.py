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

"""Terminal UI — pinned status line + keyboard input in cbreak mode.

Uses cbreak mode (not raw mode) so that Ctrl-C still sends SIGINT.
Renders a pinned status line at the bottom of the terminal. Key events
are posted to a thread-safe queue for the main thread to consume.
"""

from __future__ import annotations

import atexit
import logging
import queue
import select
import sys
import termios
import threading
import tty

logger = logging.getLogger(__name__)


class StatusLineLogHandler(logging.Handler):
    """Log handler that clears the pinned status line before printing.

    Uses rich Console for colored output. Clears the current terminal line
    before each log message so it doesn't overlap the pinned status bar.
    """

    def __init__(self):
        super().__init__()
        self._console = None
        self._rich_handler = None
        try:
            from io import StringIO
            from rich.console import Console
            from rich.logging import RichHandler

            # RichHandler with a captured console so we can prepend \r\033[K
            self._console_buf = StringIO()
            self._console = Console(
                file=self._console_buf, width=120, no_color=False, force_terminal=True,
            )
            self._rich_handler = RichHandler(
                console=self._console,
                show_path=False,
                rich_tracebacks=True,
            )
        except ImportError:
            pass

    def emit(self, record):
        try:
            if self._rich_handler and self._console:
                # Render via rich, then extract the string
                self._console_buf.truncate(0)
                self._console_buf.seek(0)
                self._rich_handler.emit(record)
                self._console_buf.seek(0)
                rendered = self._console_buf.read()
                if rendered:
                    # Clear status line, print rendered log, status redraws next tick
                    sys.stderr.write(f"\r\033[K{rendered}")
                    sys.stderr.flush()
            else:
                msg = self.format(record)
                sys.stderr.write(f"\r\033[K{msg}\n")
                sys.stderr.flush()
        except Exception:
            self.handleError(record)


class TerminalUI:
    """Terminal-based UI with pinned status line and keyboard input.

    Attributes updated by the main thread:
        state: Current recorder state string.
        episode: Current episode index.
        fps_actual: Actual recording FPS.
        teleop_hz: Actual teleop rate.
        frame_count: Frames recorded in current episode.
        encoder_drops: Total dropped frames across all encoders.

    Key events are posted to ``key_queue``:
        "space" — toggle recording / episode boundary
        "rerecord" — discard and re-record current episode
        "quit" — stop recording and exit
    """

    def __init__(self):
        self.state: str = "idle"
        self.episode: int = 0
        self.fps_actual: float = 0.0
        self.teleop_hz: float = 0.0
        self.frame_count: int = 0
        self.encoder_drops: int = 0
        self.countdown: int = 0  # countdown number (3, 2, 1, 0=GO)
        # Hottest raw temperature byte across the 6 arm joints for each arm
        # (max of T_MOS and T_ROTOR). 0 = not yet reported.
        self.t_max_left: int = 0
        self.t_max_right: int = 0

        self.key_queue: queue.Queue[str] = queue.Queue()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._old_settings = None
        # Persistent copy of terminal settings for prompt_text (never cleared)
        self._saved_termios = None
        try:
            if sys.stdin.isatty():
                self._saved_termios = termios.tcgetattr(sys.stdin)
        except termios.error:
            pass

    def start(self):
        """Start the UI thread (sets terminal to cbreak mode)."""
        if not sys.stdin.isatty():
            logger.warning("stdin is not a TTY, terminal UI disabled")
            return
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="terminal-ui"
        )
        self._thread.start()

    def stop(self):
        """Stop the UI thread and restore terminal settings."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def prompt_text(self, prompt: str, default: str = "") -> str | None:
        """Temporarily restore terminal for line input. Returns text or None if cancelled.

        Thread-safe: pauses the UI thread's rendering and keyboard polling,
        restores normal terminal mode, reads a line, then re-enters cbreak.
        """
        if self._saved_termios is None:
            return None

        # Pause the UI render loop
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

        # Restore normal terminal mode for input()
        try:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._saved_termios)
        except (termios.error, ValueError, OSError):
            return None

        # Clear status line and show prompt
        sys.stdout.write(f"\r\033[K  {prompt}")
        if default:
            sys.stdout.write(f" [{default}]")
        sys.stdout.write(": ")
        sys.stdout.flush()

        try:
            text = input().strip()
        except (EOFError, KeyboardInterrupt):
            text = None

        # Restart UI thread (re-enters cbreak mode)
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="terminal-ui"
        )
        self._thread.start()

        if text is None or text == "":
            return default if default else None
        return text

    def _restore_terminal(self):
        """Restore terminal settings. Safe to call multiple times."""
        if self._old_settings is not None:
            try:
                termios.tcsetattr(
                    sys.stdin, termios.TCSADRAIN, self._old_settings
                )
            except (termios.error, ValueError, OSError):
                pass
            self._old_settings = None

    def _run(self):
        try:
            self._old_settings = termios.tcgetattr(sys.stdin)
        except termios.error:
            logger.warning("Cannot get terminal attributes, UI disabled")
            return

        # Register atexit handler so terminal is restored even if
        # the process is killed (SIGTERM) or crashes without cleanup.
        atexit.register(self._restore_terminal)

        try:
            tty.setcbreak(sys.stdin.fileno())  # cbreak preserves Ctrl-C
            while not self._stop_event.is_set():
                self._render_status()
                self._poll_keyboard(timeout=0.1)
        except Exception:
            logger.exception("Terminal UI error")
        finally:
            self._restore_terminal()
            # Clear the status line
            try:
                sys.stdout.write("\r" + " " * 80 + "\r")
                sys.stdout.flush()
            except (OSError, ValueError):
                pass

    # ANSI color codes
    _RESET = "\033[0m"
    _BOLD = "\033[1m"
    _RED = "\033[91m"
    _GREEN = "\033[92m"
    _YELLOW = "\033[93m"
    _DIM = "\033[2m"

    # Motor-temperature color thresholds (°C). The DAMIAO firmware lets you
    # configure the protective over-temperature trip (OT_Value) anywhere in
    # [80, 200) °C, and 80 °C is the floor they allow — so anything ≥70 °C is
    # within 10 °C of the minimum manufacturer cutoff.
    _TEMP_WARN_C = 50
    _TEMP_HOT_C = 70

    _STATE_COLORS = {
        "idle": _DIM,
        "starting": _YELLOW + _BOLD,
        "countdown": _YELLOW + _BOLD,
        "recording": _GREEN + _BOLD,
        "saving": _YELLOW,
        "waiting": _YELLOW,
    }

    _STATE_INDICATORS = {
        "idle": "  ",       # dim dot
        "starting": "* ",   # blinking-ish
        "countdown": "▸ ",  # countdown arrow
        "recording": "● ",  # solid green dot
        "saving": "~ ",
        "waiting": "○ ",    # hollow dot
    }

    def _temp_color(self, t: int) -> str:
        if t >= self._TEMP_HOT_C:
            return self._RED + self._BOLD
        if t >= self._TEMP_WARN_C:
            return self._YELLOW
        return ""

    def _render_status(self):
        """Render colored pinned status line."""
        drop_str = f" {self._RED}drops:{self.encoder_drops}{self._RESET}" if self.encoder_drops > 0 else ""

        if self.fps_actual > 0 and self.state == "recording":
            elapsed_s = self.frame_count / max(self.fps_actual, 1)
            time_str = f"{int(elapsed_s // 60):02d}:{int(elapsed_s % 60):02d}"
        else:
            time_str = "--:--"

        color = self._STATE_COLORS.get(self.state, "")
        indicator = self._STATE_INDICATORS.get(self.state, "  ")

        if self.state == "countdown":
            count_str = f"{self.countdown}..." if self.countdown > 0 else "GO!"
            line = (
                f"  {color}{indicator}{count_str:9s}{self._RESET} | "
                f"Ep {self.episode} | Bksp=cancel"
            )
        else:
            temp_str = ""
            if self.t_max_left > 0 or self.t_max_right > 0:
                # Raw uint8 bytes from MIT-mode reply (data[6]=T_MOS, data[7]=T_ROTOR).
                # Per the DAMIAO DM-J4310 manual these are °C; the firmware faults
                # on B (MOS) or C (coil) overtemp once OT_Value is exceeded.
                lc = self._temp_color(self.t_max_left)
                rc = self._temp_color(self.t_max_right)
                temp_str = (
                    f" | L: {lc}{self.t_max_left}°C{self._RESET} "
                    f"R: {rc}{self.t_max_right}°C{self._RESET}"
                )
            line = (
                f"  {color}{indicator}{self.state:9s}{self._RESET} | "
                f"Ep {self.episode} | "
                f"rec:{self.fps_actual:4.0f}Hz teleop:{self.teleop_hz:4.0f}Hz | "
                f"{time_str}{drop_str}{temp_str}"
            )

        sys.stdout.write(f"\r{line}\033[K")
        sys.stdout.flush()

    def _poll_keyboard(self, timeout: float):
        """Non-blocking stdin read via select (cbreak mode)."""
        try:
            rlist, _, _ = select.select([sys.stdin], [], [], timeout)
        except (ValueError, OSError):
            return

        if not rlist:
            return

        try:
            ch = sys.stdin.read(1)
        except (EOFError, OSError):
            return

        if ch == " ":
            self.key_queue.put("space")
        elif ch.lower() == "r" or ch == "\x7f":  # R or Backspace
            self.key_queue.put("rerecord")
        elif ch.lower() == "q":
            self.key_queue.put("quit")
        elif ch.lower() == "t":
            self.key_queue.put("task")
        # Ignore ESC and escape sequences (cursor keys etc.)
