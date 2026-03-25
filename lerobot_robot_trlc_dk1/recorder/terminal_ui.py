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

        self.key_queue: queue.Queue[str] = queue.Queue()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._old_settings = None

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
                f"Ep {self.episode} | Esc/Bksp=cancel"
            )
        else:
            line = (
                f"  {color}{indicator}{self.state:9s}{self._RESET} | "
                f"Ep {self.episode} | "
                f"rec:{self.fps_actual:4.0f}Hz teleop:{self.teleop_hz:4.0f}Hz | "
                f"{time_str}{drop_str}"
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
        elif ch.lower() == "q" or ch == "\x1b":
            self.key_queue.put("quit")
