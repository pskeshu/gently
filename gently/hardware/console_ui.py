"""Lightweight terminal styling for the device-layer console.

Plain ``print`` to stdout, no third-party dependency. ``rich`` is deliberately
avoided here — it has caused Unicode/encoding issues on Windows consoles (see
the stdout-suppression note in ``dispim/device_layer.py``).

The point of this module is to give the operator a readable, always-visible
picture of the device layer at the terminal — distinct from the file log. The
file log keeps the full INFO/DEBUG firehose; the console shows a curated set of
milestones and a status panel.

Robust by construction, because the device layer runs on Windows consoles:

* **Colour** (ANSI) is auto-disabled unless stdout is a TTY. On Windows we try
  to enable virtual-terminal processing first; if that fails, colour is off so
  raw escape codes never leak. ``NO_COLOR`` (https://no-color.org) and a
  ``dumb`` ``TERM`` also disable it.
* **Box-drawing** glyphs are used only when stdout's encoding is UTF-based;
  otherwise ASCII equivalents are used so a cp1252 console shows clean output.
* ``out()`` is defensive: any residual ``UnicodeEncodeError`` is caught and the
  line re-emitted with ``errors="replace"`` rather than crashing startup.
"""

from __future__ import annotations

import json
import os
import sys

# Visible width of the status panel (border rules). Content lines are written
# without a right border so coloured text never needs width arithmetic.
WIDTH = 64


def _enable_windows_vt() -> bool:
    """Best-effort: turn on ANSI escape handling for the Windows console.

    Returns True if VT processing is (now) enabled or we're not on Windows.
    """
    if sys.platform != "win32":
        return True
    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.windll.kernel32
        ENABLE_VT = 0x0004
        handle = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
        mode = wintypes.DWORD()
        if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            return False
        return bool(kernel32.SetConsoleMode(handle, mode.value | ENABLE_VT))
    except Exception:
        return False


def _detect_color() -> bool:
    if sys.stdout is None or not hasattr(sys.stdout, "isatty") or not sys.stdout.isatty():
        return False
    if os.environ.get("NO_COLOR") is not None or os.environ.get("TERM") == "dumb":
        return False
    return _enable_windows_vt()


def _detect_unicode() -> bool:
    enc = (getattr(sys.stdout, "encoding", None) or "").lower()
    return "utf" in enc


_USE_COLOR = _detect_color()
_USE_UNICODE = _detect_unicode()

# Glyphs: pretty (UTF) vs ASCII fallback.
if _USE_UNICODE:
    _HEAVY, _LIGHT, _DOT, _CHECK, _MID, _BULLET = "═", "─", "●", "✓", "·", "•"
else:
    _HEAVY, _LIGHT, _DOT, _CHECK, _MID, _BULLET = "=", "-", "*", "+", "-", "-"

# Public separator for callers that build their own value strings.
MIDDOT = f" {_MID} "

_CODES = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "dim": "\033[2m",
    "green": "\033[32m",
    "cyan": "\033[36m",
    "yellow": "\033[33m",
    "red": "\033[31m",
    "blue": "\033[34m",
    "magenta": "\033[35m",
    "grey": "\033[90m",
}


def supports_color() -> bool:
    return _USE_COLOR


def c(text, *styles: str) -> str:
    """Wrap *text* in ANSI styles, or return it unchanged when colour is off."""
    if not _USE_COLOR or not styles:
        return str(text)
    prefix = "".join(_CODES.get(s, "") for s in styles)
    return f"{prefix}{text}{_CODES['reset']}"


def out(text: str = "") -> None:
    """Print one line to stdout, flushing so it shows immediately.

    Never raises on encoding: a console that can't represent a character gets
    a replacement rather than a crashed startup.
    """
    try:
        print(text, flush=True)
    except UnicodeEncodeError:
        enc = getattr(sys.stdout, "encoding", None) or "ascii"
        sys.stdout.write(text.encode(enc, "replace").decode(enc, "replace") + "\n")
        sys.stdout.flush()


def rule(heavy: bool = True, style: str = "grey") -> None:
    out(c((_HEAVY if heavy else _LIGHT) * WIDTH, style))


def header(title: str, badge: str | None = None, badge_style: str = "yellow") -> None:
    """Top of a panel: a heavy rule, a title row (optional right-aligned badge),
    and a closing heavy rule."""
    rule(heavy=True)
    line = "  " + c(title, "bold", "cyan")
    if badge:
        # Right-align using uncoloured widths so padding ignores ANSI codes.
        pad = max(1, WIDTH - len("  " + title) - len(badge) - 1)
        line += " " * pad + c(badge, "bold", badge_style)
    out(line)
    rule(heavy=True)


def row(label: str, value: str, label_w: int = 12, label_style: str = "grey") -> None:
    """A ``  label   value`` line inside a panel."""
    out(f"  {c(label.ljust(label_w), label_style)}{value}")


def sub(label: str, value: str, label_w: int = 10) -> None:
    """An indented sub-row, e.g. a device-group breakdown."""
    out(f"    {c(label.ljust(label_w), 'grey')}{value}")


_last_step: tuple[int, int, str] | None = None


def progress_event(**payload) -> None:
    """Emit one machine-readable startup-progress event on stdout.

    The DeviceLayerSupervisor drains the device layer's stdout pipe — the only
    channel that crosses the process boundary before the device layer's HTTP
    port opens — and parses these ``@@GENTLY_PROGRESS@@`` lines into a per-stage
    progress readout for the UI. No-op on an interactive terminal (isatty), so a
    human running the device layer directly never sees the sentinel noise; it
    only fires when stdout is captured (piped) by the supervisor.
    """
    try:
        if sys.stdout.isatty():
            return
    except (AttributeError, ValueError):
        return
    payload.setdefault("v", 1)
    try:
        print("@@GENTLY_PROGRESS@@ " + json.dumps(payload), flush=True)
    except Exception:
        pass


def step(n: int, total: int, label: str) -> None:
    """A startup progress line: ``  [2/5] Starting Micro-Manager core``"""
    global _last_step
    _last_step = (n, total, label)
    out(f"  {c(f'[{n}/{total}]', 'cyan')} {label}")
    progress_event(i=n, n=total, status="start", label=label)


def step_done(detail: str = "ok") -> None:
    """A check-mark continuation under the most recent step."""
    out(f"        {c(_CHECK, 'green')} {c(detail, 'grey')}")
    if _last_step is not None:
        i, total, label = _last_step
        progress_event(i=i, n=total, status="ok", label=label, detail=detail)


def note(text: str, style: str = "grey") -> None:
    out(f"  {c(text, style)}")


def bullet(text: str) -> None:
    out(f"    {c(_BULLET, 'cyan')} {text}")


def error_panel(
    title: str, summary: str, details: str | None = None, hints=None, log_file=None
) -> None:
    """A red FAILED panel: one-line summary, optional detail, fix hints, log path.

    Used at the top-level startup catch so an operator sees a plain-language
    diagnosis instead of a Python traceback (which still goes to the log file).
    """
    out()
    header(title, badge="FAILED", badge_style="red")
    note(summary, "yellow")
    if details:
        out()
        row("Details", details, label_w=10)
    if hints:
        out()
        note("Try this:", "bold")
        for h in hints:
            bullet(h)
    if log_file:
        out()
        row("Full log", str(log_file), label_w=10)
    rule(heavy=True)
    out()
