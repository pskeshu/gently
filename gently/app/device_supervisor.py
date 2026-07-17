"""Managed child subprocess for the device layer.

Today gently only connects to ``start_device_layer.py`` as an HTTP client, so
there is no in-app way to bring hardware up or down — you run it yourself in a
second terminal. ``DeviceLayerSupervisor`` gives the gently process *ownership*
of that child: it spawns ``start_device_layer.py``, holds the handle, tails its
console output, reports liveness, and stops it (gracefully, then forcibly). That
ownership is what makes start/stop-from-UI and no-orphans-on-exit possible.

Design: ``docs/superpowers/specs/2026-07-02-unified-launcher-design.md`` (RFC #78).

Scope of this scaffold
----------------------
- Real, working spawn / stop / status / log-tail + atexit cleanup.
- **External device layer is respected:** if something is already listening on
  the port, we surface it as ``external`` and never try to manage (or stop) what
  we did not start.

Deliberately deferred (RFC "open questions"):
- Background/non-blocking startup with per-stage boot progress.
- Truly graceful shutdown on Windows — that needs ``start_device_layer.py`` to
  handle ``SIGBREAK`` (see ``stop`` below). Until then Windows falls back to a
  hard terminate after the grace period.
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path

from gently.settings import settings

logger = logging.getLogger(__name__)

# Repo root: gently/app/device_supervisor.py -> parents[2]
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEVICE_SCRIPT = _PROJECT_ROOT / "start_device_layer.py"

# How many console lines to retain for the log-tail shown in the Devices panel.
_LOG_TAIL_LINES = 500
# Grace period (seconds) for a clean shutdown before we hard-kill.
_STOP_GRACE_SECONDS = 5.0


class DeviceLayerSupervisor:
    """Own the ``start_device_layer.py`` child process for one gently instance.

    Thread-safe: ``start``/``stop`` take an internal lock so overlapping calls
    from async route handlers can't race on the process handle. A single reader
    thread drains the child's stdout into a bounded ring buffer for the tail.
    """

    def __init__(
        self,
        port: int = settings.network.device_port,
        host: str = settings.network.device_host,
        sam_device: str = "cuda",
        config_path: str = "config/config.yml",
    ):
        self.port = port
        self.host = host
        self.sam_device = sam_device
        self.config_path = config_path

        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self._log: deque[str] = deque(maxlen=_LOG_TAIL_LINES)
        self._reader: threading.Thread | None = None
        self._started_at: float | None = None
        # True once *we* asked the child to stop, so a resulting non-zero exit
        # reads as "stopped" rather than "crashed".
        self._stopping = False

        # Latest structured startup-progress / failure events parsed from the
        # child's stdout (@@GENTLY_PROGRESS@@ lines). Single dict refs, swapped
        # atomically by the reader thread and read without the lock (GIL).
        self._progress: dict | None = None
        self._failure: dict | None = None

        atexit.register(self._atexit_cleanup)

    # ── lifecycle ────────────────────────────────────────────────────────

    def start(self, *, sam_device: str | None = None, config_path: str | None = None) -> dict:
        """Spawn the device layer child if it isn't already up.

        Idempotent: if we already own a live child, returns its status. If an
        *external* (unmanaged) device layer is already listening on the port,
        refuses to start a duplicate and returns the external status.
        """
        with self._lock:
            if self._alive():
                assert self._proc is not None  # _alive() guarantees this
                logger.debug("Device layer already managed (pid=%s)", self._proc.pid)
                return self._status_locked()

            if self._port_open():
                # Something is already listening we didn't start — don't stomp it.
                logger.info(
                    "Device layer already running externally on %s:%s — not spawning a child",
                    self.host,
                    self.port,
                )
                return self._status_locked()

            # GENTLY_DEVICE_LAYER_SCRIPT overrides the launcher — lets an operator
            # point at an alternate device-layer entry point (and lets tests point
            # at a harmless stand-in instead of touching real hardware).
            script = Path(os.environ.get("GENTLY_DEVICE_LAYER_SCRIPT") or _DEVICE_SCRIPT)
            if not script.exists():
                raise FileNotFoundError(f"device layer script not found: {script}")

            if sam_device is not None:
                self.sam_device = sam_device
            if config_path is not None:
                self.config_path = config_path

            # -u = unbuffered stdout: CPython block-buffers stdout when it's a
            # pipe, which would stall the [N/5] progress + @@GENTLY_PROGRESS@@
            # lines in the child until the buffer fills. Unbuffered makes the
            # boot readout live.
            cmd = [
                sys.executable,
                "-u",
                str(script),
                "--port",
                str(self.port),
                "--sam-device",
                self.sam_device,
                "--config",
                self.config_path,
            ]

            # A new process group on Windows is what later lets us send a
            # CTRL_BREAK for a (best-effort) clean shutdown instead of only a
            # hard TerminateProcess. CREATE_NO_WINDOW keeps the device layer from
            # popping its own console window when gently is launched as a desktop
            # app — its output is still captured via the stdout pipe (below) and
            # its own device_layer_*.log.
            creationflags = 0
            if sys.platform == "win32":
                creationflags = (
                    subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
                    | subprocess.CREATE_NO_WINDOW  # type: ignore[attr-defined]
                )

            logger.info("Spawning device layer: %s", " ".join(cmd))
            self._log.clear()
            self._progress = None
            self._failure = None
            self._stopping = False
            self._proc = subprocess.Popen(
                cmd,
                cwd=str(_PROJECT_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                creationflags=creationflags,
            )
            self._started_at = time.monotonic()

            self._reader = threading.Thread(
                target=self._drain_output,
                args=(self._proc,),
                name="device-layer-log",
                daemon=True,
            )
            self._reader.start()

            return self._status_locked()

    def stop(self, *, force: bool = False, timeout: float = _STOP_GRACE_SECONDS) -> dict:
        """Stop the managed child. Graceful first, hard-kill fallback.

        We only stop a child *we* spawned — an external device layer is left
        alone (you can't cleanly stop what you don't own).

        ``force=True`` skips the grace period and hard-kills immediately.
        """
        with self._lock:
            if not self._alive():
                # Nothing of ours to stop (may still be external — status says so).
                return self._status_locked()

            self._stopping = True
            proc = self._proc
            assert proc is not None

            if force:
                self._hard_kill(proc)
                return self._status_locked()

            # Graceful request.
            try:
                if sys.platform == "win32":
                    # Best-effort clean stop. NOTE: start_device_layer.py handles
                    # SIGINT/SIGTERM but not SIGBREAK, so today this usually falls
                    # through to the hard-kill below. Teaching the device layer to
                    # handle SIGBREAK (RFC follow-up) makes this a clean shutdown.
                    proc.send_signal(signal.CTRL_BREAK_EVENT)  # type: ignore[attr-defined]
                else:
                    proc.terminate()  # SIGTERM -> device layer's clean shutdown path
            except (ProcessLookupError, OSError) as e:
                logger.debug("Signal to device layer failed (already exiting?): %s", e)

            try:
                proc.wait(timeout=timeout)
                logger.info("Device layer stopped cleanly (rc=%s)", proc.returncode)
            except subprocess.TimeoutExpired:
                logger.warning(
                    "Device layer didn't exit within %.1fs — forcing termination", timeout
                )
                self._hard_kill(proc)

            return self._status_locked()

    def restart(self, **kwargs) -> dict:
        """Stop (if running) then start — convenience for the Devices panel."""
        self.stop()
        return self.start(**kwargs)

    # ── introspection ────────────────────────────────────────────────────

    def status(self) -> dict:
        """Current state of the device layer (see ``_status_locked``)."""
        with self._lock:
            return self._status_locked()

    def log_tail(self, limit: int = 200) -> list[str]:
        """The most recent captured console lines (oldest → newest)."""
        lines = list(self._log)
        return lines[-limit:] if limit else lines

    # ── internals ────────────────────────────────────────────────────────

    def _status_locked(self) -> dict:
        """Compute status. Caller must hold ``_lock``.

        state (our managed child, in lifecycle order):
          starting     — spawned, no progress events yet, HTTP port not up
          initializing — [N/5] progress flowing, port not up yet (carries progress)
          ready        — child alive AND port open (the server binds only after
                         initialize() finishes, so an open port == init done)
          failed       — child exited non-zero and we captured a structured
                         failure reason (carries failure summary + hints)
          crashed      — child exited non-zero with no diagnosis (unexpected)
          external     — port listening but not our child
          stopped      — nothing running (never started, or cleanly stopped)
        """
        managed_alive = self._alive()
        port_open = self._port_open()
        progress = self._progress  # atomic ref read (no lock needed)
        failure = self._failure

        if managed_alive:
            if port_open:
                state = "ready"
            elif progress is not None:
                state = "initializing"
            else:
                state = "starting"
        elif (
            self._proc is not None and self._proc.returncode not in (None, 0) and not self._stopping
        ):
            state = "failed" if failure else "crashed"
        elif port_open:
            state = "external"
        else:
            state = "stopped"

        pid = self._proc.pid if (self._proc and managed_alive) else None
        rc = self._proc.returncode if self._proc else None
        uptime = (
            round(time.monotonic() - self._started_at, 1)
            if (managed_alive and self._started_at)
            else None
        )

        return {
            "state": state,
            "managed": managed_alive,
            "pid": pid,
            "returncode": rc,
            "host": self.host,
            "port": self.port,
            "port_open": port_open,
            "sam_device": self.sam_device,
            "uptime_seconds": uptime,
            "progress": progress,
            "failure": failure,
            "log_tail": self.log_tail(50),
        }

    def _alive(self) -> bool:
        """True if we own a child that is still running."""
        return self._proc is not None and self._proc.poll() is None

    def _port_open(self, timeout: float = 0.25) -> bool:
        """True if anything is accepting connections on the device port.

        Used to detect an externally-started device layer. ``host`` may be a
        bind address like ``0.0.0.0``; probe loopback for that case.
        """
        probe_host = "127.0.0.1" if self.host in ("0.0.0.0", "", None) else self.host
        try:
            with socket.create_connection((probe_host, self.port), timeout=timeout):
                return True
        except OSError:
            return False

    def _hard_kill(self, proc: subprocess.Popen) -> None:
        """Terminate a process forcibly and reap it. Caller holds ``_lock``."""
        try:
            proc.kill()
            proc.wait(timeout=3.0)
        except subprocess.TimeoutExpired:
            logger.error("Device layer pid=%s would not die after kill()", proc.pid)
        except (ProcessLookupError, OSError):
            pass

    # Sentinel prefix for machine-readable startup events on the child's stdout
    # (emitted by gently.hardware.console_ui.progress_event).
    _PROGRESS_PREFIX = "@@GENTLY_PROGRESS@@ "

    def _drain_output(self, proc: subprocess.Popen) -> None:
        """Reader-thread body: copy child stdout into the ring buffer, peeling
        off structured @@GENTLY_PROGRESS@@ events into _progress / _failure."""
        stream = proc.stdout
        if stream is None:
            return
        try:
            for raw in iter(stream.readline, ""):
                line = raw.rstrip("\n")
                if line.startswith(self._PROGRESS_PREFIX):
                    try:
                        ev = json.loads(line[len(self._PROGRESS_PREFIX) :])
                    except ValueError:
                        self._log.append(line)  # malformed — keep it visible
                        continue
                    # Atomic single-ref swaps; read lock-free in _status_locked.
                    if ev.get("status") == "failed":
                        self._failure = ev
                    else:
                        self._progress = ev
                    continue  # keep the sentinel out of the human-visible tail
                self._log.append(line)
        except (ValueError, OSError):
            # Stream closed underneath us during shutdown — fine.
            pass

    def _atexit_cleanup(self) -> None:
        """Kill an orphaned child when the gently process exits."""
        proc = self._proc
        if proc is not None and proc.poll() is None:
            logger.info("atexit: terminating managed device layer (pid=%s)", proc.pid)
            self._stopping = True
            try:
                proc.kill()
            except (ProcessLookupError, OSError):
                pass
