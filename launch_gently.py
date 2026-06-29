#!/usr/bin/env python3
"""
Launch the Microscopy Agent

Conversational AI agent for diSPIM microscope control.

Starts the agent + web visualization server, then opens the browser UI.
The web UI is the control surface (the legacy Ink TUI is retired — its
source is kept in the tree but no longer launched).

Usage:
    python launch_gently.py                      # Start server + open browser
    python launch_gently.py --no-browser         # Start server, don't open a browser
    python launch_gently.py --offline            # Run without the device layer
    python launch_gently.py --no-api             # UI-only: boot the web UI without an API key
    python launch_gently.py --sessions           # List sessions and exit
    python launch_gently.py --resume             # Resume most recent session
    python launch_gently.py --resume latest      # Resume most recent session
    python launch_gently.py --resume <id>        # Resume specific session
    python launch_gently.py -v                   # Verbose (INFO) logging
    python launch_gently.py --debug              # Debug logging
"""

import argparse
import asyncio
import logging
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml

# Load a project-root .env (if present) so ANTHROPIC_API_KEY and other
# settings can live in a file instead of being exported every session.
# Existing environment variables take precedence.
try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent / ".env")
except ImportError:
    pass

# The gently imports below pull in heavy dependencies (anthropic, torch, scipy,
# perception) and take several seconds. Print immediate feedback first so the
# terminal isn't silent during that load. Skipped for --help/--version.
if not any(flag in sys.argv for flag in ("-h", "--help")):
    print("Starting gently — loading modules (this can take a few seconds)...", flush=True)

from gently.app.agent import MicroscopyAgent
from gently.core.file_store import FileStore
from gently.core.log_bridge import configure_log_bridge
from gently.hardware import get_hardware, load_hardware
from gently.log_config import configure_logging
from gently.organisms import load_organism
from gently.settings import settings

logger = logging.getLogger(__name__)


def _format_elapsed(last_active: str) -> str:
    """Format an ISO timestamp into a human-readable relative time."""
    if not last_active:
        return ""
    try:
        dt = datetime.fromisoformat(last_active)
        elapsed = (datetime.now() - dt).total_seconds()
        if elapsed < 60:
            return "just now"
        elif elapsed < 3600:
            return f"{int(elapsed / 60)}m ago"
        elif elapsed < 86400:
            return f"{int(elapsed / 3600)}h ago"
        else:
            return f"{int(elapsed / 86400)}d ago"
    except Exception:
        return ""


def _build_session_items(store: FileStore) -> list:
    """Build a list of session dicts for the Ink picker."""
    sessions = store.list_sessions()
    items = []
    for session in sessions:
        session_id = session.get("session_id", "unknown")
        embryos = store.list_embryos(session_id)
        embryo_count = len(embryos) if embryos else 0
        items.append(
            {
                "session_id": session_id,
                "embryo_count": embryo_count,
                "time": _format_elapsed(session.get("last_active", "")),
            }
        )
    return items


def list_sessions(store: FileStore):
    """List all available sessions (plain text, no Rich)."""
    items = _build_session_items(store)
    if not items:
        print("\nNo saved sessions found.")
        return

    print("\nAvailable Sessions")
    print("-" * 50)
    for item in items:
        time_str = f"  ({item['time']})" if item["time"] else ""
        print(f"  {item['session_id']}  {item['embryo_count']} embryos{time_str}")
    print()
    print("Use: python launch_gently.py --resume <id>")


def _print_banner(viz_url, device_connected, offline, storage_dir, log_file, resumed, no_api=False):
    """Print a human-readable launch banner to the terminal.

    This is the "what you see when you open it" surface now that the
    server (not a TUI) is the long-running process.
    """
    line = "─" * 56
    if offline:
        dev = "○ offline (--offline)"
    elif device_connected:
        dev = "● connected"
    else:
        dev = "○ offline — run:  python start_device_layer.py"
    agent_status = "○ disabled — UI only (--no-api)" if no_api else "● enabled"
    url = viz_url or "(viz server failed to start — check the log)"
    tag = "  [resumed session]" if resumed else ""
    print()
    print(f"  ✦ Gently is running.{tag}")
    print(f"    {line}")
    print(f"    Open:    {url}")
    print(f"    Agent:   {agent_status}")
    print(f"    Device:  {dev}")
    print(f"    Storage: {storage_dir}")
    print(f"    Logs:    {log_file}")
    print("    Stop:    Ctrl-C")
    print(f"    {line}")
    print()


def _open_browser(url: str) -> None:
    """Open the web UI, preferring Google Chrome.

    Override with GENTLY_BROWSER (a webbrowser name like 'firefox', or a full
    path to a browser executable). Falls back to the OS default browser if
    Chrome can't be found, so this never blocks startup.
    """
    import webbrowser

    override = os.environ.get("GENTLY_BROWSER", "").strip()

    # 1) Registered browser names (override first, then Chrome aliases).
    for name in ([override] if override else []) + [
        "chrome",
        "google-chrome",
        "chromium",
    ]:
        try:
            webbrowser.get(name).open(url)
            return
        except Exception:
            pass

    # 2) Explicit executables (an override path, then known Chrome locations).
    candidates = [override] if override else []
    candidates += [
        shutil.which("chrome"),
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    ]
    for exe in candidates:
        try:
            if exe and Path(exe).exists():
                webbrowser.register(
                    "gently-browser",
                    None,
                    webbrowser.BackgroundBrowser(exe),
                    preferred=True,
                )
                webbrowser.get("gently-browser").open(url)
                return
        except Exception:
            pass

    # 3) Fall back to the OS default.
    try:
        webbrowser.open(url)
    except Exception:
        pass


def run_ink_picker(tui_dist: Path, sessions_json: str) -> str | None:
    """
    Spawn the Ink TUI in session-picker mode and capture the selection.

    Retired: kept for reference / potential reuse by a future web session
    picker. No longer called by the launcher.

    Returns the selected session ID, or None for a new session.
    """
    proc = subprocess.run(
        ["node", str(tui_dist), "--pick-session", sessions_json],
        stdin=sys.stdin,
        stdout=subprocess.PIPE,
        stderr=sys.stderr,
        text=True,
    )

    # Clear the screen so the picker output doesn't linger
    # when the main TUI takes over.
    os.system("cls" if os.name == "nt" else "clear")

    # Parse the SESSION:<id> protocol line from stdout
    for line in (proc.stdout or "").splitlines():
        if line.startswith("SESSION:"):
            selected = line[len("SESSION:") :].strip()
            return selected if selected else None

    return None


async def main(
    offline: bool = False,
    resume_session: str | None = None,
    show_sessions: bool = False,
    pick_session: bool = False,
    log_level: str = "WARNING",
    no_browser: bool = False,
    no_api: bool = False,
    no_auth: bool = False,
):
    # Set up log file in storage directory
    # Unified with FileStore: logs live under the same root as data
    # (settings.storage.base_path reads GENTLY_STORAGE_PATH). Previously this
    # read a separate GENTLY_STORAGE env var, so setting only one split logs
    # from data.
    storage_base = settings.storage.base_path
    log_dir = storage_base / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = str(log_dir / f"gently_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    # File always gets INFO+, console uses the requested level
    configure_logging(level=log_level, log_file=log_file)
    # Mirror gently / gently_perception log lines onto the EventBus so the
    # Events page in the viz server shows them too. Env vars control level
    # and whether to include noisy third-party loggers (off by default).
    configure_log_bridge()
    logger.info("Logging to %s (console level: %s)", log_file, log_level)

    # Load organism module from config
    config_path = Path(__file__).parent / "config" / "config.yml"
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f) or {}
    else:
        config = {}
    load_organism(config.get("organism", "celegans"))
    load_hardware(config.get("hardware", "dispim"))

    # Storage directory (unified with FileStore)
    storage_dir = settings.storage.base_path
    storage_dir.mkdir(exist_ok=True)

    # Create unified store (FileStore) early for session queries
    from gently.core.gently_manifest import write_manifest

    write_manifest(storage_dir)
    store = FileStore(storage_dir)

    # ── Accounts / auth ───────────────────────────────────────────
    # Self-managed user accounts gate microscope control on the LAN. On first
    # run we bootstrap an admin and print its one-time password in the banner.
    # Pass --no-auth (or set GENTLY_NO_AUTH=1) to disable accounts (localhost-control mode).
    admin_creds = None
    auth_disabled = no_auth or os.environ.get("GENTLY_NO_AUTH", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    if auth_disabled:
        logger.warning("Accounts disabled (--no-auth) — control is open on this host")
    if not auth_disabled:
        try:
            from gently.ui.web.accounts import AccountStore, set_account_store

            account_store = AccountStore(storage_dir / "auth")
            set_account_store(account_store)
            admin_creds = account_store.bootstrap_admin_if_empty()
        except Exception as e:
            logger.error("Account store init failed (continuing without auth): %s", e)

    # Handle --sessions (just list and exit)
    if show_sessions:
        list_sessions(store)
        store.close()
        return

    # Web-only: the TUI is retired. The browser is the control surface and
    # the launcher just starts the server — no Node/dist requirement.

    # Handle --resume. Interactive session picking has moved to the browser;
    # without an explicit ID ("latest" or bare --resume) we resume the most
    # recent session.
    session_to_resume = None
    if pick_session or resume_session == "latest":
        sessions = store.list_sessions()
        if sessions:
            session_to_resume = sessions[0].get("session_id")
            if pick_session:
                print(
                    f"Resuming most recent session: {session_to_resume} "
                    "(interactive session picking is moving into the browser)"
                )
        else:
            print("No sessions found - starting fresh")
    elif resume_session:
        session_to_resume = resume_session

    # Create log directory
    log_dir = storage_dir / "logs"
    log_dir.mkdir(exist_ok=True)

    # Log file path for launch info
    session_name = datetime.now().strftime("%Y%m%d")
    log_file = log_dir / f"{session_name}.log"

    # Connect to device layer via hardware module's client factory
    client = None
    if not offline:
        hw = get_hardware()
        http_url = f"http://{settings.network.device_host}:{settings.network.device_port}"
        if hasattr(hw, "create_client"):
            client = hw.create_client(http_url=http_url)
        else:
            # Fallback for hardware modules without create_client
            from gently.app.queue_server_client import QueueServerClient

            client = QueueServerClient(http_url=http_url)
        connected = await client.connect()
        if not connected:
            # Keep client object (not None) so microscope tools remain in
            # Claude's tool schema.  Tools check is_connected at runtime and
            # return clear error messages. Setting client = None causes all
            # requires_microscope tools to vanish from the schema, which
            # makes Claude hallucinate XML tool calls as plain text.
            logger.debug(
                "Device layer not reachable at %s — microscope tools "
                "available but will return errors until connected",
                http_url,
            )

    # Configure device session for zero-copy volume transfer
    if client and client.is_connected:
        try:
            incoming = str(store.incoming_dir)
            await client.configure_device_session(incoming)
            logger.info("Device session configured: volume_dir=%s", incoming)
        except Exception as e:
            logger.error("Failed to configure device session (volumes will be slow): %s", e)

    # Register auto-generated microscope tools from device layer plan schemas
    if client and client.is_connected:
        try:
            from gently.harness.microscope import register_microscope_tools

            n = register_microscope_tools(client)
            if n:
                logger.info("Registered %d microscope tools from device layer", n)
        except Exception as e:
            logger.debug("Auto-tool registration skipped: %s", e)

    # Create agent
    agent = MicroscopyAgent(
        microscope_client=client,
        storage_path=storage_dir,
        session_id=session_to_resume,
        store=store,
        no_api=no_api,
    )

    # Generate TLS certificate for mesh communication
    cert_path, key_path = None, None
    try:
        from gently.mesh.tls import ensure_tls_cert, get_cert_fingerprint

        _config_dir = Path(__file__).parent / "config"
        cert_path, key_path = ensure_tls_cert(_config_dir)
    except Exception:
        pass

    # Start visualization server for real-time feedback (plain HTTP —
    # self-signed certs trigger browser "unsafe" warnings for visitors).
    await agent.start_viz_server(port=settings.network.viz_port)
    scheme = "http"
    viz_url = (
        f"{scheme}://localhost:{settings.network.viz_port}"
        if agent.viz_server is not None
        else None
    )

    # ── Mesh discovery ──────────────────────────────────────────────
    mesh = None
    try:
        import uuid as _uuid

        from gently.mesh import MeshService, register_mesh_routes
        from gently.mesh.audit import MeshAuditLog
        from gently.mesh.pairing import PairingManager

        # Persistent instance ID
        instance_id_path = Path(__file__).parent / "config" / "mesh_instance_id"
        if instance_id_path.exists():
            instance_id = instance_id_path.read_text().strip()
        else:
            instance_id = str(_uuid.uuid4())
            instance_id_path.parent.mkdir(parents=True, exist_ok=True)
            instance_id_path.write_text(instance_id)

        def _capability_provider():
            caps = {
                "has_microscope": client.is_connected if client else False,
                "has_sam": client.has_sam if client else False,
                "has_gpu": False,
                "gpu_name": "",
                "gpu_vram_gb": 0.0,
                "storage_free_gb": 0.0,
                "organism": config.get("organism", "celegans"),
                "hardware_profile": config.get("hardware", "dispim"),
                "tool_categories": [],
            }
            # GPU detection — try torch first, fall back to nvidia-smi
            try:
                import torch

                if torch.cuda.is_available():
                    caps["has_gpu"] = True
                    caps["gpu_name"] = torch.cuda.get_device_name(0)
                    caps["gpu_vram_gb"] = round(
                        torch.cuda.get_device_properties(0).total_memory / (1024**3), 1
                    )
            except ImportError:
                try:
                    import subprocess as _sp

                    out = _sp.check_output(
                        [
                            "nvidia-smi",
                            "--query-gpu=name,memory.total",
                            "--format=csv,noheader,nounits",
                        ],
                        timeout=5,
                        text=True,
                    ).strip()
                    if out:
                        parts = out.split(",", 1)
                        caps["has_gpu"] = True
                        caps["gpu_name"] = parts[0].strip()
                        if len(parts) > 1:
                            caps["gpu_vram_gb"] = round(float(parts[1].strip()) / 1024, 1)
                except Exception:
                    pass
            # Free disk space
            try:
                usage = shutil.disk_usage(str(storage_dir))
                caps["storage_free_gb"] = round(usage.free / (1024**3), 1)
            except OSError:
                pass
            return caps

        def _status_provider():
            import gently as _gently

            return {
                "session_id": agent.session_id or "",
                "acquisition_status": "idle",
                "embryo_count": len(agent.experiment.embryos),
                "total_timepoints": 0,
                "uptime_seconds": 0.0,
                "agent_mode": agent.mode,
                "active_plan": "",
                "version": getattr(_gently, "__version__", "dev"),
            }

        import socket as _socket

        config_dir = Path(__file__).parent / "config"
        audit_log = MeshAuditLog(config_dir)
        pairing_mgr = PairingManager(
            instance_id=instance_id,
            hostname=_socket.gethostname(),
            config_dir=config_dir,
            audit_log=audit_log,
        )

        # Set TLS cert fingerprint on pairing manager
        if cert_path:
            try:
                pairing_mgr.cert_fingerprint = get_cert_fingerprint(cert_path)
            except Exception:
                pass

        mesh = MeshService(
            instance_id=instance_id,
            viz_port=settings.network.viz_port,
            capability_provider=_capability_provider,
            status_provider=_status_provider,
            pairing_manager=pairing_mgr,
            audit_log=audit_log,
        )

        if agent.viz_server is not None:
            register_mesh_routes(agent.viz_server, mesh, audit_log=audit_log)
            agent.viz_server.mesh_service = mesh

        await mesh.start()
    except Exception as e:
        import logging as _log

        _log.getLogger(__name__).warning(f"Mesh discovery failed to start: {e}")
        mesh = None
    # ── End mesh ────────────────────────────────────────────────────

    # Attach the agent bridge to the viz server
    from gently.harness.bridge import AgentBridge

    bridge = AgentBridge(agent)

    bridge.set_launch_info(
        {
            "device_connected": client.is_connected if client else False,
            "sam_available": client.has_sam if client else False,
            "offline": offline or (client is None) or not client.is_connected,
            "store_path": str(storage_dir),
            "viz_url": viz_url,
            "log_path": str(log_file),
            "resumed": session_to_resume is not None,
            "mesh_service": mesh,
        }
    )

    # Initialize startup wizard (gap-driven onboarding)
    from gently.harness.memory.file_store import FileContextStore

    agent_dir = storage_dir / "agent"
    context_store = FileContextStore(agent_dir)
    agent.set_context_store(context_store)
    bridge.init_wizard(context_store=context_store, claude_client=agent.claude)

    if agent.viz_server is not None:
        agent.viz_server.agent_bridge = bridge
        agent.viz_server.set_context_store(context_store)
        # If launched into an existing session, rehydrate its persisted
        # imagery so the galleries/filmstrips show data from the start.
        if session_to_resume:
            try:
                agent.viz_server.rehydrate_session(session_to_resume)
            except Exception:
                logger.debug("Startup rehydrate failed", exc_info=True)

    # ── Banner + serve ──────────────────────────────────────────────
    # The viz server runs in-process (uvicorn in a background task). With
    # the TUI retired, the launcher's job is to keep that server alive and
    # point the operator at the browser.
    _print_banner(
        viz_url=viz_url,
        device_connected=bool(client and client.is_connected),
        offline=offline,
        storage_dir=storage_dir,
        log_file=log_file,
        resumed=session_to_resume is not None,
        no_api=no_api,
    )

    if admin_creds:
        _u, _p = admin_creds
        print("  First-run admin account created — sign in at the URL above:")
        print(f"      username: {_u}")
        print(f"      password: {_p}")
        print("  (Save this now. Add users via the admin API; GENTLY_NO_AUTH=1 disables auth.)\n")

    if viz_url and not no_browser:
        _open_browser(viz_url)

    # Keep the event loop alive so the in-process viz server keeps serving.
    # On Windows the Proactor loop won't surface Ctrl-C while blocked on a
    # bare Event().wait(), so install signal handlers and poll on a short
    # interval (which also lets a pending KeyboardInterrupt surface).
    import signal as _signal

    _loop = asyncio.get_running_loop()
    _stop = asyncio.Event()
    try:
        _loop.add_signal_handler(_signal.SIGINT, _stop.set)
        _loop.add_signal_handler(_signal.SIGTERM, _stop.set)
    except (NotImplementedError, AttributeError, RuntimeError, ValueError):
        # Windows Proactor: add_signal_handler is unsupported — fall back to
        # signal.signal, waking the loop via call_soon_threadsafe.
        def _sig(*_a):
            _loop.call_soon_threadsafe(_stop.set)

        try:
            _signal.signal(_signal.SIGINT, _sig)
            _signal.signal(_signal.SIGTERM, _sig)
        except (ValueError, OSError):
            pass

    try:
        while not _stop.is_set():
            await asyncio.sleep(0.3)
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        # Suppress noisy CancelledError / overlapped IO errors from
        # uvicorn during shutdown on Windows.
        import logging as _logging

        _logging.getLogger("uvicorn.error").setLevel(_logging.CRITICAL)
        _logging.getLogger("uvicorn").setLevel(_logging.CRITICAL)
        # Cleanup: stop mesh service
        if mesh is not None:
            try:
                await mesh.stop()
            except (asyncio.CancelledError, RuntimeError, OSError, Exception):
                pass
        # Cleanup: stop viz server gracefully
        if agent.viz_server is not None:
            try:
                await agent.viz_server.stop()
            except (asyncio.CancelledError, RuntimeError, OSError, Exception):
                pass


def cli_main():
    """Sync entry point for ``gently`` console script (pyproject.toml)."""
    parser = argparse.ArgumentParser(description="Launch Microscopy Agent")
    parser.add_argument("--offline", action="store_true", help="Run without server connections")
    parser.add_argument(
        "--no-api",
        action="store_true",
        help="UI-only mode: boot the web UI without any Anthropic API key. "
        "Chat, perception, and plan generation are disabled.",
    )
    parser.add_argument("--sessions", action="store_true", help="List available sessions and exit")
    parser.add_argument(
        "--resume",
        nargs="?",
        const="__PICK__",
        metavar="ID",
        help="Resume a session. Without ID: shows picker. With ID: resumes that session.",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose (INFO) logging"
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging (most verbose)")
    parser.add_argument(
        "--no-auth",
        action="store_true",
        help="Disable accounts/login (localhost-control mode; same as GENTLY_NO_AUTH=1)",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not auto-open the web UI in a browser",
    )
    args = parser.parse_args()

    # An API key is required unless running in UI-only mode.
    if not args.no_api and not os.getenv("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY not set")
        if os.name == "nt":
            print("Set with: set ANTHROPIC_API_KEY=your-key")
        else:
            print("Set with: export ANTHROPIC_API_KEY=your-key")
        print("Or add it to a .env file in the project root: ANTHROPIC_API_KEY=your-key")
        print("Or run UI-only without a key: python launch_gently.py --no-api")
        exit(1)

    log_level = "WARNING"
    if args.verbose:
        log_level = "INFO"
    if args.debug:
        log_level = "DEBUG"

    pick_session = args.resume == "__PICK__"
    resume_id = args.resume if args.resume and args.resume != "__PICK__" else None

    try:
        asyncio.run(
            main(
                offline=args.offline,
                show_sessions=args.sessions,
                resume_session=resume_id,
                pick_session=pick_session,
                log_level=log_level,
                no_browser=args.no_browser,
                no_api=args.no_api,
                no_auth=args.no_auth,
            )
        )
    except (KeyboardInterrupt, RuntimeError, SystemExit):
        pass


if __name__ == "__main__":
    cli_main()
