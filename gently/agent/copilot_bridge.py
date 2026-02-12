"""
Copilot Bridge - WebSocket adapter for MicroscopyCopilot

Wraps the async-generator-based handle_message_stream() into a
future-based API suitable for WebSocket communication. Converts
the asend() bidirectional protocol into send_fn / choice_future
pairs so the WebSocket route can drive the conversation.
"""

import asyncio
import json
import logging
import time
from typing import Any, Callable, Coroutine, Dict, Optional

from .command_registry import get_command_registry, CommandCategory

logger = logging.getLogger(__name__)


class CopilotBridge:
    """
    Thin adapter between MicroscopyCopilot and a WebSocket transport.

    Converts ``handle_message_stream()`` (an async generator that uses
    ``asend()`` for choice responses) into a callback-driven API:

        await bridge.stream_response(
            message,
            send_fn=ws.send_json,
            choice_future_factory=make_future,
        )

    Parameters
    ----------
    copilot : MicroscopyCopilot
        The copilot instance to wrap.
    """

    def __init__(self, copilot):
        self.copilot = copilot
        self._active_stream: Optional[asyncio.Task] = None
        self._launch_info: Dict[str, Any] = {}
        self._wizard = None  # StartupWizard, set by init_wizard()

    def set_launch_info(self, info: Dict[str, Any]) -> None:
        """Store launch metadata to include in the connect message."""
        self._launch_info = info

    def init_wizard(self, context_store, claude_client=None) -> None:
        """Create the startup wizard from a ContextStore."""
        from gently.context.startup_wizard import StartupWizard
        self._context_store = context_store
        self._wizard = StartupWizard(
            context_store=context_store,
            session_id=self.copilot.session_id,
            claude_client=claude_client,
        )

    async def stream_response(
        self,
        message: str,
        send_fn: Callable[[Dict], Coroutine],
        choice_future_factory: Callable[[Dict], "asyncio.Future[str]"],
    ) -> None:
        """
        Stream a copilot response over WebSocket.

        Parameters
        ----------
        message : str
            User chat message.
        send_fn : async callable
            Called with each chunk dict to send to the client.
        choice_future_factory : callable
            Called with choice_data dict; must return a Future that
            resolves to the user's selected option ID string.
        """
        stream_iter = self.copilot.handle_message_stream(message).__aiter__()
        pending_choice_result = None

        try:
            while True:
                try:
                    if pending_choice_result is not None:
                        chunk = await stream_iter.asend(pending_choice_result)
                        pending_choice_result = None
                    else:
                        chunk = await stream_iter.__anext__()
                except StopAsyncIteration:
                    # Stream finished — send token usage summary
                    await send_fn({
                        "type": "stream_end",
                        "tokens": self._get_token_snapshot(),
                    })
                    return

                chunk_type = chunk.get("type")

                if chunk_type == "choice_request":
                    choice_data = chunk.get("choice_data", {})
                    # Create future first — factory writes request_id into choice_data
                    future = choice_future_factory(choice_data)
                    # Attach request_id to the outer chunk for the client
                    chunk["request_id"] = choice_data.get("request_id", "")
                    # Send the choice request to the client
                    await send_fn(chunk)
                    # Wait for the client to respond
                    pending_choice_result = await future
                else:
                    # Forward text, tool_start, tool_call chunks directly
                    await send_fn(chunk)

        except asyncio.CancelledError:
            logger.info("Stream cancelled")
            raise
        except Exception as e:
            logger.error(f"Stream error: {e}", exc_info=True)
            await send_fn({"type": "error", "error": str(e)})

    async def handle_command(
        self,
        command: str,
        send_fn: Callable[[Dict], Coroutine],
    ) -> None:
        """
        Execute a slash command and send the result.

        Delegates to the copilot's command handling. For commands that
        need Rich rendering, we return structured data instead.

        Parameters
        ----------
        command : str
            The slash command string (e.g., "/status").
        send_fn : async callable
            Called with the result dict.
        """
        registry = get_command_registry()
        cmd_name = command.strip().lower().split()[0]
        cmd_def = registry.get(cmd_name)

        if not cmd_def:
            await send_fn({
                "type": "command_result",
                "command": command,
                "error": f"Unknown command: {cmd_name}",
            })
            return

        # Handle commands that return structured data
        cmd = command.strip().lower()

        if cmd in ("/quit", "/exit", "/q"):
            await send_fn({
                "type": "command_result",
                "command": cmd,
                "action": "quit",
            })
            return

        if cmd == "/status":
            status = self._get_status_data()
            await send_fn({
                "type": "command_result",
                "command": "/status",
                "content": status,
            })
            return

        if cmd == "/embryos" or cmd.startswith("/embryos "):
            parts = cmd.split(maxsplit=1)
            embryo_id = parts[1].strip() if len(parts) > 1 else None
            data = self._get_embryos_data(embryo_id)
            await send_fn({
                "type": "command_result",
                "command": "/embryos",
                "content": data,
            })
            return

        if cmd == "/tokens":
            data = self._get_tokens_data()
            await send_fn({
                "type": "command_result",
                "command": "/tokens",
                "content": data,
            })
            return

        if cmd == "/help" or cmd.startswith("/help "):
            parts = command.strip().split(maxsplit=1)
            help_cmd = parts[1] if len(parts) > 1 else None
            if help_cmd:
                text = registry.generate_command_help(help_cmd)
                if not text:
                    text = f"Unknown command: {help_cmd}"
            else:
                text = registry.generate_help_markdown()
            await send_fn({
                "type": "command_result",
                "command": "/help",
                "content": {"text": text},
            })
            return

        if cmd.startswith("/theme"):
            parts = cmd.split()
            if len(parts) > 1:
                from .theme import set_theme, get_theme
                try:
                    set_theme(parts[1])
                    theme = get_theme()
                    await send_fn({
                        "type": "command_result",
                        "command": "/theme",
                        "content": {"theme": theme.name, "changed": True},
                    })
                except ValueError as e:
                    await send_fn({
                        "type": "command_result",
                        "command": "/theme",
                        "error": str(e),
                    })
            else:
                from .theme import list_themes, get_theme
                current = get_theme()
                themes = {k: v.name for k, v in list_themes().items()}
                await send_fn({
                    "type": "command_result",
                    "command": "/theme",
                    "content": {"themes": themes, "current": current.name},
                })
            return

        if cmd == "/sessions":
            sessions = []
            if hasattr(self.copilot, 'store') and self.copilot.store:
                raw = self.copilot.store.list_sessions()
                for s in raw:
                    sid = s.get("session_id", "unknown")
                    embryos = self.copilot.store.list_embryos(sid)
                    sessions.append({
                        "session_id": sid,
                        "name": s.get("name", ""),
                        "embryo_count": len(embryos) if embryos else 0,
                        "last_active": s.get("last_active", ""),
                    })
            await send_fn({
                "type": "command_result",
                "command": "/sessions",
                "content": {"sessions": sessions},
            })
            return

        if cmd == "/timelapse" or cmd == "/timelapse watch":
            data = self._get_timelapse_data()
            await send_fn({
                "type": "command_result",
                "command": "/timelapse",
                "content": data,
            })
            return

        if cmd.startswith("/timeline"):
            parts = command.strip().split()
            data = self._get_timeline_data(parts[1:] if len(parts) > 1 else [])
            await send_fn({
                "type": "command_result",
                "command": "/timeline",
                "content": data,
            })
            return

        if cmd == "/detectors":
            data = self._get_detectors_data()
            await send_fn({
                "type": "command_result",
                "command": "/detectors",
                "content": data,
            })
            return

        if cmd == "/history":
            data = self._get_history_data()
            await send_fn({
                "type": "command_result",
                "command": "/history",
                "content": data,
            })
            return

        if cmd == "/save":
            success = self.copilot.save_session()
            if success:
                await send_fn({
                    "type": "command_result",
                    "command": "/save",
                    "content": {"text": f"Session saved: {self.copilot.session_id}"},
                })
            else:
                await send_fn({
                    "type": "command_result",
                    "command": "/save",
                    "error": "Failed to save session",
                })
            return

        if cmd == "/reset-context":
            cs = getattr(self, "_context_store", None)
            if cs is None:
                await send_fn({
                    "type": "command_result",
                    "command": "/reset-context",
                    "error": "Context store not available",
                })
            else:
                counts = cs.reset()
                total = sum(counts.values())
                # Re-create the wizard so it re-assesses gaps
                claude_client = self._wizard.claude_client if self._wizard else None
                self.init_wizard(cs, claude_client)
                if total > 0:
                    details = ", ".join(f"{v} {k}" for k, v in counts.items())
                    msg = f"Context cleared: {total} entries removed ({details}).\nRun /wizard to set up again."
                else:
                    msg = "Context already empty — nothing to clear."
                await send_fn({
                    "type": "command_result",
                    "command": "/reset-context",
                    "content": {"text": msg},
                })
            return

        if cmd == "/wizard":
            # Handled by the WebSocket route (copilot_ws.py), not the bridge.
            # If we reach here, it means the wizard loop called handle_command
            # — i.e. /wizard was typed while the wizard is already running.
            await send_fn({
                "type": "command_result",
                "command": "/wizard",
                "content": {"text": "The wizard is already running."},
            })
            return

        if cmd == "/clear":
            await send_fn({
                "type": "command_result",
                "command": "/clear",
                "action": "clear",
            })
            return

        if cmd.startswith("/resume"):
            parts = command.strip().split(maxsplit=1)
            if len(parts) > 1:
                session_id = parts[1].strip()
                success = self.copilot.resume_session(session_id)
                if success:
                    embryo_count = len(self.copilot.experiment.embryos)
                    msg_count = len(self.copilot.conversation_history)
                    await send_fn({
                        "type": "command_result",
                        "command": "/resume",
                        "content": {
                            "text": f"Session resumed: {session_id}\n  {embryo_count} embryos, {msg_count} messages",
                        },
                    })
                else:
                    await send_fn({
                        "type": "command_result",
                        "command": "/resume",
                        "error": f"Session '{session_id}' not found",
                    })
            else:
                # No session ID — list available sessions for the user
                sessions = self._get_sessions_list()
                if sessions:
                    lines = ["Available sessions (use /resume <id>):"]
                    for s in sessions:
                        lines.append(
                            f"  {s['session_id']} — {s['embryo_count']} embryos"
                        )
                    await send_fn({
                        "type": "command_result",
                        "command": "/resume",
                        "content": {"text": "\n".join(lines)},
                    })
                else:
                    await send_fn({
                        "type": "command_result",
                        "command": "/resume",
                        "content": {"text": "No saved sessions found."},
                    })
            return

        if cmd.startswith("/import-embryos"):
            parts = command.strip().split(maxsplit=1)
            if len(parts) > 1:
                arg = parts[1].strip()
                # Handle 'last' shortcut
                if arg.lower() == "last":
                    sessions = self._get_sessions_list()
                    sessions_with = [s for s in sessions if s["embryo_count"] > 0]
                    if not sessions_with:
                        await send_fn({
                            "type": "command_result",
                            "command": "/import-embryos",
                            "error": "No sessions with embryos found.",
                        })
                        return
                    session_id = sessions_with[0]["session_id"]
                else:
                    session_id = arg

                result = self.copilot.import_embryos_from_session(session_id)
                if result.get("success"):
                    imported = result.get("imported", [])
                    skipped = result.get("skipped", [])
                    lines = [f"Imported {len(imported)} embryo(s) from {session_id}"]
                    if imported:
                        lines.append(f"  {', '.join(imported)}")
                    if skipped:
                        lines.append(f"  Skipped (exist): {', '.join(skipped)}")
                    await send_fn({
                        "type": "command_result",
                        "command": "/import-embryos",
                        "content": {"text": "\n".join(lines)},
                    })
                else:
                    await send_fn({
                        "type": "command_result",
                        "command": "/import-embryos",
                        "error": result.get("error", "Import failed"),
                    })
            else:
                sessions = self._get_sessions_list()
                sessions_with = [s for s in sessions if s["embryo_count"] > 0]
                if sessions_with:
                    lines = ["Sessions with embryos (use /import-embryos <id>):"]
                    for s in sessions_with[:10]:
                        lines.append(
                            f"  {s['session_id']} — {s['embryo_count']} embryos"
                        )
                    await send_fn({
                        "type": "command_result",
                        "command": "/import-embryos",
                        "content": {"text": "\n".join(lines)},
                    })
                else:
                    await send_fn({
                        "type": "command_result",
                        "command": "/import-embryos",
                        "content": {"text": "No sessions with embryos found."},
                    })
            return

        if cmd.startswith("/make-video"):
            parts = command.strip().split()
            embryo_id = None
            fps = 10
            i = 1
            while i < len(parts):
                if parts[i] == "--fps" and i + 1 < len(parts):
                    try:
                        fps = int(parts[i + 1])
                    except ValueError:
                        pass
                    i += 2
                elif not parts[i].startswith("--"):
                    embryo_id = parts[i]
                    i += 1
                else:
                    i += 1

            session_id = self.copilot.session_id
            if not session_id:
                await send_fn({
                    "type": "command_result",
                    "command": "/make-video",
                    "error": "No active session",
                })
                return

            try:
                from .video_maker import discover_volumes, create_timelapse_video
                storage_path = self.copilot.storage_path
                session_images_dir = storage_path / "images" / session_id

                if not session_images_dir.exists():
                    await send_fn({
                        "type": "command_result",
                        "command": "/make-video",
                        "error": f"No images found for session {session_id}",
                    })
                    return

                all_volumes = discover_volumes(session_images_dir, embryo_id)
                if not all_volumes:
                    await send_fn({
                        "type": "command_result",
                        "command": "/make-video",
                        "content": {"text": "No timelapse volumes found."},
                    })
                    return

                lines = [f"Creating timelapse videos (fps={fps})..."]
                for eid, vol_paths in all_volumes.items():
                    output_path = session_images_dir / f"{eid}_timelapse.mp4"
                    create_timelapse_video(vol_paths, str(output_path), fps=fps)
                    lines.append(f"  {eid}: {len(vol_paths)} frames → {output_path.name}")

                await send_fn({
                    "type": "command_result",
                    "command": "/make-video",
                    "content": {"text": "\n".join(lines)},
                })
            except ImportError:
                await send_fn({
                    "type": "command_result",
                    "command": "/make-video",
                    "error": "Video maker module not available.",
                })
            except Exception as e:
                await send_fn({
                    "type": "command_result",
                    "command": "/make-video",
                    "error": str(e),
                })
            return

        if cmd.startswith("/benchmark"):
            parts = command.strip().split()
            n_volumes = 5
            n_slices = 50
            n_warmup = 1
            i = 1
            while i < len(parts):
                p = parts[i]
                if p in ("--volumes", "-n") and i + 1 < len(parts):
                    try:
                        n_volumes = int(parts[i + 1])
                    except ValueError:
                        pass
                    i += 2
                elif p in ("--slices", "-s") and i + 1 < len(parts):
                    try:
                        n_slices = int(parts[i + 1])
                    except ValueError:
                        pass
                    i += 2
                elif p in ("--warmup", "-w") and i + 1 < len(parts):
                    try:
                        n_warmup = int(parts[i + 1])
                    except ValueError:
                        pass
                    i += 2
                else:
                    i += 1

            try:
                from .benchmark import run_benchmark
                await send_fn({
                    "type": "command_result",
                    "command": "/benchmark",
                    "content": {"text": f"Running benchmark ({n_volumes} volumes, {n_slices} slices, {n_warmup} warmup)..."},
                })
                results = await run_benchmark(
                    self.copilot,
                    n_volumes=n_volumes,
                    n_slices=n_slices,
                    n_warmup=n_warmup,
                )
                lines = [
                    "Benchmark Results",
                    f"  Volumes: {results.n_volumes}",
                    f"  Mean acquisition: {results.mean_acquisition:.3f}s",
                    f"  Mean storage: {results.mean_storage:.3f}s",
                    f"  Mean total: {results.mean_total:.3f}s",
                    f"  FPS: {results.fps:.1f}",
                ]
                await send_fn({
                    "type": "command_result",
                    "command": "/benchmark",
                    "content": {"text": "\n".join(lines)},
                })
            except ImportError:
                await send_fn({
                    "type": "command_result",
                    "command": "/benchmark",
                    "error": "Benchmark module not available.",
                })
            except Exception as e:
                await send_fn({
                    "type": "command_result",
                    "command": "/benchmark",
                    "error": str(e),
                })
            return

        # Fallback for truly unimplemented commands
        await send_fn({
            "type": "command_result",
            "command": cmd,
            "content": {"text": f"Command `{cmd}` is not yet available in the TUI."},
        })

    def get_commands_json(self) -> list:
        """Serialize the command registry for the TUI client."""
        registry = get_command_registry()
        commands = []
        for cmd in registry.get_all():
            commands.append({
                "name": cmd.name,
                "description": cmd.description,
                "aliases": cmd.aliases,
                "category": cmd.category.name,
                "usage": cmd.usage_string(),
                "arg_hint": cmd.arg_hint_string(),
                "subcommands": [
                    {"name": s.name, "description": s.description}
                    for s in cmd.subcommands
                ],
            })
        return commands

    # ------------------------------------------------------------------
    # Private helpers for structured command data
    # ------------------------------------------------------------------

    def _get_status_data(self) -> dict:
        """Build structured status data."""
        exp = self.copilot.experiment
        client = self.copilot.client

        return {
            "session_id": self.copilot.session_id,
            "connected": client.is_connected if client else False,
            "embryo_count": len(exp.embryos),
            "embryo_ids": list(exp.embryos.keys()),
            "has_sam": client.has_sam if client else False,
        }

    def _get_embryos_data(self, embryo_id: str = None) -> dict:
        """Build structured embryo data."""
        exp = self.copilot.experiment
        if embryo_id:
            embryo = exp.get_embryo_by_any_name(embryo_id)
            if embryo:
                return {
                    "embryo": {
                        "id": embryo_id,
                        "nickname": embryo.nickname,
                        "user_label": embryo.user_label,
                        "stage_position": embryo.stage_position,
                    }
                }
            return {"error": f"Embryo '{embryo_id}' not found"}

        embryos = []
        for eid, emb in exp.embryos.items():
            embryos.append({
                "id": eid,
                "nickname": emb.nickname,
                "user_label": emb.user_label,
            })
        return {"embryos": embryos}

    def _get_tokens_data(self) -> dict:
        """Build structured token usage data."""
        return self._get_token_snapshot()

    def _get_token_snapshot(self) -> dict:
        """Current token usage from the copilot."""
        c = self.copilot
        input_t = getattr(c, "total_input_tokens", 0)
        output_t = getattr(c, "total_output_tokens", 0)
        api_calls = getattr(c, "api_call_count", 0)
        return {
            "input_tokens": input_t,
            "output_tokens": output_t,
            "total_tokens": input_t + output_t,
            "api_calls": api_calls,
        }

    def get_connect_metadata(self) -> dict:
        """Metadata sent to the TUI on connect."""
        import gently
        exp = self.copilot.experiment
        meta = {
            "session_id": self.copilot.session_id,
            "commands": self.get_commands_json(),
            "version": getattr(gently, "__version__", "dev"),
            "tokens": self._get_token_snapshot(),
            "embryo_count": len(exp.embryos),
            # Launch info fields (set by launch_copilot.py for TUI mode)
            "device_connected": self._launch_info.get("device_connected", False),
            "sam_available": self._launch_info.get("sam_available", False),
            "offline": self._launch_info.get("offline", False),
            "store_path": self._launch_info.get("store_path", ""),
            "viz_url": self._launch_info.get("viz_url", None),
            "log_path": self._launch_info.get("log_path", ""),
            "resumed": self._launch_info.get("resumed", False),
        }
        # Wizard metadata (if initialized)
        if self._wizard is not None:
            meta["wizard"] = self._wizard.gap_summary
        return meta

    def _get_sessions_list(self) -> list:
        """Return a list of saved sessions with metadata."""
        sessions = []
        if hasattr(self.copilot, "store") and self.copilot.store:
            raw = self.copilot.store.list_sessions()
            for s in raw:
                sid = s.get("session_id", "unknown")
                embryos = self.copilot.store.list_embryos(sid)
                sessions.append({
                    "session_id": sid,
                    "name": s.get("name", ""),
                    "embryo_count": len(embryos) if embryos else 0,
                    "last_active": s.get("last_active", ""),
                })
        return sessions

    def _get_timelapse_data(self) -> dict:
        """Build structured timelapse status."""
        orch = getattr(self.copilot, "timelapse_orchestrator", None)
        if not orch:
            return {"text": "No timelapse running."}

        state = orch.get_status()
        return state.to_dict()

    def _get_timeline_data(self, args: list) -> dict:
        """Build structured timeline data."""
        tm = getattr(self.copilot, "timeline_manager", None)
        if not tm:
            return {"text": "Timeline not available."}

        # Parse filters
        event_filter = None
        embryo_filter = None
        show_all = False
        i = 0
        while i < len(args):
            arg = args[i].lower() if isinstance(args[i], str) else ""
            if arg == "--filter" and i + 1 < len(args):
                event_filter = args[i + 1].lower()
                i += 2
            elif arg == "--embryo" and i + 1 < len(args):
                embryo_filter = args[i + 1]
                i += 2
            elif arg == "--all":
                show_all = True
                i += 1
            else:
                i += 1

        events = tm.get_events(
            event_type=event_filter,
            embryo_id=embryo_filter,
            session_id="all" if show_all else "current",
            limit=50,
        )

        if not events:
            return {"text": "No timeline events found."}

        event_list = [e.to_dict() for e in events]
        # Also render a text summary for display
        lines = []
        for e in events:
            ts = e.timestamp.strftime("%H:%M:%S")
            emb = f" [{e.embryo_id}]" if e.embryo_id else ""
            det = f" ({e.detector_name})" if e.detector_name else ""
            tp = f" tp{e.timepoint}" if e.timepoint is not None else ""
            lines.append(f"  {e.icon} {ts}{emb}{tp}{det} {e.event_subtype}")

        return {
            "events": event_list,
            "text": "Timeline\n" + "\n".join(lines),
        }

    def _get_detectors_data(self) -> dict:
        """Build structured detector/perception data."""
        pm = getattr(self.copilot, "perception_manager", None)
        if not pm or not pm.sessions:
            return {"text": "No active perception sessions."}

        lines = ["Perception Sessions"]
        for embryo_id, session in pm.sessions.items():
            stage = session.get_current_stage() or "unknown"
            obs_count = len(session.observations)
            lines.append(f"  {embryo_id}: stage={stage}, {obs_count} observations")

        return {"text": "\n".join(lines)}

    def _get_history_data(self) -> dict:
        """Build structured conversation history."""
        history = self.copilot.conversation_history[-20:]  # Last 20 messages
        if not history:
            return {"text": "No conversation history."}

        lines = ["Recent Conversation"]
        for msg in history:
            role = msg.get("role", "?")
            content = msg.get("content", "")
            if isinstance(content, list):
                # Extract text blocks
                text_parts = []
                for block in content:
                    if hasattr(block, "text"):
                        text_parts.append(block.text)
                    elif isinstance(block, dict) and block.get("text"):
                        text_parts.append(block["text"])
                content = " ".join(text_parts)
            # Truncate long messages
            if len(content) > 120:
                content = content[:117] + "..."
            prefix = "You" if role == "user" else "Copilot"
            lines.append(f"  [{prefix}] {content}")

        return {"text": "\n".join(lines)}
