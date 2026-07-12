"""
Agent Bridge - WebSocket adapter for MicroscopyAgent

Wraps the async-generator-based handle_message_stream() into a
future-based API suitable for WebSocket communication. Converts
the asend() bidirectional protocol into send_fn / choice_future
pairs so the WebSocket route can drive the conversation.
"""

import asyncio
import logging
from collections.abc import Callable, Coroutine
from typing import Any

from .commands import get_command_registry

logger = logging.getLogger(__name__)


class AgentBridge:
    """
    Thin adapter between MicroscopyAgent and a WebSocket transport.

    Converts ``handle_message_stream()`` (an async generator that uses
    ``asend()`` for choice responses) into a callback-driven API:

        await bridge.stream_response(
            message,
            send_fn=ws.send_json,
            choice_future_factory=make_future,
        )

    Parameters
    ----------
    agent : MicroscopyAgent
        The agent instance to wrap.
    """

    def __init__(self, agent):
        self.agent = agent
        self._launch_info: dict[str, Any] = {}
        self._wizard = None  # StartupWizard, set by init_wizard()
        self._active_remote: dict[str, Any] | None = None  # {"peer": PeerInfo, "campaign_id": str}
        self._pending_import: dict | None = None  # For /import-embryos picker
        # Set by the web layer (register_display_broadcaster) so AGENT-INITIATED
        # turns (the wake-router) can stream to all chat clients + the transcript.
        self._display_broadcaster: Callable | None = None

    def register_display_broadcaster(
        self, broadcast_fn, choice_factory=None, choice_discard=None
    ) -> None:
        """Register the web layer's broadcast fn for autonomous (wake) turns.

        The wake-router has no per-connection send_fn, so to make autonomous
        turns visible we route their chunks through the same _broadcast the web
        route uses for user turns (records to the display transcript AND fans out
        to every connected chat client). Also wires the agent's dangling
        on_message_callback to this path, and (for ASK mode) the choice-future
        factory so an autonomous turn can round-trip an approval picker.
        Idempotent — last registration wins; the registered fns are router-scoped
        and fan out to whoever is connected.
        """
        self._display_broadcaster = broadcast_fn
        try:
            self.agent.on_message_callback = self.broadcast_autonomous_chunk
            if choice_factory is not None:
                self.agent._wake_choice_factory = choice_factory
            if choice_discard is not None:
                self.agent._wake_choice_discard = choice_discard
        except Exception:
            pass

    async def broadcast_autonomous_chunk(self, chunk) -> None:
        """Fan one autonomous-turn chunk to all chat clients + the transcript.

        No-op when no web client has registered a broadcaster (headless run) —
        the wake turn still executes and is persisted to the conversation/log.
        """
        fn = self._display_broadcaster
        if fn is None:
            return
        try:
            res = fn(chunk)
            if asyncio.iscoroutine(res):
                await res
        except Exception:
            logger.debug("broadcast_autonomous_chunk failed", exc_info=True)

    async def handle_choice_response(self, request_id: str, selected: str, send_fn) -> bool:
        """Handle a choice response that may belong to a bridge-initiated picker.

        Returns True if this response was consumed, False if not ours.
        """
        if self._pending_import and self._pending_import.get("request_id") == request_id:
            self._pending_import = None
            if selected:
                result = self.agent.import_embryos_from_session(selected)
                await self._send_import_result(send_fn, result, selected[:8])
            return True
        return False

    def set_launch_info(self, info: dict[str, Any]) -> None:
        """Store launch metadata to include in the connect message."""
        self._launch_info = info

    def _require_context_store(self):
        """Return the context store or None if unavailable."""
        return getattr(self, "_context_store", None)

    def _require_mesh(self):
        """Return the mesh service or None if unavailable."""
        return self._launch_info.get("mesh_service")

    def get_session_briefing(self) -> str:
        """Generate briefing for new sessions.

        Resolves plan context first: if there's exactly one unblocked
        imaging item, auto-sets it as the active plan item on both
        AgentMemory and ExperimentState before generating the briefing.

        When resolution mode will fire (see should_enter_resolution),
        returns an empty string — the agent's first turn becomes the
        opener instead of static text.
        """
        if not hasattr(self.agent, "memory") or not self.agent.memory:
            return ""

        memory = self.agent.memory

        # Restore active_plan_item_id from experiment state (session resume)
        experiment = getattr(self.agent, "experiment", None)
        if experiment and experiment.active_plan_item_id:
            memory.active_plan_item_id = experiment.active_plan_item_id
        elif experiment:
            # Resolve plan context from the dependency graph
            active_id, candidates = memory.resolve_plan_context()
            if active_id:
                experiment.active_plan_item_id = active_id
                memory.active_plan_item_id = active_id
                logger.info(f"Auto-set active plan item: {active_id}")

                # Link session to the campaign
                cs = getattr(self.agent, "context_store", None)
                if cs and self.agent.session_id:
                    try:
                        item = cs.get_plan_item(active_id)
                        if item:
                            cs.link_session_campaign(
                                self.agent.session_id,
                                item.campaign_id,
                            )
                    except Exception:
                        pass
            elif self.should_enter_resolution():
                # Resolution mode will take over the opener.
                return ""

        # Invalidate prompt cache so the system prompt picks up the
        # active plan item on the next message
        prompts = getattr(self.agent, "prompts", None)
        if prompts and memory.active_plan_item_id:
            prompts.invalidate_context_cache()

        return memory.get_session_briefing()

    def should_enter_resolution(self) -> bool:
        """True when the agent should open the session in resolution mode.

        Cheap O(1) check: new session (no ``active_plan_item_id`` yet) and
        agent memory is available. The picker itself decides what to do
        based on the user's top-level pick — candidate enumeration is
        lazy-loaded only if they pick "Continue a planned task". This
        replaces the older O(memory-scan) ``resolve_plan_context()`` gate
        that was making the picker take seconds to appear at startup.
        """
        if not hasattr(self.agent, "memory") or not self.agent.memory:
            return False
        experiment = getattr(self.agent, "experiment", None)
        if experiment is None:
            return False
        if experiment.active_plan_item_id:
            return False  # already resolved (resume or earlier auto-attach)
        return True

    # ------------------------------------------------------------------
    # Session-resolution picker (replaces LLM-driven bootstrap proposal)
    # ------------------------------------------------------------------

    def _build_resolution_candidates(self) -> list:
        """Pull and rank imaging candidates for the resolution picker.

        Returns the same list ``memory.resolve_plan_context`` returns —
        a list of ``(item, spec, campaign)`` tuples — sorted in_progress
        first, then by ``phase_order``.
        """
        if not hasattr(self.agent, "memory") or not self.agent.memory:
            return []
        try:
            _, candidates = self.agent.memory.resolve_plan_context()
        except Exception:
            return []

        def _sort_key(t):
            item, _spec, _campaign = t
            status_val = getattr(item.status, "value", str(item.status))
            in_progress = status_val == "in_progress"
            order = getattr(item, "phase_order", 0) or 0
            return (0 if in_progress else 1, order)

        candidates.sort(key=_sort_key)
        return candidates

    def _candidate_to_option(self, item, spec, campaign) -> dict:
        """Turn a ``(item, spec, campaign)`` tuple into a picker option."""
        memory = getattr(self.agent, "memory", None)
        spec_summary = ""
        if memory is not None and spec is not None:
            try:
                spec_summary = memory.format_imaging_spec_summary(spec) or ""
            except Exception:
                spec_summary = ""

        meta: dict[str, Any] = {}
        if campaign is not None:
            c_name = (
                getattr(campaign, "shorthand", None)
                or ((getattr(campaign, "description", "") or "")[:60])
            )
            if c_name:
                meta["campaign"] = c_name

        cs = getattr(self.agent, "context_store", None)
        if cs is not None and campaign is not None:
            try:
                ps = cs.get_plan_status(campaign.id)
                total = ps.get("total", 0) or 0
                done = ps.get("completed", 0) or 0
                order = getattr(item, "phase_order", 0) or 0
                if total > 0:
                    meta["sequence"] = (
                        f"{order} of {total} · {done} done" if order else f"{done}/{total} done"
                    )
            except Exception:
                pass

        status_val = getattr(item.status, "value", str(item.status))
        if status_val and status_val != "planned":
            meta["status"] = status_val

        if spec is not None:
            spec_dict: dict[str, Any] = {}
            for field in (
                "strain",
                "genotype",
                "reporter",
                "temperature_c",
                "num_slices",
                "exposure_ms",
                "laser_wavelength_nm",
                "interval_s",
                "stop_condition",
                "success_criteria",
            ):
                val = getattr(spec, field, None)
                if val is not None:
                    spec_dict[field] = val
            # Carry per-field provenance so the UI can tag inferred values
            # (e.g. "561 nm · inferred · medium") and show what to confirm.
            prov = getattr(spec, "provenance", None)
            if prov:
                spec_dict["provenance"] = prov
            if spec_dict:
                meta["spec"] = spec_dict

        return {
            "id": f"plan:{item.id}",
            "label": f"Continue · {item.title}",
            "description": spec_summary or "(no spec recorded)",
            "meta": meta,
        }

    def _build_resolution_choice_payload(
        self,
        candidates: list,
        full_list: bool = False,
    ) -> dict:
        """Build the ``choice_data`` payload for the resolution picker.

        Top 3 candidates by default; ``full_list=True`` shows up to 20
        with the standalone/plan_new escape hatches still appended.
        """
        show_n = 20 if full_list else 3
        options = []
        for item, spec, campaign in candidates[:show_n]:
            options.append(self._candidate_to_option(item, spec, campaign))

        if not full_list and len(candidates) > show_n:
            remaining = len(candidates) - show_n
            options.append(
                {
                    "id": "show_all",
                    "label": f"See all imaging tasks ({remaining} more)…",
                    "description": "Browse the full unblocked list",
                }
            )

        options.append(
            {
                "id": "standalone",
                "label": "Standalone — just exploring",
                "description": "No plan attached, default settings",
            }
        )
        options.append(
            {
                "id": "plan_new",
                "label": "Design a new plan",
                "description": "Enter plan mode first",
            }
        )

        question = (
            f"All {len(candidates)} unblocked imaging tasks — pick one:"
            if full_list
            else "What is this session for?"
        )

        return {
            "_type": "single",
            "_kind": "session_resolution",
            "question": question,
            "options": options,
            "allow_multiple": False,
        }

    async def bootstrap_resolution_picker(
        self,
        send_fn: Callable[[dict], Coroutine],
        choice_future_factory: Callable[[dict], "asyncio.Future[str]"],
    ) -> None:
        """Open the session with a deterministic resolution picker.

        Two-phase to minimize perceived startup latency:

        **Phase 1 — fast top-level picker** (no slow queries before this
        renders): the user sees a 3-option question
        (Standalone / Continue a planned task / Design a new plan)
        immediately on session start. In parallel, plan-candidate
        enumeration (``_build_resolution_candidates``, which scans the
        memory store) runs as a background ``to_thread`` task so it's
        usually done by the time the user picks anything.

        **Phase 2 — candidate picker** (only if they pick "Continue"):
        await the background enumeration and show the existing
        candidate-list picker with the standalone/plan_new escape hatches.

        Dispatch dictionary unchanged from before:

        - ``plan:<item_id>`` → ``attach_session_to_plan`` +
          ``apply_plan_acquisition_spec`` + ``applied_spec`` panel
        - ``standalone`` → ``mark_session_standalone``
        - ``plan_new`` → ``enter_plan_mode`` on the agent
        - ``show_all`` → re-emit the candidate picker with the full list
        - ``__custom__`` / free text → hand off to LLM-driven resolution
          mode with the typed text as the bootstrap turn
        """
        import uuid as _uuid

        # Kick off the slow candidate scan in the background. By the time
        # the user picks "Continue", this is almost always already done.
        candidates_task = asyncio.create_task(asyncio.to_thread(self._build_resolution_candidates))

        # --- Phase 1: fast top-level question -------------------------
        top_payload = {
            "question": "What is this session for?",
            "options": [
                {
                    "id": "standalone",
                    "label": "Standalone — just exploring",
                    "description": "No plan attached, default settings",
                },
                {
                    "id": "resume_plan",
                    "label": "Continue a planned task",
                    "description": "Pick from active plan items",
                },
                {
                    "id": "plan_new",
                    "label": "Design a new plan",
                    "description": "Enter plan mode first",
                },
            ],
            "allow_multiple": False,
        }
        top_request_id = f"resolve_top_{_uuid.uuid4().hex[:8]}"
        top_payload["request_id"] = top_request_id
        top_future = choice_future_factory(top_payload)
        await send_fn(
            {
                "type": "choice_request",
                "choice_data": top_payload,
                "request_id": top_request_id,
            }
        )

        try:
            top_choice = await top_future
        except asyncio.CancelledError:
            candidates_task.cancel()
            return

        # Anything except "resume_plan" — standalone, plan_new, free-text,
        # ESC cancel — dispatches directly without enumerating candidates.
        if top_choice != "resume_plan":
            candidates_task.cancel()
            await self._dispatch_resolution_pick(
                top_choice or "standalone",
                send_fn,
                choice_future_factory,
            )
            return

        # --- Phase 2: candidate picker (only if "resume_plan") --------
        try:
            candidates = await candidates_task
        except Exception:
            candidates = []

        if not candidates:
            # Plan-resume path with no candidates — show the briefing and
            # let the user start chatting; nothing to attach.
            briefing = self.get_session_briefing()
            if briefing:
                await send_fn({"type": "stream_start"})
                await send_fn({"type": "text", "text": briefing})
                await send_fn(
                    {
                        "type": "stream_end",
                        "tokens": self._get_token_snapshot(),
                        "mode": self.agent.mode,
                    }
                )
            else:
                await self._emit_resolution_result(
                    send_fn,
                    "No plan items available to continue. "
                    "You can design a new plan from the Plans tab.",
                )
            return

        full_list = False
        while True:
            payload = self._build_resolution_choice_payload(
                candidates,
                full_list=full_list,
            )
            # Re-label the question for the secondary picker so the user
            # knows they're now picking the specific plan item.
            payload["question"] = "Which plan item?"
            request_id = f"resolve_pick_{_uuid.uuid4().hex[:8]}"
            payload["request_id"] = request_id
            future = choice_future_factory(payload)
            await send_fn(
                {
                    "type": "choice_request",
                    "choice_data": payload,
                    "request_id": request_id,
                }
            )
            try:
                selected = await future
            except asyncio.CancelledError:
                return

            if selected == "show_all":
                full_list = True
                continue

            await self._dispatch_resolution_pick(
                selected,
                send_fn,
                choice_future_factory,
            )
            return

    async def _dispatch_resolution_pick(
        self,
        selected: str,
        send_fn: Callable[[dict], Coroutine],
        choice_future_factory: Callable[[dict], "asyncio.Future[str]"],
    ) -> None:
        """Apply the user's resolution pick.

        ``selected`` is the option id from the picker, or — for the
        auto-injected ``__custom__`` option — the user's free-text
        response. Anything that isn't one of our known ids is treated
        as custom text and handed off to the LLM-driven resolution
        flow.
        """
        from gently.harness.tools.registry import get_tool_registry

        if not selected:
            # User cancelled the picker (ESC). Leave the session in
            # default state — no plan attached. The user can ask for
            # the picker again or just start chatting.
            await self._emit_resolution_result(
                send_fn,
                "No plan attached yet. Let me know what you'd like to do.",
            )
            return

        registry = get_tool_registry()
        context = {
            "agent": self.agent,
            "client": getattr(self.agent, "client", None),
        }

        if selected.startswith("plan:"):
            plan_item_id = selected.split(":", 1)[1]
            try:
                await registry.execute(
                    "attach_session_to_plan",
                    {
                        "plan_item_id": plan_item_id,
                        "rationale": "User picked from resolution card.",
                    },
                    context=context,
                )
            except Exception as e:
                logger.error(f"attach_session_to_plan failed: {e}", exc_info=True)
                await self._emit_resolution_result(
                    send_fn,
                    "Couldn't attach the session to that plan item."
                    " You can try again or pick standalone.",
                )
                return

            try:
                await registry.execute(
                    "apply_plan_acquisition_spec",
                    {"plan_item_id": plan_item_id},
                    context=context,
                )
            except Exception as e:
                logger.error(f"apply_plan_acquisition_spec failed: {e}", exc_info=True)

            spec_dict = self._get_active_plan_spec()
            closer = self._compose_attach_closer(spec_dict)
            await self._emit_resolution_result(
                send_fn,
                closer,
                applied_spec=spec_dict,
            )
            return

        if selected == "standalone":
            try:
                await registry.execute(
                    "mark_session_standalone",
                    {"description": "Standalone exploration"},
                    context=context,
                )
            except Exception as e:
                logger.error(f"mark_session_standalone failed: {e}", exc_info=True)
            await self._emit_resolution_result(
                send_fn,
                "Standalone session — no plan attached. Ready when you are.",
            )
            return

        if selected == "plan_new":
            try:
                msg = self.agent.enter_plan_mode()
            except Exception as e:
                logger.error(f"enter_plan_mode failed: {e}", exc_info=True)
                msg = "Plan mode active."
            await self._emit_resolution_result(
                send_fn,
                msg or "Plan mode — what are we designing?",
            )
            return

        # Anything else: treat as custom text. Hand off to the
        # LLM-driven resolution flow with the user's input as the
        # opening turn so the agent can disambiguate.
        try:
            self.agent.enter_resolution_mode()
        except Exception as e:
            logger.warning(f"enter_resolution_mode failed: {e}")
            await self._emit_resolution_result(
                send_fn,
                "Couldn't enter resolution mode.",
            )
            return
        await self.stream_response(
            selected or "(no input)",
            send_fn,
            choice_future_factory,
        )

    def _get_active_plan_spec(self) -> dict | None:
        """Return the ``active_plan_spec`` dict stashed on
        ``experiment.metadata`` by ``apply_plan_acquisition_spec``."""
        try:
            spec = self.agent.experiment.metadata.get("active_plan_spec")
            return dict(spec) if spec else None
        except Exception:
            return None

    def _compose_attach_closer(self, spec_dict: dict | None) -> str:
        """One-line conversational closer to follow attach + apply."""
        title = (spec_dict or {}).get("plan_item_title") or "this plan item"
        return f"Attached to **{title}**. Mark embryo positions when you're ready."

    async def _emit_resolution_result(
        self,
        send_fn: Callable[[dict], Coroutine],
        closer_text: str,
        applied_spec: dict | None = None,
    ) -> None:
        """Emit a deterministic stream_start → text → stream_end pair,
        followed by an optional ``applied_spec`` panel message."""
        await send_fn({"type": "stream_start"})
        await send_fn({"type": "text", "text": closer_text})
        await send_fn(
            {
                "type": "stream_end",
                "tokens": self._get_token_snapshot(),
                "mode": self.agent.mode,
            }
        )
        if applied_spec:
            await send_fn(
                {
                    "type": "applied_spec",
                    "spec": applied_spec,
                }
            )

    def init_wizard(self, context_store, claude_client=None) -> None:
        """Create the startup wizard from a ContextStore."""
        from .memory.startup_wizard import StartupWizard

        self._context_store = context_store
        self._wizard = StartupWizard(
            context_store=context_store,
            session_id=self.agent.session_id,
            claude_client=claude_client,
        )

    async def stream_response(
        self,
        message: str,
        send_fn: Callable[[dict], Coroutine],
        choice_future_factory: Callable[[dict], "asyncio.Future[str]"],
    ) -> None:
        """
        Stream an agent response over WebSocket.

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
        stream_iter = self.agent.handle_message_stream(message).__aiter__()
        pending_choice_result = None

        try:
            # Open the stream envelope immediately — before the (potentially
            # slow) context build + Claude API call with extended thinking — so
            # every client shows the "Working…" trust signal the instant the
            # turn starts, not only once the first token arrives. Pairs with the
            # stream_end emitted on StopAsyncIteration below.
            await send_fn({"type": "stream_start"})

            while True:
                try:
                    if pending_choice_result is not None:
                        chunk = await stream_iter.asend(pending_choice_result)
                        pending_choice_result = None
                    else:
                        chunk = await stream_iter.__anext__()
                except StopAsyncIteration:
                    # Stream finished — send token usage summary + current mode
                    await send_fn(
                        {
                            "type": "stream_end",
                            "tokens": self._get_token_snapshot(),
                            "mode": self.agent.mode,
                        }
                    )
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
        finally:
            # Deterministically close the agent generator so its turn-lock (and
            # any other resources) release immediately. Without this, a cancelled
            # or aborted stream leaves the generator suspended at a `yield` still
            # holding self._turn_lock until non-deterministic GC, stalling the
            # next user turn and any autonomous wake turn on lock.acquire().
            try:
                await stream_iter.aclose()
            except Exception:
                pass

    async def handle_command(
        self,
        command: str,
        send_fn: Callable[[dict], Coroutine],
        choice_futures: dict | None = None,
    ) -> None:
        """
        Execute a slash command and send the result.

        Delegates to the agent's command handling. For commands that
        need Rich rendering, we return structured data instead.

        Parameters
        ----------
        command : str
            The slash command string (e.g., "/status").
        send_fn : async callable
            Called with the result dict.
        """
        registry = get_command_registry()
        cmd = command.strip().lower()
        cmd_name = cmd.split()[0]
        cmd_def = registry.get(cmd_name)
        logger.info(
            "handle_command: %s (resolved: %s)",
            cmd_name,
            cmd_def.name if cmd_def else "NOT FOUND",
        )

        if not cmd_def:
            await send_fn(
                {
                    "type": "command_result",
                    "command": command,
                    "error": f"Unknown command: {cmd_name}",
                }
            )
            return

        if cmd in ("/quit", "/exit", "/q"):
            await send_fn(
                {
                    "type": "command_result",
                    "command": cmd,
                    "action": "quit",
                }
            )
            return

        if cmd == "/status":
            status = self._get_status_data()
            await send_fn(
                {
                    "type": "command_result",
                    "command": "/status",
                    "content": status,
                }
            )
            return

        if cmd in ("/peers", "/mesh") or cmd.startswith("/peers ") or cmd.startswith("/mesh "):
            parts = command.strip().split()
            if len(parts) >= 3 and parts[2].lower() == "campaigns":
                data = await self._get_peer_campaigns(parts[1])
                await send_fn(
                    {
                        "type": "command_result",
                        "command": "/peers",
                        "content": data,
                    }
                )
            else:
                data = self._get_peers_data()
                await send_fn(
                    {
                        "type": "command_result",
                        "command": "/peers",
                        "content": data,
                    }
                )
            return

        if cmd == "/embryos" or cmd.startswith("/embryos "):
            parts = cmd.split(maxsplit=1)
            embryo_id = parts[1].strip() if len(parts) > 1 else None
            data = self._get_embryos_data(embryo_id)
            await send_fn(
                {
                    "type": "command_result",
                    "command": "/embryos",
                    "content": data,
                }
            )
            return

        if cmd == "/tokens":
            data = self._get_tokens_data()
            await send_fn(
                {
                    "type": "command_result",
                    "command": "/tokens",
                    "content": data,
                }
            )
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
            await send_fn(
                {
                    "type": "command_result",
                    "command": "/help",
                    "content": {"text": text},
                }
            )
            return

        if cmd.startswith("/theme"):
            parts = cmd.split()
            if len(parts) > 1:
                from gently.app.theme import get_theme, set_theme

                try:
                    set_theme(parts[1])
                    theme = get_theme()
                    await send_fn(
                        {
                            "type": "command_result",
                            "command": "/theme",
                            "content": {"theme": theme.name, "changed": True},
                        }
                    )
                except ValueError as e:
                    await send_fn(
                        {
                            "type": "command_result",
                            "command": "/theme",
                            "error": str(e),
                        }
                    )
            else:
                from gently.app.theme import get_theme, list_themes

                current = get_theme()
                themes = {k: v.name for k, v in list_themes().items()}
                await send_fn(
                    {
                        "type": "command_result",
                        "command": "/theme",
                        "content": {"themes": themes, "current": current.name},
                    }
                )
            return

        if cmd == "/sessions":
            await send_fn(
                {
                    "type": "command_result",
                    "command": "/sessions",
                    "content": {"sessions": self._get_sessions_list()},
                }
            )
            return

        if cmd == "/timelapse" or cmd == "/timelapse watch":
            data = self._get_timelapse_data()
            await send_fn(
                {
                    "type": "command_result",
                    "command": "/timelapse",
                    "content": data,
                }
            )
            return

        if cmd.startswith("/timeline"):
            parts = command.strip().split()
            data = self._get_timeline_data(parts[1:] if len(parts) > 1 else [])
            await send_fn(
                {
                    "type": "command_result",
                    "command": "/timeline",
                    "content": data,
                }
            )
            return

        if cmd == "/detectors":
            data = self._get_detectors_data()
            await send_fn(
                {
                    "type": "command_result",
                    "command": "/detectors",
                    "content": data,
                }
            )
            return

        if cmd == "/history":
            data = self._get_history_data()
            await send_fn(
                {
                    "type": "command_result",
                    "command": "/history",
                    "content": data,
                }
            )
            return

        if cmd == "/save":
            success = self.agent.save_session()
            if success:
                await send_fn(
                    {
                        "type": "command_result",
                        "command": "/save",
                        "content": {"text": f"Session saved: {self.agent.session_id}"},
                    }
                )
            else:
                await send_fn(
                    {
                        "type": "command_result",
                        "command": "/save",
                        "error": "Failed to save session",
                    }
                )
            return

        if cmd == "/reset-context":
            cs = self._require_context_store()
            if cs is None:
                await send_fn(
                    {
                        "type": "command_result",
                        "command": "/reset-context",
                        "error": "Context store not available",
                    }
                )
            else:
                counts = cs.reset()
                total = sum(counts.values())
                # Re-create the wizard so it re-assesses gaps
                claude_client = self._wizard.claude_client if self._wizard else None
                self.init_wizard(cs, claude_client)
                if total > 0:
                    details = ", ".join(f"{v} {k}" for k, v in counts.items())
                    msg = (
                        f"Context cleared: {total} entries removed ({details})."
                        "\nRun /wizard to set up again."
                    )
                else:
                    msg = "Context already empty — nothing to clear."
                await send_fn(
                    {
                        "type": "command_result",
                        "command": "/reset-context",
                        "content": {"text": msg},
                    }
                )
            return

        if cmd == "/wizard":
            # Handled by the WebSocket route (agent_ws.py), not the bridge.
            # If we reach here, it means the wizard loop called handle_command
            # — i.e. /wizard was typed while the wizard is already running.
            await send_fn(
                {
                    "type": "command_result",
                    "command": "/wizard",
                    "content": {"text": "The wizard is already running."},
                }
            )
            return

        if cmd == "/campaign" or cmd == "/campaigns" or cmd.startswith("/campaign "):
            parts = command.strip().split()
            subcmd = parts[1].lower() if len(parts) >= 2 else None

            if subcmd == "share" and len(parts) >= 3:
                data = self._share_campaign(parts[2])
                await send_fn(
                    {
                        "type": "command_result",
                        "command": "/campaign",
                        "content": data,
                    }
                )
            elif subcmd == "unshare" and len(parts) >= 3:
                data = self._unshare_campaign(parts[2])
                await send_fn(
                    {
                        "type": "command_result",
                        "command": "/campaign",
                        "content": data,
                    }
                )
            elif subcmd == "delete" and len(parts) >= 3:
                data = self._delete_campaign(parts[2])
                await send_fn(
                    {
                        "type": "command_result",
                        "command": "/campaign",
                        "content": data,
                    }
                )
            elif subcmd == "rename" and len(parts) >= 4:
                data = self._rename_campaign(parts[2], " ".join(parts[3:]))
                await send_fn(
                    {
                        "type": "command_result",
                        "command": "/campaign",
                        "content": data,
                    }
                )
            elif subcmd == "pause" and len(parts) >= 3:
                data = self._pause_campaign(parts[2])
                await send_fn(
                    {
                        "type": "command_result",
                        "command": "/campaign",
                        "content": data,
                    }
                )
            elif subcmd == "resume" and len(parts) >= 3:
                data = self._resume_campaign(parts[2])
                await send_fn(
                    {
                        "type": "command_result",
                        "command": "/campaign",
                        "content": data,
                    }
                )
            else:
                data = self._get_campaigns_data(command.strip())
                await send_fn(
                    {
                        "type": "command_result",
                        "command": "/campaign",
                        "content": data,
                    }
                )
            return

        if cmd.startswith("/join-campaign"):
            parts = command.strip().split()
            if len(parts) >= 3:
                data = await self._join_campaign(parts[1], parts[2])
            else:
                data = {"text": "Usage: /join-campaign <hostname> <campaign_id>"}
            await send_fn(
                {
                    "type": "command_result",
                    "command": "/join-campaign",
                    "content": data,
                }
            )
            return

        if cmd.startswith("/claim"):
            parts = command.strip().split()
            if len(parts) >= 2:
                data = await self._claim_item(parts[1])
            else:
                data = {"text": "Usage: /claim <item_id>"}
            await send_fn(
                {
                    "type": "command_result",
                    "command": "/claim",
                    "content": data,
                }
            )
            return

        if cmd == "/pair" or cmd.startswith("/pair "):
            parts = command.strip().split()
            subcmd = parts[1].lower() if len(parts) > 1 else ""
            arg = parts[2] if len(parts) > 2 else ""

            if subcmd == "accept":
                data = await self._pair_accept(send_fn)
            elif subcmd == "reject":
                data = await self._pair_reject()
            elif subcmd == "list":
                data = self._pair_list()
            elif subcmd == "unpair" and arg:
                data = self._pair_unpair(arg)
            elif subcmd == "scopes":
                extra_args = parts[3] if len(parts) > 3 else ""
                data = self._pair_scopes(arg, extra_args)
            elif subcmd and subcmd not in (
                "accept",
                "reject",
                "list",
                "unpair",
                "scopes",
            ):
                # Treat as hostname — initiate pairing
                data = await self._pair_initiate(subcmd, send_fn)
            else:
                data = {
                    "text": (
                        "Usage: /pair <hostname> | accept | reject | list"
                        " | unpair <id> | scopes [hostname] [scope_list]"
                    )
                }

            await send_fn(
                {
                    "type": "command_result",
                    "command": "/pair",
                    "content": data,
                }
            )
            return

        if cmd == "/plan" or cmd.startswith("/plan "):
            parts = command.strip().split(maxsplit=1)
            subcmd = parts[1].strip().lower() if len(parts) > 1 else None

            if subcmd == "exit":
                msg = self.agent.exit_plan_mode()
                await send_fn(
                    {
                        "type": "command_result",
                        "command": "/plan",
                        "content": {
                            "text": msg,
                            "mode": self.agent.mode,
                        },
                    }
                )
            elif subcmd == "status":
                summary = self.agent._get_active_plan_summary()
                if summary:
                    text = f"Mode: {self.agent.mode}\n\n{summary}"
                else:
                    text = f"Mode: {self.agent.mode}\nNo active campaigns found."
                await send_fn(
                    {
                        "type": "command_result",
                        "command": "/plan",
                        "content": {
                            "text": text,
                            "mode": self.agent.mode,
                        },
                    }
                )
            else:
                # Enter plan mode (or show status if already in it)
                if self.agent.mode == "plan":
                    summary = self.agent._get_active_plan_summary()
                    text = "Already in plan mode."
                    if summary:
                        text += f"\n\n{summary}"
                    text += "\n\nUse /plan exit to return to run mode."
                    await send_fn(
                        {
                            "type": "command_result",
                            "command": "/plan",
                            "content": {
                                "text": text,
                                "mode": "plan",
                            },
                        }
                    )
                else:
                    msg = self.agent.enter_plan_mode()
                    await send_fn(
                        {
                            "type": "command_result",
                            "command": "/plan",
                            "content": {
                                "text": msg,
                                "mode": "plan",
                            },
                        }
                    )
            return

        if cmd == "/clear":
            await send_fn(
                {
                    "type": "command_result",
                    "command": "/clear",
                    "action": "clear",
                }
            )
            return

        if cmd.startswith("/resume"):
            parts = command.strip().split(maxsplit=1)
            if len(parts) > 1:
                session_id = parts[1].strip()
                success = self.agent.resume_session(session_id)
                if success:
                    embryo_count = len(self.agent.experiment.embryos)
                    msg_count = len(self.agent.conversation_history)
                    await send_fn(
                        {
                            "type": "command_result",
                            "command": "/resume",
                            "content": {
                                "text": (
                                    f"Session resumed: {session_id}\n"
                                    f"  {embryo_count} embryos, {msg_count} messages"
                                ),
                            },
                        }
                    )
                else:
                    await send_fn(
                        {
                            "type": "command_result",
                            "command": "/resume",
                            "error": f"Session '{session_id}' not found",
                        }
                    )
            else:
                # No session ID — list available sessions for the user
                sessions = self._get_sessions_list()
                if sessions:
                    lines = ["Available sessions (use /resume <id>):"]
                    for s in sessions:
                        lines.append(f"  {s['session_id']} — {s['embryo_count']} embryos")
                    await send_fn(
                        {
                            "type": "command_result",
                            "command": "/resume",
                            "content": {"text": "\n".join(lines)},
                        }
                    )
                else:
                    await send_fn(
                        {
                            "type": "command_result",
                            "command": "/resume",
                            "content": {"text": "No saved sessions found."},
                        }
                    )
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
                        await send_fn(
                            {
                                "type": "command_result",
                                "command": "/import-embryos",
                                "error": "No sessions with embryos found.",
                            }
                        )
                        return
                    session_id = sessions_with[0]["session_id"]
                else:
                    session_id = arg

                result = self.agent.import_embryos_from_session(session_id)
                await self._send_import_result(send_fn, result, session_id)
            else:
                sessions = self._get_sessions_list()
                sessions_with = [s for s in sessions if s["embryo_count"] > 0]
                if sessions_with:
                    import uuid as _uuid

                    request_id = f"import_embryos_{_uuid.uuid4().hex[:8]}"
                    options = []
                    for s in sessions_with[:10]:
                        sid = s["session_id"]
                        count = s["embryo_count"]
                        name = s.get("name", "")
                        last_active = s.get("last_active", "")
                        # Format datetime for display
                        time_str = ""
                        if last_active:
                            try:
                                from datetime import datetime

                                dt = datetime.fromisoformat(last_active)
                                time_str = dt.strftime("%b %d %H:%M")
                            except (ValueError, TypeError):
                                time_str = str(last_active)[:16]
                        label = f"{sid[:8]} — {count} embryo{'s' if count != 1 else ''}"
                        if time_str:
                            label += f" ({time_str})"
                        desc = f"Import embryos from session {sid}"
                        if name:
                            desc = name
                        options.append(
                            {
                                "id": sid,
                                "label": label,
                                "description": desc,
                            }
                        )
                    await send_fn(
                        {
                            "type": "choice_request",
                            "choice_data": {
                                "_type": "single",
                                "question": "Import embryos from which session?",
                                "options": options,
                                "allow_multiple": False,
                            },
                            "request_id": request_id,
                        }
                    )
                    # Register a callback so the choice response triggers the import.
                    # We can't await here (would deadlock the REPL loop), so we
                    # store state for _handle_import_choice to pick up.
                    self._pending_import = {
                        "request_id": request_id,
                        "send_fn": send_fn,
                    }
                else:
                    await send_fn(
                        {
                            "type": "command_result",
                            "command": "/import-embryos",
                            "content": {"text": "No sessions with embryos found."},
                        }
                    )
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

            session_id = self.agent.session_id
            if not session_id:
                await send_fn(
                    {
                        "type": "command_result",
                        "command": "/make-video",
                        "error": "No active session",
                    }
                )
                return

            try:
                from gently.app.video_maker import (
                    create_timelapse_video,
                    discover_volumes,
                )

                storage_path = self.agent.storage_path
                session_images_dir = storage_path / "images" / session_id

                if not session_images_dir.exists():
                    await send_fn(
                        {
                            "type": "command_result",
                            "command": "/make-video",
                            "error": f"No images found for session {session_id}",
                        }
                    )
                    return

                all_volumes = discover_volumes(session_images_dir, embryo_id)
                if not all_volumes:
                    await send_fn(
                        {
                            "type": "command_result",
                            "command": "/make-video",
                            "content": {"text": "No timelapse volumes found."},
                        }
                    )
                    return

                lines = [f"Creating timelapse videos (fps={fps})..."]
                for eid, vol_paths in all_volumes.items():
                    output_path = session_images_dir / f"{eid}_timelapse.mp4"
                    create_timelapse_video(vol_paths, output_path, fps=fps)
                    lines.append(f"  {eid}: {len(vol_paths)} frames → {output_path.name}")

                await send_fn(
                    {
                        "type": "command_result",
                        "command": "/make-video",
                        "content": {"text": "\n".join(lines)},
                    }
                )
            except ImportError:
                await send_fn(
                    {
                        "type": "command_result",
                        "command": "/make-video",
                        "error": "Video maker module not available.",
                    }
                )
            except Exception as e:
                await send_fn(
                    {
                        "type": "command_result",
                        "command": "/make-video",
                        "error": str(e),
                    }
                )
            return

        if cmd.startswith("/test-device") or cmd.startswith("/benchmark"):
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
                from gently.app.benchmark import run_benchmark

                async def _benchmark_progress(stage, current, total, timing):
                    if stage == "warmup":
                        text = f"⏳ Warmup {current}/{total}..."
                    elif stage == "acquiring":
                        text = f"▶ Acquiring volume {current}/{total}..."
                    elif stage == "volume_done":
                        ok = "✓" if timing.success else "✗"
                        parts = [f"{ok} Volume {current}/{total}"]
                        if timing.success:
                            parts.append(f"acq={timing.acquisition_time:.3f}s")
                            parts.append(f"store={timing.storage_time:.3f}s")
                            parts.append(f"total={timing.total_time:.3f}s")
                        else:
                            parts.append(f"error: {timing.error}")
                        text = "  ".join(parts)
                    else:
                        return
                    await send_fn(
                        {
                            "type": "command_result",
                            "command": "/test-device",
                            "content": {"text": text},
                        }
                    )

                await send_fn(
                    {
                        "type": "command_result",
                        "command": "/test-device",
                        "content": {
                            "text": (
                                f"Running device test ({n_volumes} volumes,"
                                f" {n_slices} slices, {n_warmup} warmup)..."
                            )
                        },
                    }
                )
                results = await run_benchmark(
                    self.agent,
                    n_volumes=n_volumes,
                    n_slices=n_slices,
                    n_warmup=n_warmup,
                    progress_fn=_benchmark_progress,
                )

                # Format summary table
                total = results.total_stats
                acq = results.acquisition_stats
                stor = results.storage_stats
                viz = results.viz_push_stats
                lines = [
                    "━━━ Device Test Results ━━━",
                    f"  {len(results.successful)}/{len(results.timings)} volumes OK",
                    f"  Throughput: {results.fps:.2f} vol/s",
                    "",
                    "  Stage          Mean      Std       Min       Max",
                    f"  Acquisition  {acq['mean']:.3f}s  {acq['std']:.3f}s  "
                    f"{acq['min']:.3f}s  {acq['max']:.3f}s",
                    f"  Storage      {stor['mean']:.3f}s  {stor['std']:.3f}s  "
                    f"{stor['min']:.3f}s  {stor['max']:.3f}s",
                ]
                if viz["mean"] > 0:
                    lines.append(
                        f"  Viz push     {viz['mean']:.3f}s  {viz['std']:.3f}s  "
                        f"{viz['min']:.3f}s  {viz['max']:.3f}s"
                    )
                lines.extend(
                    [
                        f"  Total        {total['mean']:.3f}s  {total['std']:.3f}s  "
                        f"{total['min']:.3f}s  {total['max']:.3f}s",
                    ]
                )
                if results.avg_file_size_mb > 0:
                    lines.append(f"  File size:   {results.avg_file_size_mb:.1f} MB avg")
                if results.failed:
                    lines.append(f"  Failures:    {len(results.failed)}")

                await send_fn(
                    {
                        "type": "command_result",
                        "command": "/test-device",
                        "content": {"text": "\n".join(lines)},
                    }
                )
            except ImportError as e:
                logger.error("Benchmark import failed: %s", e, exc_info=True)
                await send_fn(
                    {
                        "type": "command_result",
                        "command": "/test-device",
                        "error": f"Benchmark module not available: {e}",
                    }
                )
            except Exception as e:
                await send_fn(
                    {
                        "type": "command_result",
                        "command": "/test-device",
                        "error": str(e),
                    }
                )
            return

        # Fallback for truly unimplemented commands
        await send_fn(
            {
                "type": "command_result",
                "command": cmd,
                "content": {"text": f"Command `{cmd}` is not yet available in the TUI."},
            }
        )

    def get_commands_json(self) -> list:
        """Serialize the command registry for the TUI client."""
        registry = get_command_registry()
        commands = []
        for cmd in registry.get_all():
            commands.append(
                {
                    "name": cmd.name,
                    "description": cmd.description,
                    "aliases": cmd.aliases,
                    "category": cmd.category.name,
                    "usage": cmd.usage_string(),
                    "arg_hint": cmd.arg_hint_string(),
                    "subcommands": [
                        {"name": s.name, "description": s.description} for s in cmd.subcommands
                    ],
                }
            )
        return commands

    def get_tools_json(self) -> list:
        """Serialize the agent tool registry for client-side autocomplete.

        Trimmed on purpose (first description line + lightweight param list) so
        the connect frame stays small. The web chat uses this for @tool-name
        completion and to show a tool's arguments inline.
        """
        try:
            from gently.harness.tools.registry import get_tool_registry

            registry = get_tool_registry()
        except Exception:
            return []
        tools = []
        for t in registry.list_all():
            desc = (t.description or "").strip().split("\n", 1)[0][:200]
            category = getattr(t.category, "name", None) or str(t.category)
            tools.append(
                {
                    "name": t.name,
                    "description": desc,
                    "category": category,
                    "params": [
                        {"name": p.name, "type": p.type, "required": bool(p.required)}
                        for p in t.parameters
                        if p.name != "context"
                    ],
                }
            )
        tools.sort(key=lambda x: x["name"])
        return tools

    # ------------------------------------------------------------------
    # Private helpers for structured command data
    # ------------------------------------------------------------------

    def _get_status_data(self) -> dict:
        """Build structured status data."""
        exp = self.agent.experiment
        client = self.agent.client

        return {
            "session_id": self.agent.session_id,
            "connected": client.is_connected if client else False,
            "embryo_count": len(exp.embryos),
            "embryo_ids": list(exp.embryos.keys()),
            "has_sam": client.has_sam if client else False,
        }

    def _get_embryos_data(self, embryo_id: str | None = None) -> dict:
        """Build structured embryo data."""
        exp = self.agent.experiment
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
            embryos.append(
                {
                    "id": eid,
                    "nickname": emb.nickname,
                    "user_label": emb.user_label,
                }
            )
        return {"embryos": embryos}

    def _get_tokens_data(self) -> dict:
        """Build structured token usage data."""
        return self._get_token_snapshot()

    def _get_token_snapshot(self) -> dict:
        """Current token usage from the agent."""
        c = self.agent
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

        exp = self.agent.experiment
        meta = {
            "session_id": self.agent.session_id,
            "commands": self.get_commands_json(),
            "tools": self.get_tools_json(),
            "version": getattr(gently, "__version__", "dev"),
            "tokens": self._get_token_snapshot(),
            "embryo_count": len(exp.embryos),
            "campaign_count": self._get_campaign_count(),
            # Launch info fields (set by launch_agent.py for TUI mode)
            "device_connected": self._launch_info.get("device_connected", False),
            "sam_available": self._launch_info.get("sam_available", False),
            "offline": self._launch_info.get("offline", False),
            "store_path": self._launch_info.get("store_path", ""),
            "viz_url": self._launch_info.get("viz_url", None),
            "log_path": self._launch_info.get("log_path", ""),
            "resumed": self._launch_info.get("resumed", False),
            "mode": self.agent.mode,
            "peer_count": self._get_peer_count(),
        }
        # Wizard metadata (if initialized)
        if self._wizard is not None:
            meta["wizard"] = self._wizard.gap_summary
        return meta

    def _get_peers_data(self) -> dict:
        """Build structured mesh peer data for the /peers command."""
        mesh = self._require_mesh()
        if mesh is None:
            return {"text": "Mesh not available."}

        peers = mesh.get_peers()
        if not peers:
            local = mesh.get_local_info()
            hostname = local.get("hostname", "unknown")
            instance_id = local.get("instance_id", "")[:8]
            return {"text": f"No peers discovered.\n  This node: {hostname} ({instance_id})"}

        lines = [f"Mesh Peers ({len(peers)})", ""]
        for p in peers:
            caps = []
            if p.capabilities.has_microscope:
                caps.append("microscope")
            if p.capabilities.has_gpu:
                gpu = p.capabilities.gpu_name or "GPU"
                vram = f" {p.capabilities.gpu_vram_gb}GB" if p.capabilities.gpu_vram_gb else ""
                caps.append(f"{gpu}{vram}")
            if p.capabilities.has_sam:
                caps.append("SAM")
            if p.capabilities.storage_free_gb:
                caps.append(f"{p.capabilities.storage_free_gb:.0f}GB free")

            cap_str = ", ".join(caps) if caps else "no special capabilities"
            status = p.status.agent_mode or "unknown"
            embryos = p.status.embryo_count

            stale = " (stale)" if p.is_stale else ""
            lines.append(f"  **{p.hostname}** ({p.instance_id[:8]}){stale}")
            lines.append(f"    {p.ip_address}:{p.viz_port} · {status} mode · {embryos} embryos")
            lines.append(f"    {cap_str}")
            lines.append("")

        return {"text": "\n".join(lines)}

    def _get_peer_count(self) -> int:
        """Return the number of live mesh peers."""
        mesh = self._require_mesh()
        if mesh is not None:
            try:
                return mesh.peer_count
            except Exception:
                logger.debug("Failed to get peer count", exc_info=True)
        return 0

    def _get_sessions_list(self) -> list:
        """Return a list of saved sessions with metadata."""
        sessions = []
        if hasattr(self.agent, "store") and self.agent.store:
            raw = self.agent.store.list_sessions()
            for s in raw:
                sid = s.get("session_id", "unknown")
                embryos = self.agent.store.list_embryos(sid)
                sessions.append(
                    {
                        "session_id": sid,
                        "name": s.get("name", ""),
                        "embryo_count": len(embryos) if embryos else 0,
                        "last_active": s.get("last_active", ""),
                    }
                )
        return sessions

    async def _send_import_result(self, send_fn, result: dict, session_label: str):
        """Format and send an import-embryos result."""
        if result.get("success"):
            imported = result.get("imported", [])
            skipped = result.get("skipped", [])
            lines = [f"Imported {len(imported)} embryo(s) from {session_label}"]
            if imported:
                lines.append(f"  {', '.join(imported)}")
            if skipped:
                lines.append(f"  Skipped (exist): {', '.join(skipped)}")
            await send_fn(
                {
                    "type": "command_result",
                    "command": "/import-embryos",
                    "content": {"text": "\n".join(lines)},
                }
            )
        else:
            await send_fn(
                {
                    "type": "command_result",
                    "command": "/import-embryos",
                    "error": result.get("error", "Import failed"),
                }
            )

    def _get_timelapse_data(self) -> dict:
        """Build structured timelapse status."""
        orch = getattr(self.agent, "timelapse_orchestrator", None)
        if not orch:
            return {"text": "No timelapse running."}

        state = orch.get_status()
        return state.to_dict()

    def _get_timeline_data(self, args: list) -> dict:
        """Build structured timeline data."""
        tm = getattr(self.agent, "timeline_manager", None)
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
        perceiver = getattr(self.agent, "perceiver", None)
        if not perceiver or not perceiver.sessions:
            return {"text": "No active perception sessions."}

        lines = ["Perception Sessions"]
        for embryo_id, session in perceiver.sessions.items():
            stage = session.current_stage or "unknown"
            obs_count = len(session.observations)
            lines.append(f"  {embryo_id}: stage={stage}, {obs_count} observations")

        return {"text": "\n".join(lines)}

    def _get_campaign_count(self) -> int:
        """Count active root campaigns."""
        cs = self._require_context_store()
        if cs is None:
            return 0
        try:
            return len(cs.get_root_campaigns())
        except Exception:
            logger.debug("Failed to get campaign count", exc_info=True)
            return 0

    def _delete_campaign(self, campaign_id: str) -> dict:
        """Delete a campaign by ID or shorthand."""
        cs = self._require_context_store()
        if cs is None:
            return {"text": "Context store not available."}

        campaign = cs.resolve_campaign(campaign_id)
        if not campaign:
            return {"text": f"Campaign '{campaign_id}' not found."}

        label = campaign.shorthand or campaign.display_name
        counts = cs.delete_campaign(campaign.id, cascade=True)
        parts = []
        if counts["campaigns"] > 0:
            parts.append(f"{counts['campaigns']} campaign{'s' if counts['campaigns'] != 1 else ''}")
        if counts["plan_items"] > 0:
            parts.append(
                f"{counts['plan_items']} plan item{'s' if counts['plan_items'] != 1 else ''}"
            )
        detail = f" ({', '.join(parts)})" if parts else ""
        return {"text": f"Deleted **{label}**{detail}."}

    def _rename_campaign(self, campaign_id: str, new_name: str) -> dict:
        """Rename a campaign's shorthand."""
        cs = self._require_context_store()
        if cs is None:
            return {"text": "Context store not available."}

        campaign = cs.resolve_campaign(campaign_id)
        if not campaign:
            return {"text": f"Campaign '{campaign_id}' not found."}

        old_name = campaign.shorthand or campaign.display_name
        cs.update_campaign(campaign.id, shorthand=new_name.strip())
        return {"text": f"Renamed **{old_name}** → **{new_name.strip()}**"}

    def _share_campaign(self, campaign_ref: str) -> dict:
        """Share a campaign on the mesh."""
        cs = self._require_context_store()
        if cs is None:
            return {"text": "Context store not available."}

        campaign = cs.resolve_campaign(campaign_ref)
        if not campaign:
            return {"text": f"Campaign '{campaign_ref}' not found."}

        cs.share_campaign(campaign.id)
        label = campaign.shorthand or campaign.display_name
        return {"text": f"Campaign **{label}** is now shared on the mesh."}

    def _unshare_campaign(self, campaign_ref: str) -> dict:
        """Stop sharing a campaign on the mesh."""
        cs = self._require_context_store()
        if cs is None:
            return {"text": "Context store not available."}

        campaign = cs.resolve_campaign(campaign_ref)
        if not campaign:
            return {"text": f"Campaign '{campaign_ref}' not found."}

        cs.unshare_campaign(campaign.id)
        label = campaign.shorthand or campaign.display_name
        return {"text": f"Campaign **{label}** is no longer shared."}

    def _pause_campaign(self, campaign_ref: str) -> dict:
        """Pause a campaign."""
        cs = self._require_context_store()
        if cs is None:
            return {"text": "Context store not available."}
        campaign = cs.resolve_campaign(campaign_ref)
        if not campaign:
            return {"text": f"Campaign '{campaign_ref}' not found."}
        from .memory.model import Status

        cs.update_campaign_status(campaign.id, Status.PAUSED)
        label = campaign.shorthand or campaign.display_name
        return {"text": f"Campaign **{label}** paused."}

    def _resume_campaign(self, campaign_ref: str) -> dict:
        """Resume a paused campaign."""
        cs = self._require_context_store()
        if cs is None:
            return {"text": "Context store not available."}
        campaign = cs.resolve_campaign(campaign_ref)
        if not campaign:
            return {"text": f"Campaign '{campaign_ref}' not found."}
        from .memory.model import Status

        cs.update_campaign_status(campaign.id, Status.ACTIVE)
        label = campaign.shorthand or campaign.display_name
        return {"text": f"Campaign **{label}** resumed."}

    async def _get_peer_campaigns(self, hostname: str) -> dict:
        """Fetch shared campaigns from a specific peer."""
        mesh = self._require_mesh()
        if mesh is None:
            return {"text": "Mesh not available."}

        peer = mesh.find_peer_by_hostname(hostname)
        if peer is None:
            return {"text": f"Peer '{hostname}' not found."}

        pc = mesh.peer_client
        if pc is None:
            return {"text": "Peer client not available."}

        info = await pc.fetch_peer_info(peer)
        if info is None:
            return {"text": f"Could not reach peer '{hostname}'."}

        shared = info.get("shared_campaigns", [])
        if not shared:
            return {"text": f"No shared campaigns on **{hostname}**."}

        lines = [f"Shared campaigns on **{hostname}**", ""]
        for c in shared:
            shorthand = c.get("shorthand") or c.get("id", "")[:8]
            desc = c.get("description", "")
            total = c.get("item_count", 0)
            done = c.get("completed_count", 0)
            lines.append(f"  **{shorthand}** ({done}/{total}) — {desc}")
            lines.append(f"    ID: {c.get('id', '')}")
        lines.append("")
        lines.append(f"Use `/join-campaign {hostname} <id>` to join.")
        return {"text": "\n".join(lines)}

    async def _join_campaign(self, hostname: str, campaign_ref: str) -> dict:
        """Join a shared campaign on a remote peer."""
        mesh = self._require_mesh()
        if mesh is None:
            return {"text": "Mesh not available."}

        peer = mesh.find_peer_by_hostname(hostname)
        if peer is None:
            return {"text": f"Peer '{hostname}' not found."}

        pc = mesh.peer_client
        if pc is None:
            return {"text": "Peer client not available."}

        import socket

        local_hostname = socket.gethostname()

        ok = await pc.join_campaign(peer, campaign_ref, mesh.instance_id, local_hostname)
        if not ok:
            return {"text": f"Failed to join campaign '{campaign_ref}' on {hostname}."}

        self._active_remote = {"peer": peer, "campaign_id": campaign_ref}
        return {
            "text": (
                f"Joined campaign **{campaign_ref}** on **{hostname}**."
                "\nUse `/claim <item_id>` to claim items."
            )
        }

    async def _claim_item(self, item_id: str) -> dict:
        """Claim a plan item from the active remote campaign."""
        if self._active_remote is None:
            return {
                "text": "No active remote campaign. Use `/join-campaign <hostname> <id>` first."
            }

        mesh = self._require_mesh()
        if mesh is None:
            return {"text": "Mesh not available."}

        pc = mesh.peer_client
        if pc is None:
            return {"text": "Peer client not available."}

        import socket

        local_hostname = socket.gethostname()

        peer = self._active_remote["peer"]
        campaign_id = self._active_remote["campaign_id"]

        ok = await pc.claim_item(peer, campaign_id, item_id, mesh.instance_id, local_hostname)
        if not ok:
            return {
                "text": (
                    f"Failed to claim item `{item_id}` — it may already be claimed by another node."
                )
            }

        return {
            "text": (
                f"Claimed item `{item_id}` from campaign **{campaign_id}** on **{peer.hostname}**."
            )
        }

    # ------------------------------------------------------------------
    # /pair helpers
    # ------------------------------------------------------------------

    async def _pair_initiate(self, hostname: str, send_fn) -> dict:
        """Initiate pairing with a peer by hostname."""
        mesh = self._require_mesh()
        if mesh is None:
            return {"text": "Mesh not available."}
        if mesh.pairing_manager is None:
            return {"text": "Pairing not configured."}

        peer = mesh.find_peer_by_hostname(hostname)
        if peer is None:
            return {"text": f"Peer **{hostname}** not found. Use `/peers` to see discovered peers."}

        pm = mesh.pairing_manager
        nonce_local = pm.create_initiation()

        # Send pairing request to the remote peer
        pc = mesh.peer_client
        if pc is None:
            return {"text": "Peer client not available."}

        resp = await pc.send_pair_request(
            peer,
            pm.instance_id,
            pm.hostname,
            nonce_local,
            cert_fingerprint=pm.cert_fingerprint,
            udp_sign_key=pm.udp_sign_key,
        )
        if resp is None:
            return {"text": f"Failed to reach **{hostname}** for pairing."}
        if "_error" in resp:
            return {"text": f"Failed to reach **{hostname}** for pairing.\n\n`{resp['_error']}`"}

        nonce_remote = resp.get("nonce", "")
        pairing_id = resp.get("pairing_id", "")
        peer_id = resp.get("responder_id", peer.instance_id)
        peer_host = resp.get("responder_hostname", hostname)
        remote_cert_fp = resp.get("cert_fingerprint", "")
        remote_udp_key = resp.get("udp_sign_key", "")

        if not nonce_remote or not pairing_id:
            return {"text": f"Invalid pairing response from **{hostname}**."}

        # Create local session and compute PIN
        session = pm.process_initiation_response(
            peer_id,
            peer_host,
            nonce_local,
            nonce_remote,
            pairing_id,
        )
        # Store remote peer's TLS cert fingerprint and UDP signing key
        session.responder_cert_fingerprint = remote_cert_fp
        session.responder_udp_sign_key = remote_udp_key
        session.initiator_cert_fingerprint = pm.cert_fingerprint
        session.initiator_udp_sign_key = pm.udp_sign_key

        # Auto-confirm initiator side on the remote
        await pc.confirm_pair_remote(peer, pairing_id, pm.instance_id)

        # Start background polling for confirmation
        asyncio.create_task(
            self._pair_poll(
                mesh,
                peer,
                pairing_id,
                send_fn,
            )
        )

        return {
            "text": (
                f"Pairing with **{peer_host}**\n\n"
                f"Verification code: **{session.pin}**\n\n"
                f"Verify this code matches on {peer_host}, then they should type `/pair accept`."
            ),
        }

    async def _pair_accept(self, send_fn) -> dict:
        """Accept the most recent pending pairing request."""
        mesh = self._require_mesh()
        if mesh is None or mesh.pairing_manager is None:
            return {"text": "Mesh/pairing not available."}

        pm = mesh.pairing_manager
        pending = pm.get_pending_sessions()
        if not pending:
            return {"text": "No pending pairing requests."}

        session = pending[-1]  # most recent
        pm.confirm_pairing(session.pairing_id, pm.instance_id)

        if session.status == "confirmed":
            mesh.mark_peer_trusted(session.initiator_id)

            from gently.core.event_bus import EventType, get_event_bus

            get_event_bus().publish(
                EventType.MESH_PAIRING_COMPLETED,
                {
                    "pairing_id": session.pairing_id,
                    "peer_hostname": session.initiator_hostname,
                },
                source="mesh",
            )

            return {"text": f"Paired with **{session.initiator_hostname}**!"}

        return {
            "text": (
                f"Confirmed pairing with **{session.initiator_hostname}**."
                " Waiting for their confirmation..."
            )
        }

    async def _pair_reject(self) -> dict:
        """Reject the most recent pending pairing request."""
        mesh = self._require_mesh()
        if mesh is None or mesh.pairing_manager is None:
            return {"text": "Mesh/pairing not available."}

        pm = mesh.pairing_manager
        pending = pm.get_pending_sessions()
        if not pending:
            return {"text": "No pending pairing requests."}

        session = pending[-1]
        pm.reject_pairing(session.pairing_id)
        return {"text": f"Rejected pairing request from **{session.initiator_hostname}**."}

    def _pair_list(self) -> dict:
        """List all trusted peers."""
        mesh = self._require_mesh()
        if mesh is None or mesh.pairing_manager is None:
            return {"text": "Mesh/pairing not available."}

        pm = mesh.pairing_manager
        trusted = pm.get_all_trusted()
        if not trusted:
            return {"text": "No trusted peers. Use `/pair <hostname>` to pair with a peer."}

        lines = ["Trusted Peers", ""]
        for tp in trusted:
            peer = mesh.get_peer(tp.instance_id)
            status = "online" if (peer and not peer.is_dead) else "offline"
            scope_str = ", ".join(tp.scopes) if tp.scopes else "none"
            lines.append(
                f"  **{tp.hostname}** ({tp.instance_id[:8]}) \u2014 {status} "
                f"\u2014 paired {tp.paired_at} \u2014 scopes: {scope_str}"
            )
        return {"text": "\n".join(lines)}

    def _pair_scopes(self, hostname: str, scope_arg: str) -> dict:
        """View or set permission scopes for a peer."""
        mesh = self._require_mesh()
        if mesh is None or mesh.pairing_manager is None:
            return {"text": "Mesh/pairing not available."}

        pm = mesh.pairing_manager

        if not hostname:
            # Show scopes for all peers
            trusted = pm.get_all_trusted()
            if not trusted:
                return {"text": "No trusted peers."}
            lines = ["Peer Scopes", ""]
            for tp in trusted:
                scope_str = ", ".join(tp.scopes) if tp.scopes else "none"
                lines.append(f"  **{tp.hostname}** ({tp.instance_id[:8]}): {scope_str}")
            from gently.mesh.pairing import ALL_SCOPES

            lines.append("")
            lines.append(f"Available scopes: {', '.join(ALL_SCOPES)}")
            return {"text": "\n".join(lines)}

        if not scope_arg:
            # Show scopes for a specific peer
            trusted = pm.get_all_trusted()
            for tp in trusted:
                if tp.hostname.lower() == hostname.lower() or tp.instance_id.startswith(hostname):
                    scope_str = ", ".join(tp.scopes) if tp.scopes else "none"
                    return {"text": f"Scopes for **{tp.hostname}**: {scope_str}"}
            return {"text": f"No trusted peer matching **{hostname}**."}

        # Set scopes
        new_scopes = [s.strip() for s in scope_arg.split(",") if s.strip()]
        from gently.mesh.pairing import ALL_SCOPES

        invalid = [s for s in new_scopes if s not in ALL_SCOPES]
        if invalid:
            return {
                "text": f"Invalid scopes: {', '.join(invalid)}. Available: {', '.join(ALL_SCOPES)}"
            }

        if pm.set_scopes(hostname, new_scopes):
            return {"text": f"Scopes for **{hostname}** updated to: {', '.join(new_scopes)}"}
        return {"text": f"No trusted peer matching **{hostname}**."}

    def _pair_unpair(self, identifier: str) -> dict:
        """Remove trust for a peer."""
        mesh = self._require_mesh()
        if mesh is None or mesh.pairing_manager is None:
            return {"text": "Mesh/pairing not available."}

        pm = mesh.pairing_manager
        removed = pm.unpair(identifier)
        if not removed:
            return {"text": f"No trusted peer matching **{identifier}**."}

        # Mark the peer as untrusted in the mesh
        for peer in mesh.get_all_peers():
            if peer.hostname.lower() == identifier.lower() or peer.instance_id.startswith(
                identifier
            ):
                peer.is_trusted = False
                break

        return {
            "text": (
                f"Unpaired from **{identifier}**."
                " They will need to re-pair to access mesh services."
            )
        }

    async def _pair_poll(self, mesh, peer, pairing_id, send_fn):
        """Background poll: wait for remote to confirm pairing."""
        pm = mesh.pairing_manager
        pc = mesh.peer_client
        if pm is None or pc is None:
            return

        for _ in range(60):  # 2s * 60 = 120s timeout
            await asyncio.sleep(2.0)

            resp = await pc.poll_pair_status(peer, pairing_id)
            if resp is None:
                continue

            status = resp.get("status", "")
            if status == "confirmed":
                # Finalize on our side
                session = pm.get_session(pairing_id)
                if session and session.status != "confirmed":
                    pm.confirm_pairing(pairing_id, session.responder_id)

                mesh.mark_peer_trusted(peer.instance_id)

                from gently.core.event_bus import EventType, get_event_bus

                get_event_bus().publish(
                    EventType.MESH_PAIRING_COMPLETED,
                    {"pairing_id": pairing_id, "peer_hostname": peer.hostname},
                    source="mesh",
                )

                await send_fn(
                    {
                        "type": "notification",
                        "level": "success",
                        "title": f"Paired with {peer.hostname}",
                    }
                )
                return

            if status in ("rejected", "expired"):
                await send_fn(
                    {
                        "type": "notification",
                        "level": "warning",
                        "title": f"Pairing {status}",
                        "body": peer.hostname,
                    }
                )
                return

        # Timeout
        await send_fn(
            {
                "type": "notification",
                "level": "warning",
                "title": "Pairing timed out",
                "body": peer.hostname,
            }
        )

    def _get_campaigns_data(self, command: str) -> dict:
        """Build structured campaign/plan data."""
        cs = self._require_context_store()
        if cs is None:
            return {"text": "Context store not available."}

        parts = command.split(maxsplit=1)
        campaign_id = parts[1].strip() if len(parts) > 1 else None

        if campaign_id:
            return self._render_campaign_detail(cs, campaign_id)
        return self._render_campaign_list(cs)

    def _render_campaign_list(self, cs) -> dict:
        """Render all campaigns as a text summary."""
        roots = cs.get_root_campaigns()
        if not roots:
            return {"text": "No campaigns yet. Use plan mode (/plan) to create experimental plans."}

        lines = ["Campaigns", ""]
        for root in roots:
            status = cs.get_plan_status(root.id)
            total = status["total"]
            completed = status["completed"]
            pct = f" ({completed}/{total})" if total > 0 else ""

            label = root.shorthand or root.display_name
            lines.append(f"  **{label}**{pct} — {root.description}")
            if root.target:
                lines.append(f"    Target: {root.target}")

            # Show subcampaigns (phases)
            children = cs.get_subcampaigns(root.id)
            for child in children:
                child_status = cs.get_plan_status(child.id)
                ct = child_status["total"]
                cc = child_status["completed"]
                child_label = child.shorthand or child.display_name
                child_pct = f" ({cc}/{ct})" if ct > 0 else ""
                lines.append(f"    · {child_label}{child_pct}")

            lines.append("")

        lines.append(
            "Use `/campaign <id>` for details, or browse at the viz server /campaigns page."
        )
        return {"text": "\n".join(lines)}

    def _render_campaign_detail(self, cs, campaign_id: str) -> dict:
        """Render a single campaign with all plan items."""
        # Try to find by ID or shorthand
        campaign = cs.get_campaign(campaign_id)
        if not campaign:
            # Try matching by shorthand
            for c in cs.get_active_campaigns():
                if c.shorthand and c.shorthand.lower() == campaign_id.lower():
                    campaign = c
                    break
            # Also check root campaigns
            if not campaign:
                for c in cs.get_root_campaigns():
                    if c.shorthand and c.shorthand.lower() == campaign_id.lower():
                        campaign = c
                        break
        if not campaign:
            return {"text": f"Campaign '{campaign_id}' not found."}

        status = cs.get_plan_status(campaign.id)
        total = status["total"]
        completed = status["completed"]
        in_progress = status["in_progress"]

        lines = [
            f"**{campaign.shorthand or campaign.display_name}** — {campaign.description}",
        ]
        if campaign.target:
            lines.append(f"Target: {campaign.target}")
        if campaign.progress:
            lines.append(f"Progress: {campaign.progress}")
        lines.append(
            f"Status: {campaign.status.value} · {completed}/{total} complete"
            f" · {in_progress} in progress"
        )
        lines.append("")

        TYPE_ICONS = {
            "imaging": "📷",
            "bench": "🧪",
            "genetics": "🧬",
            "analysis": "📊",
            "decision_point": "🚦",
        }
        STATUS_MARKS = {
            "planned": "○",
            "in_progress": "◑",
            "completed": "●",
            "skipped": "⊘",
            "blocked": "⊗",
        }

        # Show subcampaigns/phases with their items
        children = cs.get_subcampaigns(campaign.id)
        if children:
            for phase_idx, child in enumerate(children, 1):
                child_status = cs.get_plan_status(child.id)
                ct = child_status["total"]
                cc = child_status["completed"]
                phase_label = child.shorthand or child.display_name
                lines.append(f"**Phase {phase_idx}: {phase_label}** ({cc}/{ct})")

                items = cs.get_plan_items(campaign_id=child.id)
                items.sort(key=lambda x: x.phase_order)
                for task_idx, item in enumerate(items, 1):
                    icon = TYPE_ICONS.get(item.type.value, "📋")
                    mark = STATUS_MARKS.get(item.status.value, "?")
                    num = f"{phase_idx}.{task_idx}"
                    short_id = item.id[:8]
                    lines.append(f"  {mark} {icon} **{num}** {item.title}  `{short_id}`")
                    if item.imaging_spec and item.imaging_spec.strain:
                        spec = item.imaging_spec
                        details = []
                        if spec.strain:
                            details.append(spec.strain)
                        if spec.num_embryos:
                            details.append(f"{spec.num_embryos} embryos")
                        if spec.interval_s:
                            details.append(f"{spec.interval_s}s interval")
                        lines.append(f"      {' · '.join(details)}")
                lines.append("")
        else:
            # Items directly under this campaign (no phases)
            items = cs.get_plan_items(campaign_id=campaign.id)
            items.sort(key=lambda x: x.phase_order)
            for task_idx, item in enumerate(items, 1):
                icon = TYPE_ICONS.get(item.type.value, "📋")
                mark = STATUS_MARKS.get(item.status.value, "?")
                short_id = item.id[:8]
                lines.append(f"  {mark} {icon} **{task_idx}** {item.title}  `{short_id}`")

        # Next actions
        if status["next_actions"]:
            lines.append("")
            lines.append("**Next actions:**")
            for item in status["next_actions"][:5]:
                icon = TYPE_ICONS.get(item.type.value, "📋")
                lines.append(f"  → {icon} {item.title}")

        return {"text": "\n".join(lines)}

    def _get_history_data(self) -> dict:
        """Build structured conversation history."""
        history = self.agent.conversation_history[-20:]  # Last 20 messages
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
            prefix = "You" if role == "user" else "Agent"
            lines.append(f"  [{prefix}] {content}")

        return {"text": "\n".join(lines)}
