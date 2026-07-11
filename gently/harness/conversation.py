"""
ConversationManager - LLM conversation, tool execution, and token tracking.

Extracted from agent.py to separate conversation mechanics from
experiment orchestration and session persistence.
"""

import asyncio
import json
import logging
import re
import time
from typing import Any

from ..settings import settings

logger = logging.getLogger(__name__)


def _extend_tool_calls(out: list[dict[str, Any]], content_blocks) -> None:
    """Append every tool_use block in content_blocks to out.

    Tolerates absent attributes (some SDK versions / mock objects) so it
    never crashes the live agent on a content-shape surprise.
    """
    if not content_blocks:
        return
    for block in content_blocks:
        try:
            if getattr(block, "type", None) != "tool_use":
                continue
            out.append(
                {
                    "name": getattr(block, "name", None),
                    "input": getattr(block, "input", None),
                    "id": getattr(block, "id", None),
                }
            )
        except Exception:
            continue


class ConversationManager:
    """
    Manages Claude API conversations, tool execution, and token tracking.

    This class owns the conversation history and all LLM interaction logic:
    - Streaming and non-streaming API calls with retry
    - Tool execution via the tool registry
    - Token usage tracking and cost estimation
    - Quick response shortcuts (no API call needed)
    """

    def __init__(self, client, model, tool_registry):
        self.claude = client
        self.model = model
        self._tool_registry = tool_registry

        # Conversation state
        self.conversation_history: list[dict] = []

        # Token counters
        self.total_input_tokens: int = 0
        self.total_output_tokens: int = 0
        self.api_call_count: int = 0
        self.cache_creation_tokens: int = 0
        self.cache_read_tokens: int = 0

        # Set by agent after construction
        self.interaction_logger = None
        self.choice_handler = None
        self.context_store = None  # for tool_label
        self._tool_context: dict | None = None  # default execution context

        # Decision capture for orchestrator A/B testing. Set by the agent
        # alongside the EventCapture once the session folder is known. None
        # = no capture, so tests / harnesses without a session still work.
        self.decision_log = None

    # ===== Quick Response =====

    def try_quick_response(
        self, message: str, experiment, mode: str, enter_plan_fn, exit_plan_fn
    ) -> str | None:
        """
        Answer simple queries from state without LLM call.

        Parameters
        ----------
        message : str
            User message
        experiment : ExperimentState
            Current experiment state
        mode : str
            Current mode ("run" or "plan")
        enter_plan_fn : callable
            Callback to enter plan mode
        exit_plan_fn : callable
            Callback to exit plan mode

        Returns
        -------
        str or None
            Quick response if possible, None if Claude is needed
        """
        message_lower = message.lower()

        # Status query
        if "status" in message_lower and len(message.split()) < 5:
            return experiment.get_summary()

        # Plan mode switching via natural language
        plan_enter_phrases = (
            "plan mode",
            "enter plan",
            "switch to plan",
            "let's plan",
            "design an experiment",
        )
        plan_exit_phrases = ("exit plan", "leave plan", "back to run", "run mode")

        if mode != "plan" and any(p in message_lower for p in plan_enter_phrases):
            enter_plan_fn()
            if len(message.split()) <= 6:
                return "Switched to plan mode. I'm now your experimental design collaborator."
            return None

        if mode == "plan" and any(p in message_lower for p in plan_exit_phrases):
            exit_plan_fn()
            if len(message.split()) <= 6:
                return "Back to run mode."
            return None

        # Simple commands
        if message_lower in ["stop", "pause", "halt"]:
            # Note: run_engine is on agent, but this path is rarely used
            # and the experiment status update is the important part
            experiment.acquisition_status = "paused"
            return "Acquisition paused. What would you like to do next?"

        return None

    # ===== Thinking Decision =====

    def should_use_thinking(self, message: str, mode: str) -> bool:
        """
        Determine if extended thinking should be enabled for this message.

        Auto-triggers for plan mode, explicit thinking requests,
        calibration operations, image analysis, and complex queries.
        """
        if mode == "plan":
            return True

        msg_lower = message.lower()

        if re.search(r"\bthink(ing)?\b", message, re.IGNORECASE):
            return True
        if re.search(r"\bcalibrat", msg_lower):
            return True
        if re.search(r"\b(plan|timelapse|time-lapse|acquisition)\b", msg_lower):
            return True
        if re.search(
            r"\b(analy[sz]e|look at|check|inspect|review).*(image|volume|embryo)", msg_lower
        ):
            return True
        if re.search(r"\b(all|every|each)\s+(embryo|sample)", msg_lower):
            return True
        if re.search(
            r"\b(first|then|after|next|finally)\b.*\b(first|then|after|next|finally)\b", msg_lower
        ):
            return True
        if re.search(r"\b(why|problem|issue|error|wrong|fail|debug|troubleshoot)", msg_lower):
            return True

        return False

    # ===== Non-Streaming API Call =====

    async def _create_with_refusal_fallback(self, api_kwargs):
        """messages.create with main-tier resilience: if the model rejects the
        request with a 400 (e.g. Fable 5 under <30-day org data retention, or
        unavailable) OR declines it (stop_reason="refusal", empty content), retry
        the SAME request once on the fallback model (Opus 4.8) — so gently keeps
        working whether or not Fable 5 is currently serviceable. The moment the
        org retention is fixed, Fable 5 serves with no code change."""
        from anthropic import BadRequestError

        fb = settings.models.refusal_fallback
        model = api_kwargs.get("model")
        try:
            response = await self._call_api_with_retry(self.claude.messages.create, **api_kwargs)
        except BadRequestError:
            if not fb or fb == model:
                raise
            logger.warning("Model %s rejected the request (400); falling back to %s", model, fb)
            return await self._call_api_with_retry(
                self.claude.messages.create, **{**api_kwargs, "model": fb}
            )
        if response.stop_reason == "refusal" and fb and fb != model:
            logger.warning("Model %s declined the turn; retrying on %s", model, fb)
            response = await self._call_api_with_retry(
                self.claude.messages.create, **{**api_kwargs, "model": fb}
            )
        return response

    async def call_claude(
        self, user_message: str, system_prompt, tools, mode: str, auto_save_fn
    ) -> str:
        """
        Call Claude API with full context and tool access (non-streaming).

        Parameters
        ----------
        user_message : str
            User message
        system_prompt : list
            Cached system prompt blocks
        tools : list
            Tool schemas for current mode
        mode : str
            Current mode ("run" or "plan")
        auto_save_fn : callable
            Callback for auto-saving session

        Returns
        -------
        str
            Claude's response text
        """
        start_time = time.time()

        use_thinking = self.should_use_thinking(user_message, mode)

        # Start interaction logging
        interaction = None
        if self.interaction_logger:
            interaction = self.interaction_logger.start_interaction(
                user_prompt=user_message,
                system_state={
                    "acquisition_status": "unknown",
                },
            )

        # Snapshot inputs for decision capture BEFORE the tool loop starts
        # appending to conversation_history. This is the state shadow
        # candidates would need to reproduce production's input — same
        # system_prompt and same starting messages.
        decision_prompt_hash = None
        if self.decision_log is not None:
            try:
                from gently.eval import prompt_hash as _prompt_hash

                decision_prompt_hash = _prompt_hash(
                    system_prompt,
                    list(self.conversation_history),
                )
            except Exception:
                logger.exception("Failed to compute decision prompt_hash")

        tool_calls_collected: list[dict[str, Any]] = []
        assistant_message = ""
        error_occurred = None

        try:
            api_kwargs = {
                "model": self.model,
                "system": system_prompt,
                "messages": self.conversation_history,
                "tools": tools,
                "max_tokens": 16000 if use_thinking else 4096,
            }
            if use_thinking:
                # Fable 5 / Opus 4.8 reject thinking budget_tokens (400) — thinking
                # is adaptive; control depth via effort instead of a token budget.
                api_kwargs["output_config"] = {"effort": "high" if mode == "plan" else "medium"}

            response = await self._create_with_refusal_fallback(api_kwargs)
            self._track_token_usage(response)
            _extend_tool_calls(tool_calls_collected, response.content)

            # Process tool calls
            while response.stop_reason == "tool_use":
                tool_results = await self._execute_tools_with_logging(response.content, interaction)

                self.conversation_history.append({"role": "assistant", "content": response.content})
                self.conversation_history.append({"role": "user", "content": tool_results})

                api_kwargs["messages"] = self.conversation_history
                response = await self._create_with_refusal_fallback(api_kwargs)
                self._track_token_usage(response)
                _extend_tool_calls(tool_calls_collected, response.content)

            # Extract text response. Fable 5 may refuse (stop_reason="refusal")
            # with empty content — surface it instead of returning blank.
            if response.stop_reason == "refusal":
                assistant_message = (
                    "(The request was declined by the model's safety system. Try rephrasing.)"
                )
            else:
                assistant_message = ""
                for block in response.content:
                    if hasattr(block, "text"):
                        assistant_message += block.text

            self.conversation_history.append({"role": "assistant", "content": response.content})

        except Exception as e:
            import traceback

            error_occurred = str(e)
            error_tb = traceback.format_exc()
            assistant_message = f"Error: {error_occurred}"

            if interaction and self.interaction_logger:
                self.interaction_logger.complete_interaction(
                    interaction=interaction,
                    assistant_response=assistant_message,
                    total_duration_seconds=time.time() - start_time,
                    error=error_occurred,
                    error_traceback=error_tb,
                )
            self._write_production_decision(
                user_message=user_message,
                tool_calls=tool_calls_collected,
                response_text=assistant_message,
                duration_ms=(time.time() - start_time) * 1000.0,
                prompt_hash_value=decision_prompt_hash,
                error=error_occurred,
            )
            raise

        if interaction and self.interaction_logger:
            self.interaction_logger.complete_interaction(
                interaction=interaction,
                assistant_response=assistant_message,
                total_duration_seconds=time.time() - start_time,
            )

        self._write_production_decision(
            user_message=user_message,
            tool_calls=tool_calls_collected,
            response_text=assistant_message,
            duration_ms=(time.time() - start_time) * 1000.0,
            prompt_hash_value=decision_prompt_hash,
            error=None,
        )

        auto_save_fn()

        return assistant_message

    def _write_production_decision(
        self,
        *,
        user_message: str,
        tool_calls: list[dict[str, Any]],
        response_text: str,
        duration_ms: float,
        prompt_hash_value: str | None,
        error: str | None,
    ) -> None:
        """Persist one production Decision row (best-effort).

        Failures here are swallowed — decision capture must never break
        the live agent. The DecisionLog itself is also tolerant of
        serialisation errors.
        """
        if self.decision_log is None:
            return
        try:
            from datetime import datetime

            from gently.eval import Decision, DecisionTrigger

            self.decision_log.append(
                Decision(
                    timestamp=datetime.now(),
                    agent="production",
                    trigger=DecisionTrigger.USER_MESSAGE,
                    trigger_detail=(user_message or "")[:200],
                    tool_calls=tool_calls,
                    response_text=response_text,
                    prompt_hash=prompt_hash_value,
                    duration_ms=duration_ms,
                    error=error,
                )
            )
        except Exception:
            logger.exception("Failed to write production Decision")

    # ===== Dry-Run Tool Call (Benchmarking) =====

    async def get_tool_call(self, user_message: str, system_prompt, tools) -> dict | None:
        """
        Get what tool Claude would call without executing it (dry-run mode).

        Used for benchmarking tool selection accuracy.

        Parameters
        ----------
        user_message : str
            User query to analyze
        system_prompt : list
            Cached system prompt
        tools : list
            Tool schemas

        Returns
        -------
        dict or None
            Tool call info: {name, input, input_tokens, output_tokens, latency_ms}
        """
        start_time = time.time()

        messages = self.conversation_history.copy()
        messages.append({"role": "user", "content": user_message})

        try:
            api_kwargs = {
                "model": self.model,
                "system": system_prompt,
                "messages": messages,
                "tools": tools,
                "max_tokens": 4096,
            }

            response = await self._call_api_with_retry(self.claude.messages.create, **api_kwargs)

            latency_ms = (time.time() - start_time) * 1000

            input_tokens = getattr(response.usage, "input_tokens", 0)
            output_tokens = getattr(response.usage, "output_tokens", 0)

            # A refusal returns empty content — treat as "no tool call".
            if response.stop_reason == "refusal" or not response.content:
                return None
            for block in response.content:
                if block.type == "tool_use":
                    return {
                        "name": block.name,
                        "input": block.input,
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "latency_ms": latency_ms,
                    }

            return None

        except Exception as e:
            logger.error(f"Error in get_tool_call: {e}")
            raise

    # ===== Tool Execution =====

    async def _execute_tools_with_logging(self, content_blocks, interaction) -> list[dict]:
        """
        Execute Claude's tool calls with interaction logging.

        Parameters
        ----------
        content_blocks : list
            Content blocks from Claude response
        interaction : InteractionRecord
            Current interaction being logged

        Returns
        -------
        list of dict
            Tool result content blocks
        """
        from gently.app.tools.interaction_tools import CHOICE_RESPONSE_TYPE

        results = []

        for block in content_blocks:
            if block.type == "tool_use":
                start_time = time.time()
                is_error = False
                error_message = None

                try:
                    result = await self._execute_single_tool(block.name, block.input)

                    if self.choice_handler and isinstance(result, str):
                        try:
                            choice_data = json.loads(result)
                            if (
                                isinstance(choice_data, dict)
                                and choice_data.get("_type") == CHOICE_RESPONSE_TYPE
                            ):
                                user_selection = await self.choice_handler(choice_data)
                                result = user_selection
                        except (json.JSONDecodeError, TypeError):
                            pass

                except Exception as e:
                    result = f"Error: {str(e)}"
                    is_error = True
                    error_message = str(e)

                duration = time.time() - start_time

                if interaction and self.interaction_logger:
                    self.interaction_logger.record_tool_call(
                        interaction=interaction,
                        tool_name=block.name,
                        tool_input=block.input,
                        result=result,
                        duration_seconds=duration,
                        is_error=is_error,
                        error_message=error_message,
                    )

                results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                        "is_error": is_error,
                    }
                )

        return results

    # ===== Streaming API Call =====

    async def call_claude_stream(self, system_prompt, tools, tool_label_fn, auto_save_fn):
        """
        Call Claude API with streaming enabled.

        Parameters
        ----------
        system_prompt : list
            Cached system prompt blocks
        tools : list
            Tool schemas
        tool_label_fn : callable
            Function to generate human-readable tool labels
        auto_save_fn : callable
            Callback for auto-saving session

        Yields
        ------
        dict
            Chunks as they arrive from Claude
        """
        from anthropic import APIStatusError, BadRequestError

        # Live streaming: a worker thread drains the SDK's (blocking) stream and
        # pushes each event onto an asyncio queue as it arrives, so this coroutine
        # can yield text/thinking deltas in real time instead of collecting the
        # whole turn first (which left the UI on a blank spinner for the entire
        # turn). thinking=summarized surfaces the model's reasoning during the
        # wait. The full assistant content (incl. thinking blocks) is replayed from
        # final_message below, so the tool-loop continuation stays valid.
        _DONE = object()

        async def _stream_live(model, sink):
            """Stream one attempt live: yield delta dicts as they arrive; record
            events / final_message / error / full_text into `sink`."""
            loop = asyncio.get_running_loop()
            queue: asyncio.Queue = asyncio.Queue()
            state: dict = {}

            def worker():
                try:
                    with self.claude.messages.stream(
                        model=model,
                        system=system_prompt,
                        messages=self.conversation_history,
                        tools=tools,
                        max_tokens=16000,
                        # Adaptive thinking with a streamed, human-readable summary —
                        # this is what fills the "working…" wait. Opus 4.8 defaults to
                        # display="omitted" (empty thinking text), so it must be set.
                        thinking={"type": "adaptive", "display": "summarized"},
                        output_config={"effort": "medium"},
                    ) as stream:
                        for event in stream:
                            loop.call_soon_threadsafe(queue.put_nowait, event)
                        state["final"] = stream.get_final_message()
                except BaseException as exc:  # noqa: BLE001 — re-raised to caller below
                    state["error"] = exc
                finally:
                    loop.call_soon_threadsafe(queue.put_nowait, _DONE)

            task = asyncio.create_task(asyncio.to_thread(worker))
            events: list = []
            full_text: list = []
            while True:
                item = await queue.get()
                if item is _DONE:
                    break
                events.append(item)
                if item.type != "content_block_delta":
                    continue
                delta = item.delta
                dtype = getattr(delta, "type", None)
                if dtype == "thinking_delta":
                    chunk = getattr(delta, "thinking", "") or ""
                    if chunk:
                        yield {"type": "thinking", "text": chunk}
                elif dtype == "text_delta" or hasattr(delta, "text"):
                    chunk = getattr(delta, "text", "") or ""
                    if chunk:
                        full_text.append(chunk)
                        yield {"type": "text", "text": chunk}
            await task
            sink["events"] = events
            sink["full_text"] = full_text
            sink["final"] = state.get("final")
            sink["error"] = state.get("error")

        max_retries = 3
        retry_delay = 1.0
        fb = settings.models.refusal_fallback
        model_in_use = self.model
        sink: dict = {}

        for attempt in range(max_retries):
            sink = {}
            yielded_any = False
            async for chunk in _stream_live(model_in_use, sink):
                yielded_any = True
                yield chunk
            err = sink.get("error")
            if err is None:
                break
            # Fable 5 under <30-day data retention (or unavailable) rejects with a
            # 400 — fall back to Opus 4.8. Only safe before any partial was streamed.
            if isinstance(err, BadRequestError) and fb and fb != model_in_use and not yielded_any:
                logger.warning(
                    "Stream model %s rejected the request (400); falling back to %s",
                    model_in_use,
                    fb,
                )
                model_in_use = fb
                continue
            if isinstance(err, APIStatusError):
                error_type = getattr(err, "body", {})
                if isinstance(error_type, dict):
                    error_type = error_type.get("error", {}).get("type", "")
                overloaded = (
                    error_type in ("overloaded_error", "rate_limit_error")
                    or "overloaded" in str(err).lower()
                )
                if overloaded and attempt < max_retries - 1 and not yielded_any:
                    wait_time = retry_delay * (2**attempt)
                    logger.warning(
                        "API overloaded, retrying in %.1fs (attempt %d/%d)",
                        wait_time,
                        attempt + 1,
                        max_retries,
                    )
                    yield {
                        "type": "text",
                        "text": f"\n*[API busy, retrying in {wait_time:.0f}s...]*\n",
                    }
                    await asyncio.sleep(wait_time)
                    continue
            raise err
        else:
            raise RuntimeError("API overloaded after multiple retries")

        final_message = sink["final"]
        full_text = sink["full_text"]
        self._track_token_usage(final_message)

        # Refusal → retry on the fallback model. Re-streaming live is only safe when
        # the refusal came before any visible output (pre-output refusals carry empty
        # content, so nothing was yielded); otherwise we keep the partial we showed.
        if final_message.stop_reason == "refusal" and fb and model_in_use != fb and not full_text:
            logger.warning("Model %s declined the streamed turn; retrying on %s", model_in_use, fb)
            sink = {}
            async for chunk in _stream_live(fb, sink):
                yield chunk
            model_in_use = fb
            final_message = sink["final"]
            full_text = sink["full_text"]
            self._track_token_usage(final_message)

        # Last resort: if even the fallback declined, surface it and stop.
        if final_message.stop_reason == "refusal":
            logger.warning("Claude declined the request (model=%s)", model_in_use)
            yield {
                "type": "text",
                "text": "(The request was declined by the model's safety system. Try rephrasing.)",
            }
            return

        # Diagnostic: per-response counts. DEBUG, not WARNING — stop_reason=tool_use
        # with matching tool blocks is normal; the genuine anomaly is the
        # logger.error below (tool blocks present but stop_reason != tool_use).
        tool_block_count = sum(
            1 for b in final_message.content if hasattr(b, "type") and b.type == "tool_use"
        )
        logger.debug(
            "Claude response: stop_reason=%s, content_blocks=%d, "
            "tool_use_blocks=%d, tools_passed=%d, model=%s",
            final_message.stop_reason,
            len(final_message.content),
            tool_block_count,
            len(tools),
            model_in_use,
        )
        if tool_block_count > 0 and final_message.stop_reason != "tool_use":
            logger.error(
                "BUG: Claude returned %d tool_use blocks but stop_reason=%s (expected 'tool_use')",
                tool_block_count,
                final_message.stop_reason,
            )

        # Detect fake XML tool calls in text (Claude writing tool_use as text)
        joined_text = "".join(full_text)
        if "<tool_use>" in joined_text or "<function_calls>" in joined_text:
            logger.error(
                "DETECTED: Claude wrote XML tool tags as plain text instead of "
                "using API tool_use mechanism. stop_reason=%s, text_preview=%.200s",
                final_message.stop_reason,
                joined_text[:200],
            )

        response_content = final_message.content

        # Process tool calls if any
        if final_message.stop_reason == "tool_use":
            # Brief pause so the TUI can flush the streamed text before
            # the tool_start event commits it into <Static>.  Without this,
            # Ink on Windows leaves "ghost" text in the dynamic area.
            await asyncio.sleep(0.05)

            tool_results = []

            # Concurrency fast-path: run a turn's tool calls in parallel when ALL of
            # them are non-hardware and non-interactive (e.g. several strain / paper /
            # lab-history lookups). Any microscope action or ask_user_choice in the
            # batch falls back to the serial path below, so we never race hardware or
            # an interactive prompt, and ordering of stateful ops is preserved.
            tool_blocks = [b for b in response_content if getattr(b, "type", None) == "tool_use"]
            _interactive = {"ask_user_choice"}
            # Only parallelize genuinely read-only tools (independent lookups). Mutating
            # tools (create_/update_/delete_/set_…) must stay serial — they share state
            # (e.g. a campaign's plan file) and are order-dependent, so concurrent runs
            # could race or corrupt it.
            _readonly_prefixes = (
                "search_",
                "read_",
                "query_",
                "get_",
                "list_",
                "recall_",
                "find_",
                "fetch_",
                "lookup_",
            )

            def _parallel_safe(b):
                td = self._tool_registry.get(b.name)
                return (
                    td is not None
                    and not td.requires_microscope
                    and b.name not in _interactive
                    and b.name.startswith(_readonly_prefixes)
                )

            handled_parallel = False
            if len(tool_blocks) > 1 and all(_parallel_safe(b) for b in tool_blocks):
                handled_parallel = True
                starts = {b.id: time.time() for b in tool_blocks}
                for b in tool_blocks:
                    yield {
                        "type": "tool_start",
                        "tool_name": b.name,
                        "tool_input": b.input,
                        "tool_label": tool_label_fn(b.name, b.input),
                    }
                gathered = await asyncio.gather(
                    *[self._execute_single_tool(b.name, b.input) for b in tool_blocks],
                    return_exceptions=True,
                )
                for b, res in zip(tool_blocks, gathered, strict=True):
                    if isinstance(res, BaseException):
                        is_error_flag = True
                        result_text = f"Error: {res}"
                        tool_results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": b.id,
                                "content": result_text,
                                "is_error": True,
                            }
                        )
                    else:
                        is_error_flag = False
                        result_text = res if isinstance(res, str) else str(res)
                        tool_results.append(
                            {"type": "tool_result", "tool_use_id": b.id, "content": res}
                        )
                    result_summary = next(
                        (ln.strip() for ln in (result_text or "").splitlines() if ln.strip()),
                        "",
                    )
                    if len(result_summary) > 140:
                        result_summary = result_summary[:139] + "…"
                    result_full = result_text or ""
                    if len(result_full) > 4000:
                        result_full = result_full[:4000] + "\n…(truncated)"
                    yield {
                        "type": "tool_call",
                        "tool_name": b.name,
                        "tool_input": b.input,
                        "duration": time.time() - starts[b.id],
                        "result_summary": result_summary,
                        "result_full": result_full,
                        "is_error": is_error_flag,
                    }

            for block in response_content:
                if handled_parallel:
                    break
                if hasattr(block, "type") and block.type == "tool_use":
                    start_time = time.time()

                    yield {
                        "type": "tool_start",
                        "tool_name": block.name,
                        "tool_input": block.input,
                        "tool_label": tool_label_fn(block.name, block.input),
                    }

                    is_error_flag = False
                    result_text = ""
                    try:
                        tool_result = await self._execute_single_tool(block.name, block.input)

                        if isinstance(tool_result, str):
                            try:
                                from gently.app.tools.interaction_tools import CHOICE_RESPONSE_TYPE

                                choice_data = json.loads(tool_result)
                                if (
                                    isinstance(choice_data, dict)
                                    and choice_data.get("_type") == CHOICE_RESPONSE_TYPE
                                ):
                                    user_selection = yield {
                                        "type": "choice_request",
                                        "choice_data": choice_data,
                                    }
                                    tool_result = user_selection or "cancelled"
                            except (json.JSONDecodeError, TypeError):
                                pass

                        result_text = (
                            tool_result if isinstance(tool_result, str) else str(tool_result)
                        )
                        tool_results.append(
                            {"type": "tool_result", "tool_use_id": block.id, "content": tool_result}
                        )
                    except Exception as e:
                        is_error_flag = True
                        result_text = f"Error: {str(e)}"
                        tool_results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": result_text,
                                "is_error": True,
                            }
                        )

                    # First non-empty line of the result, trimmed — gives the chat
                    # UI a one-line summary so the operator can see what a tool did
                    # (or didn't do), not just that it ran.
                    result_summary = next(
                        (ln.strip() for ln in (result_text or "").splitlines() if ln.strip()),
                        "",
                    )
                    if len(result_summary) > 140:
                        result_summary = result_summary[:139] + "…"

                    # Full result (bounded) so the UI's expandable tool card can
                    # show what the tool actually returned — not just the 140-char
                    # one-liner. The web client caps/scrolls this further; keep the
                    # streamed payload sane.
                    result_full = result_text or ""
                    if len(result_full) > 4000:
                        result_full = result_full[:4000] + "\n…(truncated)"

                    yield {
                        "type": "tool_call",
                        "tool_name": block.name,
                        "tool_input": block.input,
                        "duration": time.time() - start_time,
                        "result_summary": result_summary,
                        "result_full": result_full,
                        "is_error": is_error_flag,
                    }

            self.conversation_history.append({"role": "assistant", "content": response_content})
            self.conversation_history.append({"role": "user", "content": tool_results})

            auto_save_fn()

            # Recurse for next response
            recursive_gen = self.call_claude_stream(
                system_prompt, tools, tool_label_fn, auto_save_fn
            )
            sent_value = None
            try:
                while True:
                    if sent_value is None:
                        chunk = await recursive_gen.__anext__()
                    else:
                        chunk = await recursive_gen.asend(sent_value)
                        sent_value = None
                    sent_value = yield chunk
            except StopAsyncIteration:
                pass

        else:
            # No tool calls - add final message to history
            self.conversation_history.append({"role": "assistant", "content": response_content})
            auto_save_fn()

    # ===== Tool Label =====

    def tool_label(self, tool_name: str, tool_input: dict) -> str:
        """Build a human-readable label for a tool call.

        Used in tool_start chunks so the TUI shows biologist-friendly
        summaries instead of raw UUIDs.
        """
        inp = tool_input or {}

        # Plan tools: resolve campaign/item IDs to names
        campaign_id = inp.get("campaign_id")
        if campaign_id and self.context_store:
            campaign = self.context_store.get_campaign(campaign_id)
            if campaign:
                campaign_label = campaign.shorthand or campaign.description
                if tool_name in (
                    "propose_plan",
                    "get_plan_status",
                    "export_plan",
                    "snapshot_plan",
                    "list_plan_versions",
                ):
                    return campaign_label
                if tool_name == "create_campaign" and inp.get("parent_id"):
                    return f"phase under {campaign_label}"
                if tool_name == "create_plan_item":
                    title = inp.get("title", "")
                    phase = inp.get("phase_number")
                    prefix = f"P{phase}" if phase else campaign_label
                    return f"{prefix}: {title}" if title else prefix
                if tool_name == "delete_phase":
                    phase = inp.get("phase_number")
                    return f"{campaign_label} phase {phase}" if phase else campaign_label
                if tool_name == "restore_plan_version":
                    vn = inp.get("version_number")
                    return f"{campaign_label} → v{vn}" if vn else campaign_label

        # Item reference tools
        item_ref = inp.get("item_ref") or inp.get("ref") or inp.get("item_id")
        if item_ref and tool_name in (
            "get_plan_item",
            "update_plan_item",
            "delete_plan_item",
            "move_plan_item",
        ):
            if self.context_store:
                item = self.context_store.resolve_plan_item(str(item_ref), campaign_id=campaign_id)
                if item:
                    return item.title
            return str(item_ref)

        # Research tools
        if tool_name == "search_literature":
            return inp.get("query", "")
        if tool_name == "search_strains":
            return inp.get("gene", "") or inp.get("query", "")
        if tool_name == "query_lab_history":
            return inp.get("query", "")

        # Campaign creation
        if tool_name == "create_campaign":
            return inp.get("shorthand") or inp.get("description", "")

        # Generic fallback
        for key in ("title", "description", "query", "question"):
            val = inp.get(key)
            if val and isinstance(val, str):
                return val[:60]

        return ""

    async def _execute_single_tool(
        self, tool_name: str, tool_input: dict, context: dict | None = None
    ) -> str:
        """Execute a single tool call using the tool registry.

        Parameters
        ----------
        tool_name : str
            Name of tool to execute
        tool_input : dict
            Tool input parameters
        context : dict, optional
            Execution context (agent, client, etc.).
            If None, uses self._tool_context (set by agent).
        """
        ctx = context or self._tool_context
        return await self._tool_registry.execute(tool_name, tool_input, ctx)

    # ===== Token Tracking =====

    def _track_token_usage(self, response):
        """Track token usage from API response, including cache metrics."""
        if hasattr(response, "usage"):
            usage = response.usage
            self.total_input_tokens += usage.input_tokens
            self.total_output_tokens += usage.output_tokens
            self.api_call_count += 1
            self.cache_creation_tokens += getattr(usage, "cache_creation_input_tokens", 0)
            self.cache_read_tokens += getattr(usage, "cache_read_input_tokens", 0)

    @property
    def current_context_tokens(self) -> int:
        """Estimate current context window size in tokens."""
        system_tokens = 0  # Caller manages system prompt

        tool_tokens = 10000  # ~10K tokens for 65 tools

        conv_chars = 0
        for msg in self.conversation_history:
            content = msg.get("content", "")
            if isinstance(content, str):
                conv_chars += len(content)
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        conv_chars += len(str(block.get("text", "")))
                    elif hasattr(block, "text"):
                        conv_chars += len(str(block.text))
                    else:
                        conv_chars += len(str(block))
            else:
                conv_chars += len(str(content))

        conv_tokens = conv_chars // 4

        return system_tokens + tool_tokens + conv_tokens

    @property
    def token_usage_summary(self) -> str:
        """Get human-readable token usage summary."""
        cache_read = self.cache_read_tokens
        cache_created = self.cache_creation_tokens
        total_input = self.total_input_tokens + cache_read + cache_created
        total = total_input + self.total_output_tokens

        input_cost = self.total_input_tokens * 0.003 / 1000
        cache_read_cost = cache_read * 0.0003 / 1000
        cache_write_cost = cache_created * 0.006 / 1000
        output_cost = self.total_output_tokens * 0.015 / 1000
        total_cost = input_cost + cache_read_cost + cache_write_cost + output_cost

        cost_without_cache = (total_input * 0.003 + self.total_output_tokens * 0.015) / 1000
        savings = cost_without_cache - total_cost

        summary = (
            f"Tokens: {total:,} total ({total_input:,} in, {self.total_output_tokens:,} out) | "
            f"API calls: {self.api_call_count} | Est. cost: ${total_cost:.3f}"
        )
        if cache_read > 0:
            summary += f" (cache saved ${savings:.3f})"
        return summary

    # ===== API Retry Logic =====

    async def _call_api_with_retry(self, api_func, *args, max_retries=3, **kwargs):
        """
        Call an API function with retry logic for transient errors.

        Parameters
        ----------
        api_func : callable
            The API function to call
        max_retries : int
            Maximum number of retry attempts

        Returns
        -------
        The API response

        Raises
        ------
        Exception
            If all retries fail
        """
        from anthropic import APIStatusError

        retry_delay = 1.0

        for attempt in range(max_retries):
            try:
                return await asyncio.to_thread(api_func, *args, **kwargs)
            except APIStatusError as e:
                error_type = getattr(e, "body", {})
                if isinstance(error_type, dict):
                    error_type = error_type.get("error", {}).get("type", "")

                is_retryable = (
                    error_type in ("overloaded_error", "rate_limit_error")
                    or "overloaded" in str(e).lower()
                    or "rate_limit" in str(e).lower()
                )

                if is_retryable and attempt < max_retries - 1:
                    wait_time = retry_delay * (2**attempt)
                    logger.warning(
                        f"API error ({error_type}), retrying in {wait_time:.1f}s "
                        f"(attempt {attempt + 1}/{max_retries})"
                    )
                    await asyncio.sleep(wait_time)
                    continue

                raise

        raise RuntimeError(f"API call failed after {max_retries} retries")
