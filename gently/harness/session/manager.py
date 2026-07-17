"""
SessionManager - Session persistence and lifecycle.

Extracted from agent.py to separate session management
from conversation and experiment orchestration.
"""

import json
import logging
import uuid

logger = logging.getLogger(__name__)


class SessionManager:
    """
    Manages session creation, saving, resuming, and listing.

    Works with FileStore for persistence. Does not hold references
    back to agent — receives data as parameters instead.
    """

    def __init__(self, store, storage_path):
        self.store = store
        self.storage_path = storage_path
        self._session_id: str | None = None

    @property
    def session_id(self) -> str | None:
        """Get current session ID (None before create_session())."""
        return self._session_id

    def create_session(self):
        """Create a new session in FileStore."""
        self._session_id = str(uuid.uuid4())[:8]
        self.store.create_session(self._session_id)
        logger.info(f"Created new session: {self._session_id}")

    def _resume_session(self, session_id: str, experiment):
        """
        Resume a session from FileStore.

        Restores embryo state onto the given experiment object and
        returns True if resumed successfully.

        Parameters
        ----------
        session_id : str
            Session ID to resume
        experiment : ExperimentState
            Experiment state to restore embryos into

        Returns
        -------
        tuple of (bool, list)
            (success, conversation_history)
        """
        session = self.store.get_session(session_id)
        if not session:
            logger.warning(f"Session {session_id} not found, creating new")
            self._session_id = session_id
            self.store.create_session(session_id)
            return False, []

        self._session_id = session_id

        # Resume strategy:
        #   - Conversation history: JSON snapshot only (DB doesn't store it)
        #   - Embryo state: JSON snapshot is primary (has full params like
        #     interval, num_slices, priority, etc.). DB embryo table fills
        #     in any embryos that exist in the DB but not the snapshot.
        snapshot = self.store.load_session_snapshot(session_id)
        conversation_history = []

        if snapshot:
            raw_history = snapshot.get("conversation_history", [])
            conversation_history = self.sanitize_loaded_messages(raw_history)

            experiment_data = snapshot.get("experiment_data", {})
            experiment.active_plan_item_id = experiment_data.get("active_plan_item_id")
            embryo_states = experiment_data.get("embryos", {})

            for embryo_id, embryo_data in embryo_states.items():
                pos = embryo_data.get("stage_position", {})
                experiment.add_embryo(
                    embryo_id=embryo_id,
                    position=pos,
                    calibration=embryo_data.get("calibration", {}),
                    user_label=embryo_data.get("user_label"),
                    uid=embryo_data.get("uid"),
                )
                embryo = experiment.embryos[embryo_id]
                embryo.nickname = embryo_data.get("nickname")
                embryo.interval_seconds = embryo_data.get("interval_seconds")
                embryo.num_slices = embryo_data.get("num_slices", 50)
                embryo.exposure_ms = embryo_data.get("exposure_ms", 10.0)
                embryo.priority = embryo_data.get("priority", "normal")
                embryo.timepoints_acquired = embryo_data.get("timepoints_acquired", 0)
                embryo.should_skip = embryo_data.get("should_skip", False)
                embryo.skip_reason = embryo_data.get("skip_reason")

        # Also load embryos from store's embryo table. FileStore returns
        # position_coarse / position_fine (with legacy position_x / position_y
        # backfilled into coarse on read), so both calibration stages survive
        # the resume.
        store_embryos = self.store.list_embryos(session_id)
        for e in store_embryos:
            eid = e["embryo_id"]
            if eid not in experiment.embryos:
                experiment.add_embryo(
                    embryo_id=eid,
                    position=e.get("position_coarse") or {},
                    position_fine=e.get("position_fine") or {},
                    calibration=json.loads(e["calibration"]) if e.get("calibration") else {},
                )

        self.store.touch_session(session_id)

        logger.info(f"Resumed session: {session_id}")
        return True, conversation_history

    def save_session(self, experiment, conversation_history, system_prompt) -> bool:
        """
        Save current session state to FileStore.

        Parameters
        ----------
        experiment : ExperimentState
            Current experiment state
        conversation_history : list
            Current conversation history
        system_prompt : str
            Current system prompt

        Returns
        -------
        bool
            True if saved successfully
        """
        if not self._session_id:
            return False
        try:
            self.store.save_session_snapshot(
                self._session_id,
                {
                    "conversation_history": self.serialize_messages(conversation_history),
                    "experiment_data": experiment.to_dict(),
                    "system_prompt": system_prompt,
                },
            )
            self._sync_embryos_to_db(experiment)
            self.store.touch_session(self._session_id)
            return True
        except Exception as e:
            logger.error(f"Failed to save session: {e}")
            return False

    def auto_save(self, experiment, conversation_history, system_prompt):
        """Auto-save session to FileStore (non-blocking, silent on error)."""
        if not self._session_id:
            return
        try:
            self.store.save_session_snapshot(
                self._session_id,
                {
                    "conversation_history": self.serialize_messages(conversation_history),
                    "experiment_data": experiment.to_dict(),
                    "system_prompt": system_prompt,
                },
            )
            self._sync_embryos_to_db(experiment)
            self.store.touch_session(self._session_id)
        except Exception:
            pass  # Silent fail for auto-save

    def _sync_embryos_to_db(self, experiment):
        """Sync in-memory embryo state (positions, calibration) to the DB."""
        for embryo_id, embryo in experiment.embryos.items():
            pos = embryo.stage_position or {}
            self.store.register_embryo(
                self._session_id,
                embryo_id,
                embryo_uid=getattr(embryo, "uid", None),
                nickname=getattr(embryo, "user_label", None),
                position_x=pos.get("x"),
                position_y=pos.get("y"),
                calibration=embryo.calibration,
            )

    def list_sessions(self) -> list[dict]:
        """
        List available sessions from FileStore.

        Returns
        -------
        list of dict
            Session summaries
        """
        return self.store.list_sessions()

    def resume_session(
        self, session_id: str, experiment, conversation_mgr, prompt_mgr_update_fn
    ) -> bool:
        """
        Resume a session (public interface for CLI).

        Saves current session first, then loads the target session.

        Parameters
        ----------
        session_id : str
            Session ID to resume
        experiment : ExperimentState
            Current experiment state
        conversation_mgr : ConversationManager
            Conversation manager to update history on
        prompt_mgr_update_fn : callable
            Callback to update system prompt after restore

        Returns
        -------
        bool
            True if resumed successfully
        """
        # Save current session before switching
        if self._session_id:
            self.save_session(
                experiment,
                conversation_mgr.conversation_history,
                "",  # system_prompt will be rebuilt
            )

        success, history = self._resume_session(session_id, experiment)
        if success:
            conversation_mgr.conversation_history = history

        prompt_mgr_update_fn()

        return success

    # ===== Message Serialization =====

    @staticmethod
    def sanitize_loaded_messages(messages: list[dict]) -> list[dict]:
        """Fix conversation history loaded from JSON snapshots.

        Old snapshots may contain content blocks that were serialized
        via ``default=str`` (e.g. ``"TextBlock(text='...', type='text')"``).
        These are invalid for the Claude API. This method drops any
        message whose content blocks aren't valid dicts or strings.
        """
        clean = []
        for msg in messages:
            content = msg.get("content")
            if content is None:
                continue
            if isinstance(content, str):
                clean.append(msg)
                continue
            if isinstance(content, list):
                valid_blocks: list[dict | str] = []
                for block in content:
                    if isinstance(block, dict):
                        valid_blocks.append(block)
                    elif isinstance(block, str):
                        if block.startswith(("TextBlock(", "ToolUseBlock(")):
                            continue
                        valid_blocks.append(block)
                if valid_blocks:
                    clean.append({**msg, "content": valid_blocks})
                continue
        return clean

    @staticmethod
    def serialize_messages(messages: list[dict]) -> list[dict]:
        """Convert conversation history to JSON-safe plain dicts.

        Anthropic SDK returns content blocks as objects (TextBlock,
        ToolUseBlock) that serialize via ``default=str`` into their
        repr strings. This converts everything to plain dicts so the
        history round-trips cleanly through JSON.
        """

        def _block_to_dict(block):
            if isinstance(block, dict):
                return block
            if isinstance(block, str):
                return block
            if hasattr(block, "model_dump"):
                return block.model_dump()
            if hasattr(block, "to_dict"):
                return block.to_dict()
            if hasattr(block, "type"):
                d = {"type": block.type}
                if block.type == "text" and hasattr(block, "text"):
                    d["text"] = block.text
                elif block.type == "tool_use":
                    d["id"] = getattr(block, "id", "")
                    d["name"] = getattr(block, "name", "")
                    d["input"] = getattr(block, "input", {})
                return d
            return str(block)

        serialized = []
        for msg in messages:
            content = msg.get("content")
            if isinstance(content, list):
                content = [_block_to_dict(b) for b in content]
            serialized.append({**msg, "content": content})
        return serialized
