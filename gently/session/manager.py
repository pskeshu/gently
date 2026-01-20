"""
Session manager for persistence and restoration

Handles saving, loading, and listing sessions.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, TYPE_CHECKING

from .state import SessionState, ConversationMessage

if TYPE_CHECKING:
    from ..agent.state import ExperimentState, EmbryoState


logger = logging.getLogger(__name__)


class SessionManager:
    """
    Manages session persistence and restoration

    Sessions are stored as:
    1. JSON files in the sessions directory (primary, for easy inspection)
    2. Optionally in Databroker (future, for lineage with data)

    Auto-saves on significant actions like:
    - Image/volume acquisition
    - Embryo detection
    - Calibration changes
    - Detector configuration changes
    """

    def __init__(
        self,
        sessions_dir: Path = Path("./sessions"),
        auto_save: bool = True,
    ):
        """
        Parameters
        ----------
        sessions_dir : Path
            Directory to store session JSON files
        auto_save : bool
            Whether to auto-save on significant actions
        """
        self.sessions_dir = Path(sessions_dir)
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self.auto_save = auto_save

        self._current_session: Optional[SessionState] = None
        self._dirty = False  # Track if session has unsaved changes

    @property
    def current_session(self) -> Optional[SessionState]:
        """Get current session"""
        return self._current_session

    @property
    def session_id(self) -> Optional[str]:
        """Get current session ID"""
        return self._current_session.session_id if self._current_session else None

    def create_session(
        self,
        name: Optional[str] = None,
        description: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> SessionState:
        """
        Create a new session

        Parameters
        ----------
        name : str, optional
            Human-readable name for the session
        description : str, optional
            Description of the session
        metadata : dict, optional
            Additional metadata

        Returns
        -------
        SessionState
            The new session
        """
        session = SessionState(
            name=name,
            description=description,
            metadata=metadata or {},
        )
        self._current_session = session
        self._dirty = False

        logger.info(f"Created new session: {session.session_id}")

        # Save initial state
        self.save_session()

        return session

    def save_session(self, session: Optional[SessionState] = None) -> bool:
        """
        Save session to disk

        Parameters
        ----------
        session : SessionState, optional
            Session to save (defaults to current)

        Returns
        -------
        bool
            True if saved successfully
        """
        session = session or self._current_session
        if not session:
            logger.warning("No session to save")
            return False

        try:
            # Update last active time
            session.last_active = datetime.now()

            # Save to JSON file
            session_file = self.sessions_dir / f"{session.session_id}.json"
            with open(session_file, 'w') as f:
                f.write(session.to_json(indent=2))

            self._dirty = False
            logger.info(f"Saved session {session.session_id} to {session_file}")
            return True

        except Exception as e:
            logger.error(f"Failed to save session: {e}")
            return False

    def load_session(self, session_id: str) -> Optional[SessionState]:
        """
        Load a session by ID

        Parameters
        ----------
        session_id : str
            Session ID to load

        Returns
        -------
        SessionState or None
            The loaded session, or None if not found
        """
        session_file = self.sessions_dir / f"{session_id}.json"

        if not session_file.exists():
            logger.warning(f"Session file not found: {session_file}")
            return None

        try:
            with open(session_file, 'r') as f:
                data = json.load(f)

            session = SessionState.from_dict(data)
            self._current_session = session
            self._dirty = False

            logger.info(f"Loaded session {session_id}: {session.get_summary()}")
            return session

        except Exception as e:
            logger.error(f"Failed to load session {session_id}: {e}")
            return None

    def list_sessions(self) -> List[Dict]:
        """
        List all available sessions

        Returns
        -------
        list of dict
            Session summaries with id, name, embryo_count, last_active
        """
        sessions = []

        for session_file in self.sessions_dir.glob("*.json"):
            try:
                with open(session_file, 'r') as f:
                    data = json.load(f)

                sessions.append({
                    'session_id': data.get('session_id', session_file.stem),
                    'name': data.get('name'),
                    'embryo_count': len(data.get('embryo_states', {})),
                    'message_count': len(data.get('conversation', [])),
                    'created_at': data.get('created_at'),
                    'last_active': data.get('last_active'),
                })
            except Exception as e:
                logger.warning(f"Failed to read session file {session_file}: {e}")

        # Sort by last_active (most recent first)
        sessions.sort(
            key=lambda s: s.get('last_active') or '',
            reverse=True
        )

        return sessions

    def delete_session(self, session_id: str) -> bool:
        """
        Delete a session

        Parameters
        ----------
        session_id : str
            Session ID to delete

        Returns
        -------
        bool
            True if deleted successfully
        """
        session_file = self.sessions_dir / f"{session_id}.json"

        if not session_file.exists():
            logger.warning(f"Session file not found: {session_file}")
            return False

        try:
            session_file.unlink()

            # Clear current session if it was deleted
            if self._current_session and self._current_session.session_id == session_id:
                self._current_session = None

            logger.info(f"Deleted session {session_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete session {session_id}: {e}")
            return False

    # ===== State synchronization methods =====

    def sync_from_copilot(
        self,
        conversation_history: List[Dict],
        experiment: 'ExperimentState',
        detector_registry=None,
        system_prompt: str = "",
    ):
        """
        Sync session state from copilot

        Call this to update session state from copilot's current state.

        Parameters
        ----------
        conversation_history : list of dict
            Copilot's conversation history
        experiment : ExperimentState
            Copilot's experiment state
        detector_registry : DetectorRegistry, optional
            Copilot's detector registry
        system_prompt : str
            Current system prompt
        """
        if not self._current_session:
            logger.warning("No current session to sync")
            return

        session = self._current_session

        # Sync conversation (convert raw dicts to ConversationMessage)
        session.conversation = []
        for msg in conversation_history:
            session.conversation.append(ConversationMessage(
                role=msg.get('role', 'unknown'),
                content=msg.get('content', ''),
                timestamp=datetime.now(),  # We don't have original timestamps
            ))

        # Sync experiment state
        session.experiment_data = experiment.to_dict() if hasattr(experiment, 'to_dict') else {}

        # Sync embryo states
        session.embryo_states = {}
        for embryo_id, embryo in experiment.embryos.items():
            if hasattr(embryo, 'to_dict'):
                session.embryo_states[embryo_id] = embryo.to_dict()
            else:
                session.embryo_states[embryo_id] = {
                    'id': embryo_id,
                    'stage_position': getattr(embryo, 'stage_position', {}),
                    'calibration': getattr(embryo, 'calibration', {}),
                }

            # Sync detection history
            if hasattr(embryo, 'detection_results'):
                session.detection_history[embryo_id] = []
                for detector_name, results in embryo.detection_results.items():
                    for result in results:
                        session.detection_history[embryo_id].append({
                            'detector': detector_name,
                            **result
                        })

        # Sync detector configs
        if detector_registry:
            session.detector_configs = {}
            for detector in detector_registry.list_all():
                session.detector_configs[detector.name] = detector.to_dict() if hasattr(detector, 'to_dict') else {}

        # Sync system prompt
        session.system_prompt = system_prompt

        self._dirty = True

    def sync_to_copilot(self) -> Dict:
        """
        Get state to restore to copilot

        Returns a dict with:
        - conversation_history: List of dicts for copilot.conversation_history
        - experiment_data: Dict to restore ExperimentState
        - embryo_states: Dict of embryo data
        - detector_configs: Dict of detector configurations

        Returns
        -------
        dict
            State to restore
        """
        if not self._current_session:
            return {}

        session = self._current_session

        return {
            'conversation_history': [
                msg.to_claude_format() for msg in session.conversation
            ],
            'experiment_data': session.experiment_data,
            'embryo_states': session.embryo_states,
            'detector_configs': session.detector_configs,
            'detection_history': session.detection_history,
            'system_prompt': session.system_prompt,
        }

    def update_state(
        self,
        conversation: List[Dict],
        experiment: Dict,
        system_prompt: str = "",
    ):
        """
        Update session state from copilot (simplified API for auto-save)

        Parameters
        ----------
        conversation : list of dict
            Copilot's conversation history (raw dicts)
        experiment : dict
            Experiment state as dict (from experiment.to_dict())
        system_prompt : str
            Current system prompt
        """
        if not self._current_session:
            logger.warning("No current session to update")
            return

        session = self._current_session

        # Update conversation
        session.conversation = []
        for msg in conversation:
            session.conversation.append(ConversationMessage(
                role=msg.get('role', 'unknown'),
                content=msg.get('content', ''),
                timestamp=datetime.now(),
            ))

        # Update experiment data
        session.experiment_data = experiment

        # Update embryo states from experiment dict
        session.embryo_states = experiment.get('embryos', {})

        # Update system prompt
        session.system_prompt = system_prompt

        self._dirty = True

    # ===== Auto-save trigger methods =====

    def mark_significant_action(self, action_type: str):
        """
        Mark that a significant action occurred, triggering auto-save

        Parameters
        ----------
        action_type : str
            Type of action: "acquisition", "detection", "calibration",
            "embryo_change", "detector_config"
        """
        if not self.auto_save or not self._current_session:
            return

        self._dirty = True
        logger.debug(f"Significant action: {action_type}")

        # Auto-save
        self.save_session()

    def add_message(
        self,
        role: str,
        content: Any,
        tool_calls: Optional[List[Dict]] = None,
        tool_results: Optional[List[Dict]] = None,
    ):
        """
        Add a message to current session's conversation

        Parameters
        ----------
        role : str
            Message role (user, assistant, system)
        content : any
            Message content
        tool_calls : list, optional
            Tool calls made
        tool_results : list, optional
            Tool results
        """
        if not self._current_session:
            return

        self._current_session.add_message(
            role=role,
            content=content,
            tool_calls=tool_calls,
            tool_results=tool_results,
        )
        self._dirty = True

    def add_image_ref(self, uid: str):
        """Add an image UID to the session's references"""
        if self._current_session:
            if uid not in self._current_session.image_refs:
                self._current_session.image_refs.append(uid)
                self._dirty = True

    def add_volume_ref(self, uid: str):
        """Add a volume UID to the session's references"""
        if self._current_session:
            if uid not in self._current_session.volume_refs:
                self._current_session.volume_refs.append(uid)
                self._dirty = True

    def add_analysis_ref(self, uid: str):
        """Add an analysis UID to the session's references"""
        if self._current_session:
            if uid not in self._current_session.analysis_refs:
                self._current_session.analysis_refs.append(uid)
                self._dirty = True

    # ===== Utility methods =====

    def get_recent_sessions(self, limit: int = 5) -> List[Dict]:
        """
        Get most recent sessions

        Parameters
        ----------
        limit : int
            Maximum number of sessions to return

        Returns
        -------
        list of dict
            Session summaries
        """
        return self.list_sessions()[:limit]

    def session_exists(self, session_id: str) -> bool:
        """Check if a session exists"""
        session_file = self.sessions_dir / f"{session_id}.json"
        return session_file.exists()

    def get_session_file_path(self, session_id: str) -> Path:
        """Get path to session file"""
        return self.sessions_dir / f"{session_id}.json"
