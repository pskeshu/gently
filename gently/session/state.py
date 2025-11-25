"""
Session state dataclasses for persistence

Defines the complete session state that can be saved and restored.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Optional, Any
import json
import uuid


@dataclass
class ConversationMessage:
    """
    A single message in the conversation history

    Stores both the message content and any tool interactions.
    """
    role: str  # "user", "assistant", "system"
    content: Any  # str or list of content blocks
    timestamp: datetime = field(default_factory=datetime.now)

    # Tool interactions (for assistant messages)
    tool_calls: Optional[List[Dict]] = None  # Tool calls made
    tool_results: Optional[List[Dict]] = None  # Results from tools

    # Image references (UIDs of images referenced in this message)
    image_refs: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """Serialize to dictionary"""
        return {
            'role': self.role,
            'content': self._serialize_content(self.content),
            'timestamp': self.timestamp.isoformat(),
            'tool_calls': self.tool_calls,
            'tool_results': self.tool_results,
            'image_refs': self.image_refs,
        }

    def _serialize_content(self, content: Any) -> Any:
        """Serialize content, handling both str and list of blocks"""
        if isinstance(content, str):
            return content
        elif isinstance(content, list):
            # Handle content blocks (from Claude API response)
            serialized = []
            for block in content:
                if hasattr(block, 'text'):
                    serialized.append({'type': 'text', 'text': block.text})
                elif hasattr(block, 'type'):
                    if block.type == 'tool_use':
                        serialized.append({
                            'type': 'tool_use',
                            'name': getattr(block, 'name', 'unknown'),
                            'input': getattr(block, 'input', {}),
                            'id': getattr(block, 'id', ''),
                        })
                    else:
                        # Try to convert to dict
                        try:
                            serialized.append(dict(block))
                        except:
                            serialized.append(str(block))
                elif isinstance(block, dict):
                    serialized.append(block)
                else:
                    serialized.append(str(block))
            return serialized
        else:
            return str(content)

    @classmethod
    def from_dict(cls, data: Dict) -> 'ConversationMessage':
        """Deserialize from dictionary"""
        return cls(
            role=data['role'],
            content=data['content'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            tool_calls=data.get('tool_calls'),
            tool_results=data.get('tool_results'),
            image_refs=data.get('image_refs', []),
        )

    def to_claude_format(self) -> Dict:
        """
        Convert to Claude API format for conversation history

        Returns the message in the format expected by Claude API.
        """
        return {
            'role': self.role,
            'content': self.content,
        }


@dataclass
class SessionState:
    """
    Complete session state that can be persisted and restored

    Contains everything needed to resume a copilot session exactly
    where it left off.
    """
    # Session identity
    session_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    created_at: datetime = field(default_factory=datetime.now)
    last_active: datetime = field(default_factory=datetime.now)

    # Human-readable name (optional, for easier identification)
    name: Optional[str] = None
    description: Optional[str] = None

    # Conversation history
    conversation: List[ConversationMessage] = field(default_factory=list)
    system_prompt: str = ""

    # Experiment state (serialized from ExperimentState)
    experiment_data: Dict = field(default_factory=dict)

    # Embryo states (embryo_id -> serialized EmbryoState)
    embryo_states: Dict[str, Dict] = field(default_factory=dict)

    # Detector configurations (detector_name -> config)
    detector_configs: Dict[str, Dict] = field(default_factory=dict)

    # Detection history (embryo_id -> list of detection results)
    detection_history: Dict[str, List[Dict]] = field(default_factory=dict)

    # Data lineage (UIDs of all data associated with this session)
    image_refs: List[str] = field(default_factory=list)
    volume_refs: List[str] = field(default_factory=list)
    analysis_refs: List[str] = field(default_factory=list)

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Serialize to dictionary for storage"""
        return {
            'session_id': self.session_id,
            'created_at': self.created_at.isoformat(),
            'last_active': self.last_active.isoformat(),
            'name': self.name,
            'description': self.description,
            'conversation': [msg.to_dict() for msg in self.conversation],
            'system_prompt': self.system_prompt,
            'experiment_data': self.experiment_data,
            'embryo_states': self.embryo_states,
            'detector_configs': self.detector_configs,
            'detection_history': self.detection_history,
            'image_refs': self.image_refs,
            'volume_refs': self.volume_refs,
            'analysis_refs': self.analysis_refs,
            'metadata': self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'SessionState':
        """Deserialize from dictionary"""
        return cls(
            session_id=data['session_id'],
            created_at=datetime.fromisoformat(data['created_at']),
            last_active=datetime.fromisoformat(data['last_active']),
            name=data.get('name'),
            description=data.get('description'),
            conversation=[ConversationMessage.from_dict(msg) for msg in data.get('conversation', [])],
            system_prompt=data.get('system_prompt', ''),
            experiment_data=data.get('experiment_data', {}),
            embryo_states=data.get('embryo_states', {}),
            detector_configs=data.get('detector_configs', {}),
            detection_history=data.get('detection_history', {}),
            image_refs=data.get('image_refs', []),
            volume_refs=data.get('volume_refs', []),
            analysis_refs=data.get('analysis_refs', []),
            metadata=data.get('metadata', {}),
        )

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string"""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> 'SessionState':
        """Deserialize from JSON string"""
        return cls.from_dict(json.loads(json_str))

    def add_message(
        self,
        role: str,
        content: Any,
        tool_calls: Optional[List[Dict]] = None,
        tool_results: Optional[List[Dict]] = None,
        image_refs: Optional[List[str]] = None,
    ):
        """Add a message to the conversation history"""
        msg = ConversationMessage(
            role=role,
            content=content,
            timestamp=datetime.now(),
            tool_calls=tool_calls,
            tool_results=tool_results,
            image_refs=image_refs or [],
        )
        self.conversation.append(msg)
        self.last_active = datetime.now()

    def get_conversation_for_claude(self) -> List[Dict]:
        """
        Get conversation in Claude API format

        Returns the conversation history formatted for the Claude API.
        """
        return [msg.to_claude_format() for msg in self.conversation]

    def get_summary(self) -> str:
        """Get a brief summary of the session"""
        embryo_count = len(self.embryo_states)
        msg_count = len(self.conversation)
        duration = self.last_active - self.created_at
        hours = duration.total_seconds() // 3600
        minutes = (duration.total_seconds() % 3600) // 60

        name_part = f" ({self.name})" if self.name else ""
        return (
            f"Session {self.session_id}{name_part}: "
            f"{embryo_count} embryos, {msg_count} messages, "
            f"{int(hours)}h {int(minutes)}m"
        )

    @property
    def embryo_count(self) -> int:
        """Number of embryos in this session"""
        return len(self.embryo_states)

    @property
    def message_count(self) -> int:
        """Number of messages in this session"""
        return len(self.conversation)
