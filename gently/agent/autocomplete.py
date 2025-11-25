"""
Autocomplete support for Rich CLI

Provides command completion for:
- Slash commands (/detectors, /status, /embryos, etc.)
- Embryo IDs (embryo_1, embryo_2, etc.)
- Detector names (hatching, comma, pretzel, etc.)
"""

from typing import List, Optional
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.document import Document


class CopilotCompleter(Completer):
    """
    Autocompleter for microscopy copilot CLI

    Provides intelligent completions for:
    - Slash commands
    - Embryo IDs
    - Detector names
    - Common phrases
    """

    # Slash commands
    SLASH_COMMANDS = [
        '/detectors',
        '/status',
        '/embryos',
        '/summary',
        '/help',
        '/clear',
        '/quit',
        '/pause',
        '/resume',
        '/stop',
        # Session commands
        '/sessions',
        '/save',
    ]

    # Common command patterns
    COMMON_PHRASES = [
        'What detectors do we have?',
        'Show me the detection summary',
        'Add a detector for',
        'Test the',
        'Enable',
        'Disable',
        'Remove',
        "What's the status?",
        'How is embryo',
        'Analyze embryo',
        'Skip embryo',
        'Resume embryo',
        'List all embryos',
        'Show me embryo',
        'Generate a prompt for detecting',
        'Start imaging',
        'Pause acquisition',
        'Resume acquisition',
    ]

    def __init__(self, copilot=None):
        """
        Initialize completer

        Parameters
        ----------
        copilot : MicroscopyCopilot, optional
            Copilot instance for dynamic completions (embryo IDs, detector names)
        """
        self.copilot = copilot

    def get_embryo_ids(self) -> List[str]:
        """Get list of embryo IDs from copilot"""
        if self.copilot and hasattr(self.copilot, 'experiment'):
            return list(self.copilot.experiment.embryos.keys())
        return []

    def get_detector_names(self) -> List[str]:
        """Get list of detector names from copilot"""
        if self.copilot and hasattr(self.copilot, 'detector_registry'):
            return [d.name for d in self.copilot.detector_registry.list_all()]
        return []

    def get_session_ids(self) -> List[str]:
        """Get list of session IDs from copilot"""
        if self.copilot and hasattr(self.copilot, 'list_sessions'):
            try:
                sessions = self.copilot.list_sessions()
                return [s.get('id', '') for s in sessions if s.get('id')]
            except:
                pass
        return []

    def get_completions(self, document: Document, complete_event):
        """
        Generate completions based on current input

        Parameters
        ----------
        document : Document
            Current document state
        complete_event : CompleteEvent
            Completion event

        Yields
        ------
        Completion
            Possible completions
        """
        text = document.text_before_cursor
        word = document.get_word_before_cursor(WORD=True)

        # Slash command completion
        if text.startswith('/'):
            # Check if completing /resume with session ID
            if text.startswith('/resume '):
                session_ids = self.get_session_ids()
                partial = text[8:]  # After "/resume "
                for session_id in session_ids:
                    if session_id.startswith(partial) or partial == '':
                        yield Completion(
                            session_id,
                            start_position=-len(partial),
                            display=session_id,
                            display_meta='session'
                        )
            else:
                # Regular slash command completion
                for cmd in self.SLASH_COMMANDS:
                    if cmd.startswith(text):
                        yield Completion(
                            cmd,
                            start_position=-len(text),
                            display=cmd,
                            display_meta='command'
                        )

        # Embryo ID completion
        elif 'embryo' in text.lower():
            embryo_ids = self.get_embryo_ids()
            for embryo_id in embryo_ids:
                if embryo_id.startswith(word):
                    yield Completion(
                        embryo_id,
                        start_position=-len(word),
                        display=embryo_id,
                        display_meta='embryo'
                    )
                # Also support "embryo 1" -> "embryo_1"
                elif embryo_id.startswith('embryo_') and word.isdigit():
                    embryo_num = embryo_id.split('_')[1]
                    if embryo_num.lstrip('0') == word:
                        yield Completion(
                            embryo_id,
                            start_position=-len(word),
                            display=embryo_id,
                            display_meta='embryo'
                        )

        # Detector name completion
        elif any(trigger in text.lower() for trigger in ['detector', 'test', 'enable', 'disable', 'remove']):
            detector_names = self.get_detector_names()
            for detector_name in detector_names:
                if detector_name.startswith(word):
                    yield Completion(
                        detector_name,
                        start_position=-len(word),
                        display=detector_name,
                        display_meta='detector'
                    )

        # Common phrase completion (only at start of line)
        elif len(text.strip()) < 20:  # Only suggest for short inputs
            for phrase in self.COMMON_PHRASES:
                if phrase.lower().startswith(text.lower().strip()):
                    yield Completion(
                        phrase,
                        start_position=-len(text.strip()),
                        display=phrase,
                        display_meta='suggestion'
                    )


def create_completer(copilot=None) -> CopilotCompleter:
    """
    Create autocompleter for copilot CLI

    Parameters
    ----------
    copilot : MicroscopyCopilot, optional
        Copilot instance for dynamic completions

    Returns
    -------
    CopilotCompleter
        Configured completer
    """
    return CopilotCompleter(copilot=copilot)
