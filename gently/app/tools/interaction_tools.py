"""
Interaction Tools

Tools for structured user interaction, including presenting choices
and collecting user input in a UX-friendly way.
"""

import json

from gently.harness.tools.registry import ToolCategory, ToolExample, tool

# Special marker for CLI to detect choice responses
CHOICE_RESPONSE_TYPE = "_user_choice_request"


@tool(
    name="ask_user_choice",
    description="""Present the user with selectable options instead of requiring typed input.
Use this when asking questions with discrete, enumerable options like:
- "Which session to import?" → list of available sessions
- "Which embryo to focus on?" → list of embryo IDs
- "Start the timelapse?" → Yes/No confirmation
- "Which algorithm?" → list of algorithm options

The CLI will render these as selectable options (arrow keys + enter) for better UX.
Returns the user's selected option(s).

IMPORTANT: Do NOT include a generic "Something else" or "Other" option — the UI
automatically appends a "Something else..." option with a free-text input field.
Only include specific, meaningful choices.""",
    category=ToolCategory.EXPERIMENT,
    requires_microscope=False,
    examples=[
        ToolExample(
            "Ask which session",
            {
                "question": "Which session to import from?",
                "options": [
                    {"id": "abc123", "label": "Today's session (4 embryos)"},
                    {"id": "def456", "label": "Yesterday (2 embryos)"},
                ],
            },
        ),
        ToolExample(
            "Yes/No confirmation",
            {
                "question": "Start the timelapse?",
                "options": [
                    {"id": "yes", "label": "Yes, start now"},
                    {"id": "no", "label": "No, cancel"},
                ],
            },
        ),
    ],
)
async def ask_user_choice(
    question: str,
    options: list[dict[str, str]],
    allow_multiple: bool = False,
    default_id: str | None = None,
    context: dict | None = None,
) -> str:
    """
    Present user with selectable options.

    Parameters
    ----------
    question : str
        The question to ask the user
    options : List[Dict[str, str]]
        List of options, each with 'id' and 'label' keys.
        - id: The value returned when selected
        - label: The display text shown to user
        Optional keys:
        - description: Additional detail shown below label
        - disabled: If true, option is shown but not selectable
    allow_multiple : bool
        If True, user can select multiple options
    default_id : str, optional
        ID of the option to pre-select
    context : dict
        Execution context

    Returns
    -------
    str
        JSON string with special _type marker for CLI to parse.
        CLI will render options and return user's selection.
    """
    # Validate options
    if not options or len(options) < 1:
        return "Error: Must provide at least one option"

    for i, opt in enumerate(options):
        if "id" not in opt or "label" not in opt:
            return f"Error: Option {i} missing required 'id' or 'label' field"

    # Return special format that CLI will intercept and render
    choice_request = {
        "_type": CHOICE_RESPONSE_TYPE,
        "question": question,
        "options": options,
        "allow_multiple": allow_multiple,
        "default_id": default_id,
    }

    return json.dumps(choice_request)


def parse_choice_response(response: str) -> dict | None:
    """
    Parse a tool response to check if it's a choice request.

    Parameters
    ----------
    response : str
        Tool response string

    Returns
    -------
    Dict or None
        Parsed choice request if valid, None otherwise
    """
    try:
        data = json.loads(response)
        if isinstance(data, dict) and data.get("_type") == CHOICE_RESPONSE_TYPE:
            return data
    except (json.JSONDecodeError, TypeError):
        pass
    return None


# Helper functions for common choice patterns


def yes_no_options(yes_label: str = "Yes", no_label: str = "No") -> list[dict[str, str]]:
    """Generate standard Yes/No options"""
    return [
        {"id": "yes", "label": yes_label},
        {"id": "no", "label": no_label},
    ]


def yes_no_cancel_options(
    yes_label: str = "Yes", no_label: str = "No", cancel_label: str = "Cancel"
) -> list[dict[str, str]]:
    """Generate Yes/No/Cancel options"""
    return [
        {"id": "yes", "label": yes_label},
        {"id": "no", "label": no_label},
        {"id": "cancel", "label": cancel_label},
    ]


def embryo_options(agent) -> list[dict[str, str]]:
    """Generate options from available embryos"""
    options = []
    for eid, embryo in agent.experiment.embryos.items():
        label = eid
        if embryo.nickname:
            label = f"{eid} ({embryo.nickname})"
        elif embryo.user_label:
            label = f"{eid} ({embryo.user_label})"

        pos = embryo.stage_position
        if pos:
            label += f" @ ({pos.get('x', 0):.0f}, {pos.get('y', 0):.0f})"

        options.append({"id": eid, "label": label})

    return options


def session_options(sessions: list[dict]) -> list[dict[str, str]]:
    """Generate options from available sessions"""
    options = []
    for session in sessions:
        sid = session.get("session_id", session.get("id", "unknown"))
        embryo_count = session.get("embryo_count", 0)
        message_count = session.get("message_count", 0)
        last_active = session.get("last_active", "unknown")

        label = f"{sid[:8]} - {embryo_count} embryos, {message_count} messages"
        if last_active != "unknown":
            label += f" (last: {last_active})"

        options.append({"id": sid, "label": label, "description": f"Session {sid}"})

    return options
