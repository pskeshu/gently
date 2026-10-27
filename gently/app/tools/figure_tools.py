"""Figure tools: the agent pools what sessions measured, and draws it.

pool_rows   — the rows a session tool measured across a selector: project,
              phase, tag. Each row carries its session and the selector's
              label as its condition.
draw_figure — matplotlib code the agent writes, run over those rows, filed
              under the project as a PNG. The agent decides the form; the
              tool only provides the data, the axes and the filing.
"""

import json
import logging
import traceback

import numpy as np

from gently.harness.tools.helpers import require_agent
from gently.harness.tools.registry import ToolCategory, ToolExample, tool
from gently.tools.figure import figure_path, figure_url, slug
from gently.tools.select import select_sessions
from gently.tools.twitch_hatch import SULSTON_TWITCH_TO_HATCH_MIN, session_report

logger = logging.getLogger(__name__)

# Which tools can be pooled, and how a session answers for each.
POOLERS = {"twitch_hatch": session_report}
REFERENCES = {"twitch_hatch": {"Sulston": SULSTON_TWITCH_TO_HATCH_MIN}}

STRIP_PLOT = """\\
# rows: list of dicts; ax: the axes; ref: reference values by name
cols = sorted({r['session_name'] for r in rows if r['minutes'] is not None})
for i, c in enumerate(cols):
    ys = [r['minutes'] for r in rows if r['session_name'] == c and r['minutes'] is not None]
    xs = np.full(len(ys), i) + np.linspace(-0.15, 0.15, len(ys))
    ax.scatter(xs, ys, s=28, color='C0', zorder=3)
    ax.hlines(np.median(ys), i - 0.25, i + 0.25, color='C0', linewidth=2)
for name, v in ref.items():
    ax.axhline(v, linestyle=':', color='0.5')
    ax.text(0.99, v, f'{name} {v:g}', transform=ax.get_yaxis_transform(),
            ha='right', va='bottom', color='0.4')
ax.set_xticks(range(len(cols)), cols)
plt.setp(ax.get_xticklabels(), rotation=15, ha='right')
ax.set_ylabel('minutes')
"""


def _pool(agent, tool_name: str, project, phase, tag) -> tuple[dict, str | None]:
    pooler = POOLERS.get(tool_name)
    if pooler is None:
        return {}, f"Error: no pooler for tool {tool_name!r}; known: {sorted(POOLERS)}"
    store = getattr(agent, "store", None)
    cs = getattr(agent, "context_store", None)
    if store is None or cs is None:
        return {}, "Error: store or context store unavailable"
    picked = select_sessions(store, cs, project=project, phase=phase, tag=tag)
    # Nights often share a name; a column must not. Add the date when they do.
    names = [s.get("name") or s["session_id"] for s in picked["sessions"]]
    rows = []
    for s in picked["sessions"]:
        sid = s["session_id"]
        name = s.get("name") or sid
        if names.count(name) > 1:
            name = f"{name} {sid[:4]}"
        for i, r in enumerate(pooler(store, sid), 1):
            rows.append(
                {
                    **r,
                    "n": i,
                    "session_id": sid,
                    "session_name": name,
                    "condition": picked["label"],
                }
            )
    return {"label": picked["label"], "sessions": picked["sessions"], "rows": rows}, None


@tool(
    name="pool_rows",
    description=(
        "The rows a session tool measured, pooled across the sessions a selector "
        "picks: a project (root campaign id or shorthand), a phase within it, a tag "
        "on the session, or any combination (they intersect). Each row carries its "
        "session_id, session_name and the selector's label as 'condition'. Tools: "
        "twitch_hatch (per embryo: twitch_frame, hatch_frame, twitch_at, hatch_at, "
        "minutes from first twitch to hatching, temp_c over the interval, "
        "vs_sulston_min)."
    ),
    category=ToolCategory.DATA,
    examples=[
        ToolExample(
            user_query="What twitch-to-hatch times do we have across the control project?",
            tool_input={"tool": "twitch_hatch", "project": "t2h"},
        )
    ],
)
async def pool_rows(
    tool: str = "twitch_hatch",
    project: str | None = None,
    phase: str | None = None,
    tag: str | None = None,
    context: dict | None = None,
) -> str:
    agent, err = require_agent(context)
    if err:
        return err
    pooled, err = _pool(agent, tool, project, phase, tag)
    if err:
        return err
    return json.dumps(
        {
            "label": pooled["label"],
            "sessions": [s["session_id"] for s in pooled["sessions"]],
            "rows": pooled["rows"],
        },
        default=str,
    )


@tool(
    name="draw_figure",
    description=(
        "Draw a figure from pooled rows with matplotlib code you write, and file it "
        "under the project as a PNG. The code runs with these names bound: rows (the "
        "pooled rows, list of dicts, as pool_rows returns), ax and fig (a prepared "
        "matplotlib axes and figure), plt, np, and ref (the tool's reference values by "
        "name, e.g. {'Sulston': 370.0} for twitch_hatch). Do not call plt.show() or "
        "savefig; the tool saves. Returns the figure's URL; put it in a note to keep "
        "it. Omit code for the default: a strip plot of minutes per night with "
        "medians and the reference line."
    ),
    category=ToolCategory.ANALYSIS,
    examples=[
        ToolExample(
            user_query="Plot twitch to hatch across the control nights against Sulston",
            tool_input={"project": "t2h", "title": "Twitch to hatch, control nights"},
        ),
        ToolExample(
            user_query="Histogram of twitch-to-hatch minutes for the whole project",
            tool_input={
                "project": "t2h",
                "title": "Twitch to hatch, histogram",
                "code": (
                    "ys = [r['minutes'] for r in rows if r['minutes'] is not None]\n"
                    "ax.hist(ys, bins=12); ax.axvline(ref['Sulston'], linestyle=':')\n"
                    "ax.set_xlabel('minutes'); ax.set_ylabel('embryos')"
                ),
            },
        ),
    ],
)
async def draw_figure(
    project: str,
    title: str,
    code: str | None = None,
    tool: str = "twitch_hatch",
    phase: str | None = None,
    tag: str | None = None,
    name: str | None = None,
    context: dict | None = None,
) -> str:
    agent, err = require_agent(context)
    if err:
        return err
    pooled, err = _pool(agent, tool, project, phase, tag)
    if err:
        return err
    cs = agent.context_store
    c = cs.resolve_campaign(project)
    if c is None:
        return f"Error: project {project!r} not found"
    root = cs.root_campaign_id(c.id)
    if not pooled["rows"]:
        return f"No rows under {pooled['label'] or project!r}; nothing drawn."

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4), dpi=120)
    scope = {
        "rows": pooled["rows"],
        "ax": ax,
        "fig": fig,
        "plt": plt,
        "np": np,
        "ref": dict(REFERENCES.get(tool, {})),
    }
    try:
        # The agent's own plotting code, in-process: it already drives the rig.
        exec(code or STRIP_PLOT, scope)  # noqa: S102
        ax.set_title(title)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        fig.tight_layout()
        path = figure_path(cs, root, name or title)
        fig.savefig(path)
    except Exception:
        return "Error: the plotting code failed:\n" + traceback.format_exc(limit=3)
    finally:
        plt.close(fig)
    return json.dumps(
        {
            "url": figure_url(root, name or title),
            "name": slug(name or title),
            "rows": len(pooled["rows"]),
            "label": pooled["label"],
            "reference": REFERENCES.get(tool, {}),
        }
    )
