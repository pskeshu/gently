"""
Databroker Tools

Tools for querying and retrieving data from Bluesky/Databroker.
"""

from typing import Dict, List

from ..tool_registry import tool, ToolCategory
from ..tool_helpers import require_copilot


@tool(
    name="list_runs",
    description="List recent Bluesky runs from Databroker",
    category=ToolCategory.DATA,
)
def list_runs(
    limit: int = 10,
    embryo_id: str = None,
    plan_name: str = None,
    context: Dict = None
) -> str:
    """List recent runs"""
    copilot = context.get('copilot')

    if not copilot or not copilot.databroker:
        return "Databroker not available"

    try:
        db = copilot.databroker

        query = {}
        if embryo_id:
            query['embryo_id'] = embryo_id
        if plan_name:
            query['plan_name'] = plan_name

        runs = list(db(**query))[:limit]

        if not runs:
            return "No runs found"

        lines = [f"Recent runs ({len(runs)}):", ""]

        for run_uid in runs:
            run = db[run_uid]
            start = run.metadata.get('start', {})
            lines.append(f"* {run_uid[:8]}...")
            lines.append(f"  Plan: {start.get('plan_name', 'unknown')}")
            lines.append(f"  Time: {start.get('time', 'unknown')}")
            if 'embryo_id' in start:
                lines.append(f"  Embryo: {start['embryo_id']}")
            lines.append("")

        return "\n".join(lines)

    except Exception as e:
        return f"Error listing runs: {str(e)}"


@tool(
    name="get_run_data",
    description="Get data from a specific Bluesky run",
    category=ToolCategory.DATA,
)
def get_run_data(
    run_id: str,
    data_keys: List[str] = None,
    stream: str = "primary",
    context: Dict = None
) -> str:
    """Get run data"""
    copilot = context.get('copilot')

    if not copilot or not copilot.databroker:
        return "Databroker not available"

    try:
        db = copilot.databroker

        if run_id.startswith('-'):
            run = db[int(run_id)]
        else:
            run = db[run_id]

        data = run.primary.read()

        if data_keys:
            data = {k: data[k] for k in data_keys if k in data}

        lines = [f"Run: {run.metadata['start']['uid'][:8]}...", ""]
        lines.append(f"Available keys: {list(data.keys())}")

        for key, values in data.items():
            shape = values.shape if hasattr(values, 'shape') else 'scalar'
            lines.append(f"  {key}: shape={shape}")

        return "\n".join(lines)

    except Exception as e:
        return f"Error getting run data: {str(e)}"


@tool(
    name="get_run_image",
    description="Get an image from a Bluesky run for analysis",
    category=ToolCategory.DATA,
)
async def get_run_image(
    run_id: str,
    detector: str = None,
    analyze: bool = False,
    analysis_prompt: str = None,
    context: Dict = None
) -> str:
    """Get run image"""
    copilot = context.get('copilot')

    if not copilot or not copilot.databroker:
        return "Databroker not available"

    try:
        db = copilot.databroker

        if run_id.startswith('-'):
            run = db[int(run_id)]
        else:
            run = db[run_id]

        data = run.primary.read()

        if not detector:
            for key in ['bottom_camera', 'camera', 'detector']:
                if key in data:
                    detector = key
                    break

        if detector not in data:
            return f"Detector '{detector}' not found. Available: {list(data.keys())}"

        image = data[detector]
        shape = image.shape if hasattr(image, 'shape') else 'unknown'

        result = f"Retrieved image from {detector}\nShape: {shape}"

        if analyze and analysis_prompt:
            analysis = await copilot._analyze_image_with_vision(
                image=image,
                prompt=analysis_prompt
            )
            result += f"\n\nAnalysis:\n{analysis}"

        return result

    except Exception as e:
        return f"Error getting image: {str(e)}"


@tool(
    name="search_runs",
    description="Search Databroker runs by metadata criteria",
    category=ToolCategory.DATA,
)
def search_runs(
    since: str = None,
    until: str = None,
    metadata: Dict = None,
    limit: int = 20,
    context: Dict = None
) -> str:
    """Search runs"""
    copilot = context.get('copilot')

    if not copilot or not copilot.databroker:
        return "Databroker not available"

    try:
        db = copilot.databroker

        query = metadata or {}

        if since:
            query['since'] = since
        if until:
            query['until'] = until

        runs = list(db(**query))[:limit]

        if not runs:
            return "No matching runs found"

        lines = [f"Found {len(runs)} runs:", ""]

        for run_uid in runs:
            run = db[run_uid]
            start = run.metadata.get('start', {})
            lines.append(f"* {run_uid[:8]}: {start.get('plan_name', 'unknown')}")

        return "\n".join(lines)

    except Exception as e:
        return f"Error searching runs: {str(e)}"
