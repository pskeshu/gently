"""
Plan validation — check an experimental plan for errors and warnings.

Validates hardware limits, stage consistency, duration estimates,
detector validity, missing controls, dependency cycles, and completeness.
"""

import logging

from ...tools.registry import ToolCategory, ToolExample, tool

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hardware limits (from gently/hardware/dispim/description.py)
# ---------------------------------------------------------------------------

HARDWARE_LIMITS = {
    "num_slices": (10, 200),
    "exposure_ms": (5.0, 100.0),
    "laser_power_pct": (0.0, 100.0),
    "interval_s": (10, None),  # minimum 10s, no hard max
    "piezo_amplitude_um": (None, 200.0),  # max ±200 μm
}

# Stage timing at 20°C from biology.py (minutes from fertilisation)
STAGE_TIMING_20C = {
    "early": 0,
    "bean": 350,
    "comma": 400,
    "1.5fold": 450,
    "2fold": 500,
    "pretzel": 550,
    "hatching": 800,
    "hatched": 830,
}

# Temperature scaling factors (relative to 20°C)
TEMP_SCALE = {
    15.0: 24.0 / 14.0,  # ~1.71×  slower
    20.0: 1.0,
    25.0: 10.0 / 14.0,  # ~0.71×  faster
}

CONTROL_KEYWORDS = {"control", "wildtype", "n2", "wt", "wild-type", "wild type"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_temp_factor(temperature_c: float | None) -> float:
    """Return scaling factor for developmental timing at given temperature."""
    if temperature_c is None:
        return 1.0
    # Interpolate linearly between known points
    if temperature_c <= 15.0:
        return TEMP_SCALE[15.0]
    if temperature_c >= 25.0:
        return TEMP_SCALE[25.0]
    if temperature_c <= 20.0:
        frac = (temperature_c - 15.0) / 5.0
        return TEMP_SCALE[15.0] + frac * (TEMP_SCALE[20.0] - TEMP_SCALE[15.0])
    frac = (temperature_c - 20.0) / 5.0
    return TEMP_SCALE[20.0] + frac * (TEMP_SCALE[25.0] - TEMP_SCALE[20.0])


def _check_dependency_cycles(items) -> list[str]:
    """DFS-based cycle detection on the dependency graph."""
    # Build adjacency list: item_id -> list of dependency IDs
    adj: dict[str, list[str]] = {}
    id_to_title: dict[str, str] = {}
    for item in items:
        adj[item.id] = list(item.depends_on)
        id_to_title[item.id] = item.title

    WHITE, GRAY, BLACK = 0, 1, 2
    color: dict[str, int] = {nid: WHITE for nid in adj}
    cycles: list[str] = []
    path: list[str] = []

    def dfs(node: str):
        if node not in color:
            return  # dependency points outside this set
        color[node] = GRAY
        path.append(node)
        for dep in adj.get(node, []):
            if dep not in color:
                continue
            if color[dep] == GRAY:
                # Found a cycle — report it
                cycle_start = path.index(dep)
                cycle_nodes = path[cycle_start:]
                names = [id_to_title.get(n, n) for n in cycle_nodes]
                cycles.append(
                    f"Dependency cycle: {' → '.join(names)} → {id_to_title.get(dep, dep)}"
                )
            elif color[dep] == WHITE:
                dfs(dep)
        path.pop()
        color[node] = BLACK

    for node in list(adj.keys()):
        if color.get(node) == WHITE:
            dfs(node)

    return cycles


def _stage_order(stage_name: str) -> int | None:
    """Get ordinal position of a stage, or None if unrecognised."""
    from gently_perception.organism import CELEGANS

    stages = CELEGANS.stages
    aliases = {
        "3fold": "pretzel",
        "threefold": "pretzel",
        "1.5-fold": "1.5fold",
        "2-fold": "2fold",
    }
    normed = stage_name.lower().replace("-", "").replace(" ", "")
    name = aliases.get(normed, normed)
    # Match against stages list (handle dot removal: "15fold" -> "1.5fold")
    for i, s in enumerate(stages):
        if name == s or name == s.replace(".", ""):
            return i
    return None


def _normalise_stage(name: str) -> str | None:
    """Normalise a stage name to canonical form, or None."""
    from gently_perception.organism import CELEGANS

    stages = CELEGANS.stages
    low = name.lower().strip()
    for s in stages:
        if low == s or low.replace("-", "").replace(" ", "") == s.replace(".", ""):
            return s
    aliases = {"3fold": "pretzel", "threefold": "pretzel", "3-fold": "pretzel"}
    return aliases.get(low.replace(" ", ""))


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------


@tool(
    name="validate_plan",
    description=(
        "Validate an experimental plan for errors and warnings. Checks "
        "hardware limits, stage consistency, duration estimates, detector "
        "presets, missing controls, dependency cycles, and completeness. "
        "Returns a structured report of errors (blocking) and warnings."
    ),
    category=ToolCategory.UTILITY,
    examples=[
        ToolExample(
            user_query="Check this plan for problems",
            tool_input={"campaign_id": "nrf-2026"},
        ),
    ],
)
async def validate_plan(
    campaign_id: str,
    context: dict | None = None,
) -> str:
    """Validate a plan and return errors/warnings."""
    agent = context.get("agent") if context else None
    if not agent or not hasattr(agent, "context_store") or not agent.context_store:
        return "Error: Context store not available"

    store = agent.context_store
    campaign = store.get_campaign(campaign_id)
    if not campaign:
        return f"Campaign '{campaign_id}' not found"

    # Gather all items (including sub-campaigns)
    items = store.get_plan_items(campaign_id=campaign_id, include_children=True)
    if not items:
        return f"Campaign '{campaign.description}' has no plan items to validate."

    errors: list[str] = []
    warnings: list[str] = []

    # Load detector presets for validation
    try:
        from gently.organisms import get_organism

        org = get_organism()
        presets_mod = __import__(
            f"gently.organisms.{org.ORGANISM_NAME}.detector_presets",
            fromlist=["get_detector_presets"],
        )
        valid_detectors = set(presets_mod.get_detector_presets().keys())
    except ImportError:
        valid_detectors = set()

    # ------------------------------------------------------------------
    # Per-item checks
    # ------------------------------------------------------------------
    has_control = False

    for item in items:
        label = f"[{item.type.value}] '{item.title}'"

        # Check for control mentions
        text_blob = " ".join(
            filter(
                None,
                [
                    item.title,
                    item.description,
                    item.outcome,
                ],
            )
        ).lower()
        if item.imaging_spec:
            text_blob += (
                " "
                + " ".join(
                    filter(
                        None,
                        [
                            item.imaging_spec.strain,
                            item.imaging_spec.genotype,
                            item.imaging_spec.reporter,
                            item.imaging_spec.success_criteria,
                        ],
                    )
                ).lower()
            )
        if any(kw in text_blob for kw in CONTROL_KEYWORDS):
            has_control = True

        # Imaging-specific checks
        spec = store.resolve_imaging_spec(item) if item.imaging_spec else None
        if item.type.value == "imaging" and spec:
            # Hardware limits
            for field_name, (lo, hi) in HARDWARE_LIMITS.items():
                val = getattr(spec, field_name, None)
                if val is None:
                    continue
                if lo is not None and val < lo:
                    errors.append(f"{label}: {field_name}={val} below minimum {lo}")
                if hi is not None and val > hi:
                    errors.append(f"{label}: {field_name}={val} exceeds maximum {hi}")

            # Stage consistency
            if spec.start_stage and spec.stop_condition:
                start_ord = _stage_order(spec.start_stage)
                stop_ord = _stage_order(spec.stop_condition)
                if start_ord is not None and stop_ord is not None:
                    if start_ord >= stop_ord:
                        errors.append(
                            f"{label}: start_stage '{spec.start_stage}' is not "
                            f"before stop_condition '{spec.stop_condition}'"
                        )

            # Duration estimate vs biology
            if spec.estimated_duration_h and spec.start_stage and spec.stop_condition:
                start_t = STAGE_TIMING_20C.get(_normalise_stage(spec.start_stage) or "")
                stop_t = STAGE_TIMING_20C.get(_normalise_stage(spec.stop_condition) or "")
                if start_t is not None and stop_t is not None:
                    factor = _get_temp_factor(spec.temperature_c)
                    expected_min = (stop_t - start_t) * factor
                    expected_h = expected_min / 60.0
                    if expected_h > 0:
                        ratio = spec.estimated_duration_h / expected_h
                        if ratio < 0.5 or ratio > 2.0:
                            warnings.append(
                                f"{label}: estimated_duration_h={spec.estimated_duration_h} "
                                f"differs significantly from biology estimate "
                                f"~{expected_h:.1f}h ({spec.start_stage} → {spec.stop_condition}"
                                f" at {spec.temperature_c or 20}°C)"
                            )

            # Detector validity
            if spec.detectors and valid_detectors:
                for det in spec.detectors:
                    if det not in valid_detectors:
                        errors.append(
                            f"{label}: detector '{det}' is not a valid preset. "
                            f"Available: {', '.join(sorted(valid_detectors))}"
                        )

            # Completeness — key fields for imaging items
            missing_fields = []
            if not spec.strain:
                missing_fields.append("strain")
            if not spec.num_slices:
                missing_fields.append("num_slices")
            if not spec.interval_s:
                missing_fields.append("interval_s")
            if not spec.stop_condition and not spec.target_window:
                missing_fields.append("stop_condition or target_window")
            if missing_fields:
                warnings.append(
                    f"{label}: missing imaging spec fields: {', '.join(missing_fields)}"
                )

    # ------------------------------------------------------------------
    # Plan-level checks
    # ------------------------------------------------------------------

    # Missing controls
    if not has_control:
        warnings.append(
            "No plan item mentions a control condition (control, wildtype, N2, WT). "
            "Consider adding a control group."
        )

    # Dependency cycles
    cycle_errors = _check_dependency_cycles(items)
    for cyc in cycle_errors:
        errors.append(cyc)

    # ------------------------------------------------------------------
    # Format report
    # ------------------------------------------------------------------
    lines = [f"Validation Report for '{campaign.description}'"]
    lines.append(f"Items checked: {len(items)}")
    lines.append("")

    if errors:
        lines.append(f"ERRORS ({len(errors)}):")
        for i, e in enumerate(errors, 1):
            lines.append(f"  {i}. {e}")
        lines.append("")

    if warnings:
        lines.append(f"WARNINGS ({len(warnings)}):")
        for i, w in enumerate(warnings, 1):
            lines.append(f"  {i}. {w}")
        lines.append("")

    if not errors and not warnings:
        lines.append("No issues found. Plan looks good!")

    summary = "PASS" if not errors else "FAIL"
    lines.append(f"Result: {summary} ({len(errors)} errors, {len(warnings)} warnings)")

    return "\n".join(lines)
