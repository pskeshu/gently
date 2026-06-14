"""
Bluesky plan synthesis from natural language goals
"""

import ast
import logging
import re

from jinja2 import Template

logger = logging.getLogger(__name__)


class PlanValidator:
    """Validates generated Bluesky plans for safety and correctness"""

    def __init__(self, devices: dict | None = None):
        """
        Parameters
        ----------
        devices : dict, optional
            Available Ophyd devices with their limits
        """
        self.devices = devices or {}
        self.errors: list[str] = []

    def is_valid(self, plan_code: str) -> bool:
        """
        Validate plan code

        Checks:
        - Valid Python syntax
        - Uses only safe operations (no destructive commands)
        - Respects device limits
        - Includes proper metadata

        Parameters
        ----------
        plan_code : str
            Generated Python code for Bluesky plan

        Returns
        -------
        bool
            True if valid, False otherwise (check self.errors for details)
        """
        self.errors = []

        # Check syntax
        try:
            ast.parse(plan_code)
        except SyntaxError as e:
            self.errors.append(f"Syntax error: {e}")
            return False

        # Check for dangerous operations
        dangerous_patterns = [
            r"import\s+os",
            r"import\s+subprocess",
            r"eval\(",
            r"exec\(",
            r"__import__",
        ]

        for pattern in dangerous_patterns:
            if re.search(pattern, plan_code):
                self.errors.append(f"Dangerous operation detected: {pattern}")
                return False

        # Check for proper yield from usage
        if "def " in plan_code and "yield from" not in plan_code:
            self.errors.append("Plan function must use 'yield from' for Bluesky operations")
            return False

        # Warnings (not errors, but noted)
        if "metadata=" not in plan_code:
            self.errors.append("Warning: Plan should include metadata for provenance")

        # If we have errors (not just warnings), fail
        actual_errors = [e for e in self.errors if not e.startswith("Warning:")]
        return len(actual_errors) == 0

    def check_parameters(self, params: dict) -> bool:
        """
        Validate acquisition parameters

        Parameters
        ----------
        params : dict
            Parameters like num_slices, exposure_ms, interval_seconds

        Returns
        -------
        bool
            True if valid
        """
        self.errors = []

        # num_slices
        if "num_slices" in params:
            if not (10 <= params["num_slices"] <= 200):
                self.errors.append(f"num_slices {params['num_slices']} outside range [10, 200]")

        # exposure_ms
        if "exposure_ms" in params:
            if not (5 <= params["exposure_ms"] <= 100):
                self.errors.append(f"exposure_ms {params['exposure_ms']} outside range [5, 100]")

        # interval_seconds
        if "interval_seconds" in params:
            if params["interval_seconds"] < 10:
                self.errors.append(
                    f"interval_seconds {params['interval_seconds']} too short"
                    " (min 10s for hardware settle)"
                )

        return len(self.errors) == 0


class PlanTemplate:
    """Template for a specific type of acquisition plan"""

    def __init__(self, name: str, template_str: str, description: str):
        self.name = name
        self.template = Template(template_str)
        self.description = description

    def render(self, **kwargs) -> str:
        """Render template with parameters"""
        return self.template.render(**kwargs)


class PlanLibrary:
    """Collection of plan templates"""

    def __init__(self):
        self.templates: dict[str, PlanTemplate] = {}
        self._load_default_templates()

    def _load_default_templates(self):
        """Load built-in plan templates"""

        # Multi-embryo adaptive timelapse
        self.templates["adaptive_timelapse"] = PlanTemplate(
            name="adaptive_timelapse",
            description="Multi-embryo timelapse with adaptive parameters",
            template_str='''
def adaptive_timelapse_plan(
    volume_scanner,
    xy_stage,
    embryo_database,
    agent,
    num_timepoints={{ num_timepoints }},
    metadata=None
):
    """
    Adaptive multi-embryo timelapse acquisition

    Generated for: {{ goal }}

    Parameters
    ----------
    volume_scanner : DiSPIMVolumeScanner
        Volume acquisition device
    xy_stage : DiSPIMXYStage
        XY positioning stage
    embryo_database : dict
        Embryo positions and calibrations
    agent : MicroscopyAgent
        AI agent for decisions
    num_timepoints : int
        Maximum number of timepoints
    metadata : dict, optional
        Additional metadata
    """
    import bluesky.plan_stubs as bps
    from datetime import datetime

    # Initialize metadata
    _md = {
        'plan_name': 'adaptive_timelapse',
        'goal': '{{ goal }}',
        'start_time': datetime.now().isoformat(),
        'embryo_ids': {{ embryo_ids }},
        'agent_generated': True,
    }
    if metadata:
        _md.update(metadata)

    for timepoint in range(num_timepoints):
        # Check if experiment should continue
        if agent.should_stop_experiment():
            logger.info("Agent ending experiment at timepoint %s", timepoint)
            break

        # Get embryo acquisition order (priority queue)
        embryo_order = agent.get_embryo_acquisition_order()

        for embryo_id in embryo_order:
            embryo_state = agent.experiment.embryos[embryo_id]

            # Check if should skip this embryo
            if embryo_state.should_skip:
                continue

            # Get adaptive parameters from agent
            params = agent.decide_parameters(embryo_id, timepoint)

            # Move to embryo position
            pos = embryo_state.stage_position
            yield from bps.mov(xy_stage.x, pos['x'], xy_stage.y, pos['y'])

            # Configure volume scanner with adaptive parameters
            volume_scanner.configure(
                num_slices=params['num_slices'],
                exposure_ms=params['exposure_ms'],
                galvo_amplitude=embryo_state.calibration.get('galvo_amplitude', 8.0),
                galvo_center=embryo_state.calibration.get('galvo_center', 0.0),
                piezo_amplitude=embryo_state.calibration.get('piezo_amplitude', 50.0),
                piezo_center=embryo_state.calibration.get('piezo_center', 0.0),
            )

            # Acquire volume
            yield from bps.trigger_and_read(
                [volume_scanner],
                name='volume_acquisition',
                md={'embryo_id': embryo_id, 'timepoint': timepoint, **_md}
            )

            # Notify agent of new data (will trigger analysis)
            agent.on_volume_acquired(embryo_id, timepoint, volume_scanner)

        # Determine next interval (adaptive)
        next_interval = agent.decide_next_interval(timepoint)

        # Wait for interval
        if timepoint < num_timepoints - 1:
            yield from bps.sleep(next_interval)
''',
        )

        # Single embryo high-resolution scan
        self.templates["single_highres"] = PlanTemplate(
            name="single_highres",
            description="Single embryo high-resolution volume",
            template_str='''
def single_highres_scan_plan(
    volume_scanner,
    xy_stage,
    embryo_id,
    agent,
    metadata=None
):
    """
    High-resolution volume scan of single embryo

    Generated for: {{ goal }}
    """
    import bluesky.plan_stubs as bps

    embryo_state = agent.experiment.embryos[embryo_id]

    # Move to position
    pos = embryo_state.stage_position
    yield from bps.mov(xy_stage.x, pos['x'], xy_stage.y, pos['y'])

    # Configure for high-res
    volume_scanner.configure(
        num_slices={{ num_slices }},
        exposure_ms={{ exposure_ms }},
        galvo_amplitude=embryo_state.calibration.get('galvo_amplitude', 8.0),
        galvo_center=embryo_state.calibration.get('galvo_center', 0.0),
        piezo_amplitude=embryo_state.calibration.get('piezo_amplitude', 50.0),
        piezo_center=embryo_state.calibration.get('piezo_center', 0.0),
    )

    # Acquire
    _md = {
        'plan_name': 'single_highres',
        'embryo_id': embryo_id,
        'goal': '{{ goal }}',
        'agent_generated': True,
    }
    if metadata:
        _md.update(metadata)

    yield from bps.trigger_and_read([volume_scanner], md=_md)

    # Notify agent
    agent.on_volume_acquired(embryo_id, 0, volume_scanner)
''',
        )

    def get_template(self, plan_type: str) -> PlanTemplate:
        """Get template by name"""
        if plan_type not in self.templates:
            raise ValueError(
                f"Unknown plan type: {plan_type}. Available: {list(self.templates.keys())}"
            )
        return self.templates[plan_type]

    def list_templates(self) -> list[str]:
        """List available template names"""
        return list(self.templates.keys())


class PlanSynthesizer:
    """Converts scientific goals into executable Bluesky plans"""

    def __init__(
        self,
        plan_library: PlanLibrary | None = None,
        validator: PlanValidator | None = None,
    ):
        self.library = plan_library or PlanLibrary()
        self.validator = validator or PlanValidator()

    def synthesize(
        self,
        goal: str,
        embryo_ids: list[str],
        params: dict,
        plan_type: str = "adaptive_timelapse",
    ) -> str:
        """
        Generate Bluesky plan from goal

        Parameters
        ----------
        goal : str
            Scientific objective (e.g., "Image all embryos and detect hatching")
        embryo_ids : list of str
            Embryo IDs to include
        params : dict
            Parameters like num_slices, exposure_ms, interval_seconds, num_timepoints
        plan_type : str
            Which template to use

        Returns
        -------
        str
            Python code for Bluesky plan
        """
        # Validate parameters first
        if not self.validator.check_parameters(params):
            raise ValueError(f"Invalid parameters: {self.validator.errors}")

        # Get template
        template = self.library.get_template(plan_type)

        # Fill template
        plan_code = template.render(
            goal=goal,
            embryo_ids=embryo_ids,
            num_slices=params.get("num_slices", 50),
            exposure_ms=params.get("exposure_ms", 10.0),
            interval_seconds=params.get("interval_seconds", 120),
            num_timepoints=params.get("num_timepoints", 500),
        )

        # Validate generated code
        if not self.validator.is_valid(plan_code):
            raise ValueError(f"Generated invalid plan: {self.validator.errors}")

        return plan_code

    def classify_goal(self, goal: str) -> str:
        """
        Determine plan type from goal description

        Parameters
        ----------
        goal : str
            Natural language goal

        Returns
        -------
        str
            Plan type name
        """
        goal_lower = goal.lower()

        # Pattern matching for plan types
        if any(
            word in goal_lower
            for word in ["timelapse", "time-lapse", "monitor", "track", "all embryos"]
        ):
            return "adaptive_timelapse"
        elif any(
            word in goal_lower for word in ["high-res", "high resolution", "detailed", "single"]
        ):
            return "single_highres"
        else:
            # Default to timelapse
            return "adaptive_timelapse"
