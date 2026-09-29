"""Every setting the Settings surface shows, in one list.

Gently has about a hundred configurable things, kept in seven places: the
browser, the browser tab, URL flags, a rig-wide defaults file, a
restart-required overrides file, the device layer's own files, and the
environment. The old Settings page showed about a third of them, hand-built
control by control, and eight of its nineteen view settings were read by
nothing: they were saved faithfully and changed nothing.

So a setting is declared here, once, and the surface is drawn from the
declaration. Each one says what it is about (its category), how far it
reaches, when a change takes effect, where it is kept, and who reads it.
``tests/test_settings_registry.py`` fails if a declared reader does not read
it, so a dead setting cannot come back.

Some settings are best edited where they are used — the XY region on the Map,
the SPIM centre on Bottom cam, the DIC light in the plan. Those stay there.
They are declared here as links, so Settings says where everything is without
holding a second copy of anything.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

# How far a setting reaches.
BROWSER = "browser"  # this browser only (localStorage)
RIG = "rig"  # this microscope, whoever is looking

# When a change takes effect.
NOW = "now"
LOAD = "load"  # the next time the page loads
LAUNCH = "launch"  # the next time Gently is launched
RESTART = "restart"  # after the backend restarts

CATEGORIES: list[dict[str, str]] = [
    {
        "id": "views",
        "label": "Views",
        "blurb": "How Gently looks, and what each view of the embryos shows.",
    },
    {
        "id": "alerts",
        "label": "Alerts",
        "blurb": "When Gently draws your attention, and how loudly.",
    },
    {
        "id": "experiment",
        "label": "Experiment",
        "blurb": "What a new plan, calibration or detection starts from.",
    },
    {
        "id": "microscope",
        "label": "Microscope",
        "blurb": "How Gently reaches the hardware, and where its fences and offsets are set.",
    },
    {
        "id": "recording",
        "label": "Recording",
        "blurb": "The record Gently keeps of how it was used.",
    },
    {
        "id": "assistant",
        "label": "Assistant",
        "blurb": "The AI assistant: whether it runs, and which models it uses.",
    },
    {
        "id": "system",
        "label": "System",
        "blurb": "Storage, network and timing, and everything that is in effect.",
    },
]


@dataclass(frozen=True)
class Setting:
    key: str
    label: str
    category: str
    type: str  # bool | int | float | choice | multichoice | text | readonly | link | custom
    group: str = ""
    help: str = ""
    default: Any = None
    reach: str = BROWSER
    applies: str = NOW
    # Where it is kept:
    #   "prefs:<path>"   the browser's dashboard prefs, under <path>
    #   "theme"          the browser's theme
    #   "env:<NAME>"     config/settings.local.yml, read at startup
    #   "launch:<key>"   config/launch.local.json
    #   ""               nowhere here: readonly, link and custom
    store: str = ""
    choices: tuple[tuple[Any, str], ...] = ()
    min: float | None = None
    max: float | None = None
    step: float | None = None
    unit: str = ""
    # link: where it is edited, and the tab that takes you there.
    where: str = ""
    href: str = ""
    # custom: the id of a block the page already holds.
    block: str = ""
    # (path relative to the repo, text that must be found there)
    readers: tuple[tuple[str, str], ...] = field(default_factory=tuple)
    # readonly: a dotted path into gently.settings, or a name in _COMPUTED.
    source: str = ""


_JS = "gently/ui/web/static/js"

SETTINGS: list[Setting] = [
    # ── Views ────────────────────────────────────────────────────────────
    Setting(
        key="views.theme",
        label="Theme",
        category="views",
        group="Appearance",
        type="choice",
        default="light",
        store="theme",
        choices=(("light", "Light"), ("dark", "Dark")),
        readers=((f"{_JS}/app.js", "gently-theme"),),
    ),
    Setting(
        key="views.atrium",
        label="Open in the Atrium",
        help="The canvas surface instead of the tabs. Experimental.",
        category="views",
        group="Appearance",
        type="bool",
        default=False,
        applies=LOAD,
        store="prefs:atrium",
        readers=((f"{_JS}/atrium.js", ".atrium === true"),),
    ),
    Setting(
        key="views.defaultView",
        label="Embryos opens on",
        category="views",
        group="Embryos",
        type="choice",
        default="default",
        applies=LOAD,
        store="prefs:defaultView",
        choices=(
            ("default", "Default"),
            ("board", "Board"),
            ("filmstrip", "Film"),
            ("vitals", "Vitals"),
        ),
        readers=((f"{_JS}/embryos.js", "this.dashboardConfig.defaultView"),),
    ),
    Setting(
        key="views.board.columns",
        label="Columns",
        category="views",
        group="Board",
        type="multichoice",
        default=("stage", "clock", "stereo", "pace", "eta", "sparkline", "alert"),
        store="prefs:board.columns",
        choices=(
            ("stage", "Stage"),
            ("clock", "Clock"),
            ("stereo", "Stereo"),
            ("pace", "Pace"),
            ("eta", "ETA"),
            ("sparkline", "Progression"),
            ("alert", "Alert"),
        ),
        readers=((f"{_JS}/embryos.js", "this.dashboardConfig.board.columns"),),
    ),
    Setting(
        key="views.board.sparklineLength",
        label="Progression length",
        help="How many evaluations the progression line shows.",
        category="views",
        group="Board",
        type="int",
        default=20,
        min=5,
        max=100,
        step=5,
        unit="evaluations",
        store="prefs:board.sparklineLength",
        readers=((f"{_JS}/embryos.js", "this.dashboardConfig.board.sparklineLength"),),
    ),
    Setting(
        key="views.film.thumbnailSize",
        label="Frame size",
        category="views",
        group="Film",
        type="choice",
        default=56,
        store="prefs:filmstrip.thumbnailSize",
        choices=((40, "Small"), (56, "Medium"), (72, "Large")),
        readers=((f"{_JS}/embryos.js", "config.thumbnailSize"),),
    ),
    Setting(
        key="views.film.showStageLabels",
        label="Label each frame",
        help="The stage under an embryo's frame, the time under a DIC frame.",
        category="views",
        group="Film",
        type="bool",
        default=True,
        store="prefs:filmstrip.showStageLabels",
        readers=((f"{_JS}/embryos.js", "config.showStageLabels"),),
    ),
    Setting(
        key="views.film.skipInterval",
        label="Show every Nth timepoint",
        category="views",
        group="Film",
        type="int",
        default=1,
        min=1,
        max=10,
        step=1,
        store="prefs:filmstrip.skipInterval",
        readers=((f"{_JS}/embryos.js", "config.skipInterval"),),
    ),
    Setting(
        key="views.vitals.showExpectedLine",
        label="Show the expected line",
        help="Reference developmental timing at 20 °C.",
        category="views",
        group="Vitals",
        type="bool",
        default=True,
        store="prefs:vitals.showExpectedLine",
        readers=((f"{_JS}/embryos.js", "this.dashboardConfig.vitals.showExpectedLine"),),
    ),
    # ── Alerts ───────────────────────────────────────────────────────────
    Setting(
        key="alerts.warnOvertimeRatio",
        label="Warn when an embryo is late by",
        help="Time in a stage, as a multiple of the reference time for that stage.",
        category="alerts",
        group="Pace",
        type="float",
        default=1.5,
        min=1,
        max=5,
        step=0.1,
        unit="×",
        store="prefs:board.warnOvertimeRatio",
        readers=((f"{_JS}/embryos.js", "this.dashboardConfig.board.warnOvertimeRatio"),),
    ),
    Setting(
        key="alerts.ambient.enabled",
        label="Ambient pulse",
        help="The quiet health light in the Embryos header.",
        category="alerts",
        group="Ambient pulse",
        type="bool",
        default=True,
        store="prefs:ambient.enabled",
        readers=((f"{_JS}/embryos.js", "this.dashboardConfig.ambient.enabled"),),
    ),
    Setting(
        key="alerts.ambient.sensitivity",
        label="Sensitivity",
        category="alerts",
        group="Ambient pulse",
        type="choice",
        default="normal",
        store="prefs:ambient.sensitivity",
        choices=(("low", "Low"), ("normal", "Normal"), ("high", "High")),
        readers=((f"{_JS}/embryos.js", "this.dashboardConfig.ambient.sensitivity"),),
    ),
    # ── Experiment ───────────────────────────────────────────────────────
    Setting(
        key="experiment.plan",
        label="Acquisition plan",
        help="Cadence, channels, the DIC light and how a run ends. Kept with each "
        "session; a plan can be saved under a name and run again.",
        category="experiment",
        type="link",
        reach=RIG,
        where="Devices › Operate › Acquisition",
        href="devices",
    ),
    Setting(
        key="experiment.calibration",
        label="Calibration options",
        help="Edge detection, the z buffer, and the edge search's step, range and tolerance.",
        category="experiment",
        type="link",
        where="Devices › Operate › Calibration › More…",
        href="devices",
    ),
    Setting(
        key="experiment.calibrationImages",
        label="Keep what a calibration looked at",
        help="Every exposure and plot of each run, kept with the embryo. A run is sixty "
        "to eighty exposures, so keeping them all costs disk.",
        category="experiment",
        group="Calibration",
        type="choice",
        default="all",
        reach=RIG,
        applies=RESTART,
        store="env:GENTLY_CALIBRATION_IMAGES",
        source="storage.calibration_images",
        choices=(("all", "Everything"), ("plots", "Plots only"), ("none", "Nothing")),
        readers=(("gently/core/calibration_record.py", "calibration_images"),),
    ),
    Setting(
        key="experiment.detection",
        label="Detection options",
        help="Whether Claude and SAM take part in finding embryos, and how permissive to be.",
        category="experiment",
        type="link",
        where="Devices › Operate › Bottom cam, in the Marking panel",
        href="devices",
    ),
    # ── Microscope ───────────────────────────────────────────────────────
    Setting(
        key="microscope.hardware",
        label="Start with the microscope on",
        help="Off starts Gently without connecting to the hardware.",
        category="microscope",
        group="At launch",
        type="bool",
        default=True,
        reach=RIG,
        applies=LAUNCH,
        store="launch:hardware",
        readers=(("gently/ui/web/launch_prefs.py", '"hardware"'),),
    ),
    Setting(
        key="microscope.devicelayer",
        label="Device layer",
        category="microscope",
        group="Device layer",
        type="custom",
        reach=RIG,
        applies=LAUNCH,
        block="settings-block-devicelayer",
    ),
    Setting(
        key="microscope.thermalizer",
        label="Thermalizer",
        category="microscope",
        group="Thermalizer (ACUITYnano)",
        type="custom",
        reach=RIG,
        block="settings-block-thermalizer",
    ),
    Setting(
        key="microscope.joystick",
        label="Joystick",
        category="microscope",
        group="Stage",
        type="custom",
        reach=RIG,
        block="settings-block-joystick",
    ),
    Setting(
        key="microscope.projectionView",
        label="What a projection shows",
        help="The SPIM camera's chip carries two channels side by side. A projection "
        "shows one of them, or the whole frame. The volume on disk is the whole frame "
        "whatever is chosen. Projections already drawn keep what they were drawn with.",
        category="microscope",
        group="SPIM camera",
        type="choice",
        default="left",
        reach=RIG,
        applies=RESTART,
        store="env:GENTLY_PROJECTION_VIEW",
        source="ui.projection_view",
        choices=(
            ("left", "The left channel"),
            ("right", "The right channel"),
            ("both", "The whole frame"),
        ),
        readers=(("gently/core/imaging.py", "projection_view"),),
    ),
    Setting(
        key="microscope.spimFullWidth",
        label="Width of a full readout",
        help="A frame this wide holds both channels and is divided down the middle. "
        "A narrower one is one channel already and is never divided.",
        category="microscope",
        group="SPIM camera",
        type="int",
        default=2048,
        min=2,
        max=16384,
        unit="px",
        reach=RIG,
        applies=RESTART,
        store="env:GENTLY_SPIM_FULL_WIDTH",
        source="ui.spim_full_width",
        readers=(("gently/core/imaging.py", "spim_full_width"),),
    ),
    Setting(
        key="microscope.xyRegion",
        label="XY region and limits",
        help="The region Gently keeps the stage within, and the controller's own limits.",
        category="microscope",
        group="Stage",
        type="link",
        reach=RIG,
        where="Devices › Map",
        href="devices",
    ),
    Setting(
        key="microscope.spimCentre",
        label="SPIM centre offset",
        help="Where the SPIM head looks, relative to the bottom camera's centre.",
        category="microscope",
        group="Stage",
        type="link",
        reach=RIG,
        where="Devices › Operate › Bottom cam › Advanced",
        href="devices",
    ),
    Setting(
        key="microscope.water",
        label="Water temperature and room light",
        category="microscope",
        group="Environment",
        type="link",
        reach=RIG,
        where="the rig menu in the header",
    ),
    # ── Recording ────────────────────────────────────────────────────────
    Setting(
        key="recording.enabled",
        label="Record how Gently is used",
        help="Every click and every change on screen, kept with the session so a "
        "walkthrough can be replayed.",
        category="recording",
        type="bool",
        default=True,
        reach=RIG,
        applies=RESTART,
        store="env:GENTLY_REPLAY",
        source="ui.replay",
        readers=(("gently/ui/web/routes/replay.py", "settings.ui.replay"),),
    ),
    Setting(
        key="recording.fidelity",
        label="Detail",
        help="Balanced leaves out the regions that redraw constantly, such as the map "
        "and the 3D view. Actions keeps only the clicks.",
        category="recording",
        type="choice",
        default="balanced",
        reach=RIG,
        applies=RESTART,
        store="env:GENTLY_REPLAY_FIDELITY",
        source="ui.replay_fidelity",
        choices=(("full", "Full"), ("balanced", "Balanced"), ("actions", "Actions only")),
        readers=(("gently/ui/web/routes/replay.py", "settings.ui.replay_fidelity"),),
    ),
    Setting(
        key="recording.maxTabMb",
        label="Largest recording of one browser tab",
        category="recording",
        group="Disk",
        type="float",
        default=120.0,
        min=10,
        max=2000,
        step=10,
        unit="MB",
        reach=RIG,
        applies=RESTART,
        store="env:GENTLY_REPLAY_MAX_TAB_MB",
        source="ui.replay_max_tab_mb",
        readers=(("gently/ui/web/routes/replay.py", "settings.ui.replay_max_tab_mb"),),
    ),
    Setting(
        key="recording.totalBudgetMb",
        label="All recordings together",
        help="When they pass this, the oldest are deleted. The newest three are always kept.",
        category="recording",
        group="Disk",
        type="float",
        default=1024.0,
        min=100,
        max=100000,
        step=100,
        unit="MB",
        reach=RIG,
        applies=RESTART,
        store="env:GENTLY_REPLAY_TOTAL_BUDGET_MB",
        source="ui.replay_total_budget_mb",
        readers=(("gently/ui/web/routes/replay.py", "settings.ui.replay_total_budget_mb"),),
    ),
    Setting(
        key="recording.about",
        label="What is recorded",
        category="recording",
        group="What is kept",
        type="custom",
        reach=RIG,
        block="settings-block-recording",
    ),
    # ── Assistant ────────────────────────────────────────────────────────
    Setting(
        key="assistant.enabled",
        label="Start with the assistant on",
        help="Off starts Gently without the AI assistant; no API key is needed.",
        category="assistant",
        group="At launch",
        type="bool",
        default=True,
        reach=RIG,
        applies=LAUNCH,
        store="launch:agent",
        readers=(("gently/ui/web/launch_prefs.py", '"agent"'),),
    ),
    Setting(
        key="assistant.key",
        label="API key",
        category="assistant",
        group="At launch",
        type="readonly",
        reach=RIG,
        source="computed:api_key",
    ),
    Setting(
        key="assistant.model.main",
        label="Planning and tools",
        category="assistant",
        group="Models",
        type="readonly",
        reach=RIG,
        applies=RESTART,
        source="models.main",
    ),
    Setting(
        key="assistant.model.perception",
        label="Looking at images, and chat",
        category="assistant",
        group="Models",
        type="readonly",
        reach=RIG,
        applies=RESTART,
        source="models.perception",
    ),
    Setting(
        key="assistant.model.fast",
        label="Quick checks",
        help="The verifier, and the blank-image and hatching checks.",
        category="assistant",
        group="Models",
        type="readonly",
        reach=RIG,
        applies=RESTART,
        source="models.fast",
    ),
    # ── System ───────────────────────────────────────────────────────────
    Setting(
        key="system.version",
        label="Build",
        category="system",
        group="About",
        type="readonly",
        reach=RIG,
        source="computed:build",
    ),
    Setting(
        key="system.storage",
        label="Data folder",
        category="system",
        group="About",
        type="readonly",
        reach=RIG,
        applies=RESTART,
        source="storage.base_path",
    ),
    Setting(
        key="system.vizPort",
        label="Web port",
        category="system",
        group="About",
        type="readonly",
        reach=RIG,
        applies=RESTART,
        source="network.viz_port",
    ),
    Setting(
        key="system.fiji.found",
        label="Fiji",
        help="Where Fiji was found. Images open in it from the viewer's Open in Fiji button.",
        category="system",
        group="Other programs",
        type="readonly",
        reach=RIG,
        source="computed:fiji",
    ),
    Setting(
        key="system.fiji.path",
        label="Where Fiji is",
        help="Leave empty and Gently looks where Fiji is usually unpacked. Give the "
        "program itself or the folder it is in. Micro-Manager's ImageJ is refused: "
        "starting it starts Micro-Manager, which takes the microscope.",
        category="system",
        group="Other programs",
        type="text",
        default="",
        reach=RIG,
        applies=RESTART,
        store="env:GENTLY_FIJI_PATH",
        source="ui.fiji_path",
        readers=(("gently/settings.py", "FIJI_PATH"),),
    ),
    Setting(
        key="system.timeout.volume",
        label="Volume acquisition",
        category="system",
        group="Timeouts",
        type="int",
        default=15,
        min=1,
        max=3600,
        unit="s",
        reach=RIG,
        applies=RESTART,
        store="env:GENTLY_TIMEOUT_VOLUME",
        source="timeouts.volume_acquisition",
        readers=(("gently/settings.py", "TIMEOUT_VOLUME"),),
    ),
    Setting(
        key="system.timeout.api",
        label="External API call",
        category="system",
        group="Timeouts",
        type="int",
        default=10,
        min=1,
        max=600,
        unit="s",
        reach=RIG,
        applies=RESTART,
        store="env:GENTLY_TIMEOUT_API",
        source="timeouts.api_call",
        readers=(("gently/settings.py", "TIMEOUT_API"),),
    ),
    Setting(
        key="system.mesh.broadcast",
        label="Broadcast interval",
        category="system",
        group="Mesh network",
        type="float",
        default=5.0,
        min=0.5,
        max=600,
        unit="s",
        reach=RIG,
        applies=RESTART,
        store="env:GENTLY_MESH_BROADCAST_INTERVAL",
        source="mesh.broadcast_interval_s",
        readers=(("gently/settings.py", "MESH_BROADCAST_INTERVAL"),),
    ),
    Setting(
        key="system.mesh.stale",
        label="A peer is stale after",
        category="system",
        group="Mesh network",
        type="float",
        default=15.0,
        min=1,
        max=3600,
        unit="s",
        reach=RIG,
        applies=RESTART,
        store="env:GENTLY_MESH_STALE_THRESHOLD",
        source="mesh.stale_threshold_s",
        readers=(("gently/settings.py", "MESH_STALE_THRESHOLD"),),
    ),
    Setting(
        key="system.mesh.dead",
        label="A peer is gone after",
        category="system",
        group="Mesh network",
        type="float",
        default=30.0,
        min=1,
        max=3600,
        unit="s",
        reach=RIG,
        applies=RESTART,
        store="env:GENTLY_MESH_DEAD_THRESHOLD",
        source="mesh.dead_threshold_s",
        readers=(("gently/settings.py", "MESH_DEAD_THRESHOLD"),),
    ),
    Setting(
        key="system.ncbi.tool",
        label="Tool name",
        help="How Gently identifies itself to NCBI when the assistant searches the literature.",
        category="system",
        group="Literature search",
        type="text",
        default="gently",
        reach=RIG,
        applies=RESTART,
        store="env:GENTLY_NCBI_TOOL",
        source="api.ncbi_tool",
        readers=(("gently/settings.py", "NCBI_TOOL"),),
    ),
    Setting(
        key="system.ncbi.email",
        label="Contact e-mail",
        help="NCBI asks for one. Use the lab's.",
        category="system",
        group="Literature search",
        type="text",
        default="",
        reach=RIG,
        applies=RESTART,
        store="env:GENTLY_NCBI_EMAIL",
        source="api.ncbi_email",
        readers=(("gently/settings.py", "NCBI_EMAIL"),),
    ),
    Setting(
        key="system.uxV2",
        label="The agent-first interface",
        help="Off returns to the older dashboard.",
        category="system",
        group="Interface",
        type="bool",
        default=True,
        reach=RIG,
        applies=RESTART,
        store="env:GENTLY_UX_V2",
        source="ui.ux_v2",
        readers=(("gently/settings.py", "UX_V2"),),
    ),
    Setting(
        key="system.history",
        label="History",
        category="system",
        group="Every change, kept",
        type="custom",
        reach=RIG,
        block="settings-block-history",
    ),
    Setting(
        key="system.effective",
        label="Everything in effect",
        category="system",
        group="Everything in effect",
        type="custom",
        reach=RIG,
        block="settings-block-effective",
    ),
]


def by_key() -> dict[str, Setting]:
    return {s.key: s for s in SETTINGS}


def env_settings() -> list[Setting]:
    """The settings kept in config/settings.local.yml: the only ones the
    overrides route will write."""
    return [s for s in SETTINGS if s.store.startswith("env:")]


def env_name(s: Setting) -> str:
    return s.store.split(":", 1)[1]


def override_type(s: Setting) -> str:
    """The coercion the overrides file uses for this setting."""
    return {"bool": "bool", "int": "int", "float": "float"}.get(s.type, "str")


def _dig(obj: Any, dotted: str) -> Any:
    for part in dotted.split("."):
        obj = getattr(obj, part)
    return obj


def _computed(name: str) -> Any:
    if name == "api_key":
        return "present" if os.getenv("ANTHROPIC_API_KEY") else "not set"
    if name == "fiji":
        from gently.ui.web.routes.reveal import fiji_path

        found = fiji_path()
        return str(found) if found else "not found"
    if name == "build":
        from gently._version import build_date, build_id

        stamp = build_date()
        return f"{build_id()}" + (f" · {stamp}" if stamp else "")
    return None


def current_value(s: Setting, settings_obj: Any, launch_prefs: dict | None = None) -> Any:
    """What is in effect now, for the settings the server knows the value of.
    Browser settings have no value here: the browser holds them."""
    try:
        if s.source.startswith("computed:"):
            return _computed(s.source.split(":", 1)[1])
        if s.source:
            val = _dig(settings_obj, s.source)
            return val if isinstance(val, (bool, int, float, str)) else str(val)
        if s.store.startswith("launch:") and launch_prefs is not None:
            return launch_prefs.get(s.store.split(":", 1)[1], s.default)
    except Exception:
        return None
    return None


def schema(
    settings_obj: Any,
    *,
    overridden: set[str] | None = None,
    launch_prefs: dict | None = None,
    rig_defaults: dict | None = None,
) -> dict[str, Any]:
    """The registry as the page needs it."""
    overridden = overridden or set()
    out = []
    for s in SETTINGS:
        item: dict[str, Any] = {
            "key": s.key,
            "label": s.label,
            "help": s.help,
            "category": s.category,
            "group": s.group,
            "type": s.type,
            "default": list(s.default) if isinstance(s.default, tuple) else s.default,
            "reach": s.reach,
            "applies": s.applies,
            "store": s.store,
        }
        if s.choices:
            item["choices"] = [{"value": v, "label": lab} for v, lab in s.choices]
        for name in ("min", "max", "step"):
            if getattr(s, name) is not None:
                item[name] = getattr(s, name)
        if s.unit:
            item["unit"] = s.unit
        if s.type == "link":
            item["where"] = s.where
            item["href"] = s.href
        if s.type == "custom":
            item["block"] = s.block
        value = current_value(s, settings_obj, launch_prefs)
        if value is not None:
            item["value"] = value
        if s.store.startswith("env:"):
            item["overridden"] = env_name(s) in overridden
        out.append(item)
    return {"categories": CATEGORIES, "settings": out, "rig_defaults": rig_defaults or {}}
