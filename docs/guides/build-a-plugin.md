# Build Your Own Plugin

Gently's plugin system lets you adapt the harness for different organisms and microscope hardware. This tutorial walks through creating both types of plugin.

## Architecture Overview

Gently has four layers with strict downward-only dependencies:

```
core/       → Foundation: event bus, data store, imaging, coordinates
harness/    → Reusable agent framework: tools, perception, memory, detection, plan mode
organisms/  → Organism plugins (biology, stages, perception prompts)
hardware/   → Hardware plugins (device control, acquisition plans, client)
app/        → The microscopy agent: domain tools, orchestration
```

Plugins live in Layer 3. They implement protocols defined by the harness (Layer 2) and are consumed by the application (Layer 4). The harness and core layers are reused unchanged.

## The Plugin Protocols

Three protocols define the plugin contracts. All are in `gently/harness/protocols.py`.

### OrganismProtocol

```python
@runtime_checkable
class OrganismProtocol(Protocol):
    ORGANISM_NAME: str  # e.g. "drosophila"
    ORGANISM_DISPLAY_NAME: str  # e.g. "Drosophila melanogaster"
    SAMPLE_TERM: str  # e.g. "embryo", "cell", "organoid"
    SAMPLE_TERM_PLURAL: str  # e.g. "embryos"
    STAGES: list  # Developmental stages (ordered)
    TERMINAL_STAGES: set  # e.g. {"hatched"}
    BIOLOGY_KNOWLEDGE: str  # Markdown for LLM context
    PERCEPTION_SYSTEM_PROMPT: str  # VLM classification prompt
```

### HardwareProtocol

```python
@runtime_checkable
class HardwareProtocol(Protocol):
    HARDWARE_NAME: str  # e.g. "twophoton"
    HARDWARE_DISPLAY_NAME: str  # e.g. "Two-Photon Microscope"
    HARDWARE_DESCRIPTION: str  # Markdown capabilities text
    CAPABILITIES: set  # e.g. {"xy_stage", "z_stack", "fluorescence"}
```

Standard capability names: `xy_stage`, `z_control`, `volume`, `snap`, `z_stack`, `dual_view`, `autofocus`, `detection`, `fluorescence`, `transmitted`.

### MicroscopeClientProtocol

Defines the generic interface that the agent and tools use to talk to hardware:

```python
@runtime_checkable
class MicroscopeClientProtocol(Protocol):
    is_connected: bool
    has_sam: bool

    async def connect(self) -> bool: ...
    async def disconnect(self) -> None: ...
    async def get_status(self) -> dict: ...
    async def move_to_position(self, x: float, y: float) -> dict: ...
    async def get_stage_position(self) -> tuple: ...
    async def get_z_position(self) -> float: ...
    async def acquire(self, **params) -> dict: ...
    async def snap(self, **params) -> dict: ...
    async def detect_samples(self, **kwargs) -> dict: ...
```

Hardware-specific operations (like diSPIM's `capture_lightsheet_image` or a 2P's `acquire_zstack`) live on the concrete client class, not the protocol.

### Hardware Factory Functions

Hardware modules must also provide two factory functions:

```python
def create_device_layer(config: dict):
    """Create the hardware control server. Returns a server with .run(port=N)."""
    ...


def create_client(http_url: str):
    """Create an HTTP client for the device layer. Returns a client with .connect()."""
    ...
```

These are called by `start_device_layer.py` and `launch_gently.py` respectively, so the framework never imports hardware-specific code directly.

## Tutorial: Create a Drosophila Organism Plugin

### 1. Create the plugin directory

```
gently/organisms/drosophila/
├── __init__.py
├── stages.py
├── biology.py
└── perception_prompt.py
```

### 2. Define developmental stages

```python
# gently/organisms/drosophila/stages.py
from enum import Enum


class DevelopmentalStage(str, Enum):
    """Drosophila embryo developmental stages."""

    SYNCYTIAL = "syncytial"
    CELLULARIZATION = "cellularization"
    GASTRULATION = "gastrulation"
    GERMBAND_EXTENSION = "germband_extension"
    GERMBAND_RETRACTION = "germband_retraction"
    DORSAL_CLOSURE = "dorsal_closure"
    HATCHED = "hatched"

    # Special states
    ARRESTED = "arrested"
    NO_OBJECT = "no_object"


# Ordered list for the perception engine
STAGES = list(DevelopmentalStage)

# Stages that mean "done"
TERMINAL_STAGES = {DevelopmentalStage.HATCHED}
```

### 3. Write biology knowledge

This markdown text is injected into the agent's system prompt. It gives the LLM context about the organism.

```python
# gently/organisms/drosophila/biology.py

BIOLOGY_KNOWLEDGE = """
## Drosophila melanogaster Embryogenesis

Drosophila embryonic development takes ~22 hours at 25°C and progresses
through well-characterized morphological stages visible by light microscopy.

### Key Stages

- **Syncytial blastoderm** (0-2.5h): Rapid nuclear divisions without
  cellularization. Nuclei migrate to cortex.
- **Cellularization** (2.5-3h): Membrane invagination compartmentalizes
  nuclei into individual cells.
- **Gastrulation** (3-4h): Ventral furrow formation, posterior midgut
  invagination.
- **Germband extension** (4-7h): Germband extends around posterior.
  Segmentation becomes visible.
- **Germband retraction** (7-10h): Germband shortens. Head involution.
- **Dorsal closure** (10-15h): Lateral epidermis spreads dorsally.
  Amnioserosa cells constrict.
- **Hatching** (~22h): First instar larva emerges.

### Imaging Notes

Drosophila embryos are ~500μm × 200μm, larger than C. elegans.
Autofluorescence from the vitelline membrane can complicate imaging.
Dechorionation improves image quality but requires careful handling.
"""
```

### 4. Write the perception prompt

This is the system prompt for the VLM when classifying stages. Be specific about morphological features and common confusion points.

```python
# gently/organisms/drosophila/perception_prompt.py

PERCEPTION_SYSTEM_PROMPT = """
You are an expert Drosophila developmental biologist analyzing microscopy
images of Drosophila melanogaster embryos.

## Task
Describe what you observe FIRST, then classify the developmental stage.

## Stages (in developmental order)
1. syncytial — No visible cell boundaries. Uniform cortex.
2. cellularization — Membrane furrows visible between nuclei.
3. gastrulation — Ventral furrow forming. Tissue invagination.
4. germband_extension — Posterior extension visible. Segmental grooves.
5. germband_retraction — Germband shortening. Head structures forming.
6. dorsal_closure — Lateral epidermis spreading. Amnioserosa visible.
7. hatched — Larva visible, no longer in egg shape.

## Special States
- arrested — Development halted. No progression over multiple timepoints.
- no_object — No embryo visible in the field of view.

## Critical Distinctions
- syncytial vs cellularization: Look for membrane furrows between nuclei
- gastrulation vs germband_extension: Ventral furrow is gastrulation;
  posterior extension is germband
- germband_extension vs retraction: Extension = germband wraps posteriorly;
  retraction = germband shortens back

## Output Format
Respond with a JSON object:
{
    "observed_features": {"shape": "...", "surface": "...", "internal": "..."},
    "contrastive_reasoning": {"why_not_previous": "...", "why_not_next": "..."},
    "stage": "<stage_name>",
    "is_transitional": true/false,
    "transition_between": ["stage1", "stage2"],
    "confidence": 0.0-1.0,
    "reasoning": "Brief explanation"
}
"""
```

### 5. Wire it up

```python
# gently/organisms/drosophila/__init__.py
from .stages import STAGES, TERMINAL_STAGES, DevelopmentalStage
from .biology import BIOLOGY_KNOWLEDGE
from .perception_prompt import PERCEPTION_SYSTEM_PROMPT

ORGANISM_NAME = "drosophila"
ORGANISM_DISPLAY_NAME = "Drosophila melanogaster"
SAMPLE_TERM = "embryo"
SAMPLE_TERM_PLURAL = "embryos"
```

### 6. Select the plugin

In `config/config.yml`:

```yaml
organism: "drosophila"
```

The loader in `gently/organisms/__init__.py` dynamically imports `gently.organisms.drosophila`.

## Tutorial: Create a Two-Photon Hardware Plugin

This example shows a complete hardware plugin for a two-photon microscope. Even if you only plan to use `--offline` initially, providing the full structure makes it easy to add hardware control later.

### 1. Create the plugin directory

```
gently/hardware/twophoton/
├── __init__.py
├── description.py
└── calibration.py     # Hardware-specific calibration model (optional)
```

### 2. Write the hardware description

```python
# gently/hardware/twophoton/description.py

HARDWARE_DESCRIPTION = """
## Two-Photon Microscope

Point-scanning two-photon fluorescence microscope for deep tissue imaging.

### Capabilities
- XY positioning via motorized stage
- Z-stacking via objective piezo or Z-motor
- Two-photon excitation with Ti:Sapphire laser (700-1050nm tunable)
- Power modulation via Pockels cell
- PMT detection (non-descanned)
- 488nm and 561nm single-photon channels (optional)

### Acquisition Modes
- Single-plane snap at current Z
- Z-stack: sequential planes with configurable step size and range
- Timelapse Z-stacks

### Safety
- Pockels cell blanking on error (prevents tissue damage)
- Laser shutter interlock
- Power limits enforced at all times
- Z-motor bounds checking
"""
```

### 3. Wire it up with capabilities and factories

```python
# gently/hardware/twophoton/__init__.py
from .description import HARDWARE_DESCRIPTION

HARDWARE_NAME = "twophoton"
HARDWARE_DISPLAY_NAME = "Two-Photon Microscope"
CAPABILITIES = {
    "xy_stage",
    "z_control",
    "z_stack",
    "snap",
    "fluorescence",
}


def create_device_layer(config: dict):
    """Create the 2P device layer server."""
    from .device_layer import TwoPhotonDeviceLayer

    return TwoPhotonDeviceLayer(
        config_path=config.get("config_path", "config/config.yml"),
    )


def create_client(http_url: str):
    """Create an HTTP client for the 2P device layer."""
    from .client import TwoPhotonClient

    return TwoPhotonClient(http_url=http_url)
```

### 4. Define hardware-specific calibration (optional)

Each hardware type has its own calibration model. For 2P, it's simpler than diSPIM — just Z-range finding:

```python
# gently/hardware/twophoton/calibration.py
from dataclasses import dataclass
from typing import Optional


@dataclass
class TwoPhotonCalibration:
    """Z-axis calibration for a two-photon microscope."""

    z_top: float = 0.0  # Top of sample (µm)
    z_bottom: float = 100.0  # Bottom of sample (µm)
    optimal_z: float = 50.0  # Best focal plane (µm)
    optimal_power: float = 10.0  # Laser power (mW)

    def to_dict(self) -> dict:
        return {
            "z_top": self.z_top,
            "z_bottom": self.z_bottom,
            "optimal_z": self.optimal_z,
            "optimal_power": self.optimal_power,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TwoPhotonCalibration":
        return cls(**{k: data[k] for k in cls.__dataclass_fields__ if k in data})
```

The framework stores calibration as an opaque dict in `EmbryoState.calibration` — your hardware tools interpret it.

### 5. Select the plugin

```yaml
# config/config.yml
hardware: "twophoton"
```

### 6. What happens at launch

When you run `python start_device_layer.py`, the framework:
1. Reads `config.yml` to find `hardware: "twophoton"`
2. Calls `load_hardware("twophoton")` → imports `gently.hardware.twophoton`
3. Calls `hw.create_device_layer(config)` → gets your server
4. Calls `server.run(port=60610)` → starts your HTTP API

When you run `python launch_gently.py`:
1. Calls `hw.create_client(http_url)` → gets your client
2. Passes the client to `MicroscopyAgent`
3. Tools receive it via `context['client']`

With `--offline`, no client is created. The hardware module's `HARDWARE_DESCRIPTION` and `CAPABILITIES` are still loaded into the agent's system prompt.

## Adding Custom Tools

Tools are registered with the `@tool` decorator from `gently/harness/tools/registry.py`. Parameters are extracted automatically from type hints.

```python
from gently.harness.tools.registry import tool, ToolCategory, ToolExample


@tool(
    name="measure_wing_disc",
    description="Measure the size of a wing imaginal disc in the current image",
    category=ToolCategory.ANALYSIS,
    requires_microscope=False,
    examples=[
        ToolExample("Measure the wing disc in embryo 3", {"embryo_id": "embryo_3"}),
    ],
)
async def measure_wing_disc(
    embryo_id: str,
    threshold: float = 0.5,
    context: dict = None,
) -> str:
    """Measure wing disc area from the latest acquired image."""
    agent = context.get("agent")
    # Your analysis logic here
    return f"Wing disc area: 1250 μm² (embryo {embryo_id})"
```

**Key points:**
- `category` groups tools in the UI and documentation
- `requires_microscope=True` tools are hidden in offline mode
- `context` is injected automatically with `agent`, `client`, and `databroker`
- Return a string — this becomes the tool result the LLM sees
- Tools can be async or sync

Hardware-specific tools (acquisition, calibration, focus) should live alongside their hardware module or in `app/tools/` with appropriate capability checks:

```python
@tool(name="acquire_zstack", requires_microscope=True)
async def acquire_zstack(
    embryo_id: str, num_planes: int = 50, z_step_um: float = 1.0, context: dict = None
) -> str:
    client = context.get("client")
    # This tool only works with a 2P client
    result = await client.acquire_zstack(num_planes=num_planes, z_step_um=z_step_um)
    ...
```

Register your tools by importing them in your app's tool setup. See `gently/app/tools/` for examples across all categories.

## Adding Reference Images for Perception

To enable few-shot perception for your organism:

```
gently/examples/stages/
├── syncytial/
│   ├── three_view.jpg      # Combined XY+YZ+XZ projection
│   ├── progression.jpg     # Time series view (optional)
│   └── metadata.json       # Stage description and annotations
├── cellularization/
│   └── ...
└── gastrulation/
    └── ...
```

The `metadata.json` provides context for each reference image:

```json
{
    "stage": "syncytial",
    "description": "Syncytial blastoderm. No cell boundaries visible.",
    "key_features": ["uniform cortex", "no membrane furrows"],
    "commonly_confused_with": "cellularization"
}
```

The perception engine's `ExampleStore` loads these automatically and includes them as few-shot examples in VLM classification prompts.

## Focus and Calibration Data Model

The harness provides a generic `FocusDataPoint` for tracking focus measurements:

```python
@dataclass
class FocusDataPoint:
    z: float  # Primary focus axis (µm)
    secondary_axis: float  # Second axis (galvo for diSPIM, 0.0 for single-axis)
    score: float  # Focus quality
    r_squared: float  # Fit quality (0-1)
    timestamp: datetime
    method: str  # 'calibration', 'fine_focus', 'manual'
    algorithm: str  # 'fft_bandpass', 'gradient', etc.
```

For single-axis systems (2P, confocal), set `secondary_axis=0.0`. The harness tracks focus history per-embryo and provides drift analysis and interpolation.

Hardware-specific calibration models (like diSPIM's `CalibrationPrior` with piezo-galvo slope fitting) live in the hardware module, not the harness. Store calibration data as a dict in `EmbryoState.calibration` — your tools interpret the keys.

## Testing Your Plugin

```bash
# Verify the plugin loads
python -c "from gently.organisms import load_organism; m = load_organism('drosophila'); print(m.ORGANISM_DISPLAY_NAME)"

# Verify hardware capabilities
python -c "from gently.hardware import load_hardware; hw = load_hardware('twophoton'); print(hw.CAPABILITIES)"

# Launch offline to test the full agent
python launch_gently.py --offline
```

In the agent, try:
- "What stages can you identify?" — should list your organism's stages
- "What organism are we working with?" — should show your display name
- `/plan` → "Design an experiment" — should use your biology knowledge

## Existing Plugins for Reference

| Plugin | Location | What to study |
|--------|----------|---------------|
| C. elegans organism | `gently/organisms/celegans/` | Stage definitions, biology text, perception prompt |
| diSPIM hardware | `gently/hardware/dispim/` | Device classes, acquisition plans, safety limits, calibration model, client factory |

## Next Steps

- [What Gently Can Do](capabilities.md) — full capabilities overview
- [Try Offline](try-offline.md) — test your plugin without hardware
- [Hardware Setup](hardware-setup.md) — connect real hardware
