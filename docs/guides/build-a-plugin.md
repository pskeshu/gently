# Build Your Own Plugin

Gently's plugin system lets you adapt the harness for different organisms and microscope hardware. This tutorial walks through creating both types of plugin.

## Architecture Overview

Gently has four layers with strict downward-only dependencies:

```
core/       → Foundation: event bus, data store, imaging, coordinates
harness/    → Reusable agent framework: tools, perception, memory, detection, plan mode
organisms/  → Organism plugins (biology, stages, perception prompts)
hardware/   → Hardware plugins (device control, acquisition plans)
app/        → The microscopy agent: domain tools, orchestration
```

Plugins live in Layer 3. They implement protocols defined by the harness (Layer 2) and are consumed by the application (Layer 4). The harness and core layers are reused unchanged.

## The Plugin Protocols

Two runtime-checkable protocols define the plugin contracts. Both are in `gently/harness/protocols.py`.

### OrganismProtocol

```python
@runtime_checkable
class OrganismProtocol(Protocol):
    ORGANISM_NAME: str              # e.g. "drosophila"
    ORGANISM_DISPLAY_NAME: str      # e.g. "Drosophila melanogaster"
    SAMPLE_TERM: str                # e.g. "embryo", "cell", "organoid"
    SAMPLE_TERM_PLURAL: str         # e.g. "embryos"
    STAGES: list                    # Developmental stages (ordered)
    TERMINAL_STAGES: set            # e.g. {"hatched"}
    BIOLOGY_KNOWLEDGE: str          # Markdown for LLM context
    PERCEPTION_SYSTEM_PROMPT: str   # VLM classification prompt
```

### HardwareProtocol

```python
@runtime_checkable
class HardwareProtocol(Protocol):
    HARDWARE_NAME: str              # e.g. "confocal"
    HARDWARE_DISPLAY_NAME: str      # e.g. "Spinning Disk Confocal"
    HARDWARE_DESCRIPTION: str       # Markdown capabilities text
```

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

Or modify `launch_gently.py` to accept it as an argument. The loader in `gently/organisms/__init__.py` dynamically imports `gently.organisms.drosophila`.

## Tutorial: Create a Simulated Hardware Plugin

For development and testing, a hardware stub that returns synthetic images.

### 1. Create the plugin directory

```
gently/hardware/simulator/
├── __init__.py
└── description.py
```

### 2. Write the hardware description

```python
# gently/hardware/simulator/description.py

HARDWARE_DESCRIPTION = """
## Simulated Microscope

A software-only microscope simulator for development and testing.

### Capabilities
- Simulated XY stage positioning
- Synthetic fluorescence image generation
- Configurable noise levels and sample density
- No physical hardware required

### Limitations
- Images are procedurally generated, not from real specimens
- No real optical effects (PSF, aberrations, scattering)
- Timing is instantaneous (no hardware settle time)
"""
```

### 3. Wire it up

```python
# gently/hardware/simulator/__init__.py
from .description import HARDWARE_DESCRIPTION

HARDWARE_NAME = "simulator"
HARDWARE_DISPLAY_NAME = "Simulated Microscope"
```

### 4. Select the plugin

```yaml
# config/config.yml
hardware: "simulator"
```

With `--offline`, the hardware plugin only provides its description text to the agent's system prompt. For online use with simulated hardware, you'd implement device classes following the patterns in `gently/hardware/dispim/devices/`.

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
        ToolExample(
            "Measure the wing disc in embryo 3",
            {"embryo_id": "embryo_3"}
        ),
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

## Testing Your Plugin

```bash
# Verify the plugin loads
python -c "from gently.organisms import load_organism; m = load_organism('drosophila'); print(m.ORGANISM_DISPLAY_NAME)"

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
| diSPIM hardware | `gently/hardware/dispim/` | Device classes, acquisition plans, safety limits |

## Next Steps

- [What Gently Can Do](capabilities.md) — full capabilities overview
- [Try Offline](try-offline.md) — test your plugin without hardware
- [Hardware Setup](hardware-setup.md) — connect real hardware
