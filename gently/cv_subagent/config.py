"""
Configuration for CV Subagent

Provides settings for models, GPU, and service configuration.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, TYPE_CHECKING
from pathlib import Path

if TYPE_CHECKING:
    from .context import CVContext


@dataclass
class ModelConfig:
    """Configuration for a CV model"""
    name: str
    model_type: str  # cellpose, stardist, etc.
    path: Optional[str] = None
    gpu: bool = True
    preload: bool = False


@dataclass
class AutoAnalysisConfig:
    """Configuration for automatic analysis on volume acquisition"""

    # Enable/disable auto-analysis
    enabled: bool = False

    # What to run automatically on each new volume
    auto_segment: bool = True  # Run Cellpose segmentation
    auto_stage: bool = True  # Classify developmental stage
    auto_track: bool = False  # Track cells (requires multiple timepoints)

    # Segmentation settings for auto-analysis
    segmentation_model: str = "nuclei"  # cellpose model type
    min_timepoints_for_tracking: int = 3

    # Filtering - only auto-analyze matching embryos
    embryo_filter: Optional[List[str]] = None  # None = all embryos

    # Rate limiting
    min_interval_seconds: float = 5.0  # Min time between auto-analyses


@dataclass
class CVSubagentConfig:
    """Main configuration for CV Subagent"""

    # Service settings
    host: str = "localhost"
    port: int = 8100

    # API keys
    anthropic_api_key: Optional[str] = field(
        default_factory=lambda: os.environ.get("ANTHROPIC_API_KEY")
    )

    # Data store
    data_store_url: Optional[str] = field(
        default_factory=lambda: os.environ.get("GENTLY_DATA_STORE_URL")
    )

    # GPU settings
    gpu_device: int = 0
    gpu_memory_limit_mb: Optional[int] = None

    # Task queue settings
    max_concurrent_tasks: int = 2

    # Model settings
    models: Dict[str, ModelConfig] = field(default_factory=dict)

    # Auto-analysis settings
    auto_analysis: AutoAnalysisConfig = field(default_factory=AutoAnalysisConfig)

    # Microscope and organism context (explicit, no guessing)
    microscope_type: str = "diSPIM"
    has_dual_view: bool = True
    view_a_range: tuple = (0, 1024)  # X pixel range for View A
    organism: str = "c_elegans"
    expected_cell_diameter_um: float = 5.0  # Typical nucleus diameter

    # C. elegans specific settings
    scale_um_per_px: float = 0.406  # Microscope scale (XY)
    scale_um_per_px_z: float = 1.0  # Z step size
    default_scale_bar_um: float = 10.0

    # Cellpose defaults
    cellpose_model_type: str = "cyto2"
    cellpose_diameter: Optional[float] = None  # Auto-detect
    cellpose_anisotropy: float = 2.0  # Z vs XY spacing ratio

    # StarDist defaults
    stardist_model: str = "2D_versatile_fluo"

    # Claude Vision settings
    vision_model: str = "claude-sonnet-4-5-20250929"
    vision_max_tokens: int = 4096

    # Extended thinking settings (for multi-step reasoning between tool calls)
    enable_interleaved_thinking: bool = True
    thinking_budget_tokens: int = 10000  # Budget for thinking across all steps

    @classmethod
    def from_env(cls) -> "CVSubagentConfig":
        """Create config from environment variables"""
        return cls(
            host=os.environ.get("CV_SUBAGENT_HOST", "localhost"),
            port=int(os.environ.get("CV_SUBAGENT_PORT", "8100")),
            anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY"),
            data_store_url=os.environ.get("GENTLY_DATA_STORE_URL"),
            gpu_device=int(os.environ.get("CV_SUBAGENT_GPU_DEVICE", "0")),
            max_concurrent_tasks=int(os.environ.get("CV_SUBAGENT_MAX_CONCURRENT", "2")),
        )

    def create_context(
        self,
        embryo_id: str = None,
        timepoint: int = None,
        session_id: str = None,
        additional_context: Dict = None,
    ) -> "CVContext":
        """
        Create a CVContext from this config.

        Parameters
        ----------
        embryo_id : str, optional
            Embryo ID being analyzed
        timepoint : int, optional
            Specific timepoint
        session_id : str, optional
            Session ID from copilot
        additional_context : dict, optional
            Additional context from copilot

        Returns
        -------
        CVContext
            Context ready for CV tool execution
        """
        from .context import CVContext

        return CVContext(
            microscope_type=self.microscope_type,
            scale_um_per_px_xy=self.scale_um_per_px,
            scale_um_per_px_z=self.scale_um_per_px_z,
            has_dual_view=self.has_dual_view,
            view_a_range=self.view_a_range,
            organism=self.organism,
            expected_cell_diameter_um=self.expected_cell_diameter_um,
            session_id=session_id,
            embryo_id=embryo_id,
            timepoint=timepoint,
            data_store_url=self.data_store_url,
            additional_context=additional_context or {},
        )


# Default configuration
default_config = CVSubagentConfig()


# C. elegans developmental stages with approximate nuclei counts
CELEGANS_STAGES = {
    "1-cell": {"nuclei_min": 1, "nuclei_max": 1, "description": "Single cell (zygote)"},
    "2-cell": {"nuclei_min": 2, "nuclei_max": 2, "description": "First division"},
    "4-cell": {"nuclei_min": 4, "nuclei_max": 4, "description": "Second division"},
    "8-cell": {"nuclei_min": 8, "nuclei_max": 8, "description": "Third division"},
    "16-cell": {"nuclei_min": 16, "nuclei_max": 16, "description": "Fourth division"},
    "28-cell": {"nuclei_min": 24, "nuclei_max": 32, "description": "Gastrulation begins"},
    "gastrula": {"nuclei_min": 28, "nuclei_max": 100, "description": "Gastrulation"},
    "bean": {"nuclei_min": 100, "nuclei_max": 200, "description": "Bean stage"},
    "comma": {"nuclei_min": 200, "nuclei_max": 400, "description": "Elongation begins"},
    "1.5-fold": {"nuclei_min": 400, "nuclei_max": 500, "description": "1.5x body length"},
    "2-fold": {"nuclei_min": 500, "nuclei_max": 550, "description": "2x body length"},
    "3-fold": {"nuclei_min": 550, "nuclei_max": 558, "description": "3x body length"},
    "pretzel": {"nuclei_min": 558, "nuclei_max": 558, "description": "Maximum folding"},
    "hatching": {"nuclei_min": 558, "nuclei_max": 558, "description": "Ready to hatch"},
}
