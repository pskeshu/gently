"""
CVContext - Explicit microscope and organism context for CV tools.

Eliminates the need for the CV agent to infer context from image dimensions.
All tools receive this context and can make microscope-aware decisions.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple


@dataclass
class CVContext:
    """
    Explicit microscope and organism context passed to all CV tools.

    This eliminates guessing like "This is likely a diSPIM image based on
    the wide width (2048 pixels)" by providing explicit configuration.

    Attributes
    ----------
    microscope_type : str
        Type of microscope: "diSPIM", "confocal", "widefield"
    scale_um_per_px_xy : float
        XY pixel size in micrometers
    scale_um_per_px_z : float
        Z step size in micrometers
    has_dual_view : bool
        For diSPIM: whether data contains both views side-by-side
    view_a_range : tuple
        X pixel range for View A (left half) when has_dual_view=True
    organism : str
        Organism being imaged: "c_elegans", "zebrafish", etc.
    expected_cell_diameter_um : float
        Typical nucleus/cell diameter for this organism
    session_id : str, optional
        Session ID from copilot
    embryo_id : str, optional
        Embryo ID being analyzed
    timepoint : int, optional
        Specific timepoint being analyzed
    """

    # Microscope context
    microscope_type: str = "diSPIM"
    scale_um_per_px_xy: float = 0.406
    scale_um_per_px_z: float = 1.0
    has_dual_view: bool = True
    view_a_range: Tuple[int, int] = (0, 1024)

    # Organism context
    organism: str = "c_elegans"
    expected_cell_diameter_um: float = 5.0

    # Processing options
    downsample_factor: float = 2.0  # Default 2x downsampling for speed

    # Session context (populated from copilot request)
    session_id: Optional[str] = None
    embryo_id: Optional[str] = None
    timepoint: Optional[int] = None

    # Data store reference
    data_store_url: Optional[str] = None

    # Additional metadata from copilot
    additional_context: Dict = field(default_factory=dict)

    @property
    def cell_diameter_px(self) -> float:
        """Expected cell diameter in pixels (for Cellpose)."""
        return self.expected_cell_diameter_um / self.scale_um_per_px_xy

    @property
    def anisotropy(self) -> float:
        """Z/XY anisotropy ratio for 3D segmentation."""
        return self.scale_um_per_px_z / self.scale_um_per_px_xy

    @property
    def is_dispim(self) -> bool:
        """Check if this is a diSPIM microscope."""
        return self.microscope_type.lower() == "dispim"

    @property
    def is_celegans(self) -> bool:
        """Check if organism is C. elegans."""
        return self.organism.lower() in ("c_elegans", "celegans", "c. elegans")

    def get_view_a_slice(self) -> slice:
        """Get slice object for extracting View A from dual-view image."""
        return slice(self.view_a_range[0], self.view_a_range[1])

    @classmethod
    def from_config(cls, config: "CVSubagentConfig") -> "CVContext":
        """
        Create context from CVSubagentConfig defaults.

        Parameters
        ----------
        config : CVSubagentConfig
            Configuration object

        Returns
        -------
        CVContext
            Context with default microscope/organism settings
        """
        return cls(
            scale_um_per_px_xy=config.scale_um_per_px,
            data_store_url=config.data_store_url,
        )

    @classmethod
    def from_copilot_request(
        cls,
        embryo_id: str,
        timepoint: Optional[int] = None,
        session_id: Optional[str] = None,
        additional_context: Optional[Dict] = None,
        config: Optional["CVSubagentConfig"] = None,
    ) -> "CVContext":
        """
        Build context from a copilot analysis request.

        Parameters
        ----------
        embryo_id : str
            ID of the embryo to analyze
        timepoint : int, optional
            Specific timepoint to analyze
        session_id : str, optional
            Session ID from copilot
        additional_context : dict, optional
            Additional context from copilot (current stage, etc.)
        config : CVSubagentConfig, optional
            Configuration for microscope defaults

        Returns
        -------
        CVContext
            Context ready for CV tool execution
        """
        ctx = cls() if config is None else cls.from_config(config)
        ctx.embryo_id = embryo_id
        ctx.timepoint = timepoint
        ctx.session_id = session_id
        ctx.additional_context = additional_context or {}
        return ctx

    def to_dict(self) -> Dict:
        """Serialize context to dictionary."""
        return {
            "microscope_type": self.microscope_type,
            "scale_um_per_px_xy": self.scale_um_per_px_xy,
            "scale_um_per_px_z": self.scale_um_per_px_z,
            "has_dual_view": self.has_dual_view,
            "view_a_range": list(self.view_a_range),
            "organism": self.organism,
            "expected_cell_diameter_um": self.expected_cell_diameter_um,
            "downsample_factor": self.downsample_factor,
            "session_id": self.session_id,
            "embryo_id": self.embryo_id,
            "timepoint": self.timepoint,
            "data_store_url": self.data_store_url,
            "additional_context": self.additional_context,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "CVContext":
        """Deserialize context from dictionary."""
        view_a_range = data.get("view_a_range", [0, 1024])
        if isinstance(view_a_range, list):
            view_a_range = tuple(view_a_range)

        return cls(
            microscope_type=data.get("microscope_type", "diSPIM"),
            scale_um_per_px_xy=data.get("scale_um_per_px_xy", 0.406),
            scale_um_per_px_z=data.get("scale_um_per_px_z", 1.0),
            has_dual_view=data.get("has_dual_view", True),
            view_a_range=view_a_range,
            organism=data.get("organism", "c_elegans"),
            expected_cell_diameter_um=data.get("expected_cell_diameter_um", 5.0),
            downsample_factor=data.get("downsample_factor", 2.0),
            session_id=data.get("session_id"),
            embryo_id=data.get("embryo_id"),
            timepoint=data.get("timepoint"),
            data_store_url=data.get("data_store_url"),
            additional_context=data.get("additional_context", {}),
        )
