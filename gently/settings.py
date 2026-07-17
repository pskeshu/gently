"""
Centralized settings for the Gently system.

All configurable values live here. Override via environment variables
prefixed with GENTLY_ (e.g., GENTLY_VIZ_PORT=9090).
"""

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env(key: str, default):
    """Read from GENTLY_<KEY> env var, falling back to default."""
    val = os.environ.get(f"GENTLY_{key}")
    if val is None:
        return default
    # Coerce to the type of default
    if isinstance(default, bool):
        return val.lower() in ("1", "true", "yes")
    if isinstance(default, int):
        return int(val)
    if isinstance(default, float):
        return float(val)
    if isinstance(default, Path):
        return Path(val)
    return val


def _load_local_overrides():
    """Merge config/settings.local.yml (a flat map of GENTLY_* keys) into the
    environment BEFORE settings are resolved. setdefault so a real env var still
    wins over the file. This is how the Settings panel's restart-required editors
    persist overrides — every entry point (viz, device layer, agent) picks them
    up at import."""
    try:
        import yaml

        path = Path(__file__).resolve().parents[1] / "config" / "settings.local.yml"
        if not path.exists():
            return
        data = yaml.safe_load(path.read_text()) or {}
        if isinstance(data, dict):
            for k, v in data.items():
                if v is not None:
                    os.environ.setdefault(str(k), str(v))
    except Exception:
        pass


_load_local_overrides()


@dataclass(frozen=True)
class NetworkSettings:
    """Ports, hosts, and bind addresses."""

    viz_port: int = field(default_factory=lambda: _env("VIZ_PORT", 8080))
    viz_host: str = field(default_factory=lambda: _env("VIZ_HOST", "0.0.0.0"))
    device_port: int = field(default_factory=lambda: _env("DEVICE_PORT", 60610))
    device_host: str = field(default_factory=lambda: _env("DEVICE_HOST", "127.0.0.1"))
    mesh_port: int = field(default_factory=lambda: _env("MESH_PORT", 19547))
    mesh_bind: str = field(default_factory=lambda: _env("MESH_BIND", "0.0.0.0"))


@dataclass(frozen=True)
class MeshSettings:
    """Mesh networking parameters."""

    broadcast_interval_s: float = field(
        default_factory=lambda: _env("MESH_BROADCAST_INTERVAL", 5.0)
    )
    replay_window_s: float = field(default_factory=lambda: _env("MESH_REPLAY_WINDOW", 30.0))
    reaper_interval_s: float = field(default_factory=lambda: _env("MESH_REAPER_INTERVAL", 10.0))
    status_refresh_s: float = field(default_factory=lambda: _env("MESH_STATUS_REFRESH", 30.0))
    fetch_timeout_s: float = field(default_factory=lambda: _env("MESH_FETCH_TIMEOUT", 5.0))
    stale_threshold_s: float = field(default_factory=lambda: _env("MESH_STALE_THRESHOLD", 15.0))
    dead_threshold_s: float = field(default_factory=lambda: _env("MESH_DEAD_THRESHOLD", 30.0))


@dataclass(frozen=True)
class ModelSettings:
    """Claude model identifiers — the single source of truth for every tier.

    Tiers are split by role; capability-first per the latest models:
      - main:       Opus 4.8 ($5/$25). Per-user-turn reasoning + tool
                    orchestration (plan mode) and the dopaminergic classifier
                    stage. (Fable 5 was tried here but declined benign planning
                    turns — stop_reason="refusal" — forcing a fallback on every
                    turn; set MODEL_MAIN=claude-fable-5 to retry it.)
      - perception: Opus 4.8 (high-res vision, $5/$25). Highest-frequency tier
                    (per timepoint); Opus-tier vision for perception accuracy.
      - medium:     Opus 4.8. Onboarding / wizard summaries.
      - fast:       Sonnet 4.6 ($3/$15). The cheaper/faster tier — drives the
                    verifier's parallel ensemble (ensemble_size calls per
                    verification) and blank-image / summary checks.

    API note: Opus 4.8 rejects thinking budget_tokens and sampling params
    (temperature/top_p/top_k) — adaptive thinking only, depth via effort.
    Sonnet 4.6 supports adaptive thinking. No assistant prefills anywhere
    (4.6+ family rejects them).
    """

    main: str = field(default_factory=lambda: _env("MODEL_MAIN", "claude-opus-4-8"))
    perception: str = field(default_factory=lambda: _env("MODEL_PERCEPTION", "claude-opus-4-8"))
    fast: str = field(default_factory=lambda: _env("MODEL_FAST", "claude-sonnet-4-6"))
    medium: str = field(default_factory=lambda: _env("MODEL_MEDIUM", "claude-opus-4-8"))
    # If the main tier declines a turn (stop_reason="refusal"), retry it on this
    # model instead of surfacing the refusal. Inert while main is Opus 4.8 (the
    # guard skips it when fallback == main); relevant if main is set to Fable 5.
    # Empty disables the fallback.
    refusal_fallback: str = field(
        default_factory=lambda: _env("MODEL_REFUSAL_FALLBACK", "claude-opus-4-8")
    )


@dataclass(frozen=True)
class StorageSettings:
    """File paths for data storage."""

    base_path: Path = field(default_factory=lambda: _env("STORAGE_PATH", Path("D:/Gently3")))

    @property
    def sessions_dir(self) -> Path:
        return self.base_path / "sessions"

    @property
    def traces_dir(self) -> Path:
        return self.base_path / "traces"


@dataclass(frozen=True)
class TimeoutSettings:
    """Timeout values in seconds."""

    plan_execution: int = field(default_factory=lambda: _env("TIMEOUT_PLAN", 300))
    volume_acquisition: int = field(default_factory=lambda: _env("TIMEOUT_VOLUME", 15))
    api_call: int = field(default_factory=lambda: _env("TIMEOUT_API", 10))


@dataclass(frozen=True)
class ApiSettings:
    """External API configuration."""

    ncbi_tool: str = field(default_factory=lambda: _env("NCBI_TOOL", "gently"))
    ncbi_email: str = field(default_factory=lambda: _env("NCBI_EMAIL", "pskeshu@gmail.com"))


@dataclass(frozen=True)
class MlSettings:
    """Machine learning training parameters."""

    model_cache_dir: Path = field(default_factory=lambda: _env("ML_MODEL_CACHE", Path("models")))
    default_batch_size: int = field(default_factory=lambda: _env("ML_BATCH_SIZE", 32))
    default_epochs: int = field(default_factory=lambda: _env("ML_EPOCHS", 50))
    default_lr: float = field(default_factory=lambda: _env("ML_LR", 1e-4))


@dataclass(frozen=True)
class TransferSettings:
    """Bulk transfer protocol parameters."""

    transfer_port: int = field(default_factory=lambda: _env("TRANSFER_PORT", 19548))
    chunk_size: int = field(default_factory=lambda: _env("TRANSFER_CHUNK_SIZE", 1048576))  # 1MB
    max_concurrent_transfers: int = field(
        default_factory=lambda: _env("TRANSFER_MAX_CONCURRENT", 4)
    )


@dataclass(frozen=True)
class UISettings:
    """Web UI feature flags."""

    # New agent-first UX paradigm (welcome→shell unfold, dual-rendered agent
    # asks, inference-first plan mode, shared-visibility surface). Now ON by
    # default; the v1 dashboard remains available as a fallback via
    # GENTLY_UX_V2=0 until the v1 markup is removed in a later cleanup step.
    ux_v2: bool = field(default_factory=lambda: _env("UX_V2", True))

    # Always-on session replay (rrweb capture + semantic action log) feeding
    # post-hoc human replay and agent postmortems. GENTLY_REPLAY=0 is the kill
    # switch: it drops the recorder include and 403s the ingest endpoint; the
    # player stays available for existing recordings.
    replay: bool = field(default_factory=lambda: _env("REPLAY", True))
    # Default recording fidelity — one size does not fit all. High-fidelity DOM
    # churn (the live map re-rendering at poll rate) is invaluable for a specific
    # visual debug but wasteful always-on. Levels (overridable per-load with
    # ?replay=<level>):
    #   full     — record every DOM mutation (highest churn; deep visual debug)
    #   balanced — block the machine-driven high-churn regions (map, 3D, temp
    #              graph); keeps clicks + panels. The sane always-on default.
    #   actions  — semantic action log only (clicks/nav), no rrweb DOM stream
    replay_fidelity: str = field(default_factory=lambda: _env("REPLAY_FIDELITY", "balanced"))
    # Cap one tab's rrweb stream. A dashboard tab left open re-renders the live
    # map / telemetry at poll rate, so a single long-lived tab can balloon its
    # recording to gigabytes. Past the cap, rrweb frames are dropped (a one-time
    # marker is logged); the small semantic action log keeps recording.
    replay_max_tab_mb: float = field(default_factory=lambda: _env("REPLAY_MAX_TAB_MB", 120.0))
    # Total on-disk budget for all rrweb recordings. Oldest recordings are pruned
    # (once, lazily) to keep the footprint under this.
    replay_total_budget_mb: float = field(
        default_factory=lambda: _env("REPLAY_TOTAL_BUDGET_MB", 1024.0)
    )


@dataclass(frozen=True)
class Settings:
    """Top-level settings container."""

    network: NetworkSettings = field(default_factory=NetworkSettings)
    mesh: MeshSettings = field(default_factory=MeshSettings)
    models: ModelSettings = field(default_factory=ModelSettings)
    storage: StorageSettings = field(default_factory=StorageSettings)
    timeouts: TimeoutSettings = field(default_factory=TimeoutSettings)
    api: ApiSettings = field(default_factory=ApiSettings)
    ml: MlSettings = field(default_factory=MlSettings)
    transfer: TransferSettings = field(default_factory=TransferSettings)
    ui: UISettings = field(default_factory=UISettings)


# Singleton — import this everywhere
settings = Settings()
