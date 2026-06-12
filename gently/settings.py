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
    broadcast_interval_s: float = field(default_factory=lambda: _env("MESH_BROADCAST_INTERVAL", 5.0))
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
      - main:       Claude Fable 5 ($10/$50 per MTok). Per-user-turn reasoning +
                    tool orchestration (plan mode) and the dopaminergic classifier
                    stage. Always-on thinking (no thinking budget — control depth
                    via output_config.effort), ~30%-heavier tokenizer, may refuse
                    (stop_reason="refusal", empty content); needs ≥30-day org
                    data retention.
      - perception: Opus 4.8 (high-res vision, $5/$25). Highest-frequency tier
                    (per timepoint); Opus-tier vision for perception accuracy.
      - medium:     Opus 4.8. Onboarding / wizard summaries.
      - fast:       Sonnet 4.6 ($3/$15). The cheaper/faster tier — drives the
                    verifier's parallel ensemble (ensemble_size calls per
                    verification) and blank-image / summary checks.

    API note: Fable 5 and Opus 4.8 reject thinking budget_tokens and sampling
    params (temperature/top_p/top_k) — adaptive thinking only, depth via effort.
    Sonnet 4.6 supports adaptive thinking. No assistant prefills anywhere (4.6+
    family rejects them).
    """
    main: str = field(default_factory=lambda: _env("MODEL_MAIN", "claude-fable-5"))
    perception: str = field(default_factory=lambda: _env("MODEL_PERCEPTION", "claude-opus-4-8"))
    fast: str = field(default_factory=lambda: _env("MODEL_FAST", "claude-sonnet-4-6"))
    medium: str = field(default_factory=lambda: _env("MODEL_MEDIUM", "claude-opus-4-8"))
    # When the main tier (Fable 5) declines a turn (stop_reason="refusal"), the
    # main-tier calls transparently retry it on this model instead of surfacing
    # the refusal. Empty disables the fallback.
    refusal_fallback: str = field(default_factory=lambda: _env("MODEL_REFUSAL_FALLBACK", "claude-opus-4-8"))


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
    rpc_call: int = field(default_factory=lambda: _env("TIMEOUT_RPC", 60))
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
    max_concurrent_transfers: int = field(default_factory=lambda: _env("TRANSFER_MAX_CONCURRENT", 4))


@dataclass(frozen=True)
class UISettings:
    """Web UI feature flags."""
    # New agent-first UX paradigm (welcome→shell unfold, dual-rendered agent
    # asks, inference-first plan mode, shared-visibility surface). Now ON by
    # default; the v1 dashboard remains available as a fallback via
    # GENTLY_UX_V2=0 until the v1 markup is removed in a later cleanup step.
    ux_v2: bool = field(default_factory=lambda: _env("UX_V2", True))


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
