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


@dataclass(frozen=True)
class ModelSettings:
    """Claude model identifiers."""
    main: str = field(default_factory=lambda: _env("MODEL_MAIN", "claude-opus-4-6"))
    perception: str = field(default_factory=lambda: _env("MODEL_PERCEPTION", "claude-opus-4-5-20251101"))
    fast: str = field(default_factory=lambda: _env("MODEL_FAST", "claude-haiku-4-5-20251001"))
    medium: str = field(default_factory=lambda: _env("MODEL_MEDIUM", "claude-sonnet-4-5-20250929"))


@dataclass(frozen=True)
class StorageSettings:
    """File paths for data storage."""
    base_path: Path = field(default_factory=lambda: _env("STORAGE_PATH", Path("D:/Gently2")))

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
class Settings:
    """Top-level settings container."""
    network: NetworkSettings = field(default_factory=NetworkSettings)
    mesh: MeshSettings = field(default_factory=MeshSettings)
    models: ModelSettings = field(default_factory=ModelSettings)
    storage: StorageSettings = field(default_factory=StorageSettings)
    timeouts: TimeoutSettings = field(default_factory=TimeoutSettings)
    api: ApiSettings = field(default_factory=ApiSettings)


# Singleton — import this everywhere
settings = Settings()
