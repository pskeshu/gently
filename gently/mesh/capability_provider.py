"""
DynamicCapabilityProvider — Replaces the static lambda in MeshService.

Detects GPUs, computes roles dynamically, queries FileStore for
dataset advertisements. Refreshed on each heartbeat/status response.
"""

import logging
import os
import platform
from typing import Any

from .models import DatasetAdvertisement, GpuInfo, PeerRole

logger = logging.getLogger(__name__)


def _detect_gpus() -> list[GpuInfo]:
    """Detect available NVIDIA GPUs via torch.cuda (pynvml fallback)."""
    gpus = []
    try:
        import torch

        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                # Get live utilization if pynvml available
                util_pct = 0.0
                mem_used_gb = 0.0
                try:
                    import pynvml

                    pynvml.nvmlInit()
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    util_pct = float(util.gpu)
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    mem_used_gb = mem_info.used / (1024**3)
                except Exception:
                    pass

                gpus.append(
                    GpuInfo(
                        device_index=i,
                        name=props.name,
                        vram_gb=round(props.total_mem / (1024**3), 1),
                        compute_capability=f"{props.major}.{props.minor}",
                        utilization_pct=util_pct,
                        memory_used_gb=round(mem_used_gb, 2),
                    )
                )
    except ImportError:
        pass
    except Exception as e:
        logger.debug(f"GPU detection failed: {e}")
    return gpus


def _get_system_info() -> dict[str, Any]:
    """Get CPU and RAM info."""
    cpu_cores = os.cpu_count() or 0
    ram_gb = 0.0
    try:
        if platform.system() == "Windows":
            import ctypes

            # ctypes.windll exists only on Windows; this branch is platform-guarded.
            kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
            mem_status = ctypes.c_ulonglong()
            kernel32.GetPhysicallyInstalledSystemMemory(ctypes.byref(mem_status))
            ram_gb = round(mem_status.value / (1024 * 1024), 1)
        else:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        ram_kb = int(line.split()[1])
                        ram_gb = round(ram_kb / (1024 * 1024), 1)
                        break
    except Exception:
        pass
    return {"cpu_cores": cpu_cores, "ram_gb": ram_gb}


class DynamicCapabilityProvider:
    """Provides rich, dynamic capability information for this node.

    Parameters
    ----------
    gently_store : optional
        FileStore instance for querying dataset info.
    device_layer : optional
        DeviceLayer for checking microscope connection.
    static_caps : dict
        Static capability overrides (organism, hardware_profile, etc.).
    """

    def __init__(
        self,
        gently_store=None,
        device_layer=None,
        static_caps: dict[str, Any] | None = None,
    ):
        self._store = gently_store
        self._device_layer = device_layer
        self._static = static_caps or {}
        self._gpus: list[GpuInfo] = []
        self._system_info = _get_system_info()
        # Initial GPU detection (cached, refreshed on demand)
        self._gpus = _detect_gpus()

    def refresh_gpus(self):
        """Re-detect GPUs (call periodically for live utilization)."""
        self._gpus = _detect_gpus()

    def _compute_roles(self) -> list[str]:
        """Determine dynamic roles based on current state."""
        roles = []
        # Microscope controller if device is connected and responding
        if self._device_layer is not None:
            try:
                if getattr(self._device_layer, "connected", False):
                    roles.append(PeerRole.MICROSCOPE_CONTROLLER.value)
            except Exception:
                pass
        elif self._static.get("has_microscope"):
            roles.append(PeerRole.MICROSCOPE_CONTROLLER.value)

        # ML trainer if GPU available
        if self._gpus:
            roles.append(PeerRole.ML_TRAINER.value)

        # Data server if datasets exist
        if self._get_datasets():
            roles.append(PeerRole.DATA_SERVER.value)

        # Planner always (every node can plan)
        roles.append(PeerRole.PLANNER.value)
        return roles

    def _get_datasets(self) -> list[DatasetAdvertisement]:
        """Query FileStore for dataset advertisements."""
        if self._store is None:
            return []
        datasets = []
        try:
            sessions = self._store.list_sessions()
            for sess in sessions:
                sid = sess.session_id if hasattr(sess, "session_id") else sess.get("session_id", "")
                sname = sess.name if hasattr(sess, "name") else sess.get("name", "")

                embryos = self._store.list_embryos(sid)
                embryo_count = len(embryos)
                if embryo_count == 0:
                    continue

                vol_count = 0
                gt_count = 0
                stages = set()
                for emb in embryos:
                    eid = emb.embryo_id if hasattr(emb, "embryo_id") else emb.get("embryo_id", "")
                    vols = self._store.list_volumes(sid, eid)
                    vol_count += len(vols)
                    try:
                        gts = self._store.get_ground_truth(sid, eid)
                        gt_count += len(gts)
                        for gt in gts:
                            stage = gt.stage if hasattr(gt, "stage") else gt.get("stage", "")
                            if stage:
                                stages.add(stage)
                    except Exception:
                        pass

                datasets.append(
                    DatasetAdvertisement(
                        session_id=sid,
                        session_name=sname,
                        embryo_count=embryo_count,
                        volume_count=vol_count,
                        has_ground_truth=gt_count > 0,
                        ground_truth_count=gt_count,
                        stages_covered=sorted(stages),
                    )
                )
        except Exception as e:
            logger.debug(f"Dataset advertisement failed: {e}")
        return datasets

    def _is_microscope_connected(self) -> bool:
        """Check if microscope hardware is actually responding."""
        if self._device_layer is not None:
            try:
                return bool(getattr(self._device_layer, "connected", False))
            except Exception:
                pass
        return False

    def __call__(self) -> dict[str, Any]:
        """Build the full capability dict. Called on each heartbeat."""
        datasets = self._get_datasets()
        roles = self._compute_roles()
        has_gpu = bool(self._gpus)
        gpu_name = self._gpus[0].name if self._gpus else self._static.get("gpu_name", "")
        gpu_vram = self._gpus[0].vram_gb if self._gpus else self._static.get("gpu_vram_gb", 0.0)

        # Get storage info
        storage_free_gb = self._static.get("storage_free_gb", 0.0)
        storage_total_gb = 0.0
        try:
            import shutil

            from ..settings import settings

            usage = shutil.disk_usage(str(settings.storage.base_path))
            storage_free_gb = round(usage.free / (1024**3), 1)
            storage_total_gb = round(usage.total / (1024**3), 1)
        except Exception:
            pass

        return {
            # Legacy fields (backward compat)
            "has_microscope": self._static.get("has_microscope", False),
            "has_sam": self._static.get("has_sam", False),
            "has_gpu": has_gpu,
            "gpu_name": gpu_name,
            "gpu_vram_gb": gpu_vram,
            "storage_free_gb": storage_free_gb,
            "tool_categories": self._static.get("tool_categories", []),
            "organism": self._static.get("organism", ""),
            "hardware_profile": self._static.get("hardware_profile", ""),
            # Enhanced fields
            "gpus": [g.to_dict() for g in self._gpus],
            "roles": roles,
            "datasets": [d.to_dict() for d in datasets],
            "microscope_connected": self._is_microscope_connected(),
            "cpu_cores": self._system_info["cpu_cores"],
            "ram_gb": self._system_info["ram_gb"],
            "storage_total_gb": storage_total_gb,
        }
