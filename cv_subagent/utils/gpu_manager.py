"""
GPU Resource Manager for CV Subagent

Manages GPU resources for deep learning models (Cellpose, StarDist).
Handles device selection, memory monitoring, and model caching.
"""

import logging
import threading
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class GPUManager:
    """
    Manages GPU resources for CV operations

    Features:
    - Automatic device selection (GPU if available, else CPU)
    - GPU memory monitoring
    - Model caching to avoid reloading
    - Thread-safe operations
    """

    def __init__(self, prefer_gpu: bool = True, device_id: int = 0):
        """
        Initialize GPU manager

        Parameters
        ----------
        prefer_gpu : bool
            Prefer GPU if available
        device_id : int
            GPU device ID to use
        """
        self.prefer_gpu = prefer_gpu
        self.device_id = device_id
        self._lock = threading.Lock()

        # Model cache
        self._models: Dict[str, Any] = {}

        # Check GPU availability
        self._gpu_available = False
        self._device = "cpu"
        self._check_gpu()

    def _check_gpu(self):
        """Check GPU availability"""
        try:
            import torch
            self._gpu_available = torch.cuda.is_available()
            if self._gpu_available and self.prefer_gpu:
                self._device = f"cuda:{self.device_id}"
                gpu_name = torch.cuda.get_device_name(self.device_id)
                logger.info(f"GPU available: {gpu_name}")
            else:
                self._device = "cpu"
                if not self._gpu_available:
                    logger.info("No GPU available, using CPU")
                else:
                    logger.info("GPU available but CPU preferred")
        except ImportError:
            logger.warning("PyTorch not installed, GPU features disabled")
            self._device = "cpu"

    @property
    def device(self) -> str:
        """Get current device string"""
        return self._device

    @property
    def gpu_available(self) -> bool:
        """Check if GPU is available"""
        return self._gpu_available

    def get_memory_info(self) -> Dict[str, float]:
        """
        Get GPU memory information

        Returns
        -------
        dict
            Memory info with keys: used_mb, total_mb, free_mb, utilization
        """
        if not self._gpu_available:
            return {
                "used_mb": 0,
                "total_mb": 0,
                "free_mb": 0,
                "utilization": 0,
            }

        try:
            import torch
            used = torch.cuda.memory_allocated(self.device_id) / 1024 / 1024
            total = torch.cuda.get_device_properties(self.device_id).total_memory / 1024 / 1024
            free = total - used
            utilization = used / total if total > 0 else 0

            return {
                "used_mb": used,
                "total_mb": total,
                "free_mb": free,
                "utilization": utilization,
            }
        except Exception as e:
            logger.warning(f"Could not get GPU memory info: {e}")
            return {"used_mb": 0, "total_mb": 0, "free_mb": 0, "utilization": 0}

    def clear_cache(self):
        """Clear GPU memory cache"""
        if self._gpu_available:
            try:
                import torch
                torch.cuda.empty_cache()
                logger.debug("Cleared GPU cache")
            except Exception as e:
                logger.warning(f"Could not clear GPU cache: {e}")

    def get_cellpose_model(
        self,
        model_type: str = "cyto2",
        gpu: Optional[bool] = None,
    ) -> Any:
        """
        Get or create Cellpose model

        Parameters
        ----------
        model_type : str
            Model type: "cyto", "cyto2", "nuclei", etc.
        gpu : bool, optional
            Use GPU (defaults to self.gpu_available)

        Returns
        -------
        cellpose.models.Cellpose
            Cellpose model instance
        """
        cache_key = f"cellpose_{model_type}"

        with self._lock:
            if cache_key in self._models:
                logger.debug(f"Using cached Cellpose model: {model_type}")
                return self._models[cache_key]

            try:
                from cellpose import models

                use_gpu = gpu if gpu is not None else self._gpu_available

                logger.info(f"Loading Cellpose model: {model_type} (GPU={use_gpu})")
                model = models.Cellpose(
                    gpu=use_gpu,
                    model_type=model_type,
                    device=None if not use_gpu else self.device_id,
                )

                self._models[cache_key] = model
                return model

            except ImportError:
                raise RuntimeError("Cellpose not installed: pip install cellpose")

    def get_stardist_model(
        self,
        model_name: str = "2D_versatile_fluo",
        use_3d: bool = False,
    ) -> Any:
        """
        Get or create StarDist model

        Parameters
        ----------
        model_name : str
            Model name for 2D: "2D_versatile_fluo", "2D_versatile_he"
            For 3D: "3D_demo"
        use_3d : bool
            Use 3D model

        Returns
        -------
        stardist.models.StarDist2D or StarDist3D
            StarDist model instance
        """
        cache_key = f"stardist_{model_name}_3d" if use_3d else f"stardist_{model_name}_2d"

        with self._lock:
            if cache_key in self._models:
                logger.debug(f"Using cached StarDist model: {model_name}")
                return self._models[cache_key]

            try:
                if use_3d:
                    from stardist.models import StarDist3D
                    logger.info(f"Loading StarDist3D model: {model_name}")
                    model = StarDist3D.from_pretrained(model_name)
                else:
                    from stardist.models import StarDist2D
                    logger.info(f"Loading StarDist2D model: {model_name}")
                    model = StarDist2D.from_pretrained(model_name)

                self._models[cache_key] = model
                return model

            except ImportError:
                raise RuntimeError("StarDist not installed: pip install stardist")

    def unload_model(self, model_key: str):
        """Unload a specific model from cache"""
        with self._lock:
            if model_key in self._models:
                del self._models[model_key]
                self.clear_cache()
                logger.info(f"Unloaded model: {model_key}")

    def unload_all_models(self):
        """Unload all cached models"""
        with self._lock:
            self._models.clear()
            self.clear_cache()
            logger.info("Unloaded all models")

    def list_cached_models(self) -> List[str]:
        """List currently cached model keys"""
        return list(self._models.keys())

    def get_status(self) -> Dict[str, Any]:
        """Get GPU manager status"""
        memory = self.get_memory_info()
        return {
            "gpu_available": self._gpu_available,
            "device": self._device,
            "memory_used_mb": memory["used_mb"],
            "memory_total_mb": memory["total_mb"],
            "memory_utilization": memory["utilization"],
            "cached_models": self.list_cached_models(),
        }


# Global GPU manager instance
_gpu_manager: Optional[GPUManager] = None


def get_gpu_manager(
    prefer_gpu: bool = True,
    device_id: int = 0,
) -> GPUManager:
    """
    Get or create global GPU manager

    Parameters
    ----------
    prefer_gpu : bool
        Prefer GPU if available
    device_id : int
        GPU device ID

    Returns
    -------
    GPUManager
        Global GPU manager instance
    """
    global _gpu_manager
    if _gpu_manager is None:
        _gpu_manager = GPUManager(prefer_gpu=prefer_gpu, device_id=device_id)
    return _gpu_manager
