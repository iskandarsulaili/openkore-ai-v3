"""GPU detection and device selection utility with CPU fallback.

Auto-detects the best available compute device:
1. CUDA GPU (NVIDIA) — detected via nvidia-smi
2. ROCm GPU (AMD) — detected via rocminfo
3. CPU fallback — default when no GPU found

For sklearn models, GPU means using cuml (RAPIDS) which provides
GPU-accelerated RandomForest, KNN, etc. with sklearn-compatible API.
When cuml is not available, falls back to CPU sklearn with n_jobs=-1.
"""

from __future__ import annotations

import logging
import os
import subprocess
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class DeviceInfo:
    """Detected compute device information."""
    device_type: str  # "cuda" | "rocm" | "cpu"
    device_name: str
    memory_gb: float
    compute_capability: str  # CUDA capability or ROCm version
    n_jobs: int  # CPU cores for parallel processing
    cuml_available: bool  # RAPIDS cuml library available


def detect_device() -> DeviceInfo:
    """Detect the best available compute device.
    
    Returns DeviceInfo with the detected device details.
    Priority: CUDA > ROCm > CPU
    """
    n_jobs = _get_cpu_count()
    
    # Try CUDA first (NVIDIA)
    cuda_info = _detect_cuda()
    if cuda_info is not None:
        cuml_avail = _check_cuml()
        if cuml_avail:
            logger.info("gpu_detected: cuda device=%s memory=%.1fGB compute=%s",
                       cuda_info["name"], cuda_info["memory_gb"], cuda_info["capability"])
        else:
            logger.info("gpu_detected: cuda device=%s (cuml not installed, using CPU fallback for sklearn)",
                       cuda_info["name"])
        return DeviceInfo(
            device_type="cuda",
            device_name=cuda_info["name"],
            memory_gb=cuda_info["memory_gb"],
            compute_capability=cuda_info["capability"],
            n_jobs=n_jobs,
            cuml_available=cuml_avail,
        )
    
    # Try ROCm (AMD)
    rocm_info = _detect_rocm()
    if rocm_info is not None:
        logger.info("gpu_detected: rocm device=%s", rocm_info["name"])
        return DeviceInfo(
            device_type="rocm",
            device_name=rocm_info["name"],
            memory_gb=0,
            compute_capability="",
            n_jobs=n_jobs,
            cuml_available=False,
        )
    
    # CPU fallback
    logger.info("gpu_not_found: using CPU with %d parallel jobs", n_jobs)
    return DeviceInfo(
        device_type="cpu",
        device_name=f"{n_jobs}-core CPU",
        memory_gb=0,
        compute_capability="",
        n_jobs=n_jobs,
        cuml_available=False,
    )


def _get_cpu_count() -> int:
    """Get the number of available CPU cores."""
    try:
        import os
        return len(os.sched_getaffinity(0))
    except Exception:
        pass
    try:
        import multiprocessing
        return multiprocessing.cpu_count()
    except Exception:
        return 4


def _detect_cuda() -> dict[str, Any] | None:
    """Detect CUDA-capable NVIDIA GPU via nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,compute_cap",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return None
        
        lines = result.stdout.strip().split("\n")
        # Pick the GPU with the most memory
        best = None
        best_mem = 0
        for line in lines:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                try:
                    mem = float(parts[1])
                    if mem > best_mem:
                        best_mem = mem
                        best = {
                            "name": parts[0],
                            "memory_gb": mem / 1024,
                            "capability": parts[2],
                        }
                except ValueError:
                    continue
        return best
    except Exception:
        return None


def _detect_rocm() -> dict[str, Any] | None:
    """Detect AMD ROCm GPU via rocminfo."""
    try:
        result = subprocess.run(
            ["rocminfo"], capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return None
        # Extract GPU name
        for line in result.stdout.split("\n"):
            if "Name:" in line and "GPU" in result.stdout:
                name = line.split("Name:")[-1].strip()
                if name:
                    return {"name": name}
        return None
    except Exception:
        return None


def _check_cuml() -> bool:
    """Check if RAPIDS cuml is available for GPU-accelerated sklearn."""
    try:
        import cuml  # noqa: F401
        return True
    except ImportError:
        return False


# Singleton device info
_DEVICE: DeviceInfo | None = None


def get_device() -> DeviceInfo:
    """Get the cached device info (detected once on first call)."""
    global _DEVICE
    if _DEVICE is None:
        _DEVICE = detect_device()
    return _DEVICE
