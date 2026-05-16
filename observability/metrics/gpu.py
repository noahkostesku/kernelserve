"""pynvml-based GPU metric sampler for KernelServe.

Returns an empty dict on macOS or when pynvml/CUDA is unavailable so that
tests and local dev runs degrade gracefully without GPU hardware.
"""

from __future__ import annotations

try:
    import pynvml

    pynvml.nvmlInit()
    _available = True
except (ImportError, Exception):
    _available = False


def sample_gpu_metrics(device_index: int = 0) -> dict[str, float]:
    """Sample GPU utilisation and memory from NVML.

    Returns a dict with keys ``gpu.util_pct`` and ``gpu.mem_used_gb``,
    or an empty dict when NVML is unavailable.
    """
    if not _available:
        return {}
    handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
    mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return {
        "gpu.util_pct": float(util.gpu),
        "gpu.mem_used_gb": float(mem.used) / 1024**3,
    }
