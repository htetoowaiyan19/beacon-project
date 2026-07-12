"""Model and device helpers used by chat and evaluation scripts."""

from __future__ import annotations

import torch


BYTES_PER_GIB = 1024**3


def get_gpu_info(device: int | torch.device | None = None) -> dict[str, str | float]:
    """Return lightweight GPU memory information.

    The function keeps the existing return keys used by the scripts while
    avoiding unnecessary CUDA calls when the project is running on CPU.
    """

    if not torch.cuda.is_available():
        return {
            "gpu_name": "CPU",
            "vram_allocated": 0.0,
            "vram_reserved": 0.0,
        }

    if device is None:
        cuda_device = torch.device("cuda", torch.cuda.current_device())
    elif isinstance(device, int):
        cuda_device = torch.device("cuda", device)
    else:
        cuda_device = torch.device(device)

    if cuda_device.type != "cuda":
        cuda_device = torch.device("cuda", 0)

    return {
        "gpu_name": torch.cuda.get_device_name(cuda_device),
        "vram_allocated": torch.cuda.memory_allocated(cuda_device) / BYTES_PER_GIB,
        "vram_reserved": torch.cuda.memory_reserved(cuda_device) / BYTES_PER_GIB,
    }
