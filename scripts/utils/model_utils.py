import torch

def get_gpu_info():

    gpu_name = "CPU"
    vram_allocated = 0
    vram_reserved = 0

    if torch.cuda.is_available():

        gpu_name = torch.cuda.get_device_name(0)

        vram_allocated = (
            torch.cuda.memory_allocated()
            / 1024**3
        )

        vram_reserved = (
            torch.cuda.memory_reserved()
            / 1024**3
        )

    return {
        "gpu_name": gpu_name,
        "vram_allocated": vram_allocated,
        "vram_reserved": vram_reserved
    }