"""Single source of truth for the training/inference device.

Env var `KDV_DEVICE` selects the device:
  - "cpu" / "cuda" / "mps"   force that device (errors if unavailable)
  - "auto"                    (default) cuda → mps → cpu, first available
"""
import os
import torch


def get_device():
    name = os.environ.get("KDV_DEVICE", "auto").lower()
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("KDV_DEVICE=cuda but CUDA is unavailable")
        return torch.device("cuda")
    if name == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("KDV_DEVICE=mps but MPS is unavailable")
        return torch.device("mps")
    if name != "auto":
        raise ValueError(f"KDV_DEVICE must be cpu/cuda/mps/auto, got {name!r}")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
