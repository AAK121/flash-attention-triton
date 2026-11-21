"""Device capability queries used to gate dtypes and prune autotune configs."""

from __future__ import annotations

import os

import torch


def interpreting() -> bool:
    return os.environ.get("TRITON_INTERPRET", "0") == "1"


def compute_capability(device: torch.device | None = None) -> tuple[int, int]:
    if not torch.cuda.is_available():
        return (0, 0)
    return torch.cuda.get_device_capability(device)


def bf16_supported(device: torch.device | None = None) -> bool:
    """bf16 tensor-core MMA exists from Ampere (sm80) onward."""
    if interpreting():
        # The interpreter evaluates blocks with numpy, which has no bfloat16.
        return False
    return compute_capability(device) >= (8, 0)


def _device_index(device: torch.device | None) -> int:
    if device is None or device.index is None:
        return torch.cuda.current_device()
    return device.index


def max_shared_memory(device: torch.device | None = None) -> int:
    """Largest shared-memory allocation a single program may request, in bytes.

    Asks Triton's driver first, since that is the limit Triton itself checks
    when it rejects a compiled kernel.
    """
    if not torch.cuda.is_available():
        return 48 * 1024
    idx = _device_index(device)
    try:
        import triton

        return int(triton.runtime.driver.active.utils.get_device_properties(idx)["max_shared_mem"])
    except Exception:
        pass
    props = torch.cuda.get_device_properties(idx)
    opt_in = getattr(props, "shared_memory_per_block_optin", None)
    if opt_in:
        return int(opt_in)
    return 48 * 1024
