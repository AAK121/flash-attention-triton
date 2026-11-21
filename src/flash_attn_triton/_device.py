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


def max_shared_memory(device: torch.device | None = None) -> int:
    """Largest dynamic shared-memory allocation a single block may request, in bytes."""
    if not torch.cuda.is_available():
        return 48 * 1024
    props = torch.cuda.get_device_properties(device)
    opt_in = getattr(props, "shared_memory_per_block_optin", None)
    if opt_in:
        return int(opt_in)
    major, minor = props.major, props.minor
    # Turing 64 KB, Ampere GA100 163 KB, GA10x 99 KB, Hopper 227 KB.
    if (major, minor) == (7, 5):
        return 64 * 1024
    if (major, minor) == (8, 0):
        return 163 * 1024
    if major == 8:
        return 99 * 1024
    if major >= 9:
        return 227 * 1024
    return 48 * 1024
