"""Device capability queries used to gate dtypes and prune autotune configs."""

from __future__ import annotations

import os
import re
import warnings

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


_checked_dot_support = False


def warn_if_dot_lacks_tensor_cores(device: torch.device | None = None) -> None:
    """Warn once when the installed Triton is known to miss tensor cores here.

    Measured on a Tesla T4 (sm75): Triton 3.2 lowers tl.dot to mma.sync, 3.3
    fails to compile fp16 dots, and 3.4 to 3.6 fall back to scalar FMA, which
    makes every kernel 10x to 50x slower while still producing correct results.
    Ampere and newer are unaffected.
    """
    global _checked_dot_support
    if _checked_dot_support or interpreting() or not torch.cuda.is_available():
        return
    _checked_dot_support = True
    if compute_capability(device) >= (8, 0):
        return
    import triton

    m = re.match(r"(\d+)\.(\d+)", triton.__version__)
    if m and (int(m.group(1)), int(m.group(2))) >= (3, 3):
        warnings.warn(
            f"Triton {triton.__version__} does not use tensor cores on compute capability "
            f"{compute_capability(device)}; attention will be correct but slow. "
            "Triton 3.2.0 does (pip install triton==3.2.0).",
            RuntimeWarning,
            stacklevel=3,
        )
