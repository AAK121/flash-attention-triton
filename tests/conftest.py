from __future__ import annotations

import os

import pytest
import torch

# Fixed block sizes keep the suite fast and reproducible; block-size coverage
# comes from test_block_configs, and test_autotuned_path covers the tuner.
os.environ.setdefault("FLASH_ATTN_TRITON_AUTOTUNE", "0")

from flash_attn_triton._device import bf16_supported, interpreting  # noqa: E402

INTERPRET = interpreting()
HAS_CUDA = torch.cuda.is_available()

# The interpreter executes the kernel in Python, so only small shapes are practical.
INTERPRET_MAX_SEQ = 256


def pytest_collection_modifyitems(config, items):
    if HAS_CUDA or INTERPRET:
        return
    skip = pytest.mark.skip(reason="needs a CUDA GPU or TRITON_INTERPRET=1")
    for item in items:
        item.add_marker(skip)


@pytest.fixture(scope="session")
def device() -> torch.device:
    if HAS_CUDA and not INTERPRET:
        return torch.device("cuda")
    return torch.device("cpu")


@pytest.fixture
def require_dtype(device):
    def check(dtype: torch.dtype) -> None:
        if dtype == torch.bfloat16 and not bf16_supported(device):
            pytest.skip("bfloat16 needs sm80 or newer")
    return check


@pytest.fixture
def require_size():
    def check(*seqs: int) -> None:
        if INTERPRET and max(seqs) > INTERPRET_MAX_SEQ:
            pytest.skip("sequence too long for the Triton interpreter")
    return check

