# flash-attention-triton

FlashAttention-2 written from scratch in Triton: forward and backward kernels
behind a `torch.autograd.Function`. Work in progress: benchmarks, GQA, sliding
window and the training demo follow.

## Install

```
pip install -e .
```

Requires torch >= 2.4 and triton >= 3.0. Developed against a Kaggle 2x T4
environment (sm75, fp16); the code does not assume a particular architecture.

## Usage

```python
import torch
from flash_attn_triton import flash_attention

q = torch.randn(2, 8, 1024, 64, device="cuda", dtype=torch.float16)
k = torch.randn(2, 8, 1024, 64, device="cuda", dtype=torch.float16)
v = torch.randn(2, 8, 1024, 64, device="cuda", dtype=torch.float16)
out = flash_attention(q, k, v, causal=True)
out.sum().backward()  # if q, k, v require grad
```

Inputs are shaped `(batch, heads, seq, head_dim)` and may have arbitrary strides.
With `causal=True` and unequal lengths, query `i` attends keys `j <= i + seq_k - seq_q`.

## Status

| Feature | State |
| --- | --- |
| Forward, causal and non-causal, head_dim 16/32/64/128 | done |
| Arbitrary and unequal Q/KV lengths, strided inputs | done |
| Backward (dQ, dK, dV), deterministic, no atomics | done |
| GQA/MQA, sliding window | planned |
| Benchmarks, training demo | planned |

## Tests

```
pytest                         # GPU
TRITON_INTERPRET=1 pytest      # CPU, small shapes, fp16 only
```

The suite runs with fixed block sizes (`FLASH_ATTN_TRITON_AUTOTUNE=0`) and covers
block-size choices explicitly; one test exercises the autotuner on GPU.

## Results

Benchmark numbers will be added after the scripts are run on the target
hardware. No numbers are reported until then.
