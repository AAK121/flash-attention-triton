# flash-attention-triton

FlashAttention-2 written from scratch in Triton: forward and backward kernels
behind a `torch.autograd.Function`, with grouped-query and sliding-window
attention. Work in progress: benchmarks and the training demo follow.

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

Inputs are shaped `(batch, heads, seq, head_dim)` and may have arbitrary strides;
`k` and `v` may have fewer heads than `q` (grouped-query attention). Query `i`
attends keys `j` with `i + off - left <= j <= i + off + right`, `off = seq_k - seq_q`:
`causal=True` sets `right = 0`, and `window_size=(left, right)` bounds either side
(`-1` for unbounded).

## Status

| Feature | State |
| --- | --- |
| Forward, causal and non-causal, head_dim 16/32/64/128 | done |
| Arbitrary and unequal Q/KV lengths, strided inputs | done |
| Backward (dQ, dK, dV), deterministic, no atomics | done |
| GQA/MQA (`kv_heads` dividing `heads`), dK/dV summed over the group in-kernel | done |
| Sliding window `window_size=(left, right)`, with tile skipping outside it | done |
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
