---
tags:
  - kernel
license: apache-2.0
---

# my_softmax_function

Fused **softmax-attention** kernel written in the [CUTLASS Python DSL](https://github.com/NVIDIA/cutlass)
(`cutlass.cute`), packaged for the Hugging Face [`kernels`](https://github.com/huggingface/kernels) Hub.

Computes, in a single online pass (max-shifted for numerical stability):

```
out = softmax(scale * Q @ Kᵀ) @ V
```

## Usage

```python
import torch
from kernels import get_kernel

k = get_kernel("monteiro-t/my_softmax_function")   # replace with your repo id

Q = torch.randn(128, 64, device="cuda", dtype=torch.float32)
K = torch.randn(256, 64, device="cuda", dtype=torch.float32)
V = torch.randn(256, 64, device="cuda", dtype=torch.float32)

out = k.attention(Q, K, V)          # scale defaults to 1/sqrt(d)
out = k.attention(Q, K, V, scale=0.5)
```

## API

| Function | Signature | Notes |
|---|---|---|
| `attention(Q, K, V, scale=None)` | `(M,d),(N,d),(N,d) -> (M,d)` | Convenience wrapper; `scale` defaults to `1/sqrt(d)`. |
| `softmax_attention(Q, K, V, scale)` | registered `torch.ops` op | `scale` required; `torch.compile`-friendly (has a fake/meta impl). |

## Supported shapes / dtypes / hardware

| Property | Support |
|---|---|
| Dtype | `float32` only |
| Layout | 2-D, contiguous (inputs are forced contiguous) |
| Shapes | `Q: (M, d)`, `K: (N, d)`, `V: (N, d)` → `out: (M, d)` |
| Batch / heads | **Not supported** (single 2-D attention; no leading batch/head dims) |
| Masking | **None** (dense attention only) |
| Backward pass | **None** (inference / forward only) |
| Device | CUDA only |

## Limitations

Naive one-row-per-thread implementation intended as a reference/demo kernel. No
batching, masking, dropout, or gradients. Not performance-tuned.
