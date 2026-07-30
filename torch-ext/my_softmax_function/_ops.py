"""Public torch custom-op wrapper around the CUTLASS-DSL softmax-attention kernel.

This is the API a Hugging Face `kernels` consumer sees: a normal, registered
`torch.ops` op that works with autograd tracing / `torch.compile`, not the raw
`@cute.kernel`.
"""

import math

import torch
from cutlass.cute.runtime import from_dlpack

from .attention_v3 import solve

_OP_NAME = "my_softmax_function::softmax_attention"


@torch.library.custom_op(_OP_NAME, mutates_args=())
def softmax_attention(
    Q: torch.Tensor,        # (M, d)
    K: torch.Tensor,        # (N, d)
    V: torch.Tensor,        # (N, d)
    scale: float,
) -> torch.Tensor:          # (M, d)
    """Fused, max-shifted softmax attention: softmax(scale * Q @ K^T) @ V.

    Q, K, V must be contiguous, 2-D, float32, and on the same CUDA device.
    """
    if not (Q.is_cuda and K.is_cuda and V.is_cuda):
        raise ValueError("Q, K, V must be CUDA tensors")
    if not (Q.dim() == K.dim() == V.dim() == 2):
        raise ValueError("Q, K, V must be 2-D (M,d)/(N,d)/(N,d)")

    M, d = Q.shape
    N = K.shape[0]
    if K.shape[1] != d or V.shape[1] != d or V.shape[0] != N:
        raise ValueError("shape mismatch between Q/K/V")

    Q = Q.contiguous()
    K = K.contiguous()
    V = V.contiguous()
    output = torch.empty((M, d), dtype=torch.float32, device=Q.device)

    solve(
        from_dlpack(Q),
        from_dlpack(K),
        from_dlpack(V),
        from_dlpack(output),
        M,
        N,
        d,
        float(scale),
    )
    return output


@softmax_attention.register_fake
def _(Q, K, V, scale):
    # Shape/dtype/device metadata only — no compute. Lets torch.compile trace.
    M, d = Q.shape
    return Q.new_empty((M, d))


def attention(Q, K, V, scale=None):
    """Convenience entry point with a default 1/sqrt(d) scale."""
    if scale is None:
        scale = 1.0 / math.sqrt(Q.shape[-1])
    return torch.ops.my_softmax_function.softmax_attention(Q, K, V, scale)
