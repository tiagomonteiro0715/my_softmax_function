"""my_softmax_function — fused softmax-attention kernel (CUTLASS Python DSL).

Public API exposed to `kernels` consumers.
"""

from ._kernel import attention, softmax_attention

__all__ = ["attention", "softmax_attention"]
