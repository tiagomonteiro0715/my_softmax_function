"""Test the built kernel through the `kernels` loader against a torch reference.

Run AFTER building (so build/torch-universal/my_softmax_function exists):

    pytest tests/ -v
"""

import math
from pathlib import Path

import pytest
import torch

from kernels import get_local_kernel

REPO_ROOT = Path(__file__).resolve().parent.parent

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="kernel requires CUDA"
)


def _load():
    # Loads build/torch-universal/my_softmax_function from the local repo,
    # exactly the way kernels.get_kernel(...) would load it from the Hub.
    return get_local_kernel(REPO_ROOT, "my_softmax_function")


def _reference(Q, K, V, scale):
    scores = (Q @ K.transpose(-1, -2)) * scale
    probs = torch.softmax(scores, dim=-1)
    return probs @ V


@pytest.mark.parametrize("M,N,d", [(1, 1, 1), (17, 33, 8), (128, 256, 64)])
def test_matches_reference(M, N, d):
    kernel = _load()
    torch.manual_seed(0)
    Q = torch.randn(M, d, device="cuda", dtype=torch.float32)
    K = torch.randn(N, d, device="cuda", dtype=torch.float32)
    V = torch.randn(N, d, device="cuda", dtype=torch.float32)
    scale = 1.0 / math.sqrt(d)

    out = kernel.attention(Q, K, V)
    ref = _reference(Q, K, V, scale)

    torch.testing.assert_close(out, ref, atol=1e-3, rtol=1e-3)


def test_custom_scale():
    kernel = _load()
    Q = torch.randn(8, 16, device="cuda", dtype=torch.float32)
    K = torch.randn(12, 16, device="cuda", dtype=torch.float32)
    V = torch.randn(12, 16, device="cuda", dtype=torch.float32)

    out = kernel.attention(Q, K, V, scale=0.5)
    ref = _reference(Q, K, V, 0.5)
    torch.testing.assert_close(out, ref, atol=1e-3, rtol=1e-3)
