import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

NEG_BIG = -3.0e38  # finite stand-in for -inf


def transpose_matrix(T: cute.Tensor, i, j):
    """Element (i, j) of T-transpose: a pure index swap, no data movement."""
    return T[j, i]


def matrix_multiplication(acc, A: cute.Tensor, B: cute.Tensor, row, col, k,
                          transpose_b=False):
    """One MAC step of C[row, col] = sum_k A[row, k] * B[k, col].
    transpose_b is a Python bool, resolved at trace time."""
    b = transpose_matrix(B, k, col) if transpose_b else B[k, col]
    return acc + A[row, k] * b


def softmax_computation(score, row_max, denom):
    """One softmax probability, max-shifted for stability. Pass denom=1.0 for the
    unnormalized numerator (online softmax normalizes once at the end)."""
    return cute.math.exp(score - row_max) / denom


@cute.kernel
def softmax_attention(
    Q: cute.Tensor,        # (M, d)
    K: cute.Tensor,        # (N, d)
    V: cute.Tensor,        # (N, d)
    output: cute.Tensor,   # (M, d)
    M: cutlass.Int32,
    N: cutlass.Int32,
    d: cutlass.Int32,
    host_scale: cutlass.Float32,
):
    bx, _, _ = cute.arch.block_idx()
    bdx, _, _ = cute.arch.block_dim()
    tx, _, _ = cute.arch.thread_idx()

    row = bx * bdx + tx

    if row < M:
        scale = host_scale

        # output[row, :] is the running accumulator
        for k in cutlass.range(d):
            output[row, k] = cutlass.Float32(0.0)

        running_max = cutlass.Float32(NEG_BIG)
        running_sum = cutlass.Float32(0.0)

        # single pass: score, max, denominator and V-accumulation together
        for n in cutlass.range(N):
            s = cutlass.Float32(0.0)
            for k in cutlass.range(d):
                s = matrix_multiplication(s, Q, K, row, n, k, transpose_b=True)
            s = s * scale

            new_max = running_max
            if s > new_max:
                new_max = s

            # rescales everything accumulated under the old max
            correction = softmax_computation(running_max, new_max, 1.0)
            p = softmax_computation(s, new_max, 1.0)

            running_sum = running_sum * correction + p
            running_max = new_max

            for k in cutlass.range(d):
                output[row, k] = output[row, k] * correction + p * V[n, k]

        # normalize once, with a reciprocal instead of d divisions
        inv_sum = 1.0 / running_sum
        for k in cutlass.range(d):
            output[row, k] = output[row, k] * inv_sum


@cute.jit
def solve(
    Q: cute.Tensor,
    K: cute.Tensor,
    V: cute.Tensor,
    output: cute.Tensor,
    M: cutlass.Int32,
    N: cutlass.Int32,
    d: cutlass.Int32,
    host_scale: cutlass.Float32,
):
    block_size = 256
    grid_size = (M + block_size - 1) // block_size

    softmax_attention(Q, K, V, output, M, N, d, host_scale).launch(
        grid=(grid_size, 1, 1),
        block=(block_size, 1, 1),
    )
