"""ttnn baseline for the X280 MLP Y = silu(X @ W1) @ W2: µs per forward.

Usage: TT_VISIBLE_DEVICES=1 python bench/ttnn_mlp.py
"""

import argparse

import torch
import ttnn

from ttnn_common import compute_config, open_device, time_eager, time_trace

# (M, K, F), as in examples/mlp.py
SHAPES = [(320, 256, 384), (640, 1024, 768), (1280, 2048, 1536), (2560, 2048, 3072)]


def bench(device, config, m, k, f, iters):
    """One shape: rel-RMS against fp64, eager and trace µs per forward."""
    torch.manual_seed(0)
    x_t = torch.rand(m, k) * 2 - 1
    w1_t = (torch.rand(k, f) * 2 - 1) * 0.1
    w2_t = (torch.rand(f, f) * 2 - 1) * 0.1
    x, w1, w2 = (
        ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        for t in (x_t, w1_t, w2_t)
    )
    kw = dict(compute_kernel_config=config, dtype=ttnn.bfloat16)

    def forward():
        h = ttnn.matmul(x, w1, **kw)
        h2 = ttnn.silu(h)
        ttnn.deallocate(h)
        y = ttnn.matmul(h2, w2, **kw)
        ttnn.deallocate(h2)
        return y

    # Reference with the device's bf16 rounding of inputs and of H.
    x64, w1_64, w2_64 = (t.bfloat16().double() for t in (x_t, w1_t, w2_t))
    h_ref = (x64 @ w1_64).bfloat16().double()
    reference = torch.nn.functional.silu(h_ref).bfloat16().double() @ w2_64
    result = ttnn.to_torch(forward()).double()
    rel_rms = float((result - reference).norm() / reference.norm())

    eager = time_eager(device, forward, iters)
    trace = time_trace(device, forward, iters)
    for t in (x, w1, w2):
        ttnn.deallocate(t)
    return {
        "kernel": "ttnn_mlp",
        "shape": (m, k, f),
        "eager_us": round(eager, 1),
        "trace_us": round(trace, 1),
        "rel_rms": float(f"{rel_rms:.3e}"),
    }


def main():
    """Run every shape and print one record per shape."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=64)
    args = parser.parse_args()

    device = open_device()
    config = compute_config(device, "HiFi2")
    for m, k, f in SHAPES:
        print(bench(device, config, m, k, f, args.iters), flush=True)
    ttnn.close_device(device)


if __name__ == "__main__":
    main()
