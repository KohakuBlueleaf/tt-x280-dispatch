"""ttnn baseline for the X280 matmul shapes: µs per ttnn.matmul, eager and trace.

Usage: TT_VISIBLE_DEVICES=1 python bench/ttnn_matmul.py [--fidelity HiFi2]
"""

import argparse

import torch
import ttnn

from ttnn_common import compute_config, open_device, time_eager, time_trace

# (M, N, K)
SHAPES = [
    (320, 384, 256),
    (640, 768, 1024),
    (1280, 1536, 2048),
    (2560, 3072, 2048),
    (2560, 3072, 4096),
]


def bench(device, config, m, n, k, iters):
    """One shape: rel-RMS against fp64, eager and trace µs per matmul."""
    torch.manual_seed(0)
    a_t, b_t = torch.rand(m, k) * 2 - 1, torch.rand(k, n) * 2 - 1
    a, b = (
        ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        for t in (a_t, b_t)
    )

    def forward():
        return ttnn.matmul(a, b, compute_kernel_config=config, dtype=ttnn.bfloat16)

    reference = a_t.bfloat16().double() @ b_t.bfloat16().double()
    result = ttnn.to_torch(forward()).double()
    rel_rms = float((result - reference).norm() / reference.norm())

    eager = time_eager(device, forward, iters)
    trace = time_trace(device, forward, iters, reps=16)
    ttnn.deallocate(a)
    ttnn.deallocate(b)
    return {
        "kernel": "ttnn_matmul",
        "shape": (m, n, k),
        "eager_us": round(eager, 1),
        "trace_us": round(trace, 1),
        "trace_tflops": round(2 * m * n * k / trace / 1e6, 1),
        "rel_rms": float(f"{rel_rms:.3e}"),
    }


def main():
    """Run every shape and print one record per shape."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--fidelity", default="HiFi2")
    parser.add_argument("--iters", type=int, default=64)
    args = parser.parse_args()

    device = open_device()
    config = compute_config(device, args.fidelity)
    for m, n, k in SHAPES:
        print(bench(device, config, m, n, k, args.iters), flush=True)
    ttnn.close_device(device)


if __name__ == "__main__":
    main()
