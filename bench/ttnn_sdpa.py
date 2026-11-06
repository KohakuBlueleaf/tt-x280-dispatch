"""ttnn baseline for the X280 flash attention: non-causal SDPA, µs per call.

Usage: TT_VISIBLE_DEVICES=1 python bench/ttnn_sdpa.py [--exp-approx 0|1]
"""

import argparse

import torch
import ttnn

from ttnn_common import compute_config, open_device, time_eager, time_trace

# ((heads, seq, dim), (q_chunk, k_chunk)), as in examples/flash_attn.py
SHAPES = [
    ((12, 1280, 64), (128, 128)),
    ((8, 2048, 128), (128, 128)),
    ((16, 512, 64), (64, 64)),
]


def bench(device, config, shape, chunks, exp_approx, iters):
    """One shape: eager and trace µs per SDPA call, and TFLOPS."""
    nh, s, d = shape
    torch.manual_seed(0)
    q, k, v = (
        ttnn.from_torch(
            torch.randn(1, nh, s, d),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        for _ in range(3)
    )
    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
        q_chunk_size=chunks[0],
        k_chunk_size=chunks[1],
        exp_approx_mode=exp_approx,
    )

    def forward():
        return ttnn.transformer.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=False,
            program_config=program_config,
            compute_kernel_config=config,
        )

    eager = time_eager(device, forward, iters)
    trace = time_trace(device, forward, iters)
    for t in (q, k, v):
        ttnn.deallocate(t)
    return {
        "kernel": "ttnn_sdpa",
        "shape": shape,
        "chunk": chunks,
        "exp_approx": exp_approx,
        "eager_us": round(eager, 1),
        "trace_us": round(trace, 1),
        "trace_tflops": round(4 * nh * s * s * d / trace / 1e6, 2),
    }


def main():
    """Run every shape and print one record per shape."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=32)
    parser.add_argument("--exp-approx", type=int, default=1)
    args = parser.parse_args()

    device = open_device()
    config = compute_config(device, "HiFi2")
    for shape, chunks in SHAPES:
        record = bench(device, config, shape, chunks, bool(args.exp_approx), args.iters)
        print(record, flush=True)
    ttnn.close_device(device)


if __name__ == "__main__":
    main()
