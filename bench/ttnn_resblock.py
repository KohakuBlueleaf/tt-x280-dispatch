"""ttnn baseline for the X280 ResNet block: relu(conv(relu(conv(X, W1)), W2) + X).

Usage: TT_VISIBLE_DEVICES=1 python bench/ttnn_resblock.py [--fidelity HiFi2]
"""

import argparse

import torch
import ttnn

from ttnn_common import compute_config, open_device, time_eager, time_trace

# (N, H, W, C), as in examples/resblock.py
SHAPES = [(1, 56, 56, 64), (2, 28, 28, 128), (1, 64, 64, 64)]


def conv_reference(inp, w):
    """3x3 same-padding conv of an NHWC float64 tensor."""
    out = torch.nn.functional.conv2d(
        inp.permute(0, 3, 1, 2), w.bfloat16().double(), padding=1
    )
    return out.permute(0, 2, 3, 1)


def bench(device, config, n, hh, ww, c, iters):
    """One shape: error against fp64, eager and trace µs per block."""
    torch.manual_seed(0)
    x_t = torch.randn(n, hh, ww, c)
    w1_t, w2_t = (torch.randn(c, c, 3, 3) * (2.0 / (9 * c)) ** 0.5 for _ in range(2))
    x_rm = ttnn.from_torch(x_t.reshape(1, 1, -1, c), dtype=ttnn.bfloat16, device=device)
    x_tile = ttnn.to_layout(x_rm, ttnn.TILE_LAYOUT)
    weights = [ttnn.from_torch(w, dtype=ttnn.bfloat16) for w in (w1_t, w2_t)]
    conv_config = ttnn.Conv2dConfig(weights_dtype=ttnn.bfloat16)

    def conv(inp, i):
        out, [w_dev, _] = ttnn.conv2d(
            input_tensor=inp,
            weight_tensor=weights[i],
            in_channels=c,
            out_channels=c,
            device=device,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            batch_size=n,
            input_height=hh,
            input_width=ww,
            conv_config=conv_config,
            compute_config=config,
            return_weights_and_bias=True,
            dtype=ttnn.bfloat16,
        )
        weights[i] = w_dev  # prepared on the device after the first call
        return out

    def forward():
        h = ttnn.relu(conv(x_rm, 0))
        y = ttnn.to_memory_config(conv(h, 1), ttnn.DRAM_MEMORY_CONFIG)
        return ttnn.relu(ttnn.add(y, x_tile)), h

    # Reference with the device's bf16 rounding of inputs and of H.
    y, _ = forward()
    x64 = x_t.bfloat16().double()
    h_ref = torch.relu(conv_reference(x64, w1_t)).bfloat16().double()
    reference = torch.relu(conv_reference(h_ref, w2_t) + x64)
    diff = ttnn.to_torch(y).double().reshape(n, hh, ww, c) - reference
    ref_rms = reference.pow(2).mean().sqrt()

    eager = time_eager(device, forward, iters)
    trace = time_trace(device, forward, iters, reps=4)
    return {
        "kernel": "ttnn_resblock",
        "shape": (n, hh, ww, c),
        "eager_us": round(eager, 1),
        "trace_us": round(trace, 1),
        "rel_rms": float(f"{diff.pow(2).mean().sqrt() / ref_rms:.3e}"),
        "mean_bias": float(f"{diff.mean() / ref_rms:.3e}"),
    }


def main():
    """Run every shape and print one record per shape."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=32)
    parser.add_argument("--fidelity", default="HiFi2")
    args = parser.parse_args()

    device = open_device()
    config = compute_config(device, args.fidelity)
    for n, hh, ww, c in SHAPES:
        print(bench(device, config, n, hh, ww, c, args.iters), flush=True)
    ttnn.close_device(device)


if __name__ == "__main__":
    main()
