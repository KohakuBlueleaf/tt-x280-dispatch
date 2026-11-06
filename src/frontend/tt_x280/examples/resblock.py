"""ResNet basic block Y = relu(conv(relu(conv(X, W1)), W2) + X): one command per conv.

3×3 convs as implicit GEMMs on NHWC row-major tensors; both weights stay in L1.

Usage: TT_VISIBLE_DEVICES=0 python -m tt_x280.examples.resblock --shape 1 56 56 64
"""

import argparse
import os
import time

import torch
import ttnn

import tt_x280
from tt_x280.layout import split_range
from tt_x280.stats import error_stats


def build_resblock_program(plan, x, h, y, w1, w2, shape, n_cmds, fidelity):
    """The resident ResNet-block program over a contiguous range of output tiles a core.

    Command 2i + 1 runs conv1 (X -> H), command 2i + 2 conv2 (H, X -> Y).
    """
    n, hh, ww, c = shape
    ct, m_tiles = c // 32, n * hh * ww // 32
    tile, row = 2048, c * 2
    kdir = tt_x280.KERNEL_DIR
    cores = [(cx, cy) for cy in range(plan.grid_y) for cx in range(plan.grid_x)]
    units = split_range(m_tiles, cores)

    program = tt_x280.ResidentProgram(plan)
    program.add_cb(0, ttnn.bfloat16, row, 64)  # tap rows, double-buffered
    program.add_cb(9, ttnn.bfloat16, row, 32 + 2 * ww + 2)  # input halo of one tile
    program.add_cb(1, ttnn.bfloat16, row, 32)  # residual rows
    program.add_cb(2, ttnn.bfloat16, tile, 9 * ct)  # tilized im2col tile row
    program.add_cb(3, ttnn.bfloat16, tile, 9 * ct * ct)  # W1, resident
    program.add_cb(4, ttnn.bfloat16, tile, 9 * ct * ct)  # W2, resident
    program.add_cb(6, ttnn.bfloat16, tile, 2 * ct)  # activated output tiles
    program.add_cb(7, ttnn.bfloat16, tile, ct)  # tilized residual
    program.add_cb(16, ttnn.bfloat16, tile, 2 * ct)  # untilized output rows

    rm_accessor = ttnn.TensorAccessorArgs(x).get_compile_time_args()
    w_accessor = ttnn.TensorAccessorArgs(w1).get_compile_time_args()
    addr = {
        name: t.buffer_address()
        for name, t in dict(x=x, h=h, y=y, w1=w1, w2=w2).items()
    }

    reader = ttnn.DataMovementConfigDescriptor(
        processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.RISCV_1_default
    )
    program.add_kernel(
        f"{kdir}/resblock/reader.cpp",
        cores,
        reader,
        lambda cx, cy: [
            addr["x"],
            addr["h"],
            addr["w1"],
            addr["w2"],
            *units[(cx, cy)],
            n_cmds,
        ],
        [n, hh, ww, ct, *rm_accessor, *w_accessor],
        dispatch=True,
    )
    writer = ttnn.DataMovementConfigDescriptor(
        processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.RISCV_0_default
    )
    program.add_kernel(
        f"{kdir}/resblock/writer.cpp",
        cores,
        writer,
        lambda cx, cy: [addr["h"], addr["y"], *units[(cx, cy)], n_cmds],
        [ct, *rm_accessor],
        dispatch=True,
    )
    compute = ttnn.ComputeConfigDescriptor(
        math_fidelity=fidelity, fp32_dest_acc_en=True
    )
    program.add_kernel(
        f"{kdir}/resblock/compute.cpp",
        cores,
        compute,
        lambda cx, cy: [units[(cx, cy)][1]],
        [ct, n_cmds],
    )
    return program


def main():
    """Run the ResNet block, check it against fp64 and print one record."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shape",
        type=int,
        nargs=4,
        default=[1, 56, 56, 64],
        metavar=("N", "H", "W", "C"),
    )
    parser.add_argument("--forwards", type=int, default=32)
    parser.add_argument("--pll", type=int, default=1750)
    parser.add_argument(
        "--fidelity", default="HiFi2", choices=["LoFi", "HiFi2", "HiFi3", "HiFi4"]
    )
    parser.add_argument("--trace", default=None, help="save a per-core trace (.pt)")
    args = parser.parse_args()
    n, hh, ww, c = args.shape
    fidelity = getattr(ttnn.MathFidelity, args.fidelity)
    assert (n * hh * ww) % 32 == 0 and c % 32 == 0 and c <= 128

    device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    device.disable_and_clear_program_cache()
    runtime = tt_x280.Runtime(device, pll_mhz=args.pll)
    plan = runtime.plan(depth=1)

    torch.manual_seed(0)
    x_t = torch.randn(n, hh, ww, c)
    w1_t, w2_t = (torch.randn(c, c, 3, 3) * (2.0 / (9 * c)) ** 0.5 for _ in range(2))

    def rows(t):
        """An NHWC tensor as row-major [1, 1, N*H*W, C] on the device."""
        return ttnn.from_torch(
            t.reshape(1, 1, -1, c),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
        )

    def weight_matrix(w):
        """Conv weights [Cout, Cin, 3, 3] as a tiled [(tap, Cin), Cout] matrix."""
        mat = w.permute(2, 3, 1, 0).reshape(9 * c, c)
        return ttnn.from_torch(
            mat, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )

    x, h, y = rows(x_t), rows(torch.zeros_like(x_t)), rows(torch.zeros_like(x_t))
    w1, w2 = weight_matrix(w1_t), weight_matrix(w2_t)

    n_cmds = 2 * args.forwards
    for _ in range(2):  # first run compiles
        program = build_resblock_program(
            plan, x, h, y, w1, w2, args.shape, n_cmds, fidelity
        )
        start = time.perf_counter()
        ticks = program.run([x, h, y, w1, w2], n_cmds)
        wall = time.perf_counter() - start
    us = ticks / tt_x280.MTIME_HZ * 1e6 / args.forwards

    # One more run with per-core timestamps.
    if args.trace:
        program.run([x, h, y, w1, w2], n_cmds, trace=True)
        program.trace.save(
            args.trace,
            kernel="resblock",
            shape=(n, hh, ww, c),
            ops=["conv1", "conv2"],
            us_per_forward=us,
        )

    def conv(inp, w):
        """3×3 same-padding conv of an NHWC float64 tensor with bf16-rounded w."""
        out = torch.nn.functional.conv2d(
            inp.permute(0, 3, 1, 2), w.bfloat16().double(), padding=1
        )
        return out.permute(0, 2, 3, 1)

    # fp64 reference with the device's bf16 rounding of inputs and of H.
    x64 = x_t.bfloat16().double()
    h_ref = torch.relu(conv(x64, w1_t)).bfloat16().double()
    reference = torch.relu(conv(h_ref, w2_t) + x64)
    result = ttnn.to_torch(y).double().reshape(n, hh, ww, c)
    flop = 2 * (2 * n * hh * ww * 9 * c * c)
    record = {
        "kernel": "resblock",
        "shape": (n, hh, ww, c),
        "fidelity": args.fidelity,
        "us_per_forward": round(us, 2),
        "tflops": round(flop / (us * 1e-6) / 1e12, 2),
        "host_wall_ms": round(wall * 1e3, 1),
        **error_stats(result, reference),
        "h": error_stats(ttnn.to_torch(h).double().reshape(n, hh, ww, c), h_ref),
    }
    print(record)
    ttnn.close_mesh_device(device)


if __name__ == "__main__":
    os.environ.setdefault("TT_METAL_SLOW_DISPATCH_MODE", "1")
    main()
