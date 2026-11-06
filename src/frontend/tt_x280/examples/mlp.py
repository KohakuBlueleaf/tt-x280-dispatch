"""MLP forward Y = silu(X @ W1) @ W2: three dependent X280 commands per forward.

Usage: TT_VISIBLE_DEVICES=1 python -m tt_x280.examples.mlp --shape 320 256 384
"""

import argparse
import os
import time

import torch
import ttnn

import tt_x280
from tt_x280.stats import error_stats


def pick_subblock(pc_m, pc_n, max_tiles=4):
    """Output subblock (h, w) dividing a core's pc_m × pc_n tile block, h * w <= max."""
    for h, w in ((2, 2), (4, 1), (1, 4), (3, 1), (1, 3), (2, 1), (1, 2), (1, 1)):
        if h * w <= max_tiles and pc_m % h == 0 and pc_n % w == 0:
            return h, w
    return 1, 1


def build_mlp_program(plan, x, w1, w2, h, h2, y, n_cmds, block_w, fidelity):
    """The resident MLP program: a 2D-multicast matmul tiling over the whole grid.

    Command k runs op (k - 1) % 3: GEMM1 (X, W1 -> H), SiLU (H -> H2), GEMM2 (H2, W2 ->
    Y). Each core owns a pc_m × pc_n output tile block; panel-left cores multicast X
    blocks along their row, top-row cores multicast W blocks down their column.
    """
    gx, gy = plan.grid_x, plan.grid_y
    m_t, k_t = x.shape[0] // 32, x.shape[1] // 32
    f_t = w1.shape[1] // 32
    pc_m, pc_n = m_t // gy, f_t // gx
    sb_h, sb_w = pick_subblock(pc_m, pc_n)
    tile = 2048
    kdir = tt_x280.KERNEL_DIR

    program = tt_x280.ResidentProgram(plan)
    program.add_cb(0, ttnn.bfloat16, tile, 2 * pc_m * block_w)
    program.add_cb(1, ttnn.bfloat16, tile, 2 * pc_n * block_w)
    program.add_cb(2, ttnn.bfloat16, tile, 2)
    program.add_cb(16, ttnn.bfloat16, tile, pc_m * pc_n, shared_indices=(24,))
    sem = [program.add_semaphore() for _ in range(4)]
    accessor = ttnn.TensorAccessorArgs(x).get_compile_time_args()
    addr = {
        name: t.buffer_address()
        for name, t in dict(x=x, w1=w1, w2=w2, h=h, h2=h2, y=y).items()
    }

    def panel_of(cx):
        """The multicast panel (x0, x1) holding column cx."""
        return next(p for p in plan.panels if p[0] <= cx <= p[1])

    def reader_args(cx, cy):
        """Reader args of core (cx, cy): both GEMMs, the SiLU block, both multicasts."""
        x0, x1 = panel_of(cx)
        left, left1, right = (
            plan.translated(x0, cy),
            plan.translated(min(x0 + 1, x1), cy),
            plan.translated(x1, cy),
        )
        top, top1, bottom = (
            plan.translated(cx, 0),
            plan.translated(cx, 1),
            plan.translated(cx, gy - 1),
        )
        gemm1 = [
            addr["x"], k_t * pc_m * cy, k_t,
            addr["w1"], pc_n * cx, f_t, block_w * f_t, k_t // block_w,
        ]  # fmt: skip
        gemm2 = [
            addr["h2"], f_t * pc_m * cy, f_t,
            addr["w2"], pc_n * cx, f_t, block_w * f_t, f_t // block_w,
        ]  # fmt: skip
        silu = [addr["h"], pc_m * cy * f_t + pc_n * cx, f_t, sb_h, sb_w]
        row_mcast = [
            *right, *left1, x1 - x0, *left, sem[0], sem[1],
        ]  # fmt: skip
        col_mcast = [
            *bottom, *top1, gy - 1, *top, sem[2], sem[3],
        ]  # fmt: skip
        return [
            *gemm1,
            *gemm2,
            block_w,
            pc_m,
            pc_n,
            *silu,
            *row_mcast,
            *col_mcast,
            n_cmds,
        ]

    def writer_args(cx, cy):
        """Writer args of core (cx, cy): the three outputs and its block in them."""
        outputs = [addr["h"], addr["h2"], addr["y"]]
        block = [
            cx * pc_n + cy * pc_m * f_t,
            f_t,
            sb_w,
            sb_h,
            pc_n // sb_w,
            pc_m // sb_h,
        ]
        return [*outputs, *block, n_cmds]

    # Readers: four roles by whether the core sends the row and/or column multicast.
    cores = [(cx, cy) for cx in range(gx) for cy in range(gy)]
    for in0_sender in (False, True):
        for in1_sender in (False, True):
            role_cores = [
                (cx, cy)
                for cx, cy in cores
                if (cx == panel_of(cx)[0]) == in0_sender and (cy == 0) == in1_sender
            ]
            defines = [("IN0_SENDER", "1")] * in0_sender + [
                ("IN1_SENDER", "1")
            ] * in1_sender
            config = ttnn.DataMovementConfigDescriptor(
                processor=ttnn.DataMovementProcessor.RISCV_1,
                noc=(
                    ttnn.NOC.RISCV_0_default
                    if (in0_sender or in1_sender)
                    else ttnn.NOC.RISCV_1_default
                ),
            )
            program.add_kernel(
                f"{kdir}/mlp/reader.cpp",
                role_cores,
                config,
                reader_args,
                accessor,
                defines,
                dispatch=True,
            )

    # Writers: on the NOC their core's reader does not multicast on.
    for edge in (True, False):
        writer_cores = [
            (cx, cy) for cx, cy in cores if (cx == panel_of(cx)[0] or cy == 0) == edge
        ]
        config = ttnn.DataMovementConfigDescriptor(
            processor=ttnn.DataMovementProcessor.RISCV_0,
            noc=ttnn.NOC.RISCV_1_default if edge else ttnn.NOC.RISCV_0_default,
        )
        program.add_kernel(
            f"{kdir}/mlp/writer.cpp",
            writer_cores,
            config,
            writer_args,
            accessor,
            dispatch=True,
        )

    # Compute: the matmul blocking and the SiLU tile count, all compile-time.
    compute_args = [
        block_w, pc_m // sb_h, pc_m * block_w, sb_h * block_w,
        pc_n // sb_w, pc_n * block_w, pc_n,
        k_t // block_w, f_t // block_w, sb_h, sb_w, sb_h * sb_w,
        n_cmds, pc_m * pc_n,
    ]  # fmt: skip
    compute = ttnn.ComputeConfigDescriptor(
        math_fidelity=fidelity, fp32_dest_acc_en=True
    )
    program.add_kernel(
        f"{kdir}/mlp/compute.cpp", cores, compute, lambda cx, cy: [], compute_args
    )
    return program


def main():
    """Run the MLP, check it against fp64 and print one record."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shape", type=int, nargs=3, default=[320, 256, 384], metavar=("M", "K", "F")
    )
    parser.add_argument("--forwards", type=int, default=64)
    parser.add_argument("--block-w", type=int, default=4)
    parser.add_argument("--pll", type=int, default=1750)
    parser.add_argument(
        "--host-fed",
        action="store_true",
        help="enqueue every op from Python (streaming)",
    )
    parser.add_argument(
        "--trace", default=None, help="save the per-core command trace to this .pt file"
    )
    args = parser.parse_args()
    m, k, f = args.shape

    device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    device.disable_and_clear_program_cache()
    runtime = tt_x280.Runtime(device, pll_mhz=args.pll)
    plan = runtime.plan(depth=1)

    torch.manual_seed(0)
    x_t = torch.rand(m, k) * 2 - 1
    w1_t = (torch.rand(k, f) * 2 - 1) * 0.1
    w2_t = (torch.rand(f, f) * 2 - 1) * 0.1

    def to_device(t):
        return ttnn.from_torch(
            t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )

    x, w1, w2 = to_device(x_t), to_device(w1_t), to_device(w2_t)
    h, h2, y = (to_device(torch.zeros(m, f)) for _ in range(3))

    n_cmds = 3 * args.forwards
    feed = {}

    def submit_all(p):
        """Host-fed mode: enqueue every command, wait for the last, record timings."""
        start = time.perf_counter()
        for seq in range(1, n_cmds + 1):
            p.submit(seq)
        enqueued = time.perf_counter()
        p.wait(n_cmds)
        feed["host_us_per_op"] = round((enqueued - start) * 1e6 / n_cmds, 3)
        feed["streaming_us_per_forward"] = round(
            (time.perf_counter() - start) * 1e6 / args.forwards, 2
        )

    for _ in range(2):  # first run compiles
        program = build_mlp_program(
            plan, x, w1, w2, h, h2, y, n_cmds, args.block_w, ttnn.MathFidelity.HiFi2
        )
        start = time.perf_counter()
        ticks = program.run(
            [x, w1, w2, h, h2, y], n_cmds, host_fed=args.host_fed, submit=submit_all
        )
        wall = time.perf_counter() - start
    us = ticks / tt_x280.MTIME_HZ * 1e6 / args.forwards

    # One more run with per-core timestamps.
    if args.trace:
        tensors = [x, w1, w2, h, h2, y]
        program.run(tensors, n_cmds, args.host_fed, submit_all, trace=True)
        program.trace.save(
            args.trace,
            kernel="mlp",
            shape=(m, k, f),
            ops=["gemm1", "silu", "gemm2"],
            us_per_forward=us,
        )

    # fp64 reference with the device's bf16 rounding of inputs and of H.
    x64, w1_64, w2_64 = (t.bfloat16().double() for t in (x_t, w1_t, w2_t))
    h_ref = (x64 @ w1_64).bfloat16().double()
    reference = torch.nn.functional.silu(h_ref).bfloat16().double() @ w2_64
    result = ttnn.to_torch(y).double()
    flop = 2 * m * k * f + 2 * m * f * f
    record = {
        "kernel": "mlp",
        "shape": (m, k, f),
        "us_per_forward": round(us, 2),
        "tflops": round(flop / (us * 1e-6) / 1e12, 2),
        "host_wall_ms": round(wall * 1e3, 1),
        **feed,
        **error_stats(result, reference),
    }
    print(record)
    ttnn.close_mesh_device(device)


if __name__ == "__main__":
    os.environ.setdefault("TT_METAL_SLOW_DISPATCH_MODE", "1")
    main()
