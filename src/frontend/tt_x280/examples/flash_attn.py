"""Flash attention softmax(Q K^T / sqrt(D)) V over [1, NH, S, D], one command a forward.

Compute is ttnn's sdpa_standard; readers share K/V chunks across lanes of cores.

Usage: TT_VISIBLE_DEVICES=1 python -m tt_x280.examples.flash_attn --shape 12 1280 64
"""

import argparse
import math
import os
import struct
import time

import torch
import ttnn

import tt_x280
from tt_x280.layout import granularity, largest_subblock, plan_lanes
from tt_x280.stats import error_stats


def build_fa_program(plan, q, k, v, o, n_cmds, q_chunk, k_chunk, fidelity, exp_approx):
    """The resident flash-attention program: one forward per command.

    A job is one head's run of `lane_size` consecutive Q chunks. Lanes of cores take
    jobs round-robin; a lane's first core reads each K/V chunk once and multicasts it.
    """
    _, nh, s, d = q.shape
    dh_t, s_t = d // 32, s // 32
    sq_t, sk_t = q_chunk // 32, k_chunk // 32
    q_chunks = s_t // sq_t
    tile, tile32 = 2048, 4096
    kdir = tt_x280.KERNEL_DIR
    grid = (plan.grid_x, plan.grid_y)

    # Lanes, and per core its schedule and multicast rectangle (translated NOC0).
    lane_size, lanes = plan_lanes(grid, plan.panels, nh, q_chunks)
    num_jobs = nh * q_chunks // lane_size
    schedule = {}  # core -> [lane, pos, num_lanes, num_jobs, lane_size]
    multicast = {}  # core -> [lo_x, lo_y, hi_x, hi_y, receivers, sender_x, sender_y]
    for li, lane in enumerate(lanes):
        sender = plan.translated(*lane[0])
        receivers = [plan.translated(*c) for c in lane[1:]] or [sender]
        lo = [min(t[i] for t in receivers) for i in (0, 1)]
        hi = [max(t[i] for t in receivers) for i in (0, 1)]
        for p, core in enumerate(lane):
            schedule[core] = [li, p, len(lanes), num_jobs, lane_size]
            multicast[core] = [*lo, *hi, len(lane) - 1, *sender]
    senders = [lane[0] for lane in lanes]
    receivers = [c for lane in lanes for c in lane[1:]]

    # Cores outside every lane get no jobs but still take part in the dispatch.
    for cy in range(plan.grid_y):
        for cx in range(plan.grid_x):
            if (cx, cy) not in schedule:
                t = plan.translated(cx, cy)
                schedule[(cx, cy)] = [0, 0, 1, 0, 1]
                multicast[(cx, cy)] = [*t, *t, 0, *t]
                senders.append((cx, cy))

    # CBs in sdpa_standard's layout: inputs, constants, output, then intermediates.
    program = tt_x280.ResidentProgram(plan)
    program.add_cb(0, ttnn.bfloat16, tile, 2 * sq_t * dh_t)  # Q
    program.add_cb(1, ttnn.bfloat16, tile, 2 * sk_t * dh_t)  # K^T chunk
    program.add_cb(2, ttnn.bfloat16, tile, 2 * sk_t * dh_t)  # V chunk
    program.add_cb(4, ttnn.bfloat16, tile, 1)  # reduce scaler
    program.add_cb(5, ttnn.bfloat16, tile, 1)  # column identity
    program.add_cb(16, ttnn.bfloat16, tile, sq_t * dh_t)  # output chunk
    program.add_cb(24, ttnn.float32, tile32, sq_t * sk_t)  # QK^T scores
    program.add_cb(25, ttnn.bfloat16, tile, sq_t * dh_t)  # output accumulator A
    program.add_cb(26, ttnn.bfloat16, tile, sq_t * dh_t)  # output accumulator B
    program.add_cb(27, ttnn.bfloat16, tile, sq_t)  # row max (ping-pong)
    program.add_cb(28, ttnn.bfloat16, tile, sq_t)
    program.add_cb(29, ttnn.float32, tile32, sq_t)  # row sum (ping-pong)
    program.add_cb(30, ttnn.float32, tile32, sq_t)
    program.add_cb(31, ttnn.bfloat16, tile, sq_t)  # exp(max_prev - max_cur)

    accessor = ttnn.TensorAccessorArgs(q).get_compile_time_args()
    addr = {name: t.buffer_address() for name, t in dict(q=q, k=k, v=v, o=o).items()}

    sem_ready, sem_valid = program.add_semaphore(), program.add_semaphore()

    def reader_args(cx, cy):
        """Reader args: Q/K/V, the core's schedule, its lane multicast, semaphores."""
        core = (cx, cy)
        return [
            addr["q"],
            addr["k"],
            addr["v"],
            *schedule[core],
            *multicast[core],
            sem_ready,
            sem_valid,
            n_cmds,
        ]

    def writer_args(cx, cy):
        """Writer args: O and the core's schedule."""
        return [addr["o"], *schedule[(cx, cy)], n_cmds]

    # Senders read and multicast on NOC0, receivers on NOC1; writers on the other one.
    identity = 0x3F803F80  # two bf16 1.0
    for role_cores, reader_noc, writer_noc in (
        (senders, ttnn.NOC.RISCV_0_default, ttnn.NOC.RISCV_1_default),
        (receivers, ttnn.NOC.RISCV_1_default, ttnn.NOC.RISCV_0_default),
    ):
        if not role_cores:
            continue
        reader = ttnn.DataMovementConfigDescriptor(
            processor=ttnn.DataMovementProcessor.RISCV_1, noc=reader_noc
        )
        program.add_kernel(
            f"{kdir}/flash_attn/reader.cpp",
            role_cores,
            reader,
            reader_args,
            [dh_t, s_t, s_t, sq_t, sk_t, *accessor],
            dispatch=True,
        )
        writer = ttnn.DataMovementConfigDescriptor(
            processor=ttnn.DataMovementProcessor.RISCV_0, noc=writer_noc
        )
        program.add_kernel(
            f"{kdir}/flash_attn/writer.cpp",
            role_cores,
            writer,
            writer_args,
            [dh_t, s_t, sq_t, identity, *accessor],
            dispatch=True,
        )
    cores = senders + receivers

    def compute_args_rt(cx, cy):
        """Compute args: (first unit, number of units) of the core per command."""
        lane, _, num_lanes, jobs, _ = schedule[(cx, cy)]
        return [0, len(range(lane, jobs, num_lanes))]

    # Compute: blocking and defines as ttnn's SDPA program factory sets them.
    qk_h, qk_w = largest_subblock(sq_t, sk_t)
    out_h, out_w = largest_subblock(sq_t, dh_t)
    scale_bits = struct.unpack("<I", struct.pack("<f", 1.0 / math.sqrt(d)))[0]
    compute_args = [
        s_t,
        dh_t,
        sq_t,
        q_chunks,
        sk_t,
        s_t // sk_t,
        qk_w,
        qk_h,
        out_w,
        out_h,
        scale_bits,
        n_cmds,
    ]
    defines = [
        ("STATS_GRANULARITY", str(granularity(sq_t, 4))),
        ("SUB_EXP_GRANULARITY", str(granularity(sk_t, 4))),
        ("DHT_GRANULARITY", str(granularity(dh_t, 4))),
        ("REDUCE_GRANULARITY", str(granularity(sq_t, 2))),
        ("EXP_APPROX_MODE", str(int(exp_approx))),
    ]
    compute = ttnn.ComputeConfigDescriptor(
        math_fidelity=fidelity, fp32_dest_acc_en=True
    )
    program.add_kernel(
        f"{kdir}/flash_attn/compute.cpp",
        cores,
        compute,
        compute_args_rt,
        compute_args,
        defines,
    )
    program.lane_size = lane_size
    return program


def ttnn_sdpa(device, q, k, v, q_chunk, k_chunk, fidelity, exp_approx, grid):
    """ttnn's SDPA output with the same chunking and compute config."""
    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=grid,
        q_chunk_size=q_chunk,
        k_chunk_size=k_chunk,
        exp_approx_mode=exp_approx,
    )
    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=fidelity,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )

    return ttnn.transformer.scaled_dot_product_attention(
        q,
        k,
        v,
        is_causal=False,
        program_config=program_config,
        compute_kernel_config=compute_config,
    )


def main():
    """Run flash attention, check it against fp64 and ttnn SDPA, print one record."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shape", type=int, nargs=3, default=[12, 1280, 64], metavar=("NH", "S", "D")
    )
    parser.add_argument(
        "--chunk", type=int, nargs=2, default=[128, 128], metavar=("Q", "K")
    )
    parser.add_argument("--forwards", type=int, default=32)
    parser.add_argument("--pll", type=int, default=1750)
    parser.add_argument("--exp-approx", type=int, default=1)
    parser.add_argument(
        "--no-ttnn", action="store_true", help="skip the ttnn SDPA comparison"
    )
    parser.add_argument("--trace", default=None, help="save a per-core trace (.pt)")
    args = parser.parse_args()
    nh, s, d = args.shape
    q_chunk, k_chunk = args.chunk
    fidelity = ttnn.MathFidelity.HiFi2

    device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    device.disable_and_clear_program_cache()
    runtime = tt_x280.Runtime(device, pll_mhz=args.pll)
    plan = runtime.plan(depth=1)

    torch.manual_seed(0)
    q_t, k_t, v_t = (torch.randn(1, nh, s, d) for _ in range(3))

    def to_device(t):
        return ttnn.from_torch(
            t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )

    q, k, v = to_device(q_t), to_device(k_t), to_device(v_t)
    o = to_device(torch.zeros(1, nh, s, d))

    for _ in range(2):  # first run compiles
        program = build_fa_program(
            plan, q, k, v, o, args.forwards, q_chunk, k_chunk, fidelity, args.exp_approx
        )
        start = time.perf_counter()
        ticks = program.run([q, k, v, o], args.forwards)
        wall = time.perf_counter() - start
    us = ticks / tt_x280.MTIME_HZ * 1e6 / args.forwards

    # One more run with per-core timestamps.
    if args.trace:
        program.run([q, k, v, o], args.forwards, trace=True)
        program.trace.save(
            args.trace,
            kernel="flash_attn",
            shape=(nh, s, d),
            ops=["attention"],
            us_per_forward=us,
        )

    # fp64 reference from the bf16-rounded inputs.
    q64, k64, v64 = (t.bfloat16().double() for t in (q_t, k_t, v_t))
    reference = torch.softmax(q64 @ k64.transpose(-1, -2) / math.sqrt(d), dim=-1) @ v64
    result = ttnn.to_torch(o).double()
    flop = 4 * nh * s * s * d
    record = {
        "kernel": "flash_attn",
        "shape": (nh, s, d),
        "chunk": (q_chunk, k_chunk),
        "kv_lane": program.lane_size,
        "us_per_forward": round(us, 2),
        "tflops": round(flop / (us * 1e-6) / 1e12, 2),
        "host_wall_ms": round(wall * 1e3, 1),
        **error_stats(result, reference),
    }
    # ttnn's SDPA on the same inputs and settings, for numerics (timing: bench/).
    if not args.no_ttnn:
        device.enable_program_cache()
        grid = ttnn.CoreCoord(plan.grid_x, plan.grid_y)
        out = ttnn_sdpa(
            device, q, k, v, q_chunk, k_chunk, fidelity, bool(args.exp_approx), grid
        )
        ttnn_result = ttnn.to_torch(out).double()
        record["ttnn"] = error_stats(ttnn_result, reference)
        record["max_abs_vs_ttnn"] = float((result - ttnn_result).abs().max())
    print(record)
    ttnn.close_mesh_device(device)


if __name__ == "__main__":
    os.environ.setdefault("TT_METAL_SLOW_DISPATCH_MODE", "1")
    main()
