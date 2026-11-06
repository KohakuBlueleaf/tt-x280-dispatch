"""The README's measured figures, from results.py and docs/measurements/traces.

machine.jpg: per-core waterfall of a traced MLP run with its step breakdown.
dispatch.jpg: dispatch cost per command by model and X280 clock.
kernels.jpg: X280-sequenced kernels against ttnn trace and eager.
precision.jpg: error against fp64 per kernel and setting.
host.jpg: host submission paths and host-fed streaming.

    python scripts/docs/plots.py docs/assets
"""

import argparse
import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import results as r
from matplotlib import ticker
from style import (
    EAGER,
    GOLD,
    GREEN,
    INK,
    LINE,
    MUTED,
    OPS,
    ORANGE,
    OURS,
    PAPER,
    TTNN,
    VIOLET,
    save,
    style,
    title,
)

ROOT = Path(__file__).resolve().parents[2]
TRACES = ROOT / "docs" / "measurements" / "traces"


def load_trace(name):
    """Load a saved trace with tt_x280/trace.py, without importing the device stack."""
    path = ROOT / "src" / "frontend" / "tt_x280" / "trace.py"
    spec = importlib.util.spec_from_file_location("tt_x280_trace", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Trace.load(TRACES / name)


def legend(fig, pairs, y=0.995):
    """A one-row legend of coloured bars across the top of the figure."""
    marks = [plt.Line2D([], [], color=c, lw=6, label=n) for n, c in pairs]
    fig.legend(
        handles=marks,
        loc="upper center",
        ncol=len(marks),
        frameon=False,
        fontsize=12,
        bbox_to_anchor=(0.5, y),
    )


def machine(out: Path) -> None:
    """Waterfall: every core's commands over time, plus the step anatomy."""
    trace = load_trace("mlp_320.pt")
    ops = trace.meta["ops"]
    gx, gy = trace.grid
    first, count = 3 * 3, 3 * 4  # forwards 4..7, after warm-up
    t0 = trace.seen[:, first].min()

    fig = plt.figure(figsize=(19, 9), facecolor=PAPER)
    left = fig.add_axes((0.05, 0.08, 0.56, 0.8))
    top = fig.add_axes((0.67, 0.56, 0.31, 0.32))
    bottom = fig.add_axes((0.67, 0.08, 0.31, 0.34))

    # Left: one row per core (column-major so column groups sit together).
    order = [y * gx + x for x in range(gx) for y in range(gy)]
    for row, core in enumerate(order):
        for i in range(first, first + count):
            colour = OPS.get(ops[i % len(ops)], OURS)
            s, d, a = (
                float(v[core, i] - t0) for v in (trace.seen, trace.done, trace.acked)
            )
            left.barh(row, d - s, left=s, height=0.86, color=colour, lw=0)
            left.barh(row, a - d, left=d, height=0.86, color=GOLD, lw=0)
    for x in range(1, gx):
        left.axhline(x * gy - 0.5, color=LINE, lw=0.6)
    left.set_ylim(len(order) - 0.5, -0.5)
    left.set_yticks(
        [x * gy + gy / 2 - 0.5 for x in range(gx)], [f"col {x}" for x in range(gx)]
    )
    left.set_xlabel("µs", color=MUTED)
    style(left, grid="x")
    shape = "×".join(str(v) for v in trace.meta["shape"])
    title(
        left,
        f"MLP {shape}, 4 forwards on 120 cores (rows: cores by column group)",
        "bars: core busy with the command   gold: completion to its leader / X280",
    )

    # Right top: per command, fan-out, busy and gather.
    steps = trace.steps()
    idx = range(first, first + count)
    x = list(range(count))
    busy = [float(steps["slowest"][i]) for i in idx]
    fan = [float(steps["fan_out"][i]) for i in idx]
    gather = [float(steps["gather"][i]) for i in idx]
    gap = [
        float(steps["step"][i]) - b - f - g
        for i, b, f, g in zip(idx, busy, fan, gather)
    ]
    colours = [OPS.get(ops[i % len(ops)], OURS) for i in idx]
    top.bar(x, busy, color=colours, width=0.7)
    top.bar(x, fan, bottom=busy, color=VIOLET, width=0.7)
    top.bar(x, gather, bottom=[b + f for b, f in zip(busy, fan)], color=GOLD, width=0.7)
    top.bar(
        x,
        gap,
        bottom=[b + f + g for b, f, g in zip(busy, fan, gather)],
        color=TTNN,
        width=0.7,
    )
    top.set_xticks(x, [ops[i % len(ops)] for i in idx], rotation=45, fontsize=9)
    top.set_ylabel("µs", color=MUTED)
    style(top)
    title(
        top,
        "one step = slowest core + dispatch",
        f"fan-out {sum(fan) / count:.2f} µs, gather {sum(gather) / count:.2f} µs, "
        f"other {sum(gap) / count:.2f} µs per command",
    )

    # Right bottom: when each core saw one command, relative to the first core.
    cmd = first + 1
    seen = (trace.seen[:, cmd] - trace.seen[:, cmd].min()) * 1000
    grid = seen.reshape(gy, gx)
    im = bottom.imshow(grid.numpy(), cmap="Purples", aspect="auto")
    bottom.set_xticks(range(gx))
    bottom.set_yticks(range(gy))
    bottom.set_xlabel("logical x (west panel 0–5, east panel 6–11)", color=MUTED)
    bottom.set_ylabel("logical y", color=MUTED)
    bottom.tick_params(labelsize=9, colors=MUTED)
    cbar = fig.colorbar(im, ax=bottom, fraction=0.04)
    cbar.set_label("ns", color=MUTED)
    title(
        bottom,
        "command arrival over the 12 × 10 grid",
        f"all 120 cores within {float(seen.max()):.0f} ns (two panel multicasts)",
    )

    legend(
        fig,
        [(op, OPS[op]) for op in ops]
        + [("fan-out", VIOLET), ("gather", GOLD), ("X280 issue", TTNN)],
    )
    save(fig, out, "machine.jpg")


def dispatch(out: Path) -> None:
    """Dispatch cost per command: models at three X280 clocks."""
    fig, ax = plt.subplots(figsize=(12, 5.2), facecolor=PAPER)
    clocks = sorted(r.DISPATCH_PLL)
    width = 0.26
    for j, clock in enumerate(clocks):
        values = r.DISPATCH_PLL[clock]
        xs = [i + (j - 1) * width for i in range(len(values))]
        bars = ax.bar(
            xs, values, width, color=[TTNN, VIOLET, OURS][j], label=f"{clock} MHz"
        )
        ax.bar_label(bars, fmt="%.2f", fontsize=9.5, color=INK, padding=2)
    ax.set_xticks(range(len(r.DISPATCH_MODELS)), r.DISPATCH_MODELS, fontsize=11.5)
    ax.set_ylabel("µs per command, empty work", color=MUTED)
    ax.legend(frameon=False, fontsize=11)
    style(ax)
    title(
        ax,
        "Dispatch cost per command, 120 cores",
        "grouped completion makes it nearly clock-independent; "
        "at the 200 MHz boot clock v1 sync is 1.04 µs (flat: 19.9)",
    )
    fig.tight_layout()
    save(fig, out, "dispatch.jpg")


def hbars(ax, labels, series, name, sub):
    """Grouped horizontal bars on a log axis: series = [(label, colour, values)]."""
    n = len(series)
    height = 0.8 / n
    for j, (_label, colour, values) in enumerate(series):
        ys = [i + (j - (n - 1) / 2) * height for i in range(len(labels))]
        for y, v in zip(ys, values):
            if v is None:
                continue
            ax.barh(y, v, height * 0.92, color=colour)
            ax.annotate(
                f"{v:g}",
                (v, y),
                xytext=(3, 0),
                textcoords="offset points",
                va="center",
                fontsize=9.5,
                color=INK,
            )
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())
    ax.set_yticks(range(len(labels)), labels, fontsize=11)
    ax.set_ylim(len(labels) - 0.5, -0.5)
    style(ax, grid="x")
    ax.tick_params(axis="y", colors=INK, left=False)
    title(ax, name, sub)


def kernels(out: Path) -> None:
    """Four workloads: X280 v1 against ttnn trace and eager, µs per op or forward."""
    fig, axes = plt.subplots(2, 2, figsize=(19, 10.5), facecolor=PAPER)
    mm = r.MATMUL
    hbars(
        axes[0, 0],
        [f"{m}×{n}×{k}" for m, n, k, *_ in mm],
        [
            ("X280 ring 4", OURS, [x[5] for x in mm]),
            ("X280 sync", VIOLET, [x[4] for x in mm]),
            ("ttnn trace", TTNN, [x[6] for x in mm]),
            ("ttnn eager", EAGER, [x[7] for x in mm]),
        ],
        "matmul",
        "µs per matmul; ring 4 overlaps consecutive matmuls",
    )
    mlp = r.MLP
    hbars(
        axes[0, 1],
        [f"{m}×{k}→{f}→{f}" for m, k, f, *_ in mlp],
        [
            ("X280", OURS, [x[3] for x in mlp]),
            ("ttnn trace", TTNN, [x[4] for x in mlp]),
            ("ttnn eager", EAGER, [x[5] for x in mlp]),
        ],
        "MLP forward  silu(X W1) W2",
        "three dependent commands per forward",
    )
    fa = r.FLASH_ATTN
    hbars(
        axes[1, 0],
        [x[0] for x in fa],
        [
            ("X280, K/V multicast lanes", OURS, [x[2] for x in fa]),
            ("X280, reader without sharing", VIOLET, [x[1] for x in fa]),
            ("ttnn trace", TTNN, [x[3] for x in fa]),
            ("ttnn eager", EAGER, [x[4] for x in fa]),
        ],
        "flash attention (heads × seq × dim)",
        "compute = ttnn's sdpa_standard; output bit-identical to ttnn",
    )
    rb = r.RESBLOCK
    hbars(
        axes[1, 1],
        [x[0] for x in rb],
        [
            ("X280, halo reader", OURS, [x[1] for x in rb]),
            ("X280, per-row reader", VIOLET, [x[2] for x in rb]),
            ("ttnn trace", TTNN, [x[3] for x in rb]),
            ("ttnn eager", EAGER, [x[4] for x in rb]),
        ],
        "ResNet basic block (N×H×W×C)",
        "two dependent conv commands; ttnn: six ops",
    )
    legend(
        fig,
        [
            ("X280-sequenced, 120 cores", OURS),
            ("X280, earlier kernel or sync", VIOLET),
            ("ttnn trace, 110 cores", TTNN),
            ("ttnn eager", EAGER),
        ],
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, out, "kernels.jpg")


def precision(out: Path) -> None:
    """rel-RMS error against fp64, X280 and ttnn side by side."""
    fig, ax = plt.subplots(figsize=(12, 5.4), facecolor=PAPER)
    labels = [f"{k}\n{s}" for k, s, *_ in r.PRECISION]
    xs = range(len(labels))
    ours = [x[2] for x in r.PRECISION]
    ttnn = [x[3] for x in r.PRECISION]
    ax.bar([x - 0.2 for x in xs], ours, 0.38, color=OURS, label="X280 kernels")
    ax.bar(
        [x + 0.2 for x in xs if ttnn[x]],
        [v for v in ttnn if v],
        0.38,
        color=TTNN,
        label="ttnn",
    )
    ax.set_yscale("log")
    ax.set_xticks(xs, labels, fontsize=10)
    ax.set_ylabel("rel-RMS error vs fp64", color=MUTED)
    ax.legend(frameon=False, fontsize=11)
    style(ax)
    title(
        ax,
        "Error against an fp64 reference",
        "matches ttnn; HiFi4 removes the conv bias; exact exp halves SDPA error",
    )
    fig.tight_layout()
    save(fig, out, "precision.jpg")


def host(out: Path) -> None:
    """Host submission cost per path and host-fed streaming against ttnn."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(17, 5), facecolor=PAPER)
    names = [x[0] for x in r.HOST_IO]
    hbars(
        a,
        names,
        [("µs per command", GREEN, [x[1] for x in r.HOST_IO])],
        "Host → X280 submission",
        "µs per command written to the X280's LIM",
    )
    hf = r.HOST_FED
    hbars(
        b,
        [f"MLP {m}×{k}→{f}" for m, k, f, *_ in hf],
        [
            ("X280 self-issued", OURS, [x[3] for x in hf]),
            ("host-fed streaming", GREEN, [x[4] for x in hf]),
            ("host-fed, sync each forward", ORANGE, [x[5] for x in hf]),
            ("ttnn trace", TTNN, [x[6] for x in hf]),
            ("ttnn eager", EAGER, [x[7] for x in hf]),
        ],
        "Every op enqueued from the host",
        "µs per forward; streaming is device-bound at the larger shape",
    )
    b.legend(
        handles=[
            plt.Line2D([], [], color=c, lw=6, label=n)
            for n, c in (
                ("self-issued", OURS),
                ("host-fed stream", GREEN),
                ("host-fed sync", ORANGE),
                ("ttnn trace", TTNN),
                ("ttnn eager", EAGER),
            )
        ],
        frameon=False,
        fontsize=10,
        loc="upper center",
        bbox_to_anchor=(0.45, -0.08),
        ncol=5,
    )
    fig.tight_layout()
    save(fig, out, "host.jpg")


def main():
    """Write every measured figure into the given directory."""
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    machine(args.out)
    dispatch(args.out)
    kernels(args.out)
    precision(args.out)
    host(args.out)


if __name__ == "__main__":
    main()
