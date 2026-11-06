"""The README's drawn figures: the teaser floorplan, the architecture and the protocol.

python scripts/docs/diagrams.py docs/assets
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import patches
from style import (
    FILL,
    GOLD,
    GREEN,
    INK,
    LINE,
    MUTED,
    OURS,
    PAPER,
    TTNN,
    VIOLET,
    arrow,
    blank,
    box,
    pale,
    save,
)

# Physical NOC0 columns of our harvested p150a: kind per column x = 0..16.
COLUMNS = {0: "dram", 7: "off", 8: "l2cpu", 9: "dram", 10: "off"}
TENSIX_X = [x for x in range(1, 17) if x not in COLUMNS]
ROWS = list(range(2, 12))  # Tensix rows y = 2..11
X280_Y = 3
L2CPU_Y = (3, 5, 7, 9)  # the four L2CPU tiles; ours is the one at y = 3
LABEL = dict(boxstyle="round,pad=0.25", fc=PAPER, ec="none", alpha=0.92)


def note(ax, xy, text, colour, size=9.5, **kw):
    """Text with a white backing so it stays readable over the grid."""
    ax.text(*xy, text, fontsize=size, color=colour, bbox=LABEL, zorder=5, **kw)


def floorplan(ax, x0, y0, w, h):
    """Draws the chip: Tensix grid, harvested/DRAM/L2CPU columns, panels, groups."""
    cw, rh = w / 17, h / 12

    def cell(x, y):
        return x0 + x * cw, y0 + (11 - y) * rh

    # Panels behind the cores: physically contiguous Tensix runs.
    for xs, name in ((range(1, 7), "west panel"), (range(11, 17), "east panel")):
        px, py = cell(xs[0], 11)
        ax.add_patch(
            patches.FancyBboxPatch(
                (px - 0.002, py - 0.004),
                cw * len(xs) + 0.004,
                rh * 10 + 0.008,
                boxstyle="round,pad=0,rounding_size=0.008",
                fc=pale(OURS, 0.08),
                ec=OURS,
                lw=1.4,
            )
        )
        ax.text(
            px + cw * len(xs) / 2,
            py + rh * 10 + 0.012,
            name,
            ha="center",
            fontsize=10.5,
            color=OURS,
            fontweight="bold",
        )

    # Cores: column leaders (bottom row) gold, other Tensix plain.
    for x in range(17):
        kind = COLUMNS.get(x, "tensix")
        for y in ROWS:
            cx, cy = cell(x, y)
            if kind == "tensix":
                leader = y == ROWS[-1]
                fc = pale(GOLD, 0.55) if leader else PAPER
                ec = GOLD if leader else LINE
            elif kind == "l2cpu" and y in L2CPU_Y:
                fc = VIOLET if y == X280_Y else pale(VIOLET, 0.25)
                ec = VIOLET
            elif kind == "dram":
                fc, ec = pale(TTNN, 0.25), TTNN
            else:
                fc, ec = FILL, LINE
            ax.add_patch(
                patches.Rectangle(
                    (cx + cw * 0.08, cy + rh * 0.1),
                    cw * 0.84,
                    rh * 0.8,
                    fc=fc,
                    ec=ec,
                    lw=0.9,
                )
            )

    # Column labels under the grid.
    for x, label in (
        (0, "DRAM"),
        (7, "harv."),
        (8, "L2CPU"),
        (9, "DRAM"),
        (10, "harv."),
    ):
        cx, cy = cell(x, ROWS[-1])
        ax.text(cx + cw / 2, cy - 0.022, label, ha="center", fontsize=8, color=MUTED)
    return cell, cw, rh


def teaser(out: Path) -> None:
    """Floorplan with the command and completion paths, and the headline numbers."""
    fig, ax = blank(19, 8.2)
    ax.text(0.015, 0.955, "tt-x280-dispatch", fontsize=22, fontweight="bold", color=INK)
    ax.text(
        0.015,
        0.91,
        "The X280 inside a Blackhole p150a sequences resident kernels on all "
        "120 Tensix: "
        "no host per op, no dispatch cores.",
        fontsize=13,
        color=MUTED,
    )
    cell, cw, rh = floorplan(ax, 0.13, 0.1, 0.45, 0.74)

    # X280 tile and the two command multicasts.
    xx, xy = cell(8, X280_Y)
    x280 = (xx + cw / 2, xy + rh / 2)
    note(
        ax,
        (xx + cw / 2, xy + rh * 1.25),
        "X280",
        VIOLET,
        11,
        ha="center",
        fontweight="bold",
    )
    for tx in (3.5, 13.5):
        px, py = cell(tx, 6)
        arrow(
            ax, x280, (px, py + rh / 2), VIOLET, lw=2.2, rad=0.25 if tx < 8 else -0.25
        )
    note(ax, cell(1.3, 7.3), "command k: one posted\nstore per panel", VIOLET)

    # One column: members write their slot in the leader, the leader acks to LIM.
    for y in (2, 5, 8):
        a, b = cell(14, y), cell(14, ROWS[-1])
        arrow(
            ax,
            (a[0] + cw * 0.9, a[1] + rh / 2),
            (b[0] + cw * 0.9, b[1] + rh * 0.8),
            GOLD,
            lw=1.3,
            rad=-0.3,
        )
    lx, ly = cell(14, ROWS[-1])
    arrow(
        ax,
        (lx + cw / 2, ly + rh * 0.1),
        (x280[0] + 0.004, x280[1] - rh * 0.4),
        GOLD,
        lw=2.0,
        rad=-0.55,
    )
    note(ax, cell(14.3, 9.4), "members →\ncolumn leader", GOLD)
    note(ax, cell(9.6, 9.6), "12 leaders →\nX280: done k", GOLD)

    # Host link: submit through a write-combining window, done count to pinned memory.
    box(ax, 0.01, 0.42, 0.07, 0.17, "host", "", edge=GREEN, head_colour=GREEN)
    arrow(ax, (0.08, 0.54), (x280[0] - cw * 0.4, x280[1]), GREEN, lw=1.6, rad=-0.2)
    arrow(
        ax,
        (x280[0] - cw * 0.4, x280[1] - rh * 0.3),
        (0.08, 0.46),
        GREEN,
        lw=1.6,
        ls="--",
        rad=0.2,
    )
    note(ax, (0.01, 0.62), "submit k: 0.63 µs", GREEN)
    note(ax, (0.01, 0.37), "done count →\npinned host memory", GREEN)

    # Headline numbers on the right.
    sub = fig.add_axes((0.735, 0.14, 0.25, 0.66))
    rows = [
        ("dispatch step, dependent", 0.77, None),
        ("dispatch step, ring of 4", 0.46, None),
        ("MLP 320×256→384→384", 19.1, 25.3),
        ("flash attention 12×1280×64", 97.6, 173.1),
        ("ResNet block 1×56×56×64", 75.3, 178.3),
        ("matmul 2560×3072×4096", 410.0, 464.0),
    ]
    names = [r[0] for r in rows]
    for i, (_, ours, ttnn) in enumerate(rows):
        y = i - 0.18 if ttnn else i
        sub.barh(y, ours, 0.34, color=OURS)
        sub.annotate(
            f"{ours:g} µs",
            (ours, y),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            fontsize=10,
            color=INK,
        )
        if ttnn:
            sub.barh(i + 0.18, ttnn, 0.34, color=TTNN)
            sub.annotate(
                f"{ttnn:g}",
                (ttnn, i + 0.18),
                xytext=(4, 0),
                textcoords="offset points",
                va="center",
                fontsize=10,
                color=MUTED,
            )
    sub.set_xscale("log")
    sub.set_xlim(0.3, 2500)
    sub.set_yticks(range(len(rows)), names, fontsize=11)
    sub.set_ylim(len(rows) - 0.5, -0.5)
    sub.spines[["top", "right", "left"]].set_visible(False)
    sub.spines["bottom"].set_color(LINE)
    sub.tick_params(colors=MUTED, left=False)
    sub.tick_params(axis="y", colors=INK)
    sub.grid(axis="x", color=LINE, lw=0.6)
    sub.set_axisbelow(True)
    sub.set_xlabel("µs, log scale", color=MUTED)
    sub.set_title(
        "X280, 120 cores   vs   ttnn trace, 110 cores",
        loc="left",
        fontsize=12.5,
        color=INK,
        fontweight="bold",
    )
    marks = [
        plt.Line2D([], [], color=c, lw=6, label=n)
        for n, c in (("X280-sequenced", OURS), ("ttnn trace", TTNN))
    ]
    sub.legend(
        handles=marks,
        loc="upper left",
        bbox_to_anchor=(0, -0.1),
        ncol=2,
        frameon=False,
        fontsize=10.5,
    )
    save(fig, out, "teaser.jpg")


ARCH_COLUMNS = [
    (
        "Host",
        "Python on ttnn",
        GREEN,
        [
            (
                "ResidentProgram",
                "CBs, semaphores, kernels per core\n"
                "→ ttnn.ProgramDescriptor\nlaunched once: ttnn.generic_op",
            ),
            (
                "Plan (_ttx280)",
                "panels from NOC0 coordinates\n"
                "column groups, LIM layout\n16 dispatch args per core",
            ),
            (
                "Host-fed stream",
                "submit(k): one 4 B write-combining\n"
                "PCIe store into a LIM ring, 0.63 µs\nwait(k): read pinned host memory",
            ),
        ],
    ),
    (
        "X280 sequencer",
        "hart 0, image in its LIM",
        VIOLET,
        [
            (
                "Per command k",
                "1  take k: self-issued or host ring\n"
                "2  depth 1: wait for done k−1\n    depth 4: up to 4 in flight\n"
                "3  issue k: a posted store per panel\n4  sweep the 12 leader slots\n"
                "5  done count → host memory",
            ),
            (
                "TLB windows",
                "2 MiB multicast windows over\nphysically contiguous Tensix\n"
                "rectangles, one unicast window\nto the PCIe tile",
            ),
        ],
    ),
    (
        "Command path",
        "two posted multicasts",
        OURS,
        [
            (
                "Panels",
                "west: logical x 0–5 = NOC0 1–6\neast: logical x 6–11 = NOC0 11–16\n"
                "never across harvested, DRAM\nor L2CPU columns",
            ),
            (
                "Command ring",
                "control CB 8 in every core\nslot = k mod depth, 16 B\n"
                "the reader spins on its own L1",
            ),
        ],
    ),
    (
        "Tensix worker × 120",
        "one resident program",
        INK,
        [
            (
                "reader",
                "Worker.wait(k)\nop = table[k]: read inputs,\nmulticast, push CBs",
            ),
            ("compute", "same op table, consumes CBs\nmatmul, SDPA, conv, SiLU"),
            (
                "writer",
                "write outputs, barrier\nWorker.complete(k)\noptional trace stamps",
            ),
        ],
    ),
    (
        "Completion path",
        "12 column groups",
        GOLD,
        [
            (
                "Write-gather",
                "each core writes tag<<16 | k\ninto its own 16 B slot in its\n"
                "column leader: no atomics",
            ),
            (
                "Leader ack",
                "all 9 slots ≥ k → the leader\nwrites k to its LIM slot;\n"
                "0.5 µs from the last core done\nto the X280",
            ),
        ],
    ),
]


def architecture(out: Path) -> None:
    """Five columns: host, X280 sequencer, commands, Tensix workers, completion."""
    fig, ax = blank(20, 8.4)
    width, gap, line = 0.18, 0.02, 0.034
    tops = []
    for i, (head, sub, colour, items) in enumerate(ARCH_COLUMNS):
        x = 0.012 + i * (width + gap)
        ax.text(x, 0.95, head, fontsize=18, fontweight="bold", color=colour)
        ax.text(x, 0.905, sub, fontsize=12.5, color=MUTED)

        # Boxes sized to their text, stacked from the top.
        y = 0.86
        for title_text, body in items:
            h = 0.075 + line * (body.count("\n") + 1)
            edge = colour if colour != INK else "#424a53"
            box(
                ax,
                x,
                y - h,
                width,
                h,
                title_text,
                body,
                edge=edge,
                head_colour=colour,
                body_size=11.5,
                head_size=13,
            )
            tops.append((i, y - h / 2))
            y -= h + 0.035

    # Flow: host → X280 → panels → workers → completion, and back to the X280.
    mid = [(0.012 + i * (width + gap), 0.62) for i in range(5)]
    for i, colour in enumerate((GREEN, VIOLET, OURS, INK)):
        arrow(ax, (mid[i][0] + width, 0.66), (mid[i + 1][0], 0.66), colour, lw=2.2)
    arrow(
        ax,
        (mid[4][0] + width / 2, 0.42),
        (mid[1][0] + width / 2, 0.32),
        GOLD,
        lw=2.0,
        ls="--",
        rad=-0.32,
    )
    ax.text(
        0.6,
        0.1,
        "done k from all 12 leaders  →  X280 issues k + 1",
        ha="center",
        fontsize=13,
        color=GOLD,
        fontweight="bold",
    )
    save(fig, out, "architecture.jpg")


def main():
    """Write every drawn figure into the given directory."""
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    teaser(args.out)
    architecture(args.out)


if __name__ == "__main__":
    main()
