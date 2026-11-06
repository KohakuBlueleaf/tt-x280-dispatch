"""Shared colours and drawing helpers for the README figures."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import colors, patches

INK, MUTED, PAPER, LINE, FILL = "#1f2328", "#57606a", "#ffffff", "#d0d7de", "#f6f8fa"
OURS, TTNN, EAGER = "#0969da", "#6e7781", "#afb8c1"
VIOLET, GOLD, GREEN, ORANGE, RED = "#8250df", "#bf8700", "#1a7f37", "#e6730d", "#cf222e"
PINK = "#bf3989"
OPS = {
    "gemm1": OURS,
    "silu": GREEN,
    "gemm2": PINK,
    "conv1": OURS,
    "conv2": PINK,
    "attention": OURS,
}

plt.rcParams["font.family"] = "DejaVu Sans"


def pale(colour, alpha=0.18):
    """The colour at reduced opacity, for fills."""
    return colors.to_rgba(colour, alpha)


def save(fig, out: Path, name: str) -> None:
    """Write a figure as a JPEG into out."""
    fig.savefig(out / name, dpi=150, facecolor=PAPER, pil_kwargs={"quality": 90})
    plt.close(fig)


def style(ax, grid="y") -> None:
    """Light axes: no top/right spines, muted ticks, a grid on one axis."""
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(LINE)
    ax.tick_params(colors=MUTED, labelsize=11)
    ax.grid(axis=grid, color=LINE, lw=0.6)
    ax.set_axisbelow(True)


def title(ax, text: str, sub: str | None = None) -> None:
    """A bold left title with an optional muted subtitle."""
    ax.set_title(text, loc="left", fontsize=13, color=INK, fontweight="bold", pad=24)
    if sub:
        ax.annotate(
            sub,
            (0, 1),
            xycoords="axes fraction",
            xytext=(0, 7),
            textcoords="offset points",
            fontsize=10.5,
            color=MUTED,
        )


def box(
    ax,
    x,
    y,
    w,
    h,
    head,
    body="",
    edge=LINE,
    fill=FILL,
    head_colour=INK,
    lw=1.4,
    head_size=11.5,
    body_size=9.6,
):
    """A rounded box with a bold heading and muted body text, top-left aligned."""
    ax.add_patch(
        patches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0,rounding_size=0.012",
            fc=fill,
            ec=edge,
            lw=lw,
        )
    )
    ax.text(
        x + 0.012,
        y + h - 0.016,
        head,
        fontsize=head_size,
        fontweight="bold",
        color=head_colour,
        va="top",
    )
    if body:
        ax.text(
            x + 0.012,
            y + h - 0.016 - head_size * 0.0042,
            body,
            fontsize=body_size,
            color=MUTED,
            va="top",
            linespacing=1.35,
        )


def arrow(ax, start, end, colour=INK, lw=1.6, style="-|>", ls="-", rad=0.0):
    """An arrow between two points in axes coordinates."""
    ax.add_patch(
        patches.FancyArrowPatch(
            start,
            end,
            arrowstyle=style,
            mutation_scale=14,
            color=colour,
            lw=lw,
            ls=ls,
            connectionstyle=f"arc3,rad={rad}",
        )
    )


def blank(fig_w, fig_h):
    """A figure with one invisible full-size axes in unit coordinates."""
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor=PAPER)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return fig, ax
