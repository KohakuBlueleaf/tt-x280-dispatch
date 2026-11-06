"""Work layout for resident programs: per-core ranges, lanes and block sizes.

No device needed. Grids are (grid_x, grid_y); panels are inclusive logical column
ranges, as in Plan.panels.
"""

# Output subblocks in ttnn SDPA's preference order.
_SUBBLOCKS = (
    (2, 4), (4, 2), (1, 8), (8, 1), (1, 7), (7, 1), (2, 3), (3, 2), (1, 6), (6, 1),
    (1, 5), (5, 1), (2, 2), (1, 4), (4, 1), (1, 3), (3, 1), (1, 2), (2, 1), (1, 1),
)  # fmt: skip


def split_range(total, cores):
    """Contiguous (start, count) of `total` items per core, as a dict keyed by core.

    The first total % len(cores) cores take one extra item.
    """
    base, extra = divmod(total, len(cores))
    ranges, start = {}, 0
    for i, core in enumerate(cores):
        count = base + (i < extra)
        ranges[core] = (start, count)
        start += count
    return ranges


def largest_subblock(height, width, dst_tiles=4):
    """Largest (h, w) subblock dividing (height, width) with h * w <= dst_tiles."""
    for h, w in _SUBBLOCKS:
        if h * w <= dst_tiles and height % h == 0 and width % w == 0:
            return h, w
    return 1, 1


def granularity(tiles, limit):
    """Largest g <= limit that divides tiles."""
    g = min(tiles, limit)
    while g > 1 and tiles % g:
        g -= 1
    return g


def plan_lanes(grid, panels, groups, group_size):
    """Lanes of r cores sharing one data stream, for `groups` groups of `group_size`.

    A lane is a column segment, or a row segment inside one multicast panel; r divides
    group_size and a lane takes r units of one group per job. Picks the fewest job
    rounds, then the largest r. Returns (r, lanes): lanes of (x, y) cores, sender first.
    """
    gx, gy = grid
    widths = [x1 - x0 + 1 for x0, x1 in panels]
    best = None
    for r in range(1, max(gy, *widths) + 1):
        if group_size % r:
            continue

        # Column segments and panel-local row segments of r cores.
        cols = [
            [(x, s * r + p) for p in range(r)]
            for x in range(gx)
            for s in range(gy // r)
        ]
        rows = [
            [(x0 + s * r + p, y) for p in range(r)]
            for x0, x1 in panels
            for y in range(gy)
            for s in range((x1 - x0 + 1) // r)
        ]
        lanes = max(cols, rows, key=len)

        rounds = -(-groups * group_size // r // len(lanes))
        if best is None or (rounds, -r) <= (best[0], -best[1]):
            best = (rounds, r, lanes)
    return best[1], best[2]
