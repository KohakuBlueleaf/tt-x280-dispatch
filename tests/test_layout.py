"""Work layout: per-core ranges, subblocks and K/V lanes."""

import pytest
from tt_x280.layout import granularity, largest_subblock, plan_lanes, split_range

GRID = (12, 10)
PANELS = [(0, 5), (6, 11)]  # the two physically contiguous panels of our p150a cards


def test_split_range_covers_every_item_once():
    cores = [(x, y) for y in range(10) for x in range(12)]
    ranges = split_range(98, cores)
    covered = [
        i for start, count in ranges.values() for i in range(start, start + count)
    ]
    assert covered == list(range(98))
    assert max(c for _, c in ranges.values()) - min(c for _, c in ranges.values()) <= 1


@pytest.mark.parametrize(
    ("h", "w", "expected"),
    [(4, 4, (2, 2)), (4, 2, (2, 2)), (1, 6, (1, 3)), (3, 5, (3, 1))],
)
def test_largest_subblock(h, w, expected):
    sh, sw = largest_subblock(h, w)
    assert (sh, sw) == expected
    assert h % sh == 0 and w % sw == 0 and sh * sw <= 4


def test_granularity_divides():
    for tiles in range(1, 17):
        g = granularity(tiles, 4)
        assert 1 <= g <= 4 and tiles % g == 0


def lane_ok(lane):
    """A lane is a straight run of adjacent cores inside one panel."""
    xs, ys = {x for x, _ in lane}, {y for _, y in lane}
    if len(xs) == 1:
        return sorted(ys) == list(range(min(ys), min(ys) + len(lane)))
    panel = next(p for p in PANELS if p[0] <= min(xs) <= p[1])
    return len(ys) == 1 and max(xs) <= panel[1]


@pytest.mark.parametrize(
    ("heads", "q_chunks", "lane_size"),
    [(12, 10, 10), (8, 16, 8), (16, 8, 8), (5, 7, 7)],
)
def test_plan_lanes(heads, q_chunks, lane_size):
    r, lanes = plan_lanes(GRID, PANELS, heads, q_chunks)
    assert r == lane_size and q_chunks % r == 0
    assert all(len(lane) == r and lane_ok(lane) for lane in lanes)
    cores = [c for lane in lanes for c in lane]
    assert len(cores) == len(set(cores))


def test_plan_lanes_full_grid_single_round():
    r, lanes = plan_lanes(GRID, PANELS, 12, 10)
    assert len(lanes) * r == 120 and 12 * 10 // r == len(lanes)
