"""Trace: wall-clock records to microseconds and the per-command step breakdown."""

import torch
from tt_x280.trace import Trace


def synthetic_records(cores=4, commands=3, aiclk_mhz=1000, start=0xFFFF_FF00):
    """Records that wrap 32 bits mid-run: seen, done (+5 µs), acked (+0.5 µs)."""
    rows = []
    for c in range(cores):
        for i in range(commands):
            seen = start + (i * 10_000 + c * 100) * aiclk_mhz // 1000
            stamps = [seen, seen + 5 * aiclk_mhz, seen + 5 * aiclk_mhz + aiclk_mhz // 2]
            rows.append([s % (1 << 32) for s in stamps])

    # Stored as the device stores them: 32-bit, read back as int32.
    records = torch.tensor(rows, dtype=torch.int64).reshape(cores, commands, 3)
    return torch.where(records >= 1 << 31, records - (1 << 32), records).to(torch.int32)


def test_from_records_unwraps_and_scales():
    trace = Trace.from_records(synthetic_records(), (2, 2), 1000)
    assert torch.isclose(trace.seen.min(), torch.tensor(0.0, dtype=torch.float64))
    assert torch.allclose(trace.done - trace.seen, torch.full_like(trace.seen, 5.0))
    assert torch.allclose(
        trace.seen[0], torch.tensor([0.0, 10.0, 20.0], dtype=torch.float64)
    )


def test_steps():
    steps = Trace.from_records(synthetic_records(), (2, 2), 1000).steps()
    assert torch.allclose(steps["fan_out"], torch.full((3,), 0.3, dtype=torch.float64))
    assert torch.allclose(steps["busy"], torch.full((3,), 5.0, dtype=torch.float64))
    assert torch.allclose(steps["gather"], torch.full((3,), 0.5, dtype=torch.float64))
    assert torch.allclose(
        steps["step"][:2], torch.full((2,), 10.0, dtype=torch.float64)
    )
    assert torch.isnan(steps["step"][2])


def test_save_load_roundtrip(tmp_path):
    trace = Trace.from_records(synthetic_records(), (2, 2), 1000)
    trace.save(tmp_path / "t.pt", kernel="x", ops=["a"])
    loaded = Trace.load(tmp_path / "t.pt")
    assert loaded.meta == {"kernel": "x", "ops": ["a"]} and loaded.grid == (2, 2)
    assert torch.equal(loaded.acked, trace.acked)
