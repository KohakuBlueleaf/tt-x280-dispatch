"""Per-core command traces recorded by the dispatch template (run(..., trace=True))."""

import torch


class Trace:
    """Microseconds since the first command seen, per core c = y * grid_x + x.

    seen[c, i]: command i + 1 seen; done[c, i]: outputs written; acked[c, i]: completion
    sent (for a column leader: after its whole column reported).
    """

    def __init__(self, seen, done, acked, grid, aiclk_mhz, meta=None):
        """A trace from µs tensors [cores, commands], the grid, AICLK and metadata."""
        self.seen, self.done, self.acked = seen, done, acked
        self.grid = tuple(grid)
        self.aiclk_mhz = aiclk_mhz
        self.meta = dict(meta or {})

    @classmethod
    def from_records(cls, records, grid, aiclk_mhz):
        """Build from raw 32-bit wall-clock records [cores, commands, 3]."""
        raw = records.to(torch.int64) & 0xFFFFFFFF

        # Signed 32-bit deltas to core 0's first stamp, then shifted to start at 0.
        delta = (raw - raw[0, 0, 0] + (1 << 31)) % (1 << 32) - (1 << 31)
        delta -= delta[:, :, 0].min()
        us = delta.double() / aiclk_mhz
        return cls(us[:, :, 0], us[:, :, 1], us[:, :, 2], grid, aiclk_mhz)

    def save(self, path, **meta):
        """Write the trace and `meta` (kernel, shape, op names, ...) to a .pt file."""
        self.meta.update(meta)
        state = dict(seen=self.seen, done=self.done, acked=self.acked, grid=self.grid)
        torch.save(dict(state, aiclk_mhz=self.aiclk_mhz, meta=self.meta), path)

    @classmethod
    def load(cls, path):
        """Read a trace written by save."""
        d = torch.load(path)
        return cls(
            d["seen"], d["done"], d["acked"], d["grid"], d["aiclk_mhz"], d["meta"]
        )

    def steps(self):
        """Per command, in µs: fan_out (last seen - first seen), busy (median done -
        seen), slowest (max done - seen), gather (last acked - last done), step (first
        seen of the next command - first seen of this one; NaN for the last)."""
        first_seen = self.seen.min(dim=0).values
        busy = self.done - self.seen
        step = torch.full_like(first_seen, float("nan"))
        step[:-1] = first_seen[1:] - first_seen[:-1]
        return {
            "fan_out": self.seen.max(dim=0).values - first_seen,
            "busy": busy.median(dim=0).values,
            "slowest": busy.max(dim=0).values,
            "gather": self.acked.max(dim=0).values - self.done.max(dim=0).values,
            "step": step,
        }
