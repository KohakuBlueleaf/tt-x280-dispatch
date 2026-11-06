"""tt_x280: resident Tensix programs sequenced by the Blackhole X280, on ttnn.

device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))  # slow dispatch: 12x10 grid
plan = tt_x280.Runtime(device, pll_mhz=1750).plan(depth=1)  # 1: dependent ops
program = tt_x280.ResidentProgram(plan)
program.add_cb(...); program.add_kernel(..., dispatch=True)
ticks = program.run(io_tensors, n_cmds)
"""

from _ttx280 import (
    CTL_BYTES,
    CTL_CB,
    IMAGE_DIR,
    KERNEL_DIR,
    MTIME_HZ,
    TRACE_CB,
    Plan,
    Runtime,
    Session,
)

from .program import ResidentProgram
from .trace import Trace

__all__ = [
    "CTL_BYTES",
    "CTL_CB",
    "IMAGE_DIR",
    "KERNEL_DIR",
    "MTIME_HZ",
    "TRACE_CB",
    "Plan",
    "ResidentProgram",
    "Runtime",
    "Session",
    "Trace",
]
