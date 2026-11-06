"""ResidentProgram: a ttnn program whose kernels the X280 drives command by command."""

import torch
import ttnn
from _ttx280 import CTL_BYTES, CTL_CB, TRACE_CB

from tt_x280.trace import Trace


class ResidentProgram:
    """CBs, semaphores and kernels for every core of a dispatch plan.

    Launched once per run with ttnn.generic_op. Runtime args come from callables
    (x, y) -> list[int], rebuilt on every run; kernels added with dispatch=True get the
    plan's 16 dispatch args appended to theirs.
    """

    def __init__(self, plan):
        """An empty program on plan's grid, holding the control CB on every core."""
        self.plan = plan
        self.grid = (plan.grid_x, plan.grid_y)
        self.cbs = []
        self.semaphores = []
        self.kernels = []
        self.trace = None
        self.add_cb(CTL_CB, ttnn.uint32, CTL_BYTES, 1)

    def all_cores(self):
        """The whole grid as a CoreRangeSet."""
        gx, gy = self.grid
        return ttnn.CoreRangeSet(
            [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, gy - 1))]
        )

    def add_cb(self, index, dtype, page_size, num_pages, shared_indices=(), cores=None):
        """Add a circular buffer of num_pages pages on `cores` (default: every core).

        shared_indices are extra CB indices that alias the same memory.
        """
        formats = [
            ttnn.CBFormatDescriptor(
                buffer_index=i, data_format=dtype, page_size=page_size
            )
            for i in (index, *shared_indices)
        ]
        self.cbs.append(
            ttnn.CBDescriptor(
                total_size=page_size * num_pages,
                core_ranges=cores or self.all_cores(),
                format_descriptors=formats,
            )
        )

    def add_semaphore(self, initial_value=0):
        """Add a semaphore on every core and return its id."""
        self.semaphores.append(
            ttnn.SemaphoreDescriptor(
                id=len(self.semaphores),
                core_type=ttnn.CoreType.WORKER,
                core_ranges=self.all_cores(),
                initial_value=initial_value,
            )
        )
        return len(self.semaphores) - 1

    def add_kernel(
        self, source, cores, config, args, compile_args=(), defines=(), dispatch=False
    ):
        """Add a kernel from file `source` on `cores`, a list of (x, y).

        args(x, y) gives a core's runtime args; dispatch=True appends the plan's
        dispatch args (for the reader and the writer).
        """
        self.kernels.append(
            dict(
                source=source,
                cores=list(cores),
                config=config,
                args=args,
                compile_args=list(compile_args),
                defines=list(defines),
                dispatch=dispatch,
            )
        )

    def _descriptor(self):
        """The ttnn.ProgramDescriptor, with runtime args built for the current run."""
        kernels = []
        for k in self.kernels:
            runtime_args = ttnn.RuntimeArgs()
            for x, y in k["cores"]:
                values = list(k["args"](x, y))
                if k["dispatch"]:
                    values += list(self.plan.worker_args(x, y))
                runtime_args[x][y] = values

            ranges = ttnn.CoreRangeSet(
                [
                    ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y))
                    for x, y in k["cores"]
                ]
            )
            kernels.append(
                ttnn.KernelDescriptor(
                    kernel_source=k["source"],
                    source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                    core_ranges=ranges,
                    compile_time_args=k["compile_args"],
                    defines=k["defines"],
                    runtime_args=runtime_args,
                    config=k["config"],
                )
            )
        return ttnn.ProgramDescriptor(
            kernels=kernels, semaphores=self.semaphores, cbs=self.cbs
        )

    def run(self, io_tensors, n_cmds, host_fed=False, submit=None, trace=False):
        """Run n_cmds commands and return the X280's ticks (50 MHz), first to last.

        Starts the X280 sequencer, launches the program on io_tensors and waits. With
        host_fed, submit(plan) must enqueue the commands with plan.submit. With trace,
        self.trace holds every core's timestamps afterwards. Raises if the run fails.
        """
        io_tensors = list(io_tensors)
        n_cbs = len(self.cbs)
        gx, gy = self.grid

        # Trace: a DRAM page per core plus a CB that stages it in L1.
        trace_tensor = None
        if trace:
            trace_tensor = ttnn.from_torch(
                torch.zeros(1, 1, gx * gy, n_cmds * 4, dtype=torch.int32),
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=io_tensors[0].device(),
            )
            self.add_cb(TRACE_CB, ttnn.uint32, n_cmds * 16, 1)
            io_tensors.append(trace_tensor)

        # The sequencer starts first: it picks the tag the runtime args carry.
        session = self.plan.start(n_cmds, host_fed)
        self.plan.trace_addr = trace_tensor.buffer_address() if trace else 0
        try:
            ttnn.generic_op(io_tensors, self._descriptor())
            if host_fed and submit is not None:
                submit(self.plan)
            ticks = self.plan.finish(session)
        finally:
            self.plan.trace_addr = 0
            del self.cbs[n_cbs:]

        if ticks == 0:
            raise RuntimeError("X280 dispatch run failed or timed out")
        if trace:
            records = ttnn.to_torch(trace_tensor).reshape(gx * gy, n_cmds, 4)
            self.trace = Trace.from_records(
                records[:, :, :3], self.grid, self.plan.aiclk_mhz()
            )
        return ticks
