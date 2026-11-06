# Writing a resident kernel

A resident op is an ordinary Metalium program (reader / compute / writer per core) whose
data-movement kernels loop over commands instead of running once. The X280 decides when
each command may start; the template handles the protocol.

## Device side

[`src/dispatcher/device/ttx_dispatch.hpp`](../src/dispatcher/device/ttx_dispatch.hpp),
included by relative path from `src/kernels/<name>/`.

The reader waits for each command before it touches the inputs:

```cpp
#include "../../dispatcher/device/ttx_dispatch.hpp"

void kernel_main() {
    uint32_t a = 0;
    // ... this kernel's own runtime args ...
    const uint32_t n_cmds = get_arg_val<uint32_t>(a++);
    const auto disp = ttx::Args::read(a);  // the 16 args the plan appends

    ttx::Worker worker(disp);
    worker.start();  // ring clear, check-in or READY, hello
    for (uint32_t k = 1; k <= n_cmds; k++) {
        if (!worker.wait(k)) {
            return;  // the X280 went away: bounded spin
        }
        const uint32_t op = (k - 1) % kOps;
        // ... read the inputs of `op`, push CBs ...
    }
}
```

The writer finishes last on the core and reports completion:

```cpp
    ttx::Worker worker(disp);  // no start() here
    for (uint32_t k = 1; k <= n_cmds; k++) {
        // ... write the outputs of op(k) ...
        noc_async_write_barrier();  // outputs visible before completion
        if (!worker.complete(k)) {
            return;
        }
    }
    worker.flush_trace(n_cmds);  // no-op unless the run is traced
```

Compute kernels need nothing from the template: they loop over the same op table.

Rules:
- One kernel per core calls `start()` and `wait()`, one calls `complete()`.
- `complete(k)` only after the core's outputs of `k` are written.
- CB 8 is the control block and CB 10 the trace buffer: leave both to the template.

## Host side in Python

[`src/frontend/tt_x280`](../src/frontend/tt_x280) on ttnn objects. The resident program is
launched once with `ttnn.generic_op`, in slow dispatch (12×10 grid) with ttnn's program
cache disabled (the dispatch tag changes per run).

```python
device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
device.disable_and_clear_program_cache()
plan = tt_x280.Runtime(device, pll_mhz=1750).plan(depth=1)

program = tt_x280.ResidentProgram(plan)
program.add_cb(0, ttnn.bfloat16, 2048, 8)
sem = program.add_semaphore()
program.add_kernel(
    f"{tt_x280.KERNEL_DIR}/mine/reader.cpp",
    cores,
    ttnn.DataMovementConfigDescriptor(
        processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.RISCV_1_default
    ),
    lambda x, y: [a.buffer_address(), n_cmds],  # the plan's 16 args are appended
    ttnn.TensorAccessorArgs(a).get_compile_time_args(),
    dispatch=True,
)
# ... writer (dispatch=True) and compute (dispatch=False) the same way ...
ticks = program.run([a, out], n_cmds)  # 50 MHz X280 ticks, command 1 to last done
```

- `plan.panels`: the physically contiguous multicast rectangles; keep your own Tensix
  multicasts inside one panel.
- `plan.translated(x, y)`: NOC0 coordinates of a logical core, for multicast corners.
- `tt_x280.layout`: per-core ranges (`split_range`), lanes that share a data stream
  (`plan_lanes`), subblock and granularity choices.
- `tt_x280.stats.error_stats(result, reference)`: rel-RMS, max-abs, mean bias, PCC.

Host-fed streaming enqueues each command as it becomes ready:

```python
def feed(plan):
    for k in range(1, n_cmds + 1):
        plan.submit(k)  # one write-combining store, 0.63 µs
    plan.wait(n_cmds)  # reads the done count from pinned host memory

program.run(tensors, n_cmds, host_fed=True, submit=feed)
```

Tracing records, per core and command, when the command was seen, when the outputs were
written and when completion was sent (wall-clock cycles, converted with the chip's AICLK):

```python
program.run(tensors, n_cmds, trace=True)
program.trace.save("trace.pt", kernel="mine", ops=["a", "b"])
program.trace.steps()  # fan_out, busy, slowest, gather, step per command
```

## Chaining ops

- Dependent op k + 1 reads op k's outputs: `depth=1`, the X280 issues k + 1 only after
  every column reported k.
- Independent commands (the same op on new data): `depth=4`.
- Tensors are bound per op through runtime args: see
  [`kernels/mlp/reader.cpp`](../src/kernels/mlp/reader.cpp) (GEMM1, SiLU, GEMM2).

## Sharing a data stream

Cores that need the same data per command form a lane: its first core reads from DRAM and
multicasts to the rest, with a ready / valid semaphore pair. Lanes are column segments or
row segments inside one panel, and all cores of a lane must have the same CB history (the
multicast lands at the same L1 address). [`kernels/flash_attn/reader.cpp`](../src/kernels/flash_attn/reader.cpp)
shares K/V chunks this way; `tt_x280.layout.plan_lanes` picks the lanes.

## Reusing ttnn compute

Include it by its tt-metal path and supply the defines and CB layout its program factory
would: [`kernels/flash_attn/compute.cpp`](../src/kernels/flash_attn/compute.cpp) calls
`sdpa_standard` from
`cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/compute_common.hpp`.

## Debugging

1. Run the kernels without the X280 first (watcher cannot run while it is open).
2. A run that never completes raises after `finish`'s timeout; the Tensix kernels keep
   spinning, and the card needs `tt-smi -r <card>` and a PCIe retrain.
3. The C++ bench (`src/hosts/bench.cpp`) shows the post-mortem pattern: on a timeout every
   core writes {stage, expected k, seen value} into the X280's LIM.

## Examples

| Example | Ops | Files |
|---|---|---|
| protocol only | empty commands, every completion model | `kernels/protocol/`, `hosts/bench.cpp` |
| matmul | 2D-multicast GEMM, repeated | `kernels/matmul/`, `hosts/matmul.cpp` |
| MLP forward | GEMM → SiLU → GEMM, dependent | `kernels/mlp/`, `hosts/mlp.cpp`, `examples/mlp.py` |
| flash attention | SDPA per forward, K/V lanes | `kernels/flash_attn/`, `examples/flash_attn.py` |
| ResNet basic block | conv → ReLU → conv + residual → ReLU | `kernels/resblock/`, `examples/resblock.py` |
