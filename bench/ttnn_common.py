"""Helpers for the ttnn baselines: device, compute config, eager and trace timing.

The baselines run ttnn v0.81.1 in fast dispatch (110 cores) on one card.
"""

import time

import ttnn


def open_device():
    """Card TT_VISIBLE_DEVICES as device 0, with room for traces and conv halos."""
    return ttnn.open_device(
        device_id=0, l1_small_size=32768, trace_region_size=256 << 20
    )


def compute_config(device, fidelity="HiFi2", approx=False):
    """bf16 compute with fp32 destination accumulation at the given fidelity."""
    return ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=getattr(ttnn.MathFidelity, fidelity),
        math_approx_mode=approx,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )


def time_eager(device, forward, iters):
    """µs per call of forward(), enqueued back to back, after 4 warm-up calls.

    forward() returns the tensors it allocated; they are freed after each call.
    """
    for _ in range(4):
        release(forward())
    ttnn.synchronize_device(device)

    start = time.perf_counter()
    for _ in range(iters):
        release(forward())
    ttnn.synchronize_device(device)
    return (time.perf_counter() - start) * 1e6 / iters


def time_trace(device, forward, iters, reps=8):
    """µs per call of forward() replayed from a trace of `reps` calls, or NaN.

    Returns NaN when the ops cannot be captured.
    """
    try:
        trace_id = ttnn.begin_trace_capture(device, cq_id=0)
        outs = [forward() for _ in range(reps)]
        ttnn.end_trace_capture(device, trace_id, cq_id=0)
    except RuntimeError as exc:
        print(f"# trace capture failed: {str(exc).splitlines()[0]}", flush=True)
        return float("nan")
    ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)

    loops = max(1, iters // reps)
    start = time.perf_counter()
    for _ in range(loops):
        ttnn.execute_trace(device, trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(device)
    us = (time.perf_counter() - start) * 1e6 / (loops * reps)

    ttnn.release_trace(device, trace_id)
    for out in outs:
        release(out)
    return us


def release(tensors):
    """Deallocate a tensor or a tuple of tensors."""
    for t in tensors if isinstance(tensors, tuple) else (tensors,):
        ttnn.deallocate(t)
