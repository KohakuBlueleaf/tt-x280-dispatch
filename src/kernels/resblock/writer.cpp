// Resident ResNet basic-block writer (X280 dispatch v1): per command writes this core's
// untilized output pixel rows to H (op 0) or Y (op 1), then reports completion.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "../../dispatcher/device/ttx_dispatch.hpp"

// Per command: writes this core's output rows to H or Y, then completes.
void kernel_main() {
    constexpr uint32_t Ct = get_compile_time_arg_val(0);
    constexpr auto rm_args = TensorAccessorArgs<1>();

    uint32_t a = 0;
    const uint32_t h_addr = get_arg_val<uint32_t>(a++);
    const uint32_t y_addr = get_arg_val<uint32_t>(a++);
    const uint32_t m_start = get_arg_val<uint32_t>(a++);
    const uint32_t m_count = get_arg_val<uint32_t>(a++);
    const uint32_t n_cmds = get_arg_val<uint32_t>(a++);
    const auto disp = ttx::Args::read(a);

    constexpr uint32_t cb_out = 16, row_bytes = Ct * 32 * 2;
    const auto h = TensorAccessor(rm_args, h_addr, row_bytes);
    const auto y = TensorAccessor(rm_args, y_addr, row_bytes);

    ttx::Worker worker(disp);
    for (uint32_t it = 1; it <= n_cmds; it++) {
        const auto& dst = ((it - 1) % 2) == 1 ? y : h;
        for (uint32_t m = m_start; m < m_start + m_count; m++) {
            cb_wait_front(cb_out, Ct);
            uint32_t l1 = get_read_ptr(cb_out);
            for (uint32_t r = 0; r < 32; r++) {
                noc_async_write_page(m * 32 + r, dst, l1);
                l1 += row_bytes;
            }
            noc_async_writes_flushed();
            cb_pop_front(cb_out, Ct);
        }
        noc_async_write_barrier();
        if (!worker.complete(it)) {
            return;
        }
    }
    worker.flush_trace(n_cmds);
}
