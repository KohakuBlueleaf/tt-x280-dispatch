// Persistent MLP-forward writer: for command i writes this core's output
// block (cb 16, subblock order) to out_addr[(i - 1) % 3] (H, H2, Y), then reports
// completion through the X280 dispatch model.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "../../dispatcher/device/ttx_dispatch.hpp"

// Per command k: writes this core's block of H, H2 or Y, then completes.
void kernel_main() {
    uint32_t a = 0;
    uint32_t out_addr[3];
    for (uint32_t k = 0; k < 3; k++) {
        out_addr[k] = get_arg_val<uint32_t>(a++);
    }
    const uint32_t out_start = get_arg_val<uint32_t>(a++);
    const uint32_t out_stride_h = get_arg_val<uint32_t>(a++);  // Ft
    const uint32_t sbw = get_arg_val<uint32_t>(a++);
    const uint32_t sbh = get_arg_val<uint32_t>(a++);
    const uint32_t num_sb_w = get_arg_val<uint32_t>(a++);
    const uint32_t num_sb_h = get_arg_val<uint32_t>(a++);
    const uint32_t n_cmds = get_arg_val<uint32_t>(a++);
    const auto disp = ttx::Args::read(a);

    constexpr uint32_t cb_out = 16;
    const uint32_t tile_bytes = get_tile_size(cb_out);
    constexpr auto acc_args = TensorAccessorArgs<0>();
    ttx::Worker worker(disp);

    for (uint32_t it = 1; it <= n_cmds; it++) {
        const auto s = TensorAccessor(acc_args, out_addr[(it - 1) % 3]);
        uint32_t sbh_start = out_start;
        for (uint32_t i = 0; i < num_sb_h; i++) {
            uint32_t sbw_start = sbh_start;
            for (uint32_t j = 0; j < num_sb_w; j++) {
                uint32_t row = sbw_start;
                cb_wait_front(cb_out, sbw * sbh);
                uint32_t l1 = get_read_ptr(cb_out);
                for (uint32_t h = 0; h < sbh; h++) {
                    for (uint32_t w = 0; w < sbw; w++) {
                        noc_async_write_page(row + w, s, l1);
                        l1 += tile_bytes;
                    }
                    row += out_stride_h;
                }
                noc_async_writes_flushed();
                cb_pop_front(cb_out, sbw * sbh);
                sbw_start += sbw;
            }
            sbh_start += sbh * out_stride_h;
        }
        noc_async_write_barrier();
        if (!worker.complete(it)) {
            return;
        }
    }
    worker.flush_trace(n_cmds);
}
