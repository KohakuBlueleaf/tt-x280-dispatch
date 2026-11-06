// ResNet basic-block reader: loads both 3x3 weight matrices once; per command (0: conv1
// on X, 1: conv2 on H + residual X) reads each output tile's input halo once.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "../../dispatcher/device/ttx_dispatch.hpp"

namespace {

// Reads `tiles` consecutive tiles starting at tile 0 into `cb` and pushes them.
template <typename Accessor>
void load_tiles(const Accessor& src, uint32_t cb, uint32_t tiles) {
    const uint32_t tile_bytes = get_tile_size(cb);
    cb_reserve_back(cb, tiles);
    uint32_t l1 = get_write_ptr(cb);
    for (uint32_t t = 0; t < tiles; t++) {
        noc_async_read_page(t, src, l1);
        l1 += tile_bytes;
    }
    noc_async_read_barrier();
    cb_push_back(cb, tiles);
}

// Zeroes `bytes` of L1 at `l1`.
void zero_row(uint32_t l1, uint32_t bytes) {
    volatile tt_l1_ptr uint32_t* z = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1);
    for (uint32_t i = 0; i < bytes / 4; i++) {
        z[i] = 0;
    }
}

}  // namespace

// Loads the weights once; per command reads each tile's halo, then builds its 9 taps.
void kernel_main() {
    constexpr uint32_t N = get_compile_time_arg_val(0);
    constexpr uint32_t H = get_compile_time_arg_val(1);
    constexpr uint32_t W = get_compile_time_arg_val(2);
    constexpr uint32_t Ct = get_compile_time_arg_val(3);
    constexpr auto rm_args = TensorAccessorArgs<4>();
    constexpr auto w_args =
        TensorAccessorArgs<rm_args.next_compile_time_args_offset()>();

    uint32_t a = 0;
    const uint32_t x_addr = get_arg_val<uint32_t>(a++);
    const uint32_t h_addr = get_arg_val<uint32_t>(a++);
    const uint32_t w1_addr = get_arg_val<uint32_t>(a++);
    const uint32_t w2_addr = get_arg_val<uint32_t>(a++);
    const uint32_t m_start = get_arg_val<uint32_t>(a++);
    const uint32_t m_count = get_arg_val<uint32_t>(a++);
    const uint32_t n_cmds = get_arg_val<uint32_t>(a++);
    const auto disp = ttx::Args::read(a);

    constexpr uint32_t cb_taps = 0, cb_res = 1, cb_w1 = 3, cb_w2 = 4, cb_halo = 9;
    constexpr uint32_t row_bytes = Ct * 32 * 2, pixels = N * H * W;
    constexpr uint32_t halo_rows = 32 + 2 * W + 2;
    constexpr int32_t halo_before = W + 1;  // halo row 0 = pixel (tile start - W - 1)
    const auto x = TensorAccessor(rm_args, x_addr, row_bytes);
    const auto h = TensorAccessor(rm_args, h_addr, row_bytes);
    load_tiles(TensorAccessor(w_args, w1_addr), cb_w1, 9 * Ct * Ct);
    load_tiles(TensorAccessor(w_args, w2_addr), cb_w2, 9 * Ct * Ct);

    // The halo CB is scratch: reserved once, never pushed.
    cb_reserve_back(cb_halo, halo_rows);
    const uint32_t halo = get_write_ptr(cb_halo);

    ttx::Worker worker(disp);
    worker.start();
    for (uint32_t it = 1; it <= n_cmds; it++) {
        if (!worker.wait(it)) {
            return;
        }
        const bool second = ((it - 1) % 2) == 1;
        const auto& src = second ? h : x;

        for (uint32_t m = m_start; m < m_start + m_count; m++) {
            // Halo: every input pixel a tap of this tile can touch, one read per row.
            const int32_t p0 = static_cast<int32_t>(m * 32) - halo_before;
            for (uint32_t i = 0; i < halo_rows; i++) {
                const int32_t p = p0 + static_cast<int32_t>(i);
                if (p >= 0 && p < static_cast<int32_t>(pixels)) {
                    noc_async_read_page(p, src, halo + i * row_bytes);
                }
            }
            noc_async_read_barrier();

            // Tap row r is halo row r + shift when its shifted pixel lies in the image;
            // runs of valid rows are one local copy, the other rows are zeroed.
            for (uint32_t tap = 0; tap < 9; tap++) {
                const int32_t dy = static_cast<int32_t>(tap / 3) - 1;
                const int32_t dx = static_cast<int32_t>(tap % 3) - 1;
                const uint32_t shift = halo_before + dy * static_cast<int32_t>(W) + dx;
                cb_reserve_back(cb_taps, 32);
                const uint32_t dst = get_write_ptr(cb_taps);
                uint32_t run_start = 0, run_len = 0;
                for (uint32_t r = 0; r <= 32; r++) {
                    bool valid = false;
                    if (r < 32) {
                        const uint32_t p = m * 32 + r;
                        const int32_t y = static_cast<int32_t>((p / W) % H) + dy;
                        const int32_t xx = static_cast<int32_t>(p % W) + dx;
                        valid = y >= 0 && y < static_cast<int32_t>(H) && xx >= 0 &&
                                xx < static_cast<int32_t>(W);
                    }
                    if (valid) {
                        run_start = run_len == 0 ? r : run_start;
                        run_len++;
                        continue;
                    }
                    if (run_len > 0) {
                        noc_async_read(
                            get_noc_addr(halo + (run_start + shift) * row_bytes),
                            dst + run_start * row_bytes,
                            run_len * row_bytes
                        );
                        run_len = 0;
                    }
                    if (r < 32) {
                        zero_row(dst + r * row_bytes, row_bytes);
                    }
                }
                noc_async_read_barrier();
                cb_push_back(cb_taps, 32);
            }

            // Residual rows of X for conv2.
            if (second) {
                cb_reserve_back(cb_res, 32);
                uint32_t l1 = get_write_ptr(cb_res);
                for (uint32_t r = 0; r < 32; r++) {
                    noc_async_read_page(m * 32 + r, x, l1);
                    l1 += row_bytes;
                }
                noc_async_read_barrier();
                cb_push_back(cb_res, 32);
            }
        }
    }
}
