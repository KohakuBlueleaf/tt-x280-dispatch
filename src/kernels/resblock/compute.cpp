// Resident ResNet basic-block compute (X280 dispatch v1): per output pixel tile,
// tilizes the 9 gathered taps, runs the implicit GEMM against the resident weights in
// DST, adds the residual (op 1), applies ReLU and untilizes.
#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/relu.h"
#include "api/compute/matmul.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"

// Per output tile: tilize taps, GEMM against resident weights, residual, ReLU,
// untilize.
void kernel_main() {
    constexpr uint32_t Ct = get_compile_time_arg_val(0);
    constexpr uint32_t n_cmds = get_compile_time_arg_val(1);
    const uint32_t m_count = get_arg_val<uint32_t>(0);

    constexpr uint32_t cb_taps = 0, cb_res = 1, cb_a = 2, cb_w1 = 3, cb_w2 = 4,
                       cb_act = 6, cb_res_t = 7;
    constexpr uint32_t cb_out = 16, k_tiles = 9 * Ct;

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_a, cb_w1, cb_act);
    cb_wait_front(cb_w1, k_tiles * Ct);
    cb_wait_front(cb_w2, k_tiles * Ct);

    for (uint32_t it = 1; it <= n_cmds; it++) {
        const bool second = ((it - 1) % 2) == 1;
        const uint32_t cb_w = second ? cb_w2 : cb_w1;
        for (uint32_t m = 0; m < m_count; m++) {
            for (uint32_t tap = 0; tap < 9; tap++) {
                compute_kernel_lib::tilize<Ct, cb_taps, cb_a>(1, 32);
            }
            if (second) {
                compute_kernel_lib::tilize<Ct, cb_res, cb_res_t>(1, 32);
                cb_wait_front(cb_res_t, Ct);
            }
            cb_wait_front(cb_a, k_tiles);
            reconfig_data_format(cb_w, cb_a);
            pack_reconfig_data_format(cb_act);
            matmul_init(cb_a, cb_w);
            tile_regs_acquire();
            for (uint32_t k = 0; k < k_tiles; k++) {
                for (uint32_t co = 0; co < Ct; co++) {
                    matmul_tiles(cb_a, cb_w, k, k * Ct + co, co);
                }
            }
            if (second) {
                add_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_res_t);
                for (uint32_t co = 0; co < Ct; co++) {
                    add_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(
                        cb_res_t, co, co
                    );
                }
            }
            relu_tile_init();
            for (uint32_t co = 0; co < Ct; co++) {
                relu_tile(co);
            }
            tile_regs_commit();
            tile_regs_wait();
            cb_reserve_back(cb_act, Ct);
            for (uint32_t co = 0; co < Ct; co++) {
                pack_tile(co, cb_act);
            }
            cb_push_back(cb_act, Ct);
            tile_regs_release();
            cb_pop_front(cb_a, k_tiles);
            if (second) {
                cb_pop_front(cb_res_t, Ct);
            }
            compute_kernel_lib::untilize<Ct, cb_act, cb_out>(1);
        }
    }
}
