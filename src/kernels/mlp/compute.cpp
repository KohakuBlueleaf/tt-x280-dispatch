// Persistent MLP-forward compute: for command i, op = (i - 1) % 3:
//   0 / 2: block matmul (cb 0 x cb 1 -> cb 16, partials via cb 24), K blocks =
//   num_blocks1 / num_blocks2; 1: SiLU over silu_tiles tiles (cb 2 -> cb 16).
// Matmul loop derived from tt-metal bmm_large_block_zm.cpp.
#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/tile_move_copy.h"

namespace {
// Blocked matmul cb 0 x cb 1 -> cb 16, partials through cb 24.
void matmul_op(
    uint32_t in0_block_w,
    uint32_t in0_num_subblocks,
    uint32_t in0_block_num_tiles,
    uint32_t in0_subblock_num_tiles,
    uint32_t in1_num_subblocks,
    uint32_t in1_block_num_tiles,
    uint32_t in1_per_core_w,
    uint32_t num_blocks,
    uint32_t out_subblock_h,
    uint32_t out_subblock_w,
    uint32_t out_subblock_num_tiles
) {
    matmul_init(tt::CBIndex::c_0, tt::CBIndex::c_1);
    bool spill = num_blocks > 1;
    bool enable_reload = false;
    uint32_t out_num_tiles_to_wait = out_subblock_num_tiles;
    for (uint32_t block = 0; block < num_blocks; block++) {
        const bool last_out = block == (num_blocks - 1);
        cb_wait_front(tt::CBIndex::c_0, in0_block_num_tiles);
        cb_wait_front(tt::CBIndex::c_1, in1_block_num_tiles);
        int in0_index_subblock_offset = 0;
        for (uint32_t in0_subblock = 0; in0_subblock < in0_num_subblocks;
             in0_subblock++) {
            int in1_index_subblock_offset = 0;
            for (uint32_t in1_subblock = 0; in1_subblock < in1_num_subblocks;
                 in1_subblock++) {
                tile_regs_acquire();
                if (enable_reload) {
                    copy_init(tt::CBIndex::c_24);
                    cb_wait_front(tt::CBIndex::c_24, out_subblock_num_tiles);
                    for (uint32_t i = 0; i < out_subblock_num_tiles; i++) {
                        copy_tile(tt::CBIndex::c_24, i, i);
                    }
                    cb_pop_front(tt::CBIndex::c_24, out_subblock_num_tiles);
                    matmul_init(tt::CBIndex::c_0, tt::CBIndex::c_1);
                }
                int dst_index = 0;
                int in0_index_h_offset = 0;
                for (uint32_t h = 0; h < out_subblock_h; h++) {
                    for (uint32_t w = 0; w < out_subblock_w; w++) {
                        int in1_index_inner_dim_offset = 0;
                        for (uint32_t inner_dim = 0; inner_dim < in0_block_w;
                             inner_dim++) {
                            const int in0_index = in0_index_subblock_offset +
                                                  in0_index_h_offset + inner_dim;
                            const int in1_index = in1_index_subblock_offset +
                                                  in1_index_inner_dim_offset + w;
                            matmul_tiles(
                                tt::CBIndex::c_0,
                                tt::CBIndex::c_1,
                                in0_index,
                                in1_index,
                                dst_index
                            );
                            in1_index_inner_dim_offset += in1_per_core_w;
                        }
                        dst_index++;
                    }
                    in0_index_h_offset += in0_block_w;
                }
                tile_regs_commit();
                tile_regs_wait();
                if (last_out) {
                    cb_reserve_back(tt::CBIndex::c_16, out_subblock_num_tiles);
                    for (uint32_t i = 0; i < out_subblock_num_tiles; i++) {
                        pack_tile(i, tt::CBIndex::c_16);
                    }
                    cb_push_back(tt::CBIndex::c_16, out_subblock_num_tiles);
                } else {
                    if (block == 0) {
                        cb_reserve_back(tt::CBIndex::c_16, out_num_tiles_to_wait);
                        out_num_tiles_to_wait += out_subblock_num_tiles;
                    }
                    cb_reserve_back(tt::CBIndex::c_24, out_subblock_num_tiles);
                    for (uint32_t i = 0; i < out_subblock_num_tiles; i++) {
                        pack_tile(i, tt::CBIndex::c_24);
                    }
                    cb_push_back(tt::CBIndex::c_24, out_subblock_num_tiles);
                }
                tile_regs_release();
                in1_index_subblock_offset += out_subblock_w;
            }
            in0_index_subblock_offset += in0_subblock_num_tiles;
        }
        if (spill) {
            enable_reload = true;
        }
        cb_pop_front(tt::CBIndex::c_0, in0_block_num_tiles);
        cb_pop_front(tt::CBIndex::c_1, in1_block_num_tiles);
    }
}

// SiLU over `tiles` tiles, cb 2 -> cb 16.
void silu_op(uint32_t tiles) {
    copy_init(tt::CBIndex::c_2);
    silu_tile_init();
    for (uint32_t t = 0; t < tiles; t++) {
        cb_wait_front(tt::CBIndex::c_2, 1);
        tile_regs_acquire();
        copy_tile(tt::CBIndex::c_2, 0, 0);
        silu_tile(0);
        tile_regs_commit();
        tile_regs_wait();
        cb_reserve_back(tt::CBIndex::c_16, 1);
        pack_tile(0, tt::CBIndex::c_16);
        cb_push_back(tt::CBIndex::c_16, 1);
        tile_regs_release();
        cb_pop_front(tt::CBIndex::c_2, 1);
    }
}
}  // namespace

// Per command k: GEMM1, SiLU or GEMM2 by (k - 1) % 3.
void kernel_main() {
    constexpr uint32_t in0_block_w = get_compile_time_arg_val(0);
    constexpr uint32_t in0_num_subblocks = get_compile_time_arg_val(1);
    constexpr uint32_t in0_block_num_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t in0_subblock_num_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t in1_num_subblocks = get_compile_time_arg_val(4);
    constexpr uint32_t in1_block_num_tiles = get_compile_time_arg_val(5);
    constexpr uint32_t in1_per_core_w = get_compile_time_arg_val(6);
    constexpr uint32_t num_blocks1 = get_compile_time_arg_val(7);
    constexpr uint32_t num_blocks2 = get_compile_time_arg_val(8);
    constexpr uint32_t out_subblock_h = get_compile_time_arg_val(9);
    constexpr uint32_t out_subblock_w = get_compile_time_arg_val(10);
    constexpr uint32_t out_subblock_num_tiles = get_compile_time_arg_val(11);
    constexpr uint32_t n_cmds = get_compile_time_arg_val(12);
    constexpr uint32_t silu_tiles = get_compile_time_arg_val(13);

    compute_kernel_hw_startup<SrcOrder::Reverse>(
        tt::CBIndex::c_0, tt::CBIndex::c_1, tt::CBIndex::c_16
    );
    for (uint32_t i = 1; i <= n_cmds; i++) {
        const uint32_t op = (i - 1) % 3;
        if (op == 1) {
            silu_op(silu_tiles);
        } else {
            matmul_op(
                in0_block_w,
                in0_num_subblocks,
                in0_block_num_tiles,
                in0_subblock_num_tiles,
                in1_num_subblocks,
                in1_block_num_tiles,
                in1_per_core_w,
                op == 0 ? num_blocks1 : num_blocks2,
                out_subblock_h,
                out_subblock_w,
                out_subblock_num_tiles
            );
        }
    }
}
