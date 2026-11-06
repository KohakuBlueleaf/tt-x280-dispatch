// Resident flash-attention compute (X280 dispatch v1): per command, ttnn's
// sdpa_standard (non-causal, unmasked) over this core's (head, q_chunk) units. Requires
// fp32_dest_acc_en and the SDPA granularity / EXP_APPROX_MODE defines.
#include <cstdint>

#define REDUCE_OP (PoolType::MAX)
#define REDUCE_DIM (ReduceDim::REDUCE_ROW)

#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/compute_common.hpp"

// Per command: sdpa_standard over this core's (head, q_chunk) units.
void kernel_main() {
    constexpr uint32_t Skt = get_compile_time_arg_val(0);
    constexpr uint32_t DHt = get_compile_time_arg_val(1);
    constexpr uint32_t Sq_chunk_t = get_compile_time_arg_val(2);
    constexpr uint32_t q_num_chunks = get_compile_time_arg_val(3);
    constexpr uint32_t Sk_chunk_t = get_compile_time_arg_val(4);
    constexpr uint32_t k_num_chunks = get_compile_time_arg_val(5);
    constexpr uint32_t qk_subblock_w = get_compile_time_arg_val(6);
    constexpr uint32_t qk_subblock_h = get_compile_time_arg_val(7);
    constexpr uint32_t out_subblock_w = get_compile_time_arg_val(8);
    constexpr uint32_t out_subblock_h = get_compile_time_arg_val(9);
    constexpr uint32_t scale_fp32 = get_compile_time_arg_val(10);
    constexpr uint32_t n_cmds = get_compile_time_arg_val(11);

    const uint32_t unit_start = get_arg_val<uint32_t>(0);
    const uint32_t unit_count = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_q = 0, cb_k = 1, cb_v = 2, cb_mask = 3, cb_identity_scale = 4,
                       cb_col_identity = 5;
    constexpr uint32_t cb_sink = 6, cb_out = 16, cb_qk_im = 24, cb_out_im_a = 25,
                       cb_out_im_b = 26;
    constexpr uint32_t cb_max_a = 27, cb_max_b = 28, cb_sum_a = 29, cb_sum_b = 30,
                       cb_exp_max_diff = 31;

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_q, cb_k, cb_out);
    matmul_init(cb_q, cb_k);

    for (uint32_t it = 1; it <= n_cmds; it++) {
        sdpa_standard<
            cb_qk_im,
            cb_identity_scale,
            cb_sink,
            Sq_chunk_t,
            Sk_chunk_t,
            DHt,
            DHt,
            /*use_attention_sink=*/false,
            /*is_causal=*/false,
            /*use_provided_mask=*/false,
            /*use_padded_mask=*/false,
            /*is_chunked=*/false,
            scale_fp32,
            /*sliding_window_size=*/0>(
            Skt,
            /*qk_in0_block_w=*/DHt,
            qk_subblock_w,
            qk_subblock_h,
            Sq_chunk_t / qk_subblock_h,
            Sk_chunk_t / qk_subblock_w,
            /*out_in0_block_w=*/Sk_chunk_t,
            out_subblock_w,
            out_subblock_h,
            Sq_chunk_t / out_subblock_h,
            DHt / out_subblock_w,
            /*iter_q_start=*/0,
            /*iter_q_end=*/unit_count,
            q_num_chunks,
            /*local_q_start=*/unit_start,
            /*chunked_q_chunk_offset=*/0,
            k_num_chunks,
            Sq_chunk_t * DHt,
            Sk_chunk_t * DHt,
            Sk_chunk_t * DHt,
            Sq_chunk_t * Sk_chunk_t,
            Sq_chunk_t * DHt,
            cb_q,
            cb_k,
            cb_v,
            cb_mask,
            cb_col_identity,
            cb_out_im_a,
            cb_out_im_b,
            cb_max_a,
            cb_max_b,
            cb_sum_a,
            cb_sum_b,
            cb_exp_max_diff,
            cb_out
        );
    }
}
