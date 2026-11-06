// Resident flash-attention writer (X280 dispatch v1). Generates the reduce-scaler and
// column-identity tiles once (left fronted for the whole run), then per command writes
// this core's output chunks and reports completion.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "../../dispatcher/device/ttx_dispatch.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "ttnn/kernel/dataflow/generate_bcast_scalar.hpp"

// Writes the scaler tiles once; per command writes the output chunks, then completes.
void kernel_main() {
    constexpr uint32_t DHt = get_compile_time_arg_val(0);
    constexpr uint32_t Sqt = get_compile_time_arg_val(1);
    constexpr uint32_t Sq_chunk_t = get_compile_time_arg_val(2);
    constexpr uint32_t identity_packed = get_compile_time_arg_val(3);
    constexpr auto acc_args = TensorAccessorArgs<4>();

    uint32_t a = 0;
    const uint32_t o_addr = get_arg_val<uint32_t>(a++);
    const uint32_t lane = get_arg_val<uint32_t>(a++);
    const uint32_t pos = get_arg_val<uint32_t>(a++);
    const uint32_t num_lanes = get_arg_val<uint32_t>(a++);
    const uint32_t num_jobs = get_arg_val<uint32_t>(a++);
    const uint32_t lane_size = get_arg_val<uint32_t>(a++);
    const uint32_t n_cmds = get_arg_val<uint32_t>(a++);
    const auto disp = ttx::Args::read(a);

    constexpr uint32_t cb_identity_scale = 4, cb_col_identity = 5, cb_out = 16;
    constexpr uint32_t q_chunks = Sqt / Sq_chunk_t, chunk_tiles = Sq_chunk_t * DHt;
    const uint32_t tile_bytes = get_tile_size(cb_out);
    const auto o = TensorAccessor(acc_args, o_addr);

    dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<
        cb_identity_scale,
        ckernel::PoolType::MAX,
        ckernel::ReduceDim::REDUCE_ROW,
        dataflow_kernel_lib::SUM_AND_MAX_REDUCE_FACTOR>();
    generate_bcast_col_scalar(CircularBuffer(cb_col_identity), identity_packed);

    ttx::Worker worker(disp);
    for (uint32_t it = 1; it <= n_cmds; it++) {
        for (uint32_t j = lane; j < num_jobs; j += num_lanes) {
            const uint32_t jobs_per_head = q_chunks / lane_size;
            const uint32_t head = j / jobs_per_head,
                           q_chunk = (j % jobs_per_head) * lane_size + pos;
            const uint32_t first = (head * Sqt + q_chunk * Sq_chunk_t) * DHt;
            cb_wait_front(cb_out, chunk_tiles);
            uint32_t l1 = get_read_ptr(cb_out);
            for (uint32_t t = 0; t < chunk_tiles; t++) {
                noc_async_write_page(first + t, o, l1);
                l1 += tile_bytes;
            }
            noc_async_writes_flushed();
            cb_pop_front(cb_out, chunk_tiles);
        }
        noc_async_write_barrier();
        if (!worker.complete(it)) {
            return;
        }
    }
    worker.flush_trace(n_cmds);
}
