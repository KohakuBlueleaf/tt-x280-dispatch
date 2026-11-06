// Resident flash-attention reader (X280 dispatch v1). Cores form lanes of r cores
// working on one head at a time; per command the lane sender reads each K
// (tile-transposed) / V chunk from DRAM once and multicasts it to the lane.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "hostdevcommon/common_values.hpp"
#include "../../dispatcher/device/ttx_dispatch.hpp"

namespace {
// Issues reads of a rows x cols tile block starting at tile `first` (row stride `cols`)
// to L1 `base`. transpose=true stores tile (r, c) at slot c * rows + r.
template <typename Accessor>
void read_block(
    const Accessor& src,
    uint32_t base,
    uint32_t first,
    uint32_t rows,
    uint32_t cols,
    uint32_t tile_bytes,
    bool transpose
) {
    for (uint32_t r = 0; r < rows; r++) {
        for (uint32_t c = 0; c < cols; c++) {
            const uint32_t slot = transpose ? c * rows + r : r * cols + c;
            noc_async_read_page(first + r * cols + c, src, base + slot * tile_bytes);
        }
    }
}
}  // namespace

// Per command: Q per unit; K/V chunks read by the lane sender, multicast to the lane.
void kernel_main() {
    constexpr uint32_t DHt = get_compile_time_arg_val(0);
    constexpr uint32_t Sqt = get_compile_time_arg_val(1);
    constexpr uint32_t Skt = get_compile_time_arg_val(2);
    constexpr uint32_t Sq_chunk_t = get_compile_time_arg_val(3);
    constexpr uint32_t Sk_chunk_t = get_compile_time_arg_val(4);
    constexpr auto acc_args = TensorAccessorArgs<5>();

    uint32_t a = 0;
    const uint32_t q_addr = get_arg_val<uint32_t>(a++);
    const uint32_t k_addr = get_arg_val<uint32_t>(a++);
    const uint32_t v_addr = get_arg_val<uint32_t>(a++);
    // Lane schedule: job j = round * num_lanes + lane; head = j / jobs_per_head;
    // q_chunk = (j % jobs_per_head) * r + pos.
    const uint32_t lane = get_arg_val<uint32_t>(a++);
    const uint32_t pos = get_arg_val<uint32_t>(a++);
    const uint32_t num_lanes = get_arg_val<uint32_t>(a++);
    const uint32_t num_jobs = get_arg_val<uint32_t>(a++);
    const uint32_t lane_size = get_arg_val<uint32_t>(a++);
    // Multicast: receivers rectangle (translated NOC0, lo <= hi), receiver count,
    // sender core, semaphore ids.
    const uint32_t mc_lo_x = get_arg_val<uint32_t>(a++);
    const uint32_t mc_lo_y = get_arg_val<uint32_t>(a++);
    const uint32_t mc_hi_x = get_arg_val<uint32_t>(a++);
    const uint32_t mc_hi_y = get_arg_val<uint32_t>(a++);
    const uint32_t mc_num = get_arg_val<uint32_t>(a++);
    const uint32_t sender_x = get_arg_val<uint32_t>(a++);
    const uint32_t sender_y = get_arg_val<uint32_t>(a++);
    const uint32_t sem_ready = get_semaphore(get_arg_val<uint32_t>(a++));
    const uint32_t sem_valid = get_semaphore(get_arg_val<uint32_t>(a++));
    const uint32_t n_cmds = get_arg_val<uint32_t>(a++);
    const auto disp = ttx::Args::read(a);

    constexpr uint32_t cb_q = 0, cb_k = 1, cb_v = 2;
    constexpr uint32_t q_chunks = Sqt / Sq_chunk_t, k_chunks = Skt / Sk_chunk_t;
    constexpr uint32_t q_tiles = Sq_chunk_t * DHt, kv_tiles = Sk_chunk_t * DHt;
    const uint32_t tile_bytes = get_tile_size(cb_q);
    const uint32_t kv_bytes = kv_tiles * tile_bytes;
    const uint32_t jobs_per_head = q_chunks / lane_size;
    const bool sender = pos == 0;
    const auto q = TensorAccessor(acc_args, q_addr);
    const auto k = TensorAccessor(acc_args, k_addr);
    const auto v = TensorAccessor(acc_args, v_addr);

    volatile tt_l1_ptr uint32_t* ready =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_ready);
    volatile tt_l1_ptr uint32_t* valid =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_valid);
    if (sender) {
        *valid = VALID;  // source value for the release multicast
    }
    const uint64_t ready_noc = get_noc_addr(sender_x, sender_y, sem_ready);

    ttx::Worker worker(disp);
    worker.start();
    for (uint32_t it = 1; it <= n_cmds; it++) {
        if (!worker.wait(it)) {
            return;
        }
        for (uint32_t j = lane; j < num_jobs; j += num_lanes) {
            const uint32_t head = j / jobs_per_head;
            const uint32_t q_chunk = (j % jobs_per_head) * lane_size + pos;

            cb_reserve_back(cb_q, q_tiles);
            read_block(
                q,
                get_write_ptr(cb_q),
                (head * Sqt + q_chunk * Sq_chunk_t) * DHt,
                Sq_chunk_t,
                DHt,
                tile_bytes,
                false
            );
            noc_async_read_barrier();
            cb_push_back(cb_q, q_tiles);

            for (uint32_t kc = 0; kc < k_chunks; kc++) {
                cb_reserve_back(cb_k, kv_tiles);
                cb_reserve_back(cb_v, kv_tiles);
                const uint32_t k_l1 = get_write_ptr(cb_k), v_l1 = get_write_ptr(cb_v);
                if (sender) {
                    const uint32_t first = (head * Skt + kc * Sk_chunk_t) * DHt;
                    read_block(k, k_l1, first, Sk_chunk_t, DHt, tile_bytes, true);
                    read_block(v, v_l1, first, Sk_chunk_t, DHt, tile_bytes, false);
                    noc_async_read_barrier();
                    if (mc_num > 0) {
                        noc_semaphore_wait(ready, mc_num);
                        noc_semaphore_set(ready, 0);
                        noc_async_write_multicast(
                            k_l1,
                            get_noc_multicast_addr(
                                mc_lo_x, mc_lo_y, mc_hi_x, mc_hi_y, k_l1
                            ),
                            kv_bytes,
                            mc_num
                        );
                        noc_async_write_multicast(
                            v_l1,
                            get_noc_multicast_addr(
                                mc_lo_x, mc_lo_y, mc_hi_x, mc_hi_y, v_l1
                            ),
                            kv_bytes,
                            mc_num
                        );
                        noc_async_writes_flushed();
                        noc_semaphore_set_multicast(
                            sem_valid,
                            get_noc_multicast_addr(
                                mc_lo_x, mc_lo_y, mc_hi_x, mc_hi_y, sem_valid
                            ),
                            mc_num
                        );
                        noc_async_writes_flushed();
                    }
                } else {
                    noc_semaphore_set(valid, INVALID);
                    noc_semaphore_inc(ready_noc, 1);
                    noc_semaphore_wait(valid, VALID);
                }
                cb_push_back(cb_k, kv_tiles);
                cb_push_back(cb_v, kv_tiles);
            }
        }
    }
}
