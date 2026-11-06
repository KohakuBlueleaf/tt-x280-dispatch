// Persistent MLP-forward reader. Fixed op table, one X280 command per
// op, op = (i - 1) % 3:
//   0: GEMM1 H = X * W1 (2D multicast), 1: SiLU input = this core's H block, 2: GEMM2 Y
//   = H2 * W2 (2D multicast).
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "hostdevcommon/common_values.hpp"
#include "../../dispatcher/device/ttx_dispatch.hpp"

namespace {
struct GemmArgs {
    uint32_t in0_addr, in0_start, in0_stride_h, in1_addr, in1_start, in1_stride_h,
        in1_next_block, num_blocks;
};
}  // namespace

// Per command k: the inputs of op (k - 1) % 3, multicast along rows and columns.
void kernel_main() {
    uint32_t a = 0;
    GemmArgs g[2];
    for (uint32_t k = 0; k < 2; k++) {
        g[k].in0_addr = get_arg_val<uint32_t>(a++);
        g[k].in0_start = get_arg_val<uint32_t>(a++);
        g[k].in0_stride_h = get_arg_val<uint32_t>(a++);
        g[k].in1_addr = get_arg_val<uint32_t>(a++);
        g[k].in1_start = get_arg_val<uint32_t>(a++);
        g[k].in1_stride_h = get_arg_val<uint32_t>(a++);
        g[k].in1_next_block = get_arg_val<uint32_t>(a++);
        g[k].num_blocks = get_arg_val<uint32_t>(a++);
    }
    const uint32_t bw = get_arg_val<uint32_t>(a++);
    const uint32_t pcm = get_arg_val<uint32_t>(a++);
    const uint32_t pcn = get_arg_val<uint32_t>(a++);
    // SiLU input block (subblock order, matching the writer): H address, block origin
    // tile, row stride (Ft), subblocks.
    const uint32_t h_addr = get_arg_val<uint32_t>(a++);
    const uint32_t silu_start = get_arg_val<uint32_t>(a++);
    const uint32_t silu_stride_h = get_arg_val<uint32_t>(a++);
    const uint32_t sbh = get_arg_val<uint32_t>(a++);
    const uint32_t sbw = get_arg_val<uint32_t>(a++);
    // in0 multicast (along the panel row)
    const uint32_t in0_mc_start_x = get_arg_val<uint32_t>(a++);
    const uint32_t in0_mc_start_y = get_arg_val<uint32_t>(a++);
    const uint32_t in0_mc_end_x = get_arg_val<uint32_t>(a++);
    const uint32_t in0_mc_end_y = get_arg_val<uint32_t>(a++);
    const uint32_t in0_mc_num = get_arg_val<uint32_t>(a++);
    const uint32_t in0_sender_x = get_arg_val<uint32_t>(a++);
    const uint32_t in0_sender_y = get_arg_val<uint32_t>(a++);
    const uint32_t in0_sem_s = get_semaphore(get_arg_val<uint32_t>(a++));
    const uint32_t in0_sem_r = get_semaphore(get_arg_val<uint32_t>(a++));
    // in1 multicast (down the column)
    const uint32_t in1_mc_start_x = get_arg_val<uint32_t>(a++);
    const uint32_t in1_mc_start_y = get_arg_val<uint32_t>(a++);
    const uint32_t in1_mc_end_x = get_arg_val<uint32_t>(a++);
    const uint32_t in1_mc_end_y = get_arg_val<uint32_t>(a++);
    const uint32_t in1_mc_num = get_arg_val<uint32_t>(a++);
    const uint32_t in1_sender_x = get_arg_val<uint32_t>(a++);
    const uint32_t in1_sender_y = get_arg_val<uint32_t>(a++);
    const uint32_t in1_sem_s = get_semaphore(get_arg_val<uint32_t>(a++));
    const uint32_t in1_sem_r = get_semaphore(get_arg_val<uint32_t>(a++));
    const uint32_t n_cmds = get_arg_val<uint32_t>(a++);
    const auto disp = ttx::Args::read(a);

    constexpr uint32_t cb_in0 = 0, cb_in1 = 1, cb_silu = 2;
    const uint32_t tile_bytes = get_tile_size(cb_in0);
    const uint32_t in0_block_tiles = pcm * bw, in1_block_tiles = pcn * bw;
    constexpr auto acc_args = TensorAccessorArgs<0>();

    volatile tt_l1_ptr uint32_t* in0_r =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(in0_sem_r);
    volatile tt_l1_ptr uint32_t* in1_r =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(in1_sem_r);
    volatile tt_l1_ptr uint32_t* in0_s =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(in0_sem_s);
    volatile tt_l1_ptr uint32_t* in1_s =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(in1_sem_s);
#ifdef IN0_SENDER
    *in0_r = VALID;
#endif
#ifdef IN1_SENDER
    *in1_r = VALID;
#endif

    ttx::Worker worker(disp);
    worker.start();

    for (uint32_t it = 1; it <= n_cmds; it++) {
        if (!worker.wait(it)) {
            return;
        }
        const uint32_t op = (it - 1) % 3;

        if (op ==
            1) {  // SiLU: stream this core's H block in subblock order into cb_silu
            const auto h = TensorAccessor(acc_args, h_addr);
            uint32_t sbh_start = silu_start;
            for (uint32_t sh = 0; sh < pcm / sbh; sh++) {
                uint32_t sbw_start = sbh_start;
                for (uint32_t sw = 0; sw < pcn / sbw; sw++) {
                    uint32_t row = sbw_start;
                    for (uint32_t hh = 0; hh < sbh; hh++) {
                        for (uint32_t ww = 0; ww < sbw; ww++) {
                            cb_reserve_back(cb_silu, 1);
                            noc_async_read_page(row + ww, h, get_write_ptr(cb_silu));
                            noc_async_read_barrier();
                            cb_push_back(cb_silu, 1);
                        }
                        row += silu_stride_h;
                    }
                    sbw_start += sbw;
                }
                sbh_start += sbh * silu_stride_h;
            }
            continue;
        }

        const GemmArgs& ga = g[op == 0 ? 0 : 1];
        [[maybe_unused]] const auto s0 = TensorAccessor(acc_args, ga.in0_addr);
        [[maybe_unused]] const auto s1 = TensorAccessor(acc_args, ga.in1_addr);
        uint32_t in0_blk = ga.in0_start;
        uint32_t in1_blk = ga.in1_start;
        for (uint32_t block = 0; block < ga.num_blocks; block++) {
            cb_reserve_back(cb_in0, in0_block_tiles);
#ifdef IN0_SENDER
            {
                uint32_t l1 = get_write_ptr(cb_in0);
                const uint32_t start = l1;
                uint32_t row = in0_blk;
                for (uint32_t hh = 0; hh < pcm; hh++) {
                    uint32_t t = row;
                    for (uint32_t ww = 0; ww < bw; ww++) {
                        noc_async_read_page(t, s0, l1);
                        l1 += tile_bytes;
                        t++;
                    }
                    row += ga.in0_stride_h;
                }
                in0_blk += bw;
                noc_async_read_barrier();
                if (in0_mc_num > 0) {
                    noc_semaphore_wait(in0_s, in0_mc_num);
                    noc_semaphore_set(in0_s, 0);
                    const uint64_t dst = get_noc_multicast_addr(
                        in0_mc_end_x,
                        in0_mc_end_y,
                        in0_mc_start_x,
                        in0_mc_start_y,
                        start
                    );
                    noc_async_write_multicast(
                        start, dst, in0_block_tiles * tile_bytes, in0_mc_num
                    );
                    noc_async_writes_flushed();
                    const uint64_t sem = get_noc_multicast_addr(
                        in0_mc_end_x,
                        in0_mc_end_y,
                        in0_mc_start_x,
                        in0_mc_start_y,
                        in0_sem_r
                    );
                    noc_semaphore_set_multicast(in0_sem_r, sem, in0_mc_num);
                }
            }
#else
            noc_semaphore_set(in0_r, INVALID);
            noc_semaphore_inc(get_noc_addr(in0_sender_x, in0_sender_y, in0_sem_s), 1);
            noc_semaphore_wait(in0_r, VALID);
#endif
            cb_push_back(cb_in0, in0_block_tiles);

            cb_reserve_back(cb_in1, in1_block_tiles);
#ifdef IN1_SENDER
            {
                uint32_t l1 = get_write_ptr(cb_in1);
                const uint32_t start = l1;
                uint32_t row = in1_blk;
                for (uint32_t hh = 0; hh < bw; hh++) {
                    uint32_t t = row;
                    for (uint32_t ww = 0; ww < pcn; ww++) {
                        noc_async_read_page(t, s1, l1);
                        l1 += tile_bytes;
                        t++;
                    }
                    row += ga.in1_stride_h;
                }
                in1_blk += ga.in1_next_block;
                noc_async_read_barrier();
                if (in1_mc_num > 0) {
                    noc_semaphore_wait(in1_s, in1_mc_num);
                    noc_semaphore_set(in1_s, 0);
                    const uint64_t dst = get_noc_multicast_addr(
                        in1_mc_end_x,
                        in1_mc_end_y,
                        in1_mc_start_x,
                        in1_mc_start_y,
                        start
                    );
                    noc_async_write_multicast(
                        start, dst, in1_block_tiles * tile_bytes, in1_mc_num
                    );
                    noc_async_writes_flushed();
                    const uint64_t sem = get_noc_multicast_addr(
                        in1_mc_end_x,
                        in1_mc_end_y,
                        in1_mc_start_x,
                        in1_mc_start_y,
                        in1_sem_r
                    );
                    noc_semaphore_set_multicast(in1_sem_r, sem, in1_mc_num);
                }
            }
#else
            noc_semaphore_set(in1_r, INVALID);
            noc_semaphore_inc(get_noc_addr(in1_sender_x, in1_sender_y, in1_sem_s), 1);
            noc_semaphore_wait(in1_r, VALID);
#endif
            cb_push_back(cb_in1, in1_block_tiles);
        }
    }
    (void)in0_sender_x;
    (void)in0_sender_y;
    (void)in1_sender_x;
    (void)in1_sender_y;
    (void)in0_s;
    (void)in1_s;
    (void)in0_r;
    (void)in1_r;
}
