// Persistent 2D-multicast matmul reader (roles via IN0_SENDER / IN1_SENDER): for i
// = 1..n waits for X280 command i (skipped with TTX_STANDALONE), then streams one
// matmul's in0/in1 blocks. Derived from tt-metal reader_bmm_tile_layout_*.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "hostdevcommon/common_values.hpp"

// Per command: reads or receives the A and B blocks for each K block.
void kernel_main() {
    uint32_t a = 0;
    const uint32_t in0_addr = get_arg_val<uint32_t>(a++);
    const uint32_t in0_start = get_arg_val<uint32_t>(a++);
    const uint32_t in0_stride_h = get_arg_val<uint32_t>(a++);  // Kt
    const uint32_t in0_next_block = get_arg_val<uint32_t>(a++);
    const uint32_t in0_block_w = get_arg_val<uint32_t>(a++);
    const uint32_t in0_block_h = get_arg_val<uint32_t>(a++);
    const uint32_t in0_block_tiles = get_arg_val<uint32_t>(a++);
    const uint32_t in1_addr = get_arg_val<uint32_t>(a++);
    const uint32_t in1_start = get_arg_val<uint32_t>(a++);
    const uint32_t in1_stride_h = get_arg_val<uint32_t>(a++);  // Nt
    const uint32_t in1_next_block = get_arg_val<uint32_t>(a++);
    const uint32_t in1_block_w = get_arg_val<uint32_t>(a++);
    const uint32_t in1_block_h = get_arg_val<uint32_t>(a++);
    const uint32_t in1_block_tiles = get_arg_val<uint32_t>(a++);
    const uint32_t num_blocks = get_arg_val<uint32_t>(a++);
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
    // command protocol
    const uint32_t n_cmds = get_arg_val<uint32_t>(a++);
    const uint32_t l2_x = get_arg_val<uint32_t>(a++);
    const uint32_t l2_y = get_arg_val<uint32_t>(a++);
    const uint32_t mailbox = get_arg_val<uint32_t>(a++);
    const uint32_t ack_base = get_arg_val<uint32_t>(a++);
    const uint32_t idx = get_arg_val<uint32_t>(a++);
    const uint32_t leader = get_arg_val<uint32_t>(a++);
    // Grouped completion (grp_n > 0): members write tag<<16|step into the group
    // leader's gather slots; group leaders ack.
    const uint32_t grp_x = get_arg_val<uint32_t>(a++);
    const uint32_t grp_y = get_arg_val<uint32_t>(a++);
    const uint32_t grp_n =
        get_arg_val<uint32_t>(a++);  // members excluding the leader; 0 = flat acks
    const uint32_t grp_rank = get_arg_val<uint32_t>(a++);
    const uint32_t grp_slot = get_arg_val<uint32_t>(a++);
    const uint32_t is_grp_leader = get_arg_val<uint32_t>(a++);
    const uint32_t tag = get_arg_val<uint32_t>(a++) & 0xFFFFu;
    const uint32_t depth = get_arg_val<uint32_t>(a++);
    const uint32_t rs =
        get_arg_val<uint32_t>(a++) / 4;  // command ring slot stride, words

    constexpr uint32_t cb_in0 = 0;
    constexpr uint32_t cb_in1 = 1;
    constexpr uint32_t cb_ctl = 8;
    const uint32_t tile_bytes = get_tile_size(cb_in0);

    constexpr auto s0_args = TensorAccessorArgs<0>();
    [[maybe_unused]] const auto s0 = TensorAccessor(s0_args, in0_addr);
    constexpr auto s1_args =
        TensorAccessorArgs<s0_args.next_compile_time_args_offset()>();
    [[maybe_unused]] const auto s1 = TensorAccessor(s1_args, in1_addr);

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

    // Control block (CB 8, same L1 address on every core): [0..255] command ring (4
    // slots, 16 or 64 B apart), [256] reader ack msg, [272] writer ack msg, [288]
    // reader check-in source, [304] writer gather source, [320] leader hello, [384 + 16
    // r] gather slots of member rank r.
    const uint32_t ctl = get_write_ptr(cb_ctl);
    volatile tt_l1_ptr uint32_t* cmd =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ctl);
#ifndef TTX_STANDALONE
    for (uint32_t s = 0; s < 4; s++) {
        cmd[s * rs] = 0;
    }
    {
        if (grp_n > 0 && !is_grp_leader) {
            // Check in with the group leader after clearing the ring (no command can
            // precede the leader's READY).
            volatile tt_l1_ptr uint32_t* ci =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ctl + 288);
            ci[0] = tag << 16;
            noc_async_write(
                ctl + 288, get_noc_addr(grp_x, grp_y, ctl + 384 + 16 * grp_rank), 4
            );
            noc_async_write_barrier();
        } else if (grp_n > 0) {
            for (uint32_t r = 0; r < grp_n;) {
                invalidate_l1_cache();
                if ((*reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                         ctl + 384 + 16 * r
                     ) >>
                     16) == tag) {
                    r++;
                }
            }
        }
        // NOC writes need matching 16 B alignment of source and destination: ack slots
        // are 16 B apart.
        volatile tt_l1_ptr uint32_t* msg =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ctl + 256);
        msg[0] = 0xFFFFFFFFu;
        if (grp_n == 0 || is_grp_leader) {
            noc_async_write(
                ctl + 256,
                get_noc_addr(l2_x, l2_y, ack_base + (grp_n ? grp_slot : idx) * 16),
                4
            );
            noc_async_write_barrier();
        }
        if (leader) {
            volatile tt_l1_ptr uint32_t* hello =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ctl + 320);
            hello[0] = 1;
            hello[1] = ctl;
            noc_async_write(ctl + 320, get_noc_addr(l2_x, l2_y, mailbox), 8);
            noc_async_write_barrier();
        }
    }
#else
    (void)n_cmds;
    (void)l2_x;
    (void)l2_y;
    (void)mailbox;
    (void)ack_base;
    (void)idx;
    (void)leader;
    (void)cmd;
    (void)grp_x;
    (void)grp_y;
    (void)grp_n;
    (void)grp_rank;
    (void)grp_slot;
    (void)is_grp_leader;
    (void)tag;
    (void)depth;
    (void)rs;
#endif

    for (uint32_t it = 1; it <= n_cmds; it++) {
#ifndef TTX_STANDALONE
        uint32_t spins = 0;
        volatile tt_l1_ptr uint32_t* slot = cmd + (depth > 1 ? (it % depth) * rs : 0);
        do {
            invalidate_l1_cache();
            if (++spins == (1u << 30)) {
                return;  // X280 gone: give up (compute/writer will stall; host recovers
                         // by chip reset)
            }
        } while (static_cast<int32_t>(*slot - it) < 0);
#endif
        uint32_t in0_blk = in0_start;
        uint32_t in1_blk = in1_start;
        for (uint32_t block = 0; block < num_blocks; block++) {
            // ---- operand 0 ----
            cb_reserve_back(cb_in0, in0_block_tiles);
#ifdef IN0_SENDER
            {
                uint32_t l1 = get_write_ptr(cb_in0);
                const uint32_t start = l1;
                uint32_t row = in0_blk;
                for (uint32_t h = 0; h < in0_block_h; h++) {
                    uint32_t t = row;
                    for (uint32_t w = 0; w < in0_block_w; w++) {
                        noc_async_read_page(t, s0, l1);
                        l1 += tile_bytes;
                        t++;
                    }
                    row += in0_stride_h;
                }
                in0_blk += in0_next_block;
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

            // ---- operand 1 ----
            cb_reserve_back(cb_in1, in1_block_tiles);
#ifdef IN1_SENDER
            {
                uint32_t l1 = get_write_ptr(cb_in1);
                const uint32_t start = l1;
                uint32_t row = in1_blk;
                for (uint32_t h = 0; h < in1_block_h; h++) {
                    uint32_t t = row;
                    for (uint32_t w = 0; w < in1_block_w; w++) {
                        noc_async_read_page(t, s1, l1);
                        l1 += tile_bytes;
                        t++;
                    }
                    row += in1_stride_h;
                }
                in1_blk += in1_next_block;
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
