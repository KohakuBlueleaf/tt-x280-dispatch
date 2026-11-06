// Persistent matmul writer: for i = 1..n writes this core's output
// block, then (unless TTX_STANDALONE) reports completion: flat (grp_n == 0) acks LIM
// slot idx; grouped members write tag<<16|i to the group leader's gather slot and group
// leaders ack LIM slot grp_slot once all members reached i. Based on
// writer_bmm_tile_layout.cpp.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

namespace {
constexpr uint32_t kWriterMsg = 272, kWriterGatherSrc = 304, kGather = 384;
}

// Per command: writes this core's C block, then completes.
void kernel_main() {
    uint32_t a = 0;
    const uint32_t out_addr = get_arg_val<uint32_t>(a++);
    const uint32_t out_start = get_arg_val<uint32_t>(a++);
    const uint32_t out_stride_h = get_arg_val<uint32_t>(a++);  // Nt
    const uint32_t next_sb_w = get_arg_val<uint32_t>(a++);
    const uint32_t next_sb_h = get_arg_val<uint32_t>(a++);
    const uint32_t sb_w = get_arg_val<uint32_t>(a++);
    const uint32_t sb_h = get_arg_val<uint32_t>(a++);
    const uint32_t sb_tiles = get_arg_val<uint32_t>(a++);
    const uint32_t num_sb_w = get_arg_val<uint32_t>(a++);
    const uint32_t num_sb_h = get_arg_val<uint32_t>(a++);
    const uint32_t n_cmds = get_arg_val<uint32_t>(a++);
    const uint32_t l2_x = get_arg_val<uint32_t>(a++);
    const uint32_t l2_y = get_arg_val<uint32_t>(a++);
    const uint32_t ack_base = get_arg_val<uint32_t>(a++);
    const uint32_t idx = get_arg_val<uint32_t>(a++);
    const uint32_t grp_x = get_arg_val<uint32_t>(a++);
    const uint32_t grp_y = get_arg_val<uint32_t>(a++);
    const uint32_t grp_n = get_arg_val<uint32_t>(a++);
    const uint32_t grp_rank = get_arg_val<uint32_t>(a++);
    const uint32_t grp_slot = get_arg_val<uint32_t>(a++);
    const uint32_t is_grp_leader = get_arg_val<uint32_t>(a++);
    const uint32_t tag = get_arg_val<uint32_t>(a++) & 0xFFFFu;

    constexpr uint32_t cb_out = 16;
    constexpr uint32_t cb_ctl = 8;
    const uint32_t tile_bytes = get_tile_size(cb_out);
    constexpr auto s_args = TensorAccessorArgs<0>();
    const auto s = TensorAccessor(s_args, out_addr);

    const uint32_t ctl = get_write_ptr(cb_ctl);
    volatile tt_l1_ptr uint32_t* msg =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ctl + kWriterMsg);
    volatile tt_l1_ptr uint32_t* gsrc =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ctl + kWriterGatherSrc);
#ifndef TTX_STANDALONE
    const uint64_t ack =
        get_noc_addr(l2_x, l2_y, ack_base + (grp_n ? grp_slot : idx) * 16);
    const uint64_t gdst = get_noc_addr(grp_x, grp_y, ctl + kGather + 16 * grp_rank);
#else
    (void)l2_x;
    (void)l2_y;
    (void)ack_base;
    (void)idx;
    (void)msg;
    (void)gsrc;
    (void)grp_x;
    (void)grp_y;
    (void)grp_n;
    (void)grp_rank;
    (void)grp_slot;
    (void)is_grp_leader;
    (void)tag;
#endif

    for (uint32_t it = 1; it <= n_cmds; it++) {
        uint32_t sbh_start = out_start;
        for (uint32_t sbh = 0; sbh < num_sb_h; sbh++) {
            uint32_t sbw_start = sbh_start;
            for (uint32_t sbw = 0; sbw < num_sb_w; sbw++) {
                uint32_t row = sbw_start;
                cb_wait_front(cb_out, sb_tiles);
                uint32_t l1 = get_read_ptr(cb_out);
                for (uint32_t h = 0; h < sb_h; h++) {
                    uint32_t t = row;
                    for (uint32_t w = 0; w < sb_w; w++) {
                        noc_async_write_page(t, s, l1);
                        l1 += tile_bytes;
                        t++;
                    }
                    row += out_stride_h;
                }
                noc_async_writes_flushed();
                cb_pop_front(cb_out, sb_tiles);
                sbw_start += next_sb_w;
            }
            sbh_start += next_sb_h;
        }
        noc_async_write_barrier();
#ifndef TTX_STANDALONE
        if (grp_n > 0 && !is_grp_leader) {
            gsrc[0] = (tag << 16) | (it & 0xFFFFu);
            noc_async_write(ctl + kWriterGatherSrc, gdst, 4);
            noc_async_write_barrier();
            continue;
        }
        if (grp_n > 0) {
            uint32_t spins = 0;
            for (uint32_t r = 0; r < grp_n;) {
                invalidate_l1_cache();
                const uint32_t w = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                    ctl + kGather + 16 * r
                );
                if ((w >> 16) == tag &&
                    static_cast<int16_t>(
                        static_cast<uint16_t>(w) - static_cast<uint16_t>(it)
                    ) >= 0) {
                    r++;
                } else if (++spins == (1u << 30)) {
                    return;
                }
            }
        }
        msg[0] = it;
        noc_async_write(ctl + kWriterMsg, ack, 4);
        noc_async_write_barrier();
#endif
    }
}
