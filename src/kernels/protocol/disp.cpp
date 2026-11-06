// Persistent worker for dispatch-model experiments. For i = 1..n: wait
// for command i, spin `work` wall-clock ticks, then signal completion according to
// `sig` (0 own ack slot, 1 row-leader aggregation, 2 Tensix-only barrier).
#include <cstdint>

#include "api/dataflow/dataflow_api.h"

namespace {
constexpr uint32_t kSigFlat = 0, kSigHier = 1, kSigTbar = 2, kSigHierW = 3;
constexpr uint32_t kGatherOff =
    1152;  // sig 3: member rank r writes (tag << 16 | step) at leader ctl + 1152 + 16 r

// Whether a tagged slot holds this run's tag and a step >= `step`.
bool tagged_ge(volatile tt_l1_ptr uint32_t* p, uint32_t tag, uint32_t v) {
    const uint32_t w = *p;
    return (w >> 16) == tag &&
           static_cast<int16_t>(static_cast<uint16_t>(w) - static_cast<uint16_t>(v)) >=
               0;
}

// Waits until every member slot of the group reached `step`.
bool gather(uint32_t ctl, uint32_t members, uint32_t tag, uint32_t v) {
    uint32_t spins = 0;
    for (uint32_t r = 0; r < members;) {
        invalidate_l1_cache();
        if (tagged_ge(
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                    ctl + kGatherOff + 16 * r
                ),
                tag,
                v
            )) {
            r++;
        } else if (++spins == (1u << 28)) {
            return false;
        }
    }
    return true;
}

constexpr uint32_t kRingOff = 0;    // command ring: up to 16 entries, 16 or 64 B apart
constexpr uint32_t kMsgOff = 1024;  // ack source words (packed acks use +4 * (idx % 4))
constexpr uint32_t kHelloOff = 1088;  // leader hello {1, ctl}
constexpr uint32_t kRelOff = 1120;    // Tensix-barrier release source word
constexpr uint32_t kDiagOff = 1408;   // 16 B diagnostic record source

// Spins until the L1 word reaches `value`; false after a bound.
bool wait_ge(volatile tt_l1_ptr uint32_t* p, uint32_t v) {
    uint32_t spins = 0;
    do {
        invalidate_l1_cache();
        if (++spins == (1u << 28)) {
            return false;
        }
    } while (static_cast<int32_t>(*p - v) < 0);
    return true;
}
}  // namespace

// Empty commands under one of the completion models (dispatch cost only).
void kernel_main() {
    uint32_t a = 0;
    const uint32_t l2_x = get_arg_val<uint32_t>(a++);
    const uint32_t l2_y = get_arg_val<uint32_t>(a++);
    const uint32_t mailbox = get_arg_val<uint32_t>(a++);
    const uint32_t ack_base = get_arg_val<uint32_t>(a++);
    const uint32_t idx = get_arg_val<uint32_t>(a++);
    const uint32_t n = get_arg_val<uint32_t>(a++);
    const uint32_t leader = get_arg_val<uint32_t>(a++);
    const uint32_t sig = get_arg_val<uint32_t>(a++);
    const uint32_t ack_stride = get_arg_val<uint32_t>(a++);  // 16 or 4
    const uint32_t depth =
        get_arg_val<uint32_t>(a++);  // command ring depth (1 = single word)
    const uint32_t work = get_arg_val<uint32_t>(a++);  // wall-clock ticks per command
    const uint32_t grp_x =
        get_arg_val<uint32_t>(a++);  // hier: group leader; tbar: global leader
    const uint32_t grp_y = get_arg_val<uint32_t>(a++);
    const uint32_t grp_n =
        get_arg_val<uint32_t>(a++);  // hier: members incl. leader; tbar: all cores
    const uint32_t grp_slot =
        get_arg_val<uint32_t>(a++);  // hier: ack slot of this group
    const uint32_t is_grp_leader = get_arg_val<uint32_t>(a++);
    const uint32_t sem = get_semaphore(get_arg_val<uint32_t>(a++));
    uint32_t rect[2][5];  // tbar release multicast rectangles: x0, y0, x1, y1,
                          // num_dests (0 = unused)
    for (uint32_t r = 0; r < 2; r++) {
        for (uint32_t k = 0; k < 5; k++) {
            rect[r][k] = get_arg_val<uint32_t>(a++);
        }
    }
    const uint32_t self_rect =
        get_arg_val<uint32_t>(a++);  // tbar: rectangle containing the leader
    const uint32_t grp_rank =
        get_arg_val<uint32_t>(a++);  // sig 3: member rank within the group
    const uint32_t tag = get_arg_val<uint32_t>(a++) & 0xFFFFu;
    const uint32_t rs = get_arg_val<uint32_t>(a++) / 4;  // ring slot stride in words
    // Relay: the X280 unicasts commands to this core only; it forwards each command
    // word to both panel rectangles with a non-posted multicast (rect[r], self_rect
    // excluded from num_dests) and waits for every destination's ack.
    const uint32_t is_relay = get_arg_val<uint32_t>(a++);

    const uint32_t ctl = get_write_ptr(0);
    volatile tt_l1_ptr uint32_t* ring =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ctl + kRingOff);
    for (uint32_t s = 0; s < 16; s++) {
        ring[s * rs] = 0;
    }
    volatile tt_l1_ptr uint32_t* sem_p =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem);

    const bool hier = sig == kSigHier || sig == kSigHierW;
    const uint32_t my_slot = hier ? grp_slot : idx;
    const uint32_t msg_addr = ctl + kMsgOff + (ack_stride == 4 ? 4 * (my_slot % 4) : 0);
    volatile tt_l1_ptr uint32_t* msg =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(msg_addr);
    const uint64_t ack = get_noc_addr(l2_x, l2_y, ack_base + my_slot * ack_stride);
    const bool sends_ack =
        sig == kSigFlat || (hier && is_grp_leader) || (sig == kSigTbar && leader);

    const uint64_t grp_sem = get_noc_addr(grp_x, grp_y, sem);
    // Check-in round 0: members report to their leader before the leader reports READY,
    // so no command is issued before every ring is cleared. Step i then completes at
    // count (grp_n - 1) * (i + 1).
    const bool member =
        (sig == kSigHier && !is_grp_leader) || (sig == kSigTbar && !leader);
    const bool counts =
        (sig == kSigHier && is_grp_leader) || (sig == kSigTbar && leader);
    const uint32_t gather_src =
        ctl + kMsgOff + 32;  // 16 B aligned, matches the 16 B gather slots
    volatile tt_l1_ptr uint32_t* gsrc =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(gather_src);
    const uint64_t gather_dst =
        get_noc_addr(grp_x, grp_y, ctl + kGatherOff + 16 * grp_rank);
    if (sig == kSigHierW) {
        if (!is_grp_leader) {
            gsrc[0] = tag << 16;
            noc_async_write(gather_src, gather_dst, 4);
            noc_async_write_barrier();
        } else if (!gather(ctl, grp_n - 1, tag, 0)) {
            return;
        }
    } else if (member) {
        noc_semaphore_inc(grp_sem, 1);
    } else if (counts && grp_n > 1 && !wait_ge(sem_p, grp_n - 1)) {
        return;
    }
    if (sends_ack) {
        msg[0] = 0xFFFFFFFFu;
        noc_async_write(msg_addr, ack, 4);
        noc_async_write_barrier();
    }
    if (leader) {
        volatile tt_l1_ptr uint32_t* hello =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ctl + kHelloOff);
        hello[0] = 1;
        hello[1] = ctl;
        noc_async_write(ctl + kHelloOff, get_noc_addr(l2_x, l2_y, mailbox), 8);
        noc_async_write_barrier();
    }

    // On a timeout, record {0xD1A0 | stage, expected step, observed word, aux} at LIM
    // ack_base + 0x1000 + 16 * idx.
    auto diag = [&](uint32_t stage, uint32_t expect, uint32_t seen, uint32_t aux) {
        volatile tt_l1_ptr uint32_t* d =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ctl + kDiagOff);
        d[0] = 0xD1A00000u | stage;
        d[1] = expect;
        d[2] = seen;
        d[3] = aux;
        noc_async_write(
            ctl + kDiagOff, get_noc_addr(l2_x, l2_y, ack_base + 0x1000 + 16 * idx), 16
        );
        noc_async_write_barrier();
    };
    for (uint32_t i = 1; i <= n; i++) {
        volatile tt_l1_ptr uint32_t* slot = &ring[(depth > 1 ? (i % depth) : 0) * rs];
        if (!wait_ge(slot, i)) {
            invalidate_l1_cache();
            diag(1, i, *slot, ring[0]);
            return;
        }
        if (is_relay) {
            const uint32_t slot_addr = reinterpret_cast<uint32_t>(slot);
            for (uint32_t r = 0; r < 2; r++) {
                const uint32_t dests = rect[r][4] - (r == self_rect ? 1u : 0u);
                if (dests == 0) {
                    continue;
                }
                const uint64_t dst = get_noc_multicast_addr(
                    rect[r][0], rect[r][1], rect[r][2], rect[r][3], slot_addr
                );
                noc_semaphore_set_multicast(slot_addr, dst, dests);
            }
            noc_async_write_barrier();
        }
        if (work) {
            riscv_wait(work);
        }
        if (sig == kSigHierW) {
            if (is_grp_leader) {
                if (!gather(ctl, grp_n - 1, tag, i)) {
                    uint32_t r = 0;
                    uint32_t w = 0;
                    for (; r < grp_n - 1; r++) {
                        invalidate_l1_cache();
                        w = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                            ctl + kGatherOff + 16 * r
                        );
                        if (!tagged_ge(
                                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                                    ctl + kGatherOff + 16 * r
                                ),
                                tag,
                                i
                            )) {
                            break;
                        }
                    }
                    diag(2, i, w, r);
                    return;
                }
                msg[0] = i;
                noc_async_write(msg_addr, ack, 4);
                noc_async_write_barrier();
            } else {
                gsrc[0] = (tag << 16) | (i & 0xFFFFu);
                noc_async_write(gather_src, gather_dst, 4);
                noc_async_write_barrier();
            }
        } else if (sig == kSigFlat) {
            msg[0] = i;
            noc_async_write(msg_addr, ack, 4);
            noc_async_write_barrier();
        } else if (sig == kSigHier) {
            if (is_grp_leader) {
                if (grp_n > 1 && !wait_ge(sem_p, (grp_n - 1) * (i + 1))) {
                    invalidate_l1_cache();
                    diag(3, (grp_n - 1) * (i + 1), *sem_p, i);
                    return;
                }
                msg[0] = i;
                noc_async_write(msg_addr, ack, 4);
                noc_async_write_barrier();
            } else {
                noc_semaphore_inc(grp_sem, 1);
            }
        } else {  // Tensix-only barrier: all -> leader counter, leader multicasts
                  // release i + 1
            if (leader) {
                if (!wait_ge(sem_p, (grp_n - 1) * (i + 1))) {
                    return;
                }
                if (i == 1 || i == n) {
                    msg[0] = i;
                    noc_async_write(msg_addr, ack, 4);
                    noc_async_write_barrier();
                }
                if (i < n) {
                    volatile tt_l1_ptr uint32_t* rel =
                        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ctl + kRelOff);
                    rel[0] = i + 1;
                    for (uint32_t r = 0; r < 2; r++) {
                        if (rect[r][4] == 0) {
                            continue;
                        }
                        const uint64_t dst = get_noc_multicast_addr(
                            rect[r][0],
                            rect[r][1],
                            rect[r][2],
                            rect[r][3],
                            ctl + kRingOff
                        );
                        if (r == self_rect) {
                            noc_semaphore_set_multicast_loopback_src(
                                ctl + kRelOff, dst, rect[r][4]
                            );
                        } else {
                            noc_semaphore_set_multicast(ctl + kRelOff, dst, rect[r][4]);
                        }
                    }
                    noc_async_write_barrier();
                }
            } else {
                noc_semaphore_inc(grp_sem, 1);
            }
        }
    }
    noc_async_atomic_barrier();
}
