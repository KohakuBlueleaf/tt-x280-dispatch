// tt-x280-dispatch model v1 — device-side template for resident Tensix kernels (docs:
// docs/dispatch-model.md).
//
// Usage in a data-movement kernel pair:
//   reader:  auto d = ttx::Args::read(arg_index);  ttx::Worker w(d);
//   w.start();
//            for (seq = 1..n) { if (!w.wait(seq)) return; ...op seq... }
//   writer:  auto d = ttx::Args::read(arg_index);  ttx::Worker w(d);
//            for (seq = 1..n) { ...write outputs, noc_async_write_barrier()...;
//            w.complete(seq); }
// The host appends the matching 16 runtime args with ttx_dispatch_host.hpp
// (DispatchPlan::worker_args). Requires CB `kCtlCb` (>= 1024 B) on every core; its L1
// address must be identical on all cores. Tracing (host sets a trace tensor): CB
// `kTraceCb` holds 16 B per command, the writer calls flush_trace(n) last.
#pragma once

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

namespace ttx {

constexpr uint32_t kCtlCb = 8;
constexpr uint32_t kTraceCb = 10;
// Trace record per command (wall-clock cycles, low 32 bits): command seen, outputs
// written, completion sent.
constexpr uint32_t kTraceSeen = 0, kTraceDone = 4, kTraceAcked = 8, kTraceBytes = 16;
// Control block layout (byte offsets from the CB base).
constexpr uint32_t kRing = 0;  // command ring, depth <= 4, slot stride 16 or 64 B
constexpr uint32_t kReaderMsg = 256;   // READY source
constexpr uint32_t kWriterMsg = 272;   // leader ack source
constexpr uint32_t kCheckinSrc = 288;  // member check-in source
constexpr uint32_t kGatherSrc = 304;   // member completion source
constexpr uint32_t kHello = 320;       // leader hello {1, ctl}
constexpr uint32_t kGather = 384;      // leader gather slots, 16 B per member rank
constexpr uint32_t kReady = 0xFFFFFFFFu;
constexpr uint32_t kNumArgs = 16;

struct Args {
    uint32_t l2_x, l2_y, mailbox, ack_base, hello_core, grp_x, grp_y, grp_n, grp_rank,
        grp_slot, is_grp_leader, tag, depth, ring_stride, trace_addr, trace_page;

    // Reads kNumArgs runtime args starting at `a` and advances `a`.
    static Args read(uint32_t& a) {
        Args d;
        uint32_t* f = &d.l2_x;
        for (uint32_t k = 0; k < kNumArgs; k++) {
            f[k] = get_arg_val<uint32_t>(a++);
        }
        d.tag &= 0xFFFFu;
        return d;
    }
};

class Worker {
public:
    explicit Worker(const Args& d)
        : d_(d),
          ctl_(get_write_ptr(kCtlCb)),
          trace_(d.trace_addr != 0 ? get_write_ptr(kTraceCb) : 0) {
    }

    // Writer side, after the last command: copies this core's trace records to page
    // `trace_page` of the trace tensor.
    void flush_trace(uint32_t n_cmds) {
        if (trace_ == 0) {
            return;
        }
        const InterleavedAddrGen<true> dst = {
            .bank_base_address = d_.trace_addr, .page_size = n_cmds * kTraceBytes
        };
        noc_async_write(trace_, get_noc_addr(d_.trace_page, dst), n_cmds * kTraceBytes);
        noc_async_write_barrier();
    }

    // Reader side: clear the ring, check in (members) or collect check-ins and report
    // READY (leaders), send hello.
    void start() {
        for (uint32_t s = 0; s < 4; s++) {
            word(kRing + s * d_.ring_stride) = 0;
        }
        if (!d_.is_grp_leader) {
            word(kCheckinSrc) = d_.tag << 16;
            noc_async_write(ctl_ + kCheckinSrc, gather_dst(), 4);
            noc_async_write_barrier();
        } else {
            for (uint32_t r = 0; r < d_.grp_n;) {
                invalidate_l1_cache();
                if ((word(kGather + 16 * r) >> 16) == d_.tag) {
                    r++;
                }
            }
            word(kReaderMsg) = kReady;
            noc_async_write(ctl_ + kReaderMsg, ack_dst(), 4);
            noc_async_write_barrier();
        }
        if (d_.hello_core) {
            word(kHello) = 1;
            word(kHello + 4) = ctl_;
            noc_async_write(
                ctl_ + kHello, get_noc_addr(d_.l2_x, d_.l2_y, d_.mailbox), 8
            );
            noc_async_write_barrier();
        }
    }

    // Reader side: wait until command `seq` has been issued. Returns false after a
    // bounded spin (X280 gone).
    bool wait(uint32_t seq) {
        const uint32_t off =
            kRing + (d_.depth > 1 ? (seq % d_.depth) * d_.ring_stride : 0);
        uint32_t spins = 0;
        do {
            invalidate_l1_cache();
            if (++spins == (1u << 30)) {
                return false;
            }
        } while (static_cast<int32_t>(word(off) - seq) < 0);
        stamp(seq, kTraceSeen);
        return true;
    }

    // Writer side: report that this core finished command `seq` (call after its output
    // writes are complete).
    bool complete(uint32_t seq) {
        stamp(seq, kTraceDone);
        if (!d_.is_grp_leader) {
            word(kGatherSrc) = (d_.tag << 16) | (seq & 0xFFFFu);
            noc_async_write(ctl_ + kGatherSrc, gather_dst(), 4);
            noc_async_write_barrier();
            stamp(seq, kTraceAcked);
            return true;
        }
        uint32_t spins = 0;
        for (uint32_t r = 0; r < d_.grp_n;) {
            invalidate_l1_cache();
            const uint32_t w = word(kGather + 16 * r);
            if ((w >> 16) == d_.tag &&
                static_cast<int16_t>(
                    static_cast<uint16_t>(w) - static_cast<uint16_t>(seq)
                ) >= 0) {
                r++;
            } else if (++spins == (1u << 30)) {
                return false;
            }
        }
        word(kWriterMsg) = seq;
        noc_async_write(ctl_ + kWriterMsg, ack_dst(), 4);
        noc_async_write_barrier();
        stamp(seq, kTraceAcked);
        return true;
    }

private:
    void stamp(uint32_t seq, uint32_t field) {
        if (trace_ != 0) {
            const uint32_t now = *reinterpret_cast<volatile tt_reg_ptr uint32_t*>(
                RISCV_DEBUG_REG_WALL_CLOCK_L
            );
            *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                trace_ + (seq - 1) * kTraceBytes + field
            ) = now;
        }
    }

    volatile tt_l1_ptr uint32_t& word(uint32_t off) {
        return *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ctl_ + off);
    }

    uint64_t gather_dst() const {
        return get_noc_addr(d_.grp_x, d_.grp_y, ctl_ + kGather + 16 * d_.grp_rank);
    }

    uint64_t ack_dst() const {
        return get_noc_addr(d_.l2_x, d_.l2_y, d_.ack_base + d_.grp_slot * 16);
    }

    Args d_;
    uint32_t ctl_;
    uint32_t trace_;
};

}  // namespace ttx
