/* The X280 dispatch sequencer (hart 0). Issues commands 1..n to resident Tensix
 * programs through up to two multicast windows and gathers completion from `k` LIM
 * slots, one per column leader. Result: mtime ticks from command 1 issued to all
 * slots >= n, or ~k on error.
 *
 * scan 0: spin per slot; 1: fence once, sweep all slots; 2: watch slot 0 only.
 * depth D > 1: command i is issued once every slot >= i - D (D commands in flight).
 * host_ring != 0: command i waits until the host wrote i into its ring slot; the
 * completed count is published to host_done (LIM) or a pinned host buffer. */
#include <stdint.h>

#include "noc.h"
#include "rt/x280_hw.h"
#include "rt/x280_idle.h"
#include "x280_hart_protocol.h"

#define TTX_READY 0xFFFFFFFFu

/* Arguments written by the host (DispatchPlan::start) at the start of available LIM. */
typedef struct {
    uint64_t run_value;
    uint64_t mailbox;
    uint64_t acks;
    uint64_t n;
    uint64_t k;
    uint64_t nrect;
    uint32_t rect[2][4];
    uint64_t scan;
    uint64_t depth;
    uint64_t ack_stride;
    uint64_t host_ring;
    uint64_t host_depth;
    uint64_t host_done;
    /* noc_ordering_t of the command windows */
    uint64_t ordering;
    /* bytes between command ring slots: 16 or 64 */
    uint64_t ring_stride;
    /* 0: a 32-bit scalar store per command; 1: one 64 B vector store */
    uint64_t store_mode;
    /* mtime ticks (20 ns) to wait after issuing each command */
    uint64_t issue_gap;
    /* host-memory completion target: the PCIe tile (translated) and the NOC address of
     * a pinned host buffer; done_noc_addr 0 publishes completion in LIM only */
    uint64_t done_pcie_x;
    uint64_t done_pcie_y;
    uint64_t done_noc_addr;
} ttx_sequencer_args_t;

/* Writes command `v` into the 64 B line at `dst` with one vse32.v (VLEN 512 = 16 x
 * e32). */
static inline void store_line(volatile uint32_t* dst, uint32_t v) {
    uint32_t line[16] __attribute__((aligned(64)));
    line[0] = v;
    for (int j = 1; j < 16; j++) {
        line[j] = 0;
    }
    __asm__ volatile(
        ".option push\n.option arch, +v\n"
        "vsetivli zero, 16, e32, m1, ta, ma\n"
        "vle32.v v8, (%0)\n"
        "vse32.v v8, (%1)\n"
        ".option pop\n"
        :
        : "r"(line), "r"(dst)
        : "memory", "v8"
    );
}

/* The hart and run value, for stop requests. */
typedef struct {
    uint64_t hart, run;
} ttx_ctx_t;

/* Every 4096 spins, whether the host asked this run to stop. */
static int stopped(const ttx_ctx_t* c, uint32_t* spins) {
    if (++*spins == 4096u) {
        *spins = 0;
        return x280_hart_stop_requested(c->hart, c->run);
    }
    return 0;
}

/* All k slots reach value (signed compare); READY counts as reached only when value ==
 * READY. */
static int wait_all(
    const ttx_ctx_t* c,
    volatile const uint32_t* acks,
    uint64_t k,
    uint64_t stride_words,
    uint32_t value,
    uint64_t scan
) {
    uint32_t spins = 0;
    if (scan == 0) {
        for (uint64_t j = 0; j < k; j++) {
            for (;;) {
                x280_fence_ir();
                const uint32_t v = acks[j * stride_words];
                if (value == TTX_READY
                        ? v == TTX_READY
                        : (v != TTX_READY && (int32_t)(v - value) >= 0)) {
                    break;
                }
                if (stopped(c, &spins)) {
                    return 0;
                }
            }
        }
        return 1;
    }
    uint64_t from = 0; /* slots below `from` are known done */
    for (;;) {
        x280_fence_ir();
        while (from < k) {
            const uint32_t v = acks[from * stride_words];
            if (value == TTX_READY ? v != TTX_READY
                                   : (v == TTX_READY || (int32_t)(v - value) < 0)) {
                break;
            }
            from++;
        }
        if (from == k) {
            return 1;
        }
        if (stopped(c, &spins)) {
            return 0;
        }
    }
}

/* Hart 0 runs the sequencer; the other harts return at once. */
int main(uint64_t hartid) {
    if (hartid != 0) {
        return 0;
    }
    const ttx_ctx_t ctx = {hartid, x280_hart_control(hartid)->run_value};
    const ttx_ctx_t* c = &ctx;
    const volatile ttx_sequencer_args_t* a =
        (const volatile ttx_sequencer_args_t*)x280_hart_args(hartid);
    if (a == 0 || a->run_value != ctx.run || a->nrect < 1 || a->nrect > 2 ||
        a->depth < 1 || a->depth > 16 || a->ordering > 2 ||
        (a->ring_stride != 16 && a->ring_stride != 64) || a->store_mode > 1 ||
        (a->store_mode == 1 && a->ring_stride != 64)) {
        x280_hart_set_result(hartid, ~0ull);
        return 0;
    }
    volatile uint32_t* mb = (volatile uint32_t*)(uintptr_t)a->mailbox;
    volatile uint32_t* acks = (volatile uint32_t*)(uintptr_t)a->acks;
    volatile uint32_t* hring = (volatile uint32_t*)(uintptr_t)a->host_ring;
    volatile uint32_t* hdone = (volatile uint32_t*)(uintptr_t)a->host_done;
    const uint64_t n = a->n, k = a->k, nrect = a->nrect, depth = a->depth,
                   sw = a->ack_stride / 4;
    const uint64_t scan = a->scan, hdepth = a->host_depth;

    if (!x280_hart_wait_word_or_stop(hartid, ctx.run, &mb[0], 1u)) {
        return 0;
    }
    if (!wait_all(c, acks, k, sw, TTX_READY, 1)) {
        return 0;
    }
    const uint32_t cmd_addr = mb[1];
    const uint64_t store_mode = a->store_mode;
    if (store_mode == 1) {
        if (cmd_addr & 63u) {
            x280_hart_set_result(hartid, ~4ull);
            return 0;
        }
        __asm__ volatile("csrs mstatus, %0" ::"r"((3ul << 9) | (3ul << 13)));
    }
    volatile uint32_t* cmd[2];
    for (uint64_t r = 0; r < nrect; r++) {
        noc_tlb_2m_t programmed;
        const uint32_t* q = (const uint32_t*)a->rect[r];
        const int mcast = !(q[0] == q[2] && q[1] == q[3]);
        volatile void* win = noc_configure_tlb_2m_mcast(
            (uint32_t)r,
            q[0],
            q[1],
            q[2],
            q[3],
            0,
            (noc_ordering_t)a->ordering,
            mcast,
            0u,
            &programmed
        );
        if (win == 0 || noc_verify_tlb_2m((uint32_t)r, &programmed) != 0) {
            x280_hart_set_result(hartid, ~2ull);
            return 0;
        }
        cmd[r] = (volatile uint32_t*)((uintptr_t)win + cmd_addr);
    }
    const int done_to_host = a->done_noc_addr != 0;
    if (done_to_host) { /* window 2: unicast to the PCIe tile, completion count lands in
                           pinned host memory */
        noc_tlb_2m_t programmed;
        volatile void* win = noc_configure_tlb_2m_mcast(
            2u,
            (uint32_t)a->done_pcie_x,
            (uint32_t)a->done_pcie_y,
            (uint32_t)a->done_pcie_x,
            (uint32_t)a->done_pcie_y,
            a->done_noc_addr,
            NOC_ORDER_POSTED,
            0,
            0u,
            &programmed
        );
        if (win == 0 || noc_verify_tlb_2m(2u, &programmed) != 0) {
            x280_hart_set_result(hartid, ~5ull);
            return 0;
        }
        hdone = (volatile uint32_t*)win;
    }

    int ok = 1;
    uint64_t t0 = x280_mtime();
    if (scan == 2) { /* Tensix barrier: kick command 1, then watch the leader's slot */
        for (uint64_t r = 0; r < nrect; r++) {
            *cmd[r] = 1;
        }
        __asm__ volatile("fence ow, ow" ::: "memory");
        ok = wait_all(c, acks, 1, sw, 1, 0);
        t0 = x280_mtime();
        ok = ok && wait_all(c, acks, 1, sw, (uint32_t)n, 0);
        const uint64_t t1 = x280_mtime();
        for (uint64_t r = 0; r < nrect; r++) {
            noc_clear_tlb_2m((uint32_t)r);
        }
        /* n - 1 barrier steps between the two samples */
        x280_hart_set_result(hartid, ok ? (t1 - t0) : ~3ull);
        return 0;
    }
    for (uint32_t i = 1; i <= n && ok; i++) {
        if (hring) {
            uint32_t spins = 0;
            for (;;) {
                x280_fence_ir();
                if (hring[(i % hdepth) * 16] == i) { /* 64-B host slots */
                    break;
                }
                if (stopped(c, &spins)) {
                    ok = 0;
                    break;
                }
            }
            if (!ok) {
                break;
            }
            if (i == 1) {
                t0 = x280_mtime();
            }
        }
        if (depth > 1 && i > depth) {
            ok = wait_all(c, acks, k, sw, i - (uint32_t)depth, scan);
            if (hdone && ok) {
                *hdone = i - (uint32_t)depth;
            }
        }
        const uintptr_t off =
            depth > 1 ? (uintptr_t)(i % depth) * (uintptr_t)a->ring_stride : 0u;
        for (uint64_t r = 0; r < nrect; r++) {
            volatile uint32_t* p = (volatile uint32_t*)((uintptr_t)cmd[r] + off);
            if (store_mode == 1) {
                store_line(p, i);
            } else {
                *p = i;
            }
        }
        __asm__ volatile("fence ow, ow" ::: "memory");
        mb[4] = i; /* last issued command, for post-mortem */
        if (a->issue_gap) {
            const uint64_t until = x280_mtime() + a->issue_gap;
            while (x280_mtime() < until) {
            }
        }
        if (depth == 1) {
            ok = wait_all(c, acks, k, sw, i, scan);
            if (hdone && ok) {
                *hdone = i;
            }
        }
    }
    if (ok && depth > 1) {
        ok = wait_all(c, acks, k, sw, (uint32_t)n, scan);
        if (hdone && ok) {
            *hdone = (uint32_t)n;
        }
    }
    const uint64_t t1 = x280_mtime();
    __asm__ volatile("fence ow, ow" ::: "memory");
    for (uint64_t r = 0; r < nrect; r++) {
        noc_clear_tlb_2m((uint32_t)r);
    }
    if (done_to_host) {
        noc_clear_tlb_2m(2u);
    }
    x280_hart_set_result(hartid, ok ? (t1 - t0) : ~3ull);
    return 0;
}
