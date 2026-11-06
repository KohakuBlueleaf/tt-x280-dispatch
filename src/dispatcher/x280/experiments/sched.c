/* X280 command scheduler (hart 0): broadcasts command i = 1..n to K persistent Tensix
 * kernels through up to 2 multicast windows and waits for all K acks per command.
 * Result: mtime ticks for rounds 2..n, or error ~k. */
#include <stdint.h>

#include "noc.h"
#include "rt/x280_hw.h"
#include "rt/x280_idle.h"
#include "x280_hart_protocol.h"

#define TTX_READY 0xFFFFFFFFu

typedef struct {
    uint64_t run_value;
    uint64_t
        mailbox;   /* u32 [1] = Tensix command word L1 address (same on every core) */
    uint64_t acks; /* K u32 ack slots in LIM */
    uint64_t n;
    uint64_t k;
    uint64_t nrect;
    uint32_t rect[2][4]; /* x0, y0, x1, y1 translated NOC0 */
} ttx_sched_args_t;

/* Waits until all k ack slots reach `value`; 0 if stopped. */
static int wait_all(
    uint64_t hartid,
    uint64_t run,
    volatile const uint32_t* acks,
    uint64_t k,
    uint32_t value
) {
    for (uint64_t j = 0; j < k; j++) {
        uint32_t spins = 0;
        for (;;) {
            x280_fence_ir();
            if (acks[j * 4] == value) { /* ack slots are 16 B apart */
                break;
            }
            if (++spins == 4096u) {
                spins = 0;
                if (x280_hart_stop_requested(hartid, run)) {
                    return 0;
                }
            }
        }
    }
    return 1;
}

/* Hart 0 broadcasts commands 1..n and waits for every ack each time. */
int main(uint64_t hartid) {
    if (hartid != 0) {
        return 0;
    }
    const uint64_t run = x280_hart_control(hartid)->run_value;
    const volatile ttx_sched_args_t* a =
        (const volatile ttx_sched_args_t*)x280_hart_args(hartid);
    if (a == 0 || a->run_value != run || a->nrect < 1 || a->nrect > 2) {
        x280_hart_set_result(hartid, ~0ull);
        return 0;
    }
    volatile uint32_t* mb = (volatile uint32_t*)(uintptr_t)a->mailbox;
    volatile uint32_t* acks = (volatile uint32_t*)(uintptr_t)a->acks;
    const uint64_t n = a->n, k = a->k, nrect = a->nrect;

    if (!x280_hart_wait_word_or_stop(
            hartid, run, &mb[0], 1u
        )) { /* leader published mb[1] then set mb[0]=1 */
        return 0;
    }
    if (!wait_all(hartid, run, acks, k, TTX_READY)) {
        return 0;
    }
    const uint32_t cmd_addr = mb[1];
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
            NOC_ORDER_POSTED,
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

    uint64_t t0 = 0;
    int ok = 1;
    for (uint32_t i = 1; i <= n && ok; i++) {
        if (i == 2) {
            t0 = x280_mtime();
        }
        for (uint64_t r = 0; r < nrect; r++) {
            *cmd[r] = i;
        }
        __asm__ volatile("fence ow, ow" ::: "memory");
        ok = wait_all(hartid, run, acks, k, i);
    }
    const uint64_t t1 = x280_mtime();
    for (uint64_t r = 0; r < nrect; r++) {
        noc_clear_tlb_2m((uint32_t)r);
    }
    x280_hart_set_result(hartid, ok ? (t1 - t0) : ~3ull);
    return 0;
}
