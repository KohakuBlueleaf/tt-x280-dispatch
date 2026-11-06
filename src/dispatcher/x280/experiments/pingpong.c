/* Tensix<->X280 ping-pong (hart 0 only). Args: run_value, mailbox (LIM addr), n,
 * tensix_x, tensix_y (translated NOC0). Mailbox u32 words: [0] seq from Tensix, [1]
 * Tensix reply L1 addr. Result: mtime ticks for rounds 2..n, or error ~k.
 */
#include <stdint.h>

#include "noc.h"
#include "rt/x280_hw.h"
#include "rt/x280_idle.h"
#include "x280_hart_protocol.h"

typedef struct {
    uint64_t run_value;
    uint64_t mailbox;
    uint64_t n;
    uint64_t tensix_x;
    uint64_t tensix_y;
} ttx_pp_args_t;

/* Hart 0 answers n Tensix pings through the mailbox. */
int main(uint64_t hartid) {
    if (hartid != 0) {
        return 0;
    }
    const uint64_t run = x280_hart_control(hartid)->run_value;
    const volatile ttx_pp_args_t* args =
        (const volatile ttx_pp_args_t*)x280_hart_args(hartid);
    if (args == 0 || args->run_value != run) {
        x280_hart_set_result(hartid, ~0ull);
        return 0;
    }
    volatile uint32_t* mb = (volatile uint32_t*)(uintptr_t)args->mailbox;
    const uint64_t n = args->n;

    for (;;) { /* wait for the Tensix reply address */
        x280_fence_ir();
        if (mb[1] != 0) {
            break;
        }
        if (x280_hart_stop_requested(hartid, run)) {
            return 0;
        }
    }
    const x280_noc_coord_t at = {(uint32_t)args->tensix_x, (uint32_t)args->tensix_y};
    volatile void* win = 0;
    if (noc_open_window_2m(0, at, 0, NOC_ORDER_POSTED, &win) != NOC_WINDOW_OPEN) {
        x280_hart_set_result(hartid, ~2ull);
        return 0;
    }
    volatile uint32_t* reply = (volatile uint32_t*)((uintptr_t)win + mb[1]);

    uint64_t t0 = 0;
    for (uint32_t i = 1; i <= n; i++) {
        if (!x280_hart_wait_word_or_stop(hartid, run, mb, i)) {
            noc_clear_tlb_2m(0);
            return 0;
        }
        if (i == 1) {
            t0 = x280_mtime();
        }
        *reply = i;
        __asm__ volatile("fence ow, ow" ::: "memory");
    }
    const uint64_t t1 = x280_mtime();
    noc_clear_tlb_2m(0);
    x280_hart_set_result(hartid, t1 - t0);
    return 0;
}
