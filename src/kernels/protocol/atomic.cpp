// Atomic-increment stress: every member issues `k`
// noc_semaphore_inc(+1) to the leader's semaphore; the leader waits for members * k and
// writes {count, expected, wall ticks, timed_out} to DRAM page 0.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"

// Increments one semaphore n times, for the atomics stress test.
void kernel_main() {
    uint32_t a = 0;
    const uint32_t lead_x = get_arg_val<uint32_t>(a++);
    const uint32_t lead_y = get_arg_val<uint32_t>(a++);
    const uint32_t is_leader = get_arg_val<uint32_t>(a++);
    const uint32_t members = get_arg_val<uint32_t>(a++);
    const uint32_t k = get_arg_val<uint32_t>(a++);
    const uint32_t mode = get_arg_val<uint32_t>(
        a++
    );  // 0 non-posted, 1 posted, 2 non-posted + atomic barrier each
    const uint32_t sem = get_semaphore(get_arg_val<uint32_t>(a++));
    const uint32_t out_addr = get_arg_val<uint32_t>(a++);
    const uint32_t is_noise = get_arg_val<uint32_t>(
        a++
    );                  // multicasts `k` words into every core's scratch + 16
    uint32_t nrect[5];  // x0, y0, x1, y1, num_dests (NOC0 order)
    for (uint32_t j = 0; j < 5; j++) {
        nrect[j] = get_arg_val<uint32_t>(a++);
    }
    const uint32_t scratch = get_write_ptr(0);
    volatile tt_l1_ptr uint32_t* noise_word =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch + 16);
    noise_word[0] = 0;

    if (is_noise) {
        volatile tt_l1_ptr uint32_t* src =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch + 32);
        const uint64_t dst = get_noc_multicast_addr(
            nrect[0], nrect[1], nrect[2], nrect[3], scratch + 16
        );
        for (uint32_t i = 1; i <= k; i++) {
            src[0] = i;
            noc_semaphore_set_multicast(scratch + 32, dst, nrect[4]);
            noc_async_write_barrier();
        }
        return;
    }
    if (!is_leader) {
        const uint64_t dst = get_noc_addr(lead_x, lead_y, sem);
        for (uint32_t i = 0; i < k; i++) {
            if (mode == 1) {
                noc_semaphore_inc<true>(dst, 1);
            } else {
                noc_semaphore_inc(dst, 1);
                if (mode == 2) {
                    noc_async_atomic_barrier();
                }
            }
        }
        noc_async_atomic_barrier();
        return;
    }
    volatile tt_l1_ptr uint32_t* s =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem);
    const uint32_t expected = members * k;
    const uint64_t t0 = get_timestamp();
    uint32_t timed_out = 0;
    uint32_t spins = 0;
    while (true) {
        invalidate_l1_cache();
        if (*s >= expected) {
            break;
        }
        if (++spins == (1u << 27)) {
            timed_out = 1;
            break;
        }
    }
    const uint64_t t1 = get_timestamp();
    riscv_wait(1000000);  // let any late increments land before the final read
    invalidate_l1_cache();
    volatile tt_l1_ptr uint32_t* o =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch);
    o[0] = *s;
    o[1] = expected;
    o[2] = static_cast<uint32_t>(t1 - t0);
    o[3] = timed_out;
    o[4] = noise_word[0];
    constexpr auto args = TensorAccessorArgs<0>();
    const auto acc = TensorAccessor(args, out_addr);
    noc_async_write_page(0, acc, scratch);
    noc_async_write_barrier();
}
