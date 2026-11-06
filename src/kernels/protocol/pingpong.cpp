// X280 ping-pong, Tensix side: publishes its reply address to the mailbox, then writes
// i = 1..n and waits for the echo. Runtime args: l2cpu_x, l2cpu_y (translated NOC0),
// mailbox LIM address, n.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"

// Tensix side of the X280 round-trip test.
void kernel_main() {
    const uint32_t l2_x = get_arg_val<uint32_t>(0);
    const uint32_t l2_y = get_arg_val<uint32_t>(1);
    const uint32_t mailbox = get_arg_val<uint32_t>(2);
    const uint32_t n = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb = 0;
    const uint32_t base = get_write_ptr(cb);
    volatile tt_l1_ptr uint32_t* reply =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base);
    volatile tt_l1_ptr uint32_t* msg =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base + 64);
    reply[0] = 0;

    const uint64_t dst = get_noc_addr(l2_x, l2_y, mailbox);
    msg[0] = 0;
    msg[1] = base;
    noc_async_write(base + 64, dst, 8);
    noc_async_write_barrier();

    for (uint32_t i = 1; i <= n; i++) {
        msg[0] = i;
        noc_async_write(base + 64, dst, 4);
        noc_async_write_barrier();
        do {
            invalidate_l1_cache();
        } while (reply[0] != i);
    }
}
