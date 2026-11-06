// Persistent worker for the X280 scheduler: reports ready, then for i = 1..n waits for
// command i in L1 and acks it. Runtime args: l2cpu_x, l2cpu_y, mailbox LIM addr, ack
// base LIM addr, worker index, n, leader flag.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"

// Tensix side of the first X280 scheduler: wait, ack, repeat.
void kernel_main() {
    const uint32_t l2_x = get_arg_val<uint32_t>(0);
    const uint32_t l2_y = get_arg_val<uint32_t>(1);
    const uint32_t mailbox = get_arg_val<uint32_t>(2);
    const uint32_t ack_base = get_arg_val<uint32_t>(3);
    const uint32_t idx = get_arg_val<uint32_t>(4);
    const uint32_t n = get_arg_val<uint32_t>(5);
    const uint32_t leader = get_arg_val<uint32_t>(6);

    const uint32_t base = get_write_ptr(0);
    volatile tt_l1_ptr uint32_t* cmd =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base);
    volatile tt_l1_ptr uint32_t* msg =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base + 64);
    cmd[0] = 0;

    // NOC writes need matching 16 B alignment of source and destination: ack slots are
    // 16 B apart.
    const uint64_t ack = get_noc_addr(l2_x, l2_y, ack_base + idx * 16);
    msg[0] = 0xFFFFFFFFu;
    noc_async_write(base + 64, ack, 4);
    noc_async_write_barrier();
    if (leader) {
        volatile tt_l1_ptr uint32_t* hello =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base + 128);
        hello[0] = 1;
        hello[1] = base;
        noc_async_write(base + 128, get_noc_addr(l2_x, l2_y, mailbox), 8);
        noc_async_write_barrier();
    }
    for (uint32_t i = 1; i <= n; i++) {
        uint32_t spins = 0;
        do {
            invalidate_l1_cache();
            if (++spins == (1u << 28)) {
                return;
            }
        } while (cmd[0] != i);
        msg[0] = i;
        noc_async_write(base + 64, ack, 4);
        noc_async_write_barrier();
    }
}
