# Dispatch models we built and measured

All: X280 hart 0 issues command words; resident Tensix kernels on all 120 cores (slow dispatch grid); X280 PLL4 at
1750 MHz unless stated; physically clean multicast panels (see findings.md). µs = per command, no work per command,
20 000 commands, card 0/1 p150a.

| model | command path | completion path | µs (sync) | µs (ring 4) | status |
|---|---|---|---|---|---|
| **flat** (E4) | X280 multicast | every core writes its own 16 B ack slot in X280 LIM; X280 scans 120 | 2.45–2.55 | 2.24 | works; slow (120 messages into one endpoint) |
| **Tensix barrier** | X280 kick, then Tensix multicast release | 119 atomics into one leader core | 1.45 | — | works; X280 out of the loop |
| **row groups, atomic** | X280 multicast | members `noc_semaphore_inc` row leader; 10 leaders ack LIM | 0.70 | **0.36** (also depth 16) | works |
| **column groups, atomic** | X280 multicast | 9 members → bottom leader; 12 acks | **0.68** | 0.37 | works |
| **column groups, write-gather — v1** | X280 multicast | members write `tag<<16|seq` to own 16 B slot in leader L1 | 0.77 | 0.46 | **chosen convention** (template) |
| **relay** (tt-metal `dispatch_s` shape) | X280 unicast to one Tensix; it does a non-posted Tensix multicast | column write-gather | 1.47 | 1.23 | works; costs one extra hop |

Notes
- The limit is messages landing on one endpoint (~12–20 ns each), not the NOC direction (±5 % for one-word messages).
  ~√120 groups is the sweet spot (columns of 10 beat rows of 12; 20 half-row groups are slower).
- Ring depth > 1 only for independent commands; dependent ops use depth 1.
- Why v1 is write-gather rather than atomics: it carries `seq` explicitly (no cumulative counter semantics to get wrong)
  and is what the matmul/MLP were validated with; atomics are equally reliable and ~0.1 µs faster and can be swapped in.
