# tt-x280-dispatch model v1 (locked 2026-11-07)

The convention every resident kernel uses to be driven by the X280. Everything below is measured on
2× p150a (FW 19.13.1, tt-metal v0.81.1 source build, slow dispatch, 12×10 = 120 Tensix, X280 PLL4 at 1750 MHz).
Evidence and the experiments that led here: `docs/08-x280.md` E4–E7.

## Roles

| agent | does |
|---|---|
| X280 hart 0 (`ttx-sequencer.bin`) | issues command `seq = 1, 2, …` by storing `seq` into every core's command ring; waits for group completions before issuing commands it must not overlap |
| every Tensix core (resident kernel) | waits until its ring slot holds ≥ `seq`, runs op `seq`, reports completion to its group leader |
| group leader (one per column, bottom core) | waits until all members of its column completed `seq`, then writes `seq` to its ack slot in X280 LIM |
| host | builds the program (resident kernels + args), opens the X280 runtime, launches both, reads results |

## Command path (X280 → Tensix)

- **Two 2 MiB TLB multicast windows, posted stores, NOC0.**
- **Rectangles must be physically contiguous Tensix-only spans in NOC0 coordinates** of the specific chip. Derive them
  from `soc_desc.translate_coord_to(TRANSLATED → NOC0)`; split wherever NOC0 x is not contiguous. On our harvested
  cards: logical columns 0–5 | 6–11 (translated 1–6 | 7–14 → NOC0 1–6 | 11–16). *Never* use a translated rectangle
  that physically spans harvested, L2CPU or DRAM columns — that was the cause of every lost/corrupted command.
- Command word: `uint32 seq` written to `ring[(seq % D) * 16 B]` in each core's control block (same L1 address on all
  cores, published to the X280 by the leader's hello). Cores compare **≥** (signed difference), so re-delivery is
  harmless.
- Ring depth `D`: **1 for dependent ops** (op k+1 reads op k's output), **4 for independent commands** (repeat of the
  same op, independent requests).

## Completion path (Tensix → X280)

- **Column groups**: 12 groups of 10 cores; leader = bottom core of each column; X280 sweeps 12 LIM slots (16 B apart).
- Members report by a 4 B NOC write of `tag << 16 | (seq & 0xFFFF)` into their own 16 B gather slot in the leader's
  L1 (`ctl + 384 + 16·rank`); the leader waits until every slot carries the tag and `≥ seq`, then writes `seq` to LIM.
  (Atomic increments into a leader semaphore are an equally reliable alternative, ~0.1 µs faster: 0.68 vs 0.77 µs.)
- NOC writes keep **matching 16 B alignment** of source and destination.

## Start-up handshake

1. Every core clears its command ring.
2. Members write a check-in `tag << 16 | 0` to their leader.
3. The leader waits for all check-ins, then writes READY (`0xFFFFFFFF`) to its LIM slot; core (0,0) also writes the
   hello `{1, ctl}` to the X280 mailbox.
4. The X280 issues command 1 only after the hello and all 12 READY slots.
- `tag` is a **per-process random 16-bit value**: L1 is not cleared between programs/processes; reusing tags lets stale
  slots from the previous run look like valid check-ins (observed bug).

## Measured cost (120 cores, X280 at 1750 MHz)

| | µs per command |
|---|---|
| empty command, synchronous step (D = 1) | 0.77 |
| empty command, independent stream (D = 4) | 0.46 |
| for comparison: one ack per core (old E4/E5 scheme) | 2.55 |

| BF16 matmul (HiFi2, fp32 acc), one command per matmul | v1, D = 4 | v1, D = 1 | ttnn trace, 110 cores |
|---|---|---|---|
| 320×384×256 | **6.2 µs** | 8.5 µs | 9.0 µs |
| 640×768×1024 | **24.9 µs** | 30.1 µs | 31.8 µs |
| 1280×1536×2048 | **96.3 µs** | 108 µs | 98.6 µs |
| 2560×3072×4096 | **410 µs (157 TF)** | 416 µs | 464 µs (139 TF) |

Reliability: 16 M+ dispatch commands across both cards with zero failures after the rectangle fix.

## Rules / gotchas

- X280 PLL4: set to 1750 MHz once per session (`X280Runtime::set_pll`); exclusive ownership of the card required.
- Debug the Tensix kernels **without** the X280 first (watcher cannot run while the X280 runtime is open).
- A hung resident kernel needs `tt-smi -r <card>` + PCIe retrain (`setpci … CAP_EXP+10.w=0020:0020` on the card's
  root port: c0:01.1 card 0, c0:01.2 card 1). A killed process leaves the X280 in reset-required state.
- Never edit kernel sources while a run is in progress (kernels JIT-compile from source at launch).

MLP forward (GEMM → SiLU → GEMM, three dependent commands, written with the template): 19.1 µs vs ttnn trace 25.3 µs
(320×256 → 384 → 384), 61.1 vs 65.8 µs (640×1024 → 768 → 768). X280 PLL4: supported settings 15 / 200 / 800 / 1000 /
1750 MHz; with v1 real ops are within ~5 % between 800 and 1750 MHz.

## How to write a kernel for this model

Use `src/dispatcher/device/ttx_dispatch.hpp` (device side) and the Python front-end
`src/frontend/tt_x280` (or `src/dispatcher/host/ttx_dispatch_host.hpp` from C++); guide in
[writing-kernels.md](writing-kernels.md), example `src/kernels/mlp/` + `examples/mlp.py`.
