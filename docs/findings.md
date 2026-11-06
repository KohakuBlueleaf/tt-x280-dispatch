# Findings that shaped the dispatch model

## 1. Multicast rectangles must be physically Tensix-only (root cause of every lost command)

On a harvested Blackhole, translated coordinates are not physical. Our p150a cards have Tensix harvesting mask
`0b11000000` → NOC0 columns 7 and 10 disabled; translation maps translated x 1–6 → NOC0 1–6, **translated 7 → NOC0 11**,
10–14 → 12–16. A "safe-looking" translated rectangle x 1–7 is physically NOC0 1–11: two harvested columns, the
**L2CPU column 8 (including the X280's own tile (8,3))** and the DRAM column 9.

X280 TLB-window multicasts into that rectangle intermittently delivered nothing to most of it, or wrote 0 into the
command slot of the cores in the L2CPU's row — every ~10⁵–10⁶ commands, deterministically under bursts. Posted vs
non-posted, 4 B vs 64 B stores, issue gaps and window order only changed the rate. Splitting into physically contiguous
spans (translated 1–6 | 7–14) removed it: **0 failures in 16 M+ commands**, posted stores, ring depth 16.
Tensix-originated non-posted multicasts over the same span happened to work (relay, E5 matmul).

Rule: derive multicast rectangles from `soc_desc.translate_coord_to(TRANSLATED → NOC0)` per chip; split where NOC0 x is
not contiguous. `DispatchPlan::make` does this.

## 2. Check-in tags must be unique per process (our bug)

L1 survives between programs. A leader that recognised check-ins by `run & 0xFFFF` (same in every process) took the
previous process's slots as valid, reported READY before its members started, and late members wiped the first command
by clearing their ring (or their real check-in overwrote a stale "step 20000"). Fix: per-process random tag.
Atomic aggregation is immune (semaphores are re-initialised at launch).

## 3. Tensix atomics are reliable

119 cores × 100 000 `noc_semaphore_inc` into one semaphore, posted / non-posted / non-posted + barrier, NOC0 / NOC1,
with and without concurrent multicast traffic into the same L1: 0 lost in ~170 M; ~12 ticks per increment at the
endpoint. (Blackhole has an erratum for *inline writes and atomics into L1 under port back-pressure* — tt-metal emulates
inline writes; we did not hit it.)

## 4. What tt-metal's own fast dispatch does (for comparison)

`dispatch_s` (Tensix on the reserved column) sends each go signal as one **non-posted** multicast from L1 and counts
acks; workers report completion with a posted write into a **stream register** on the dispatcher (hardware counter);
the dispatcher waits for the count before the next go. Our v1 matches that shape (one multicast per command, counted
completion, barrier before dependent work) but moves the sequencer to the X280 and frees the dispatch column (120 cores).

## 5. Practical

- A failed case can poison later cases in the same process → one process per measurement.
- Keep `tilize` init and uninit around each block in the conv compute: one init for nine
  back-to-back tilize blocks (`InitOnly` / `Neither` / `UninitOnly`) produced wrong
  results there.
- The ResNet-block example does not complete with 16 commands (`--forwards 8`) and does
  with 64; the MLP and flash attention complete at any count. Open.
- Kernels JIT-compile from source at launch → never edit kernel sources while a run is in progress.
- Watcher cannot run while the X280 runtime is open → debug kernels standalone first.
- Killed processes leave the X280 in reset-required state → `tt-smi -r <card>` + PCIe retrain (both cards come back at
  Gen1 after a reset).
- X280 PLL4 boots at 200 MHz (`kSafePllMhz`); set it once per session (1750 MHz used throughout).
- Host → X280 submission: `X280Tile::write_available_lim` ≈ 1.6 ms per call (full state inspection);
  `tt::Cluster::write_core` to the L2CPU ≈ 4–7 µs; a persistent write-combining UMD IoWindow is one store
  plus `sfence`, 0.63 µs. The X280 publishes its done count into pinned host memory (a host read of LIM
  costs 23–42 µs).
