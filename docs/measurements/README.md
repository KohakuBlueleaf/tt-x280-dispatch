# Measurements

Two Blackhole p150a cards (FW 19.13.1, KMD 2.12.0, Tensix harvesting 0b11000000), X280
PLL4 at 1750 MHz unless a row says otherwise. X280 rows: 120 Tensix in slow dispatch, time
from the X280's 50 MHz clock (first command issued to last completion). ttnn rows: v0.81.1,
110 Tensix in fast dispatch, wall time over a synchronized loop (eager) or a replayed trace.
bf16 with fp32 accumulation, HiFi2 unless a row says otherwise.

| File | Holds | From |
|---|---|---|
| [data/disp_pll.csv](data/disp_pll.csv) | µs per empty command by completion model and X280 clock | `bench/run_disp.sh`, `bench/sweep_disp.py` |
| [data/matmul_e7.csv](data/matmul_e7.csv) | matmul µs: flat / grouped / ring 4 against ttnn | `hosts/matmul.cpp`, `bench/ttnn_matmul.py` |
| [data/matmul_pll.csv](data/matmul_pll.csv) | matmul µs and TFLOPS by X280 clock | `hosts/matmul.cpp` |
| [data/mlp.csv](data/mlp.csv), [data/ttnn_mlp.csv](data/ttnn_mlp.csv) | MLP forward µs, TFLOPS, rel-RMS | `hosts/mlp.cpp`, `bench/ttnn_mlp.py` |
| [data/hostfed.csv](data/hostfed.csv) | host submission paths and host-fed MLP | `hosts/mlp.cpp` (`TTX_HOSTFED`), `examples/mlp.py --host-fed` |
| [data/flash_attn.csv](data/flash_attn.csv) | flash attention µs, error stats, against ttnn SDPA | `examples/flash_attn.py`, `bench/ttnn_sdpa.py` |
| [data/resblock.csv](data/resblock.csv) | ResNet block µs, error stats, against ttnn conv2d | `examples/resblock.py`, `bench/ttnn_resblock.py` |
| [traces/](traces) | per-core command traces (`tt_x280.trace.Trace.load`) | `examples/* --trace` |

Error statistics (`tt_x280.stats.error_stats`) compare against an fp64 reference computed
from the bf16-rounded inputs, with bf16-rounded intermediates where the device stores
them in bf16: rel-RMS and mean bias are relative to the reference RMS.

The figures in [../assets](../assets) are drawn from these numbers by
`scripts/docs/plots.py` (with `scripts/docs/results.py`) and `scripts/docs/diagrams.py`.
