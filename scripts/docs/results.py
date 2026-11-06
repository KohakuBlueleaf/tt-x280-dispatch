"""Measured numbers behind the README figures, each with its source file (µs, p150a).

X280: 120 Tensix, X280 PLL4 1750 MHz. ttnn: v0.81.1, 110 Tensix, fast dispatch.
"""

# Dispatch cost per command, empty work (docs/measurements/data/disp_pll.csv).
DISPATCH_MODELS = ["flat, 120 acks", "v1 sync", "v1 ring 4", "row atomics ring 4"]
DISPATCH_PLL = {
    800: [5.09, 1.12, 0.86, 0.79],
    1000: [4.10, 0.99, 0.69, 0.66],
    1750: [2.45, 0.77, 0.46, 0.36],
}
DISPATCH_PLL_200 = {"flat, 120 acks": 19.9, "v1 sync": 1.04, "v1 ring 4": 0.86}

# Matmul, bf16 HiFi2 fp32 acc, us per matmul (docs/measurements/data/matmul_e7.csv).
MATMUL = [
    # (M, N, K, x280 flat sync, x280 v1 sync, x280 v1 ring 4, ttnn trace, ttnn eager)
    (320, 384, 256, 10.0, 8.5, 6.2, 9.0, 89.0),
    (640, 768, 1024, 31.7, 30.1, 24.9, 31.8, 50.0),
    (1280, 1536, 2048, 109.0, 108.0, 96.3, 98.6, 100.0),
    (2560, 3072, 2048, 243.0, 242.0, 237.0, 263.0, 263.0),
    (2560, 3072, 4096, 417.0, 416.0, 410.0, 464.0, 465.0),
]

# MLP forward silu(X W1) W2, three dependent commands (data/mlp.csv, data/ttnn_mlp.csv).
MLP = [
    # (M, K, F, x280 v1, ttnn trace, ttnn eager, x280 rel-RMS, ttnn rel-RMS)
    (320, 256, 384, 19.06, 25.26, 739.5, 8.32e-3, 7.99e-3),
    (640, 1024, 768, 61.05, 65.77, 707.8, 8.34e-3, 7.96e-3),
    (1280, 2048, 1536, 227.9, 198.1, 381.7, 8.93e-3, 7.75e-3),
    (2560, 2048, 3072, 733.9, 730.8, 728.8, 9.30e-3, 7.76e-3),
]

# Flash attention, non-causal, chunks 128 (64 for S=512) (data/flash_attn.csv).
FLASH_ATTN = [
    # (label, x280 first reader, x280 K/V multicast lanes, ttnn trace, ttnn eager)
    ("12×1280×64", 187.3, 97.6, 173.1, 182.6),
    ("8×2048×128", 711.8, 338.3, 340.7, 349.8),
    ("16×512×64", 121.9, 69.7, 74.3, 92.8),
]
FLASH_ATTN_EXACT_EXP = ("12×1280×64, exact exp", 266.0, 537.6)

# ResNet basic block, two dependent convs (data/resblock.csv).
RESBLOCK = [
    # (label, x280 halo reader, x280 per-row reader, ttnn trace, ttnn eager)
    ("1×56×56×64", 75.3, 103.2, 178.3, 1764.8),
    ("2×28×28×128", 73.8, 79.6, 165.0, 1724.3),
    ("1×64×64×64", 99.8, 153.0, 199.6, 1666.6),
]

# Error against an fp64 reference: rel-RMS (data/*.csv).
PRECISION = [
    # (kernel, setting, x280, ttnn)
    ("matmul 2560×3072×4096", "HiFi2", 6.2e-3, 4.0e-3),
    ("MLP 320×256→384", "HiFi2", 8.32e-3, 7.99e-3),
    ("flash attn", "approx exp", 2.43e-2, 2.43e-2),
    ("flash attn", "exact exp", 1.08e-2, 1.08e-2),
    ("ResNet block", "HiFi2", 7.62e-3, 7.59e-3),
    ("ResNet block", "HiFi4", 2.87e-3, None),
]

# Host to X280 submission (data/hostfed.csv).
HOST_IO = [
    # (path, us per command)
    ("runtime LIM write (checked)", 1562.0),
    ("cluster write per call", 5.64),
    ("persistent WC window", 0.63),
]
HOST_FED = [
    # (M, K, F, self-issued, host-fed stream, host-fed sync, ttnn trace, eager)
    (320, 256, 384, 19.06, 29.45, 44.91, 25.3, 739.5),
    (640, 1024, 768, 61.05, 61.15, 74.35, 65.8, 707.8),
]
