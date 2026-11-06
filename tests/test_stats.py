"""Error statistics against a reference."""

import torch
from tt_x280.stats import error_stats


def test_exact_result():
    ref = torch.randn(64, 64, dtype=torch.float64)
    stats = error_stats(ref.clone(), ref)
    assert stats["rel_rms"] == 0 and stats["max_abs"] == 0 and stats["pcc"] > 0.999999


def test_bias_and_scale():
    ref = torch.randn(256, 256, dtype=torch.float64)
    stats = error_stats(ref + 0.01, ref)

    # Bias and rel-RMS are relative to the reference RMS, to 4 significant digits.
    assert abs(stats["mean_bias"] - 0.01 / stats["ref_rms"]) < 1e-4
    assert abs(stats["rel_rms"] - 0.01 / stats["ref_rms"]) < 1e-4
