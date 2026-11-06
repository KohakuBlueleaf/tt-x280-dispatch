"""Precision statistics of a device result against a float64 reference."""


def error_stats(result, reference):
    """rel_rms, max_abs, mean_bias, pcc and ref_rms of result against reference.

    rel_rms and mean_bias (mean signed error) are relative to the reference RMS; values
    are rounded to 4 significant digits, pcc to 6 decimals.
    """
    result = result.double().flatten()
    reference = reference.double().flatten()
    err = result - reference
    ref_rms = reference.pow(2).mean().sqrt().item()

    # Pearson correlation of the centred vectors.
    centered_r = result - result.mean()
    centered_ref = reference - reference.mean()
    pcc = (centered_r @ centered_ref / (centered_r.norm() * centered_ref.norm())).item()

    return {
        "rel_rms": float(f"{(err.pow(2).mean().sqrt().item() / ref_rms):.3e}"),
        "max_abs": float(f"{err.abs().max().item():.3e}"),
        "mean_bias": float(f"{(err.mean().item() / ref_rms):.3e}"),
        "pcc": round(pcc, 6),
        "ref_rms": float(f"{ref_rms:.3e}"),
    }
