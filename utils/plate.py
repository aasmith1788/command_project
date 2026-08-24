"""
Evaluate a Stage 3 intent density at arbitrary plate coordinates.

Stage 3 writes each pitch's intent posterior as basis coefficients
(`p_knot_0..M-1`) rather than as bin probabilities, because where a pitcher is
aiming is continuous in both axes and binning it at the output step throws that
away. This module turns those coefficients back into a density you can evaluate
anywhere, or sample onto whatever mesh a plot wants.

The density for pitch i is

    f_i(h, s) = sum_m p_knot_m * N(h | k_m_h + base_h_i, bw^2)
                              * N(s | k_m_s + base_s_i, bw^2)

`base_h` / `base_s` are stored per pitch and carry the population aim and batter
adjustment, so the surface sits in real plate coordinates.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import (
    DENSITY_BW, DENSITY_EXTENT_H, DENSITY_EXTENT_S, DENSITY_KNOTS,
)


def knot_grid() -> np.ndarray:
    """(M, 2) knot centres in residual space — must match stage3.build_density_basis."""
    kh = np.linspace(-DENSITY_EXTENT_H, DENSITY_EXTENT_H, DENSITY_KNOTS[0])
    ks = np.linspace(-DENSITY_EXTENT_S, DENSITY_EXTENT_S, DENSITY_KNOTS[1])
    return np.array([(a, b) for a in kh for b in ks], dtype=float)


def knot_weights(rows: pd.DataFrame) -> np.ndarray:
    """(n_rows, M) coefficient matrix from an aim_distribution frame."""
    cols = sorted(
        (c for c in rows.columns if c.startswith("p_knot_")),
        key=lambda c: int(c.split("_")[-1]),
    )
    if not cols:
        raise ValueError("no p_knot_* columns — is this a V10 aim_distribution file?")
    return rows[cols].to_numpy(float)


def density_at(rows: pd.DataFrame, height: np.ndarray,
               side: np.ndarray, average: bool = True) -> np.ndarray:
    """
    Density at the given plate coordinates.

    `height` and `side` are broadcast against each other, so pass a meshgrid for
    a surface or two scalars for a point. With `average=True` the rows are
    averaged into one surface (the natural thing for "this pitcher's intent on
    sliders to lefties"); otherwise one surface per row is returned.

    Normalised to integrate to 1 over whatever mesh is passed, so it is directly
    comparable across pitchers and plottable as-is.
    """
    K = knot_grid()
    W = knot_weights(rows)
    h = np.asarray(height, dtype=float)
    s = np.asarray(side, dtype=float)

    base_h = rows["base_h"].to_numpy(float)[:, None]
    base_s = rows["base_s"].to_numpy(float)[:, None]
    # base_s and the knot side offsets are both in the mirrored frame, so the
    # sign is applied to their SUM — flipping either alone would put a lefty's
    # surface on the wrong side of the plate.
    sign = (rows["mirror_sign"].to_numpy(float)
            if "mirror_sign" in rows.columns else np.ones(len(rows)))

    # (n_rows, M, ...) would blow up memory on a fine mesh, so accumulate.
    shape = np.broadcast(h, s).shape
    out = np.zeros((1 if average else len(rows),) + shape)

    for m in range(K.shape[0]):
        mu_h = base_h[:, 0] + K[m, 0]
        mu_s = sign * (base_s[:, 0] + K[m, 1])
        for r in range(len(rows)):
            if W[r, m] <= 1e-12:
                continue
            bump = np.exp(-0.5 * (((h - mu_h[r]) / DENSITY_BW) ** 2
                                  + ((s - mu_s[r]) / DENSITY_BW) ** 2))
            out[0 if average else r] += W[r, m] * bump

    if average:
        out = out[0] / max(len(rows), 1)
    total = out.sum()
    return out / total if total > 0 else out


def surface_on_grid(rows: pd.DataFrame, n: int = 81,
                    pad: float = 0.8) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convenience wrapper: (grid_h, grid_s, density) over the region the rows
    actually occupy, padded so the tails are visible.

    Returns density with shape (n_height, n_side), ready for pcolormesh.
    """
    lo_h = rows["base_h"].min() - DENSITY_EXTENT_H - pad
    hi_h = rows["base_h"].max() + DENSITY_EXTENT_H + pad
    sgn = (rows["mirror_sign"].to_numpy(float)
           if "mirror_sign" in rows.columns else np.ones(len(rows)))
    real_s = sgn * rows["base_s"].to_numpy(float)
    lo_s = real_s.min() - DENSITY_EXTENT_S - pad
    hi_s = real_s.max() + DENSITY_EXTENT_S + pad

    gh = np.linspace(lo_h, hi_h, n)
    gs = np.linspace(lo_s, hi_s, n)
    H, S = np.meshgrid(gh, gs, indexing="ij")
    return gh, gs, density_at(rows, H, S, average=True)
