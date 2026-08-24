"""
Shared scoring utilities used by Stage 2 and Stage 3.
Computes mu_pop (population aim) and sigma_pop (execution spread) from Stage 1 results.

V9 — Stage 1's movement term is a spline surface, not a per-cluster lookup, so
both functions evaluate a stored basis instead of doing `.map(dict)`. The basis
spec (knots, centring, kept columns) travels inside the Stage 1 pickle, so the
surface is never re-derived from Stage 2/3's own data — re-fitting knots on a
different frame would shift the basis and with it every pitch's aim, silently.
"""

import numpy as np
import pandas as pd

from utils.matchup import add_matchup, type_x_matchup
from utils.splines import FEAT_VALID, add_spline_features, combo_frame

S1_VERSION = "v11-velo-height-only"


def _check(s1: dict) -> None:
    """
    Refuse a Stage 1 pickle this module cannot read.

    A V8 pickle has `u_cxb` and no `spec_move`. Reading it here would send every
    surface lookup to zero and every sigma to the intercept, which does not
    raise — it just produces a season of population-mean aim and a leaderboard
    that looks entirely plausible. Fail loudly instead.
    """
    version = s1.get("version")
    if version != S1_VERSION:
        raise ValueError(
            f"Stage 1 output is version {version!r}, expected {S1_VERSION!r}. "
            f"Re-run Stage 1 — the movement term changed from KMeans clusters "
            f"to a spline surface and the two pickles are not interchangeable."
        )


def prepare_features(df: pd.DataFrame, s1: dict) -> pd.DataFrame:
    """
    Attach the spline predictor columns, mirroring exactly as Stage 1 did.

    Mirroring is unconditional and defined once in utils.matchup, so there is
    no flag that can drift between the fit and the scoring pass.
    """
    _check(s1)
    if "mirror_sign" not in df.columns:
        df = add_matchup(df)
    return add_spline_features(df)


def compute_mu_pop(df: pd.DataFrame, s1: dict) -> np.ndarray:
    """
    Population aim in original units (feet) for each pitch.

    Unknown categorical levels default to 0 (partial-pool to grand mean), and a
    pitch with untracked movement gets a zero surface contribution via
    FEAT_VALID — the same "fall back to the population mean" behaviour the old
    unknown-cluster path had, but now it is the ONLY fallback, and it no longer
    leaks into sigma.
    """
    _check(s1)
    if FEAT_VALID not in df.columns:
        df = prepare_features(df, s1)

    # The surface is side-independent. A side-gated deviation surface was
    # fitted, measured as unidentified (tau_dev HDI touching zero, coefficients
    # sitting at their N(0,1) prior), and removed. Batter stance enters through
    # u_tps, which is indexed on it.
    # Evaluated at (pitcher x pitch type) resolution and gathered back, exactly
    # as Stage 1 fitted it. Doing it per pitch here would score every pitch
    # against a surface point Stage 1 never fitted.
    combo, cid = combo_frame(df)
    f = s1["spec_move"].transform(combo) @ np.asarray(s1["w_move"])
    # The velo smooth is fitted for height only — it is unidentified for side
    # once movement is in the surface (see stage1.py). A side pickle carries
    # spec_velo=None and contributes nothing here.
    if s1.get("spec_velo") is not None:
        f = f + s1["spec_velo"].transform(combo) @ np.asarray(s1["w_velo"])
    f = f[cid]

    mu_z = (
        s1["alpha"]
        + f
        + df["type_x_matchup"].map(s1["u_tps"]).fillna(0.0).values
        + df["type_x_count"].map(s1["u_tc"]).fillna(0.0).values
    )
    return mu_z * s1["y_sd"] + s1["y_mean"]


def compute_sigma_pop(df: pd.DataFrame, s1: dict) -> np.ndarray:
    """
    Per-pitch sigma in feet from Stage 1's log-linear sigma submodel.

    The movement term is a coarse spline surface; pitch type and count stay
    categorical. An unseen pitch type or count contributes 0 (the intercept's
    spread) rather than inheriting an arbitrary reference level's — which is
    the V8 bug this rewrite closes.
    """
    _check(s1)
    if FEAT_VALID not in df.columns:
        df = prepare_features(df, s1)

    combo, cid = combo_frame(df)
    B_sigma = s1["spec_sigma"].transform(combo)

    log_sigma_z = (
        s1["sigma_intercept"]
        + (B_sigma @ np.asarray(s1["beta_sigma_move"]))[cid]
        + df["pitch_type"].map(s1["sigma_typ"]).fillna(0.0).values
        + df["count"].map(s1["sigma_cnt"]).fillna(0.0).values
        # Execution spread differs by matchup — a pitch to a same-handed batter
        # is not thrown with the same precision as one across the platoon. This
        # term was missing until V10, so any platoon difference in SPREAD was
        # being charged to the pitcher as command.
        + df["matchup"].map(s1["sigma_mu"]).fillna(0.0).values
    )
    return np.exp(log_sigma_z) * s1["y_sd"]   # z-score scale -> feet


def add_stage1_keys(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the categorical keys `compute_mu_pop` and `compute_sigma_pop` read.

    Defined once here because Stage 2 and Stage 3 must construct them
    identically — the same reason `type_x_batter` is commented so heavily in
    both. A mismatch maps every lookup to NaN and quietly zeroes the term.
    """
    if "matchup" not in df.columns:
        df = add_matchup(df)
    else:
        df = df.copy()
    df["type_x_matchup"] = type_x_matchup(df)
    df["type_x_count"]   = df["pitch_type"] + "_" + df["count"]
    return df


def command_plus(avg_miss: np.ndarray) -> np.ndarray:
    """
    Scale avg miss distance to 100 = league average, 10 pts = 1 SD.
    Lower miss → higher Command+.
    """
    m, s = avg_miss.mean(), avg_miss.std()
    return 100 - (avg_miss - m) / s * 10
