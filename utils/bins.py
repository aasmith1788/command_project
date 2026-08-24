"""
Player skill bins — the middle level of the pooling hierarchy.

Both Stage 2 (batters) and Stage 3 (pitchers) shrink a sparse player toward a
group of comparable players rather than toward the league as a whole. The
grouping is an equal-count octile on a single skill proxy:

  - batters  : mean ExitSpeed over batted balls
  - pitchers : mean RelSpeed over Fastball/Sinker pitch types

Bins are quantiles, so each holds ~1/8 of the players and the mph width of a
bin varies (the middle of the distribution is dense, the tails are not).

Players without enough data to place get the dedicated `sparse` bin, which is
always the last index (n_bins). That bin is not a skill group and must not be
read as one — it exists so those players shrink to the global mean instead of
being assigned a skill level nobody measured. Keep it as its own level rather
than folding it into the median bin, which would assert an EV that was never
observed.

IMPORTANT — pass in only the rows for the players you are actually modeling.
Quantiles are computed over whoever appears in the frame, so binning the full
league and filtering afterward gives unequal bins (binning all pitchers then
keeping the 148 qualifiers yields bins of 11-23 instead of 18-19). Filter
first, then bin.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

N_BINS = 8
MIN_BATTED_BALLS = 10     # below this, a batter's mean EV is too noisy to bin on
FASTBALL_TYPES = ("Fastball", "Sinker")   # exact pitch_type values, not prefixes


def _quantile_bins_by(values: pd.Series, groups: pd.Series,
                      n_bins: int) -> pd.Series:
    """
    Equal-count bins computed WITHIN each group, then pooled.

    Binning on the pooled distribution lets the grouping variable leak into the
    bin index. Measured on 2024: lefty batters average 83.45 mph exit velo
    against righties' 82.30, so the lefty share of the pooled octiles drifted
    from 26% in the lowest to 56% in the highest. "Octile 5" then meant a
    different percentile depending on handedness, and the (octile x matchup)
    pooling cells were badly unbalanced at the extremes — 13 batters against 37
    in the lowest one. Binning within the group makes octile k mean "k-th
    eighth of comparable players" and balances the cells by construction.
    """
    out = pd.Series(index=values.index, dtype=int)
    for g in pd.unique(groups):
        m = (groups == g).to_numpy()
        if m.sum() == 0:
            continue
        out.loc[values.index[m]] = _quantile_bins(values[m], n_bins)
    return out


def _quantile_bins(values: pd.Series, n_bins: int) -> pd.Series:
    """
    Equal-count bins, 0-indexed. `duplicates="drop"` guards the degenerate case
    where ties collapse a cut point; it yields fewer than n_bins groups rather
    than raising, which is the right failure mode for a small season.
    """
    codes = pd.qcut(values, n_bins, labels=False, duplicates="drop")
    return codes.astype(int)


def batter_ev_bins(
    df: pd.DataFrame,
    n_bins: int = N_BINS,
    min_batted_balls: int = MIN_BATTED_BALLS,
    by: str | None = "BatterSide",
) -> pd.DataFrame:
    """
    Map every BatterId in `df` to a bin.

    Returns columns: BatterId, ev_bin, ev_mean, n_batted_balls, is_sparse.
    ExitSpeed is only populated on batted balls (~26% of pitches), so the
    batted-ball count, not the pitch count, is what decides sparseness.
    """
    ev = pd.to_numeric(df["ExitSpeed"], errors="coerce")
    tracked = df.assign(_ev=ev).dropna(subset=["_ev"])

    agg = tracked.groupby("BatterId")["_ev"].agg(
        n_batted_balls="count", ev_mean="mean"
    )

    all_ids = pd.Index(df["BatterId"].unique(), name="BatterId")
    out = pd.DataFrame(index=all_ids).join(agg)
    out["n_batted_balls"] = out["n_batted_balls"].fillna(0).astype(int)

    # Handedness of each batter, for within-group binning. Taken as the modal
    # stance so a switch hitter lands in one group rather than two.
    if by is not None and by in df.columns:
        grp = df.groupby("BatterId")[by].agg(lambda s: s.mode().iloc[0])
        out[by] = out.index.map(grp)

    measurable = out["n_batted_balls"] >= min_batted_balls
    out["ev_bin"] = n_bins                      # sparse bin
    if measurable.any():
        if by is not None and by in out.columns:
            out.loc[measurable, "ev_bin"] = _quantile_bins_by(
                out.loc[measurable, "ev_mean"], out.loc[measurable, by], n_bins
            )
        else:
            out.loc[measurable, "ev_bin"] = _quantile_bins(
                out.loc[measurable, "ev_mean"], n_bins
            )
    out["is_sparse"] = ~measurable

    return out.reset_index()


def pitcher_fb_bins(
    df: pd.DataFrame,
    n_bins: int = N_BINS,
    min_fastballs: int = 25,
    by: str | None = "PitcherThrows",
) -> pd.DataFrame:
    """
    Map every PitcherId in `df` to a bin on mean fastball velocity.

    Averages RelSpeed across all Fastball and Sinker pitches — a two-fastball
    pitcher contributes both, weighted by how often he throws each, which is
    what we want for a single "how hard does this guy throw" summary.

    Returns columns: PitcherId, velo_bin, fb_velo, n_fastballs, is_sparse.
    """
    rs = pd.to_numeric(df["RelSpeed"], errors="coerce")
    # Keyed on pitch_type, not the KMeans cluster labels. Since V9 the movement
    # clusters are diagnostic only and may be absent, and "nan".startswith(...)
    # is False — which would have silently emptied every velo octile.
    is_fb = df["pitch_type"].astype(str).isin(FASTBALL_TYPES)

    fb = df.assign(_rs=rs).loc[is_fb].dropna(subset=["_rs"])

    agg = fb.groupby("PitcherId")["_rs"].agg(n_fastballs="count", fb_velo="mean")

    all_ids = pd.Index(df["PitcherId"].unique(), name="PitcherId")
    out = pd.DataFrame(index=all_ids).join(agg)
    out["n_fastballs"] = out["n_fastballs"].fillna(0).astype(int)

    # Bin within pitcher handedness for the same reason batters are binned
    # within stance: lefties throw slightly slower on average, so a pooled
    # octile partly encodes handedness and the (octile x matchup) pooling
    # groups inherit that imbalance.
    if by is not None and by in df.columns:
        grp = df.groupby("PitcherId")[by].agg(lambda s: s.mode().iloc[0])
        out[by] = out.index.map(grp)

    measurable = out["n_fastballs"] >= min_fastballs
    out["velo_bin"] = n_bins                    # sparse bin
    if measurable.any():
        if by is not None and by in out.columns:
            out.loc[measurable, "velo_bin"] = _quantile_bins_by(
                out.loc[measurable, "fb_velo"], out.loc[measurable, by], n_bins
            )
        else:
            out.loc[measurable, "velo_bin"] = _quantile_bins(
                out.loc[measurable, "fb_velo"], n_bins
            )
    out["is_sparse"] = ~measurable

    return out.reset_index()


def describe_bins(binned: pd.DataFrame, value_col: str, bin_col: str) -> pd.DataFrame:
    """Per-bin membership and value range, for logging the split at run time."""
    g = binned.groupby(bin_col)[value_col]
    desc = g.agg(n="count", lo="min", hi="max", mean="mean").reset_index()
    return desc
