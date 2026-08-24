"""
Stage 2 — Batter x cluster population aim adjustment (PyMC).
Estimates how all pitchers collectively shift their aim for each specific
batter and pitch cluster combination.

Runs on the full dataset (all pitchers) — partial pooling handles sparse batters.

Pooling is three-level: global -> exit-velo octile -> batter x cluster. A batter
with few tracked pitches shrinks toward hitters of comparable exit velocity
rather than toward the league mean, on the premise that pitchers aim differently
against hitters of different caliber. Batters without enough batted balls to
place (see utils.bins) go in a dedicated sparse bin that shrinks to global.

Usage:
    uv run python stage2/stage2.py --year 2024
    uv run python stage2/stage2.py --year 2024 --draws 2000 --target-accept 0.99

Run with --help to see every sampler flag and its default for this stage.
"""

import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm

from config import (
    OUTPUT_DIR, MIN_PITCHES_BATTER_CLUSTER,
    STAGE2_CHAINS, STAGE2_ITER_WARMUP, STAGE2_ITER_SAMPLING,
    STAGE2_ADAPT_DELTA, STAGE2_INIT, MCMC_SEED,
)
from utils.bins import N_BINS, batter_ev_bins
from utils.cli import SamplerArgs, add_sampler_args, resolve_sampler_args
from utils.progress import SamplingProgress, report_diagnostics
from utils.scoring import (add_stage1_keys, compute_mu_pop, compute_sigma_pop,
                           prepare_features)

CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

TAU_RATE_FALLBACK = 5.0

DEFAULT_SAMPLER = SamplerArgs.defaults(
    draws=STAGE2_ITER_SAMPLING, tune=STAGE2_ITER_WARMUP, chains=STAGE2_CHAINS,
    target_accept=STAGE2_ADAPT_DELTA, init=STAGE2_INIT, seed=MCMC_SEED,
)


def load_data(year: int) -> pd.DataFrame:
    df = pd.read_parquet(OUTPUT_DIR / f"clustered_{year}.parquet")
    # pitch_cluster is no longer required: Stage 1's movement term is a spline
    # surface over the raw features, so a pitch that KMeans could not label is
    # still perfectly scoreable. Only an unnameable pitch TYPE is dropped.
    df = df[df["pitch_type"].notna()].copy()
    for col in ["PlateLocHeight", "PlateLocSide"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["PlateLocHeight", "PlateLocSide"])

    # Keys Stage 1's categorical block is indexed on. Built by the shared
    # helper so this stage and Stage 3 cannot drift apart.
    df = add_stage1_keys(df)

    # Pitch type x MATCHUP — the resolution this stage models at. Was pitch
    # type x pitcher hand x batter stance; in the mirrored frame the pairs that
    # are reflections of each other (LHP-LHB with RHP-RHB, LHP-RHB with
    # RHP-LHB) are the same cell, so this halves the levels and doubles the
    # pitches behind each one without losing any distinction that exists.
    df["type_x_batter"] = df["pitch_type"] + "_" + df["matchup"]
    return df


# Cells thinner than this still get an omega — they are simply not used to
# estimate how wide the omega prior should be. See compute_tau_rate.
MIN_CELLS_FOR_RATE = 20


def compute_tau_rate(residuals: np.ndarray, b_id: np.ndarray, B: int) -> float:
    """
    Empirical Exponential rate for a level's scale, from its between-group SD.

    Only groups with at least MIN_CELLS_FOR_RATE pitches contribute. A group's
    mean is an estimate of its effect plus execution noise / sqrt(n), and since
    no cell is pruned before fitting any more, most groups are thin: their means
    would carry the full ~0.9 ft execution spread rather than the ~0.09 ft
    between-batter spread. Including them inflates between_sd by an order of
    magnitude, which flattens the prior and lets exactly those thin cells
    overfit. Estimate the scale where it can be measured, then apply it to all.
    """
    sizes = np.bincount(b_id, minlength=B + 1)[1:]
    means = np.array([
        residuals[b_id == b].mean() if (b_id == b).any() else 0.0
        for b in range(1, B + 1)
    ])
    usable = means[sizes >= MIN_CELLS_FOR_RATE]
    # Fall back to every non-empty group if almost nothing clears the bar.
    group_means = usable if len(usable) >= 30 else means[sizes > 0]
    between_sd = float(np.std(group_means))
    if between_sd < 0.01:
        return TAU_RATE_FALLBACK
    return float(1.0 / (between_sd * 1.5))


def run_dim(df: pd.DataFrame, year: int, dim: str, sampler: SamplerArgs):
    # Side is modelled mirrored; height is invariant under the reflection.
    y_col   = "PlateLocHeight" if dim == "height" else "PlateLocSide_m"
    s1_path = OUTPUT_DIR / f"stage1_{dim}_{year}.pkl"
    with open(s1_path, "rb") as f:
        s1 = pickle.load(f)

    # Attach the spline predictors once, mirroring exactly as the fitted
    # surface did (the flag is read back from the pickle, not from config).
    df = prepare_features(df, s1)
    mu_pop    = compute_mu_pop(df, s1)
    sigma_pop = compute_sigma_pop(df, s1)
    r1        = df[y_col].values - mu_pop

    df = df.copy()
    # One effect per batter per pitch type per pitcher hand — NOT per movement
    # sub-cluster. The column keeps its historical name because Stage 3 reads
    # it back by that name.
    df["batter_x_cluster"] = df["BatterId"].astype(str) + "_" + df["type_x_batter"]

    counts = df.groupby("batter_x_cluster").size()
    valid  = counts[counts >= MIN_PITCHES_BATTER_CLUSTER].index
    mask   = df["batter_x_cluster"].isin(valid)
    df_s2  = df[mask].copy()
    r1_s2    = r1[mask.values]
    sigma_s2 = sigma_pop[mask.values]

    print(f"  [{dim}] Pitches in Stage 2: {mask.sum():,}  |  "
          f"Batter x pitch-type groups: {len(valid)}")

    b_cats = pd.Categorical(df_s2["batter_x_cluster"], categories=sorted(valid))
    b_id   = b_cats.codes.astype(int)   # 0-indexed
    B      = len(b_cats.categories)

    # Exit-velo octile for each batter_x_cluster group — the middle level of the
    # hierarchy. Binned on df_s2 (not the full frame) so the octiles hold equal
    # counts of the batters actually modeled here.
    # Octiles computed WITHIN batter stance. Pooled across it, the bin index
    # partly encodes handedness — measured on 2024, the lefty share of the
    # pooled octiles ran from 26% in the lowest to 56% in the highest, so
    # "octile 5" meant a different percentile for each hand and the
    # (octile x matchup) cells were unbalanced at the extremes.
    ev = batter_ev_bins(df_s2, by="BatterSide")
    bin_of_batter = dict(zip(ev["BatterId"].astype(str), ev["ev_bin"].astype(int)))

    # Pooling target is (EV octile x pitch type x hand): a sparse batter's omega
    # on a slider shrinks toward what comparable hitters get on sliders, not
    # toward their average over every pitch. Crossing on the raw type rather
    # than the movement sub-cluster is what keeps these groups populated —
    # against the sub-clusters this key produced 398 levels with 28% holding a
    # single batter, and tau_bin was the worst-mixing parameter in the stage.
    group_batter = [g.split("_", 1)[0] for g in b_cats.categories]
    group_type   = [g.split("_", 1)[1] for g in b_cats.categories]
    group_bin    = [bin_of_batter.get(bid, N_BINS) for bid in group_batter]
    eta_key      = pd.Categorical([f"{j}|{c}" for j, c in zip(group_bin, group_type)])
    eta_id       = eta_key.codes.astype(int)      # (B,) -> octile x type level
    J            = len(eta_key.categories)

    n_sparse = int(np.sum(np.array(group_bin) == N_BINS))
    members  = pd.Series(eta_id).value_counts()
    print(f"  [{dim}] EV octiles x pitch type: {J} levels, "
          f"median {members.median():.0f} batters/level, "
          f"{int((members == 1).sum())} singleton "
          f"({n_sparse:,} / {B:,} groups in the sparse EV bin)")

    tau_rate = compute_tau_rate(r1_s2, b_id + 1, B)
    print(f"  [{dim}] Empirical tau_rate={tau_rate:.3f}  "
          f"(prior mean={1/tau_rate:.3f} ft)")

    # Rate for the octile level, derived the same way but from the spread of
    # octile means. Between-octile variation is far smaller than between-batter
    # variation, so this prior is correspondingly tighter.
    bin_rate = compute_tau_rate(r1_s2, eta_id[b_id] + 1, J)
    print(f"  [{dim}] Empirical bin_rate={bin_rate:.3f}  "
          f"(prior mean={1/bin_rate:.3f} ft)")

    nu = float(s1["nu"])

    with pm.Model() as model:
        # Three levels: global (0) -> EV octile (eta) -> batter x cluster (omega).
        # A sparse batter now shrinks toward hitters of comparable exit velo
        # rather than toward the league as a whole, and the octiles themselves
        # shrink toward global — so a thin or unrepresentative octile degrades
        # to the old behavior instead of confidently pulling to a bad target.
        tau_bin = pm.Exponential("tau_bin", lam=bin_rate)
        eta_raw = pm.Normal("eta_raw", 0, 1, shape=J)
        eta     = pm.Deterministic("eta", eta_raw * tau_bin)          # (J,)

        # Non-centered at both levels — the same funnel that made centering
        # fail in Stage 3 applies here, and harder, since a group qualifies on
        # MIN_PITCHES_BATTER_CLUSTER pitches.
        tau       = pm.Exponential("tau", lam=tau_rate)
        omega_raw = pm.Normal("omega_raw", 0, 1, shape=B)
        omega     = pm.Deterministic("omega", eta[eta_id] + omega_raw * tau)

        pm.StudentT("obs", nu=nu, mu=omega[b_id], sigma=sigma_s2, observed=r1_s2)

        progress = SamplingProgress(
            f"[{dim}]", sampler.chains, sampler.tune, sampler.draws
        )
        # The default init ("auto" -> jitter+adapt_diag) applies U(-1,1) jitter
        # across ~5,300 unconstrained parameters, and one unlucky draw can put a
        # chain somewhere dual averaging never recovers from — that is what cost
        # the 2024 side model chain 1 (accept 0.15, omega off by 0.18 ft).
        trace = pm.sample(
            **sampler.to_sample_kwargs(),
            progressbar=False,
            callback=progress,
            return_inferencedata=True,
        )
        progress.close()

    n_div   = int(trace.sample_stats["diverging"].sum().values)
    summary = az.summary(trace, var_names=["tau"])
    print()
    report_diagnostics(f"[{dim}]", n_div,
                       float(summary["r_hat"].max()),
                       float(summary["ess_bulk"].min()))

    omega_means = trace.posterior["omega"].mean(dim=["chain", "draw"]).values

    omega_df = pd.DataFrame({
        "batter_x_cluster": list(b_cats.categories),
        f"omega_{dim}":     omega_means,
    })

    out = OUTPUT_DIR / f"stage2_{dim}_{year}.parquet"
    omega_df.to_parquet(out, index=False)
    print(f"  [{dim}] Saved -> {out}")
    if dim == "side":
        print(f"  [{dim}] NOTE: omega_side is in the MIRRORED frame; multiply by "
              f"mirror_sign before reporting.")

    ckpt = CHECKPOINT_DIR / f"stage2_{dim}_{year}.nc"
    trace.to_netcdf(str(ckpt))
    print(f"  [{dim}] Checkpoint -> {ckpt}")
    print(f"  [{dim}] omega SD={omega_means.std():.4f} ft  "
          f"range=[{omega_means.min():.3f}, {omega_means.max():.3f}]")


def main(year: int, sampler: SamplerArgs = DEFAULT_SAMPLER):
    print(f"Stage 2 | year={year}")
    print(sampler.describe())
    df = load_data(year)
    print(f"  Total qualifying pitches: {len(df):,}")
    for i, dim in enumerate(["height", "side"], start=1):
        print(f"\n--- Stage 2 model {i}/2: {dim} ---", flush=True)
        run_dim(df, year, dim, sampler)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stage 2: batter x cluster aim adjustment.",
    )
    parser.add_argument("--year", type=int, required=True,
                        help="season year (e.g. 2024)")
    add_sampler_args(parser, DEFAULT_SAMPLER)
    args = parser.parse_args()
    main(args.year, resolve_sampler_args(args, DEFAULT_SAMPLER))
