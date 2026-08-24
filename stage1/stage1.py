"""
Stage 1 — Population-level aim model (PyMC).
Fits height and side models locally via NUTS.

V9 — the movement surface replaces the KMeans clusters. `u_cxb[cluster x side]`
is gone; in its place sit a joint 2-D tensor B-spline over the movement plane,
an additive 1-D smooth over release speed, a batter-side deviation surface, and
a small categorical block carrying pitch type, pitcher hand, and batter stance.
`u_cxcount` becomes pitch_type x count. See utils/splines.py for the basis and
for why the plane is fit jointly rather than additively.

What this buys, concretely:
  - No reference-level fallthrough. 2.74% of 2024 pitches sat in clusters this
    stage never fit, and `compute_sigma_pop` handed them the alphabetically
    first cluster's execution spread. A surface interpolates instead.
  - No elbow heuristic choosing the resolution of the whole stage with no
    uncertainty attached.
  - No singleton cells. Cluster membership ran from 1 to 86 pitchers, and a
    one-pitcher cluster meant u_cxb for it WAS that pitcher's aim, absorbed
    into the population term before Stage 3 could measure it.

What it does NOT fix: movement is overwhelmingly a stable pitcher x type
property (within/between SD ratio 0.10-0.34 across the features), so the
surface is still, functionally, partly indexed by the pitcher. Evaluating it
per-pitch rather than at a pitcher's mean does not change that, and would
condition aim on the realized break of the pitch — an outcome. The residual
absorption of pitcher identity into the population term is a separate problem.

Usage:
    uv run python stage1/stage1.py --year 2024
    uv run python stage1/stage1.py --year 2024 --draws 2000 --target-accept 0.99

Run with --help to see every sampler flag and its default for this stage.
"""

import argparse
import pickle
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt

from config import (
    OUTPUT_DIR, STAGE1_CAP_PER_CELL,
    STAGE1_CHAINS, STAGE1_ITER_WARMUP, STAGE1_ITER_SAMPLING, STAGE1_ADAPT_DELTA,
    STAGE1_INIT, MCMC_SEED,
    SPLINE_DEGREE, SPLINE_N_MOVE, SPLINE_N_VELO, SPLINE_N_SIGMA_MOVE,
    TAU_CXB_RATE_H, TAU_COUNT_RATE_H, TAU_CXCOUNT_RATE_H,
    TAU_CXB_RATE_S, TAU_COUNT_RATE_S, TAU_CXCOUNT_RATE_S,
    TAU_SPLINE_DEV_RATE_H, TAU_SPLINE_DEV_RATE_S,
    TAU_VELO_RATE_H, TAU_VELO_RATE_S,
    TAU_TPS_RATE_H, TAU_TPS_RATE_S,
)
from utils.matchup import add_matchup, type_x_matchup
from utils.cli import SamplerArgs, add_sampler_args, resolve_sampler_args
from utils.progress import SamplingProgress, report_diagnostics
from utils.splines import (
    FEAT_HB, FEAT_IVB, FEAT_VELO, FEAT_VALID,
    add_spline_features, combo_frame, describe, fit_spec,
)

CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# Pickle format tag. utils.scoring refuses anything else rather than falling
# back to dict lookups that no longer exist — a silent zero here would zero the
# population aim for the whole season and still produce a plausible leaderboard.
S1_VERSION = "v11-velo-height-only"

MIN_COUNT_PITCHES = 200

# The stratification cell for the sampling cap. No cluster term any more, and
# pitcher hand is explicit because the movement surface is mirrored: if the cap
# let one hand dominate, the shared surface would be fit mostly to that hand.
# Stratify on matchup, not on the two hands separately: after mirroring
# they are the same cell, and splitting them would halve each stratum.
STRATA = ["pitch_type", "matchup", "count"]

DEFAULT_SAMPLER = SamplerArgs.defaults(
    draws=STAGE1_ITER_SAMPLING, tune=STAGE1_ITER_WARMUP, chains=STAGE1_CHAINS,
    target_accept=STAGE1_ADAPT_DELTA, init=STAGE1_INIT, seed=MCMC_SEED,
)


def load_and_prep(year: int) -> pd.DataFrame:
    path = OUTPUT_DIR / f"clustered_{year}.parquet"
    df = pd.read_parquet(path)

    # pitch_cluster is no longer a modelling axis, but a row without a pitch
    # type still has no aim to model — that is the ~3.7% Undefined/Other/
    # Knuckleball share PITCH_TYPE_MAP deliberately drops.
    df = df[df["pitch_type"].notna()].copy()

    for col in ["PlateLocHeight", "PlateLocSide"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["PlateLocHeight", "PlateLocSide"])

    # Mirror every flipping axis at once, then build the spline features from
    # the already-mirrored columns.
    df = add_matchup(df)
    df = add_spline_features(df)
    n_before = len(df)
    df = df[df[FEAT_VALID]].copy()
    if n_before != len(df):
        print(f"  Dropped {n_before - len(df):,} pitches with untracked movement "
              f"({(n_before - len(df)) / max(n_before, 1):.2%}); Stages 2-3 will "
              f"still score them at the population mean.")

    valid_counts = (
        df.groupby("count").size()
        .pipe(lambda s: s[s >= MIN_COUNT_PITCHES].index)
    )
    df = df[df["count"].isin(valid_counts)].copy()

    # Categorical block: pitch type x matchup, 14 levels. Was pitch type x
    # pitcher hand x batter side at 28; mirroring collapses the pairs that are
    # reflections of each other, so each level now carries twice the pitches.
    df["type_x_matchup"] = type_x_matchup(df)
    df["type_x_count"]   = df["pitch_type"] + "_" + df["count"]
    return df


def sample_training(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    parts = []
    for _, g in df.groupby(STRATA, observed=True):
        parts.append(
            g.sample(n=min(len(g), STAGE1_CAP_PER_CELL),
                     random_state=int(rng.integers(1e9)))
        )
    return pd.concat(parts).reset_index(drop=True)


def run_dim(df_train: pd.DataFrame, year: int, dim: str, sampler: SamplerArgs):
    # The side model is fit on the MIRRORED plate side. Height is invariant
    # under the reflection and needs no adjustment.
    y_col = "PlateLocHeight" if dim == "height" else "PlateLocSide_m"
    rates = dict(
        height=(TAU_CXB_RATE_H, TAU_COUNT_RATE_H, TAU_CXCOUNT_RATE_H,
                TAU_SPLINE_DEV_RATE_H, TAU_VELO_RATE_H, TAU_TPS_RATE_H),
        side=(TAU_CXB_RATE_S, TAU_COUNT_RATE_S, TAU_CXCOUNT_RATE_S,
              TAU_SPLINE_DEV_RATE_S, TAU_VELO_RATE_S, TAU_TPS_RATE_S),
    )[dim]
    # tau_count and tau_dev rates are unpacked but unused: both terms were
    # measured as unidentified and removed. The config constants stay so the
    # tuple shape and the height/side symmetry are unchanged.
    (tau_move_rate, _tau_count_rate, tau_tc_rate,
     _tau_dev_rate, tau_velo_rate, tau_tps_rate) = rates

    y_vals = df_train[y_col].values.astype(float)
    y_mean = float(y_vals.mean())
    y_sd   = float(y_vals.std())
    y_z    = (y_vals - y_mean) / y_sd

    # --- spline bases -------------------------------------------------------
    # Group-centred on pitch_type. Movement very nearly determines pitch type,
    # so a free surface plus a categorical type effect is a near-redundant pair
    # and NUTS crawls along the ridge between them — the first fit of this
    # model hit max treedepth on every chain and took 615s for 60 draws.
    # Sweeping the type means out leaves the surface carrying only WITHIN-type
    # movement structure and the categorical block carrying the rest.
    # The surface lives at (pitcher x pitch type) resolution, matching the
    # KMeans means it replaces. combo_id maps each pitch back to its row.
    combo, combo_id = combo_frame(df_train)

    spec_move  = fit_spec(combo, "move",  [FEAT_HB, FEAT_IVB],
                          list(SPLINE_N_MOVE), SPLINE_DEGREE,
                          group_col="pitch_type")
    # Velocity enters the HEIGHT model only.
    #
    # It is fit ORTHOGONAL to the movement surface: it explains only what
    # movement cannot. Without this the two compete for the same variance and
    # the side model does not converge — see fit_spec.
    #
    # Orthogonalising was not enough for side. The 2024 fit left tau_velo at
    # r_hat 1.50 / ESS 7 and w_velo_raw at 1.41 / 9, with every other parameter
    # in the stage at r_hat <= 1.02: chains 1-2 settled at tau_velo ~ 0.08,
    # chain 3 at 4.40, chain 0 wandered 0.006-5.95. Reconstructing w_velo showed
    # the disagreement confined to the top three basis coefficients — chain 3
    # put them at +5.27/-7.95/-7.61 against chains 1-2 at ~0.02 — i.e. huge
    # canceling values over a velocity region holding 1,803 of 140,139 pitches.
    #
    # The cause is structural, not numerical. Once horizontal movement is in the
    # surface, release speed carries no independent information about aim SIDE,
    # so tau_velo has nothing anchoring it and is free to run the funnel ridge
    # (small tau x ordinary raw, or large tau x near-zero raw). Height is the
    # opposite case: speed governs drop, the data pin the smooth, and all four
    # chains agree on w_velo to three decimals at r_hat 1.00 / ESS 894. So the
    # term is kept there and dropped here.
    fit_velo = (dim == "height")
    spec_velo = (
        fit_spec(combo, "velo", [FEAT_VELO],
                 [SPLINE_N_VELO], SPLINE_DEGREE,
                 group_col="pitch_type", ortho_to=spec_move)
        if fit_velo else None
    )
    spec_sigma = fit_spec(combo, "sigma", [FEAT_HB, FEAT_IVB],
                          list(SPLINE_N_SIGMA_MOVE), SPLINE_DEGREE,
                          group_col="pitch_type")

    B_move  = spec_move.transform(combo)
    B_velo  = spec_velo.transform(combo) if fit_velo else None
    B_sigma = spec_sigma.transform(combo)

    # No side-gated deviation surface. It was fitted and measured: tau_dev came
    # back at 0.052 with HDI [0.004, 0.103] touching zero, and the mean |raw|
    # coefficient was 0.808 against the 0.798 a pure N(0,1) prior gives — i.e.
    # all 54 coefficients were unidentified and resampling their prior every
    # iteration. tau_vdev was the same story. That is a non-centred scale pinned
    # at the zero boundary, the exact funnel Stage 3's tau_gamma note describes,
    # and with three of them the sampler ran at 0.5 draws/s against V8's 31.
    #
    # The platoon effect is not lost: u_tps is indexed on BatterSide, so "this
    # pitch type is aimed further away against a lefty" is still estimated, at
    # 28 well-populated levels. What the data declined to support is the finer
    # claim that WITHIN a pitch type, aim shifts with movement differently by
    # batter stance.

    # --- categorical block --------------------------------------------------
    tps_cats = pd.Categorical(df_train["type_x_matchup"])
    cnt_cats = pd.Categorical(df_train["count"])
    tc_cats  = pd.Categorical(df_train["type_x_count"])
    typ_cats = pd.Categorical(df_train["pitch_type"])
    mu_cats  = pd.Categorical(df_train["matchup"])

    tps_id = tps_cats.codes.astype(int)
    cnt_id = cnt_cats.codes.astype(int)
    tc_id  = tc_cats.codes.astype(int)
    typ_id = typ_cats.codes.astype(int)
    mu_id  = mu_cats.codes.astype(int)

    J_tps, J_count, J_tc, J_typ = (
        len(tps_cats.categories), len(cnt_cats.categories),
        len(tc_cats.categories), len(typ_cats.categories),
    )
    J_mu = len(mu_cats.categories)

    # Prior rates are calibrated on the FUNCTION, not on individual spline
    # coefficients. An i.i.d. Normal(0, tau) prior over the basis gives a
    # surface with pointwise SD ~ tau * basis_scale, so dividing the rate by
    # basis_scale keeps the surface as wide as the old per-cluster prior was.
    move_rate = tau_move_rate * spec_move.basis_scale
    velo_rate = tau_velo_rate * spec_velo.basis_scale if fit_velo else None

    print(f"  [{dim}] N={len(df_train):,}  combos={len(combo):,}  "
          f"type_x_matchup={J_tps}  count={J_count}  type_x_count={J_tc}  "
          f"matchup={J_mu}")
    print(f"  [{dim}] {describe(spec_move)}")
    print(f"  [{dim}] {describe(spec_velo)}" if fit_velo
          else f"  [{dim}] velo smooth: not fitted (height only)")
    print(f"  [{dim}] {describe(spec_sigma)}")
    print(f"  [{dim}] surface prior rates: move={move_rate:.2f} "
          f"velo={f'{velo_rate:.2f}' if fit_velo else 'n/a'}  "
          f"(implied surface SD "
          f"{1 / move_rate * spec_move.basis_scale:.3f} z)")

    with pm.Model() as model:
        # --- movement surface ------------------------------------------------
        # Non-centred, exactly as the cell effects were. The funnel argument is
        # unchanged: tau multiplies a raw coefficient vector, and the sparse
        # regions of the movement plane sit squarely in the funnel regime.
        tau_move  = pm.Exponential("tau_move", lam=move_rate)
        w_move_raw = pm.Normal("w_move_raw", 0, 1, shape=spec_move.n_cols)
        f_move = pt.dot(B_move, w_move_raw * tau_move)[combo_id]

        if fit_velo:
            tau_velo  = pm.Exponential("tau_velo", lam=velo_rate)
            w_velo_raw = pm.Normal("w_velo_raw", 0, 1, shape=spec_velo.n_cols)
            f_velo = pt.dot(B_velo, w_velo_raw * tau_velo)[combo_id]
        else:
            f_velo = 0.0

        # --- categorical block -----------------------------------------------
        # No global u_count. type_x_count already spans count at 83 levels, so a
        # count main effect on top of it is redundant, and it showed: tau_count
        # came back at 0.021 with HDI [0.002, 0.046], ESS 8 and r_hat 1.23 — the
        # worst-mixing parameter in the stage, and a third scale pinned at zero.
        tau_tps   = pm.Exponential("tau_tps",   lam=tau_tps_rate)
        tau_tc    = pm.Exponential("tau_tc",    lam=tau_tc_rate)

        u_tps_raw   = pm.Normal("u_tps_raw",   0, 1, shape=J_tps)
        u_tc_raw    = pm.Normal("u_tc_raw",    0, 1, shape=J_tc)

        u_tps   = pm.Deterministic("u_tps",   u_tps_raw   * tau_tps)
        u_tc    = pm.Deterministic("u_tc",    u_tc_raw    * tau_tc)

        # No separate beta_left: u_tps is already indexed on BatterSide, so a
        # global left/right intercept would be a second encoding of the same
        # contrast and a ridge for NUTS to crawl along.
        alpha = pm.Normal("alpha", 0, 0.2)

        # --- log-sigma submodel ----------------------------------------------
        # Execution spread over the movement plane is now smooth too, and it
        # keeps the count and pitch-type terms it always had. This is the piece
        # that fed the reference-level bug: an unseen cluster used to inherit
        # the alphabetically first cluster's spread, and sigma is both a
        # likelihood weight here and the Mahalanobis denominator in Command+.
        sigma_intercept = pm.Normal("sigma_intercept", 0, 0.5)
        beta_sigma_move = pm.Normal("beta_sigma_move", 0, 0.3,
                                    shape=spec_sigma.n_cols)
        beta_sigma_typ  = pm.Normal("beta_sigma_typ",  0, 0.3, shape=J_typ)
        beta_sigma_cnt  = pm.Normal("beta_sigma_cnt",  0, 0.3, shape=J_count)
        # Execution spread by matchup. Missing before V10, which meant any
        # platoon difference in how precisely a pitch is executed was left in
        # the residual and charged to the pitcher as command.
        beta_sigma_mu   = pm.Normal("beta_sigma_mu",   0, 0.3, shape=J_mu)

        log_sigma = (
            sigma_intercept
            + pt.dot(B_sigma, beta_sigma_move)[combo_id]
            + beta_sigma_typ[typ_id]
            + beta_sigma_cnt[cnt_id]
            + beta_sigma_mu[mu_id]
        )

        nu = pm.Gamma("nu", alpha=5, beta=0.9)

        mu = (
            alpha
            + f_move + f_velo
            + u_tps[tps_id]
            + u_tc[tc_id]
        )

        pm.StudentT("obs", nu=nu, mu=mu, sigma=pt.exp(log_sigma), observed=y_z)

        progress = SamplingProgress(
            f"[{dim}]", sampler.chains, sampler.tune, sampler.draws
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            trace = pm.sample(
                **sampler.to_sample_kwargs(),
                progressbar=False,
                callback=progress,
                return_inferencedata=True,
            )
        progress.close()

    # Diagnostics
    n_div  = int(trace.sample_stats["diverging"].sum().values)
    diag_vars = ["alpha", "nu", "tau_move", "tau_tps", "tau_tc"]
    if fit_velo:
        diag_vars.insert(3, "tau_velo")
    summary = az.summary(trace, var_names=diag_vars)
    print()
    report_diagnostics(f"[{dim}]", n_div,
                       float(summary["r_hat"].max()),
                       float(summary["ess_bulk"].min()))

    def pm_mean(name):
        return trace.posterior[name].mean(dim=["chain", "draw"]).values

    def pm_mean_scaled(raw_name, tau_name):
        """
        E[w_raw * tau], not E[w_raw] * E[tau].

        The two are equal only when the factors are uncorrelated, and a
        non-centred parameterisation makes them anti-correlated by construction:
        the posterior is a ridge along which small tau trades against large raw
        coefficients. The 2024 side fit showed exactly how far apart they land —
        the plug-in product gave [-0.587, -0.154, 0.013, 1.262, 0.785, -1.018,
        -0.881] where E[w] is [-0.027, 0.005, -0.013, 0.151, 1.687, -2.535,
        -2.421], a max discrepancy of 1.54 in standardised units, and the stored
        vector matched neither mode. It bites hardest on a bad fit but is wrong
        on every fit.
        """
        post = trace.posterior
        return (post[raw_name] * post[tau_name]).mean(dim=["chain", "draw"]).values

    # Coefficients are stored already multiplied by their scale, so scoring is a
    # plain dot product and never needs to know the model was non-centred.
    results = {
        "version":     S1_VERSION,
        "frame":       "mirrored",   # see utils/matchup.py
        "y_mean":      y_mean,
        "y_sd":        y_sd,
        "alpha":       float(pm_mean("alpha")),

        "spec_move":   spec_move,
        # None for side: the velo smooth is fitted for height only. Scoring
        # keys off this being None rather than off the dimension name.
        "spec_velo":   spec_velo,
        "spec_sigma":  spec_sigma,
        "w_move":      pm_mean_scaled("w_move_raw", "tau_move").tolist(),
        "w_velo":      (pm_mean_scaled("w_velo_raw", "tau_velo").tolist()
                        if fit_velo else None),

        "u_tps":       dict(zip(list(tps_cats.categories), pm_mean("u_tps").tolist())),
        "u_tc":        dict(zip(list(tc_cats.categories),  pm_mean("u_tc").tolist())),

        "sigma_intercept": float(pm_mean("sigma_intercept")),
        "beta_sigma_move": pm_mean("beta_sigma_move").tolist(),
        "sigma_typ":   dict(zip(list(typ_cats.categories),
                                pm_mean("beta_sigma_typ").tolist())),
        "sigma_cnt":   dict(zip(list(cnt_cats.categories),
                                pm_mean("beta_sigma_cnt").tolist())),
        "sigma_mu":    dict(zip(list(mu_cats.categories),
                                pm_mean("beta_sigma_mu").tolist())),
        "nu":          float(pm_mean("nu")),
    }

    out = OUTPUT_DIR / f"stage1_{dim}_{year}.pkl"
    with open(out, "wb") as f:
        pickle.dump(results, f)
    print(f"  [{dim}] Saved -> {out}  (nu={results['nu']:.2f})")

    # Realized surface spread, in feet. This is the number to watch across
    # runs: if the surface is much wider than the old u_cxb was, it is taking
    # structure that used to reach Stage 3 as pitcher deviation.
    f_hat = B_move @ np.array(results["w_move"])
    if fit_velo:
        f_hat = f_hat + B_velo @ np.array(results["w_velo"])
    f_hat = f_hat[combo_id]
    print(f"  [{dim}] Surface SD={f_hat.std() * y_sd:.4f} ft  "
          f"range=[{f_hat.min() * y_sd:+.3f}, {f_hat.max() * y_sd:+.3f}] ft")

    ckpt = CHECKPOINT_DIR / f"stage1_{dim}_{year}.nc"
    trace.to_netcdf(str(ckpt))
    print(f"  [{dim}] Checkpoint -> {ckpt}")


def main(year: int, sampler: SamplerArgs = DEFAULT_SAMPLER):
    print(f"Stage 1 | year={year}")
    print(sampler.describe())
    df = load_and_prep(year)
    print(f"  Rows after pruning: {len(df):,}")
    df_train = sample_training(df)
    print(f"  Rows after sampling cap ({STAGE1_CAP_PER_CELL}/cell over "
          f"{' x '.join(STRATA)}): {len(df_train):,}")

    for i, dim in enumerate(["height", "side"], start=1):
        print(f"\n--- Stage 1 model {i}/2: {dim} ---", flush=True)
        run_dim(df_train, year, dim, sampler)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stage 1: population-level aim model (spline movement surface).",
    )
    parser.add_argument("--year", type=int, required=True,
                        help="season year (e.g. 2024)")
    add_sampler_args(parser, DEFAULT_SAMPLER)
    args = parser.parse_args()
    main(args.year, resolve_sampler_args(args, DEFAULT_SAMPLER))
