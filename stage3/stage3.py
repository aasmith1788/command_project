"""
Stage 3 — Pitcher aim deviation estimation (PyMC), bivariate, single model (V8).

V8 change — one model for every pitcher at once, replacing 148 independent
per-pitcher fits. The old design could not pool across pitchers at all: a model
containing one pitcher has nobody to borrow from, so tau_delta, nu, and corr
were each estimated from that pitcher's ~1,000 pitches alone. That is what
produced the ESS < 100 warnings, and it is why the run took ~37 hours.

Pooling is two-level:

    delta[pitcher x pitch type x batter side] ~ Normal(0, tau_delta)

so a pitcher's aim on a given pitch type against a given hand shrinks toward the
global mean — hard when he has thrown few of them, barely when he has thrown
many.

A velocity-octile rung between the two was tried and removed. Once its groups
were properly populated (104 levels, ~11 pitcher cells each, 3.8% singleton),
tau_gamma still came back at 0.011-0.015 ft with both HDIs touching zero, and
the realized octile effects spanned ~0.003 ft. It explained nothing while
pinning a non-centered scale at the zero boundary, where the raw effects are
unidentified — that funnel was the source of this stage's divergences.

Velocity is not irrelevant to aim; it is already removed upstream. RelSpeed is
one of the features the Stage 1 movement clusters are built from (R^2 = 0.53 on
a pitcher's mean fastball velo), and Stage 1's u_cxb is indexed on those
clusters, so the velocity-correlated part of aim is gone before this stage sees
a residual. Testing the octile idea honestly would mean dropping RelSpeed from
the clustering features first.

The pitch axis here is the RAW pitch type, not the Stage 1 movement cluster.
Stage 1 has already removed the sub-cluster location tendency, so splitting a
pitcher's aim finer than "this type, this hand" only thins the cells — against
the 109 cluster x side levels, 77% of pooling groups held a single pitcher.

Deliberately stripped down for runtime. The only sampled parameters are the aim
deviations and their scale; everything else is a fixed input:

  - Pitcher x type x side aim deviations (delta), non-centered, per axis [sampled]
  - nu                    — fixed, from the Stage 1 posterior, as in Stage 2
  - height/side corr      — fixed, plug-in from Stage 2 residuals per
                            (velo octile x pitch type), shrunk toward 0 by n

Dropped relative to the earlier V8 draft, and what each drop costs:
  - Pitcher x game random intercepts (u_game). Outing-to-outing aim drift is
    no longer separated out, so a pitcher whose release drifts between starts
    now has that charged to him as miss.
  - AR(1) within-game serial correlation (rho). Consecutive misses are treated
    as independent, so the ~0.45 lag-1 correlation is unmodelled again and the
    posterior is correspondingly overconfident.
  - Sampling nu and corr. These cost a log-gamma and its derivative per pitch
    per gradient for quantities the data already pin down; fixing them removes
    that from the inner loop entirely.

Usage:
    python stage3/stage3.py --year 2024
    python stage3/stage3.py --year 2024 --draws 250 --chains 2 --cores 2

Run with --help to see every sampler flag and its default for this stage.
This is the longest stage by far, so --draws/--tune/--chains are the knobs
for a quick shakeout run, and --cores caps parallelism on a memory-tight box.
"""

import argparse
import logging
import os
import pickle
import sys
import warnings
from pathlib import Path

# Keep BLAS to one thread per process — this is a single large model sampled
# with parallel chains, so nested thread pools only contend.
for _thread_var in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_thread_var, "1")

sys.path.insert(0, str(Path(__file__).parent.parent))

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
from scipy.optimize import minimize
from scipy.special import logsumexp

from config import (
    OUTPUT_DIR,
    MIN_PITCHES_PITCHER, MIN_PITCHES_PITCHER_CLUSTER, MIN_PITCHES_SCORED,
    STAGE3_CHAINS, STAGE3_ITER_WARMUP, STAGE3_ITER_SAMPLING,
    STAGE3_ADAPT_DELTA, STAGE3_INIT, MCMC_SEED,
    DENSITY_KNOTS, DENSITY_EXTENT_H, DENSITY_EXTENT_S, DENSITY_BW,
    DENSITY_RANK, DENSITY_OUT_GRID,
    MIN_PITCHES_CELL_FREE, TAU_G_RATE, TAU_PI_RATE,
)
from utils.bins import N_BINS, pitcher_fb_bins
from utils.matchup import unmirror_side
from utils.cli import SamplerArgs, add_sampler_args, resolve_sampler_args
from utils.progress import SamplingProgress, report_diagnostics
from utils.scoring import (add_stage1_keys, command_plus, compute_mu_pop,
                           compute_sigma_pop, prepare_features)

CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

logging.getLogger("pymc").setLevel(logging.ERROR)

# Column order for every (N, 2) array in this module: height first, then side.
DIMS = ["h", "s"]

DEFAULT_SAMPLER = SamplerArgs.defaults(
    draws=STAGE3_ITER_SAMPLING, tune=STAGE3_ITER_WARMUP, chains=STAGE3_CHAINS,
    target_accept=STAGE3_ADAPT_DELTA, init=STAGE3_INIT, seed=MCMC_SEED,
)


def load_inputs(year: int):
    df = pd.read_parquet(OUTPUT_DIR / f"clustered_{year}.parquet")
    # pitch_cluster is no longer a modelling axis anywhere. Stage 1's movement
    # term is a spline surface over the raw features, so a pitch KMeans could
    # not label is still scoreable; only an unnameable pitch TYPE is dropped.
    df = df[df["pitch_type"].notna()].copy()
    for col in ["PlateLocHeight", "PlateLocSide"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["PlateLocHeight", "PlateLocSide"])

    # Keys Stage 1's categorical block is indexed on — shared helper so this
    # stage and Stage 2 cannot drift apart.
    df = add_stage1_keys(df)

    # Pitch type x MATCHUP. MUST match stage2.py exactly: Stage 2 wrote its
    # omega table keyed on this, and a mismatch fails silently, mapping every
    # lookup to NaN and quietly zeroing the batter adjustment for the season.
    df["type_x_batter"]    = df["pitch_type"] + "_" + df["matchup"]
    df["batter_x_cluster"] = df["BatterId"].astype(str) + "_" + df["type_x_batter"]

    s1_h = pickle.load(open(OUTPUT_DIR / f"stage1_height_{year}.pkl", "rb"))
    s1_s = pickle.load(open(OUTPUT_DIR / f"stage1_side_{year}.pkl",   "rb"))

    # Spline predictors, mirrored exactly as the fitted surface was. Both dims
    # share one mirroring flag, so preparing once is safe; assert it rather
    # than assume, since a mismatch would shift one axis and not the other.
    if s1_h.get("frame") != s1_s.get("frame"):
        raise ValueError("stage1 height/side pickles disagree on the mirroring frame")
    df = prepare_features(df, s1_h)

    s2_h = pd.read_parquet(OUTPUT_DIR / f"stage2_height_{year}.parquet")
    s2_s = pd.read_parquet(OUTPUT_DIR / f"stage2_side_{year}.parquet")
    omega_h = dict(zip(s2_h["batter_x_cluster"], s2_h["omega_height"]))
    omega_s = dict(zip(s2_s["batter_x_cluster"], s2_s["omega_side"]))

    df["mu_pop_h"]    = compute_mu_pop(df, s1_h)
    df["mu_pop_s"]    = compute_mu_pop(df, s1_s)
    df["sigma_pop_h"] = compute_sigma_pop(df, s1_h)
    df["sigma_pop_s"] = compute_sigma_pop(df, s1_s)

    df["omega_h"] = df["batter_x_cluster"].map(omega_h).fillna(0.0)
    df["omega_s"] = df["batter_x_cluster"].map(omega_s).fillna(0.0)

    # Side residual is in the MIRRORED frame, matching mu_pop_s and omega_s,
    # which Stages 1 and 2 fit there. Un-mirrored only at output.
    df["r2_h"] = df["PlateLocHeight"]   - df["mu_pop_h"] - df["omega_h"]
    df["r2_s"] = df["PlateLocSide_m"]   - df["mu_pop_s"] - df["omega_s"]

    return df


def _empirical_rate(values: np.ndarray, group_ids: np.ndarray, floor: float = 0.01,
                    fallback: float = 10.0) -> np.ndarray:
    """
    Exponential rate from the between-group spread, one per axis.
    Mirrors the Stage 2 convention: rate = 1 / (1.5 * between-group SD).
    """
    rates = []
    uniq = np.unique(group_ids)
    for d in range(values.shape[1]):
        means = np.array([values[group_ids == g, d].mean() for g in uniq])
        sd = float(np.std(means))
        rates.append(1.0 / (sd * 1.5) if sd > floor else fallback)
    return np.array(rates, dtype=float)


def build_design(df_all: pd.DataFrame, year: int):
    """
    Filter to qualifying pitchers and build every index array the model needs.

    Returns a dict of arrays plus the filtered/sorted frame, so scoring can
    reuse exactly the same row order the likelihood saw.
    """
    counts = df_all.groupby("PitcherId").size()
    qualifying = counts[counts >= MIN_PITCHES_PITCHER].index
    df = df_all[df_all["PitcherId"].isin(qualifying)].copy()
    print(f"  Qualifying pitchers (>={MIN_PITCHES_PITCHER} pitches): {len(qualifying)}")

    # Aim cells are pitcher x cluster x batter side — deliberately NOT split by
    # count. Stage 1 already removes the population count effect (u_count,
    # u_cxcount); the only thing a per-count delta would add is the pitcher's
    # own count tendency, and granting him a separate target per situation
    # reclassifies genuine miss as intent. Splitting by count also fragmented
    # the pooling: it left ~1.9 delta cells per gamma level, so a pitcher's
    # "reference class" was effectively himself, and thin cells were dropped
    # wholesale by the MIN_PITCHES_PITCHER_CLUSTER prune.
    # One aim per pitcher per pitch type per batter side — NOT per movement
    # sub-cluster. Stage 1 already removed the sub-cluster location tendency,
    # so a finer split here only thins the cells and empties the octile groups.
    df["pitcher_x_cluster"] = (
        df["PitcherId"].astype(str) + "|" + df["type_x_batter"]
    )
    combo = df.groupby("pitcher_x_cluster").size()
    keep = combo[combo >= MIN_PITCHES_PITCHER_CLUSTER].index
    df = df[df["pitcher_x_cluster"].isin(keep)].copy()
    print(f"  Pitches after pruning (>={MIN_PITCHES_PITCHER_CLUSTER}/combo): "
          f"{len(df):,}  |  pitcher x pitch-type combos: {len(keep):,}")

    # Velocity octiles, computed on the modeled pitchers only so bins are equal-count.
    # Octiles within pitcher handedness, for the same reason batters are binned
    # within stance: lefties throw slightly slower, so a pooled octile partly
    # encodes handedness and the pooling groups inherit that imbalance.
    velo = pitcher_fb_bins(df, by="PitcherThrows")
    bin_of_pitcher = dict(zip(velo["PitcherId"], velo["velo_bin"].astype(int)))
    df["velo_bin"] = df["PitcherId"].map(bin_of_pitcher).fillna(N_BINS).astype(int)
    dist = df.groupby("velo_bin")["PitcherId"].nunique()
    print(f"  FB-velo octiles: {dict(dist)}")

    df = df.sort_values(["PitcherId", "GameID", "PitchNo"]).reset_index(drop=True)

    # --- index arrays -------------------------------------------------------
    pk_cats = pd.Categorical(df["pitcher_x_cluster"])
    p_cats  = pd.Categorical(df["PitcherId"])
    # Pooling target: (velo octile, pitch type, batter side) — the level a
    # pitcher shrinks toward. Crossed with the movement sub-clusters instead,
    # this key gave 267 levels of which 77% held exactly one pitcher, which
    # left gamma unidentified: with a single member the data cannot separate
    # "this octile aims here" from "this pitcher aims here".
    df["gm_key"] = df["velo_bin"].astype(str) + "|" + df["type_x_batter"]
    gm_cats = pd.Categorical(df["gm_key"])
    # Fixed likelihood shape: (velo octile, pitch type) for the plug-in corr.
    # Matchup included: the height/side miss ellipse tilts differently across
    # the platoon, and leaving it out forced one correlation on both.
    df["vc_key"] = (df["velo_bin"].astype(str) + "|" + df["pitch_type"]
                    + "|" + df["matchup"])
    vc_cats = pd.Categorical(df["vc_key"])

    pk_id = pk_cats.codes.astype(np.int32)
    p_id  = p_cats.codes.astype(np.int32)
    gm_id = gm_cats.codes.astype(np.int32)
    vc_id = vc_cats.codes.astype(np.int32)

    # gamma is indexed per delta row, not per pitch: for each pitcher x cluster
    # combo, which (octile, cluster) target does it shrink toward.
    pk_to_gm = (
        pd.DataFrame({"pk": pk_id, "gm": gm_id})
        .drop_duplicates("pk").sort_values("pk")["gm"].to_numpy(np.int32)
    )

    # How many pitchers sit in each pooling group. This is the number that
    # decides whether gamma is identifiable at all, so it belongs in the log:
    # a group with one member cannot separate its own level from that member's.
    members = pd.Series(pk_to_gm).value_counts()
    print(f"  Pooling groups (velo octile x pitch type x side): "
          f"{len(gm_cats.categories)} levels, "
          f"median {members.median():.0f} pitcher-cells/level, "
          f"{int((members == 1).sum())} singleton "
          f"({(members == 1).mean():.1%})")

    r2    = df[["r2_h", "r2_s"]].to_numpy(float)
    sigma = df[["sigma_pop_h", "sigma_pop_s"]].to_numpy(float)
    # nu and the height/side correlation are FIXED inputs, not sampled. Sampling
    # them cost a log-gamma (and its derivative) per pitch per gradient, plus
    # VC Beta parameters, for quantities Stage 1 and the residuals already pin
    # down well. nu comes from Stage 1 exactly as it does in Stage 2; corr is a
    # plug-in estimate from the Stage 2 residuals, shrunk toward 0 by sample
    # size so a thin cell stays circular rather than confidently tilted.
    with open(OUTPUT_DIR / f"stage1_height_{year}.pkl", "rb") as f:
        nu_h = float(pickle.load(f)["nu"])
    with open(OUTPUT_DIR / f"stage1_side_{year}.pkl", "rb") as f:
        nu_s = float(pickle.load(f)["nu"])
    nu_fixed = 0.5 * (nu_h + nu_s)

    zz_h = df["r2_h"].to_numpy() / df["sigma_pop_h"].to_numpy()
    zz_s = df["r2_s"].to_numpy() / df["sigma_pop_s"].to_numpy()
    VC = len(vc_cats.categories)
    corr_vc = np.zeros(VC)
    CORR_SHRINK = 50.0
    for v in range(VC):
        m = vc_id == v
        n = int(m.sum())
        if n < 3:
            continue
        a, b = zz_h[m], zz_s[m]
        sa, sb = a.std(), b.std()
        if sa <= 0 or sb <= 0:
            continue
        r = float(((a - a.mean()) * (b - b.mean())).mean() / (sa * sb))
        corr_vc[v] = np.clip(r, -0.95, 0.95) * (n / (n + CORR_SHRINK))
    print(f"  Fixed nu={nu_fixed:.2f} (height {nu_h:.2f}, side {nu_s:.2f})  |  "
          f"plug-in corr over {VC} cells: mean {corr_vc.mean():+.3f}, "
          f"range [{corr_vc.min():+.3f}, {corr_vc.max():+.3f}]")

    # Cells with enough pitches to carry their own repertoire. Everyone else
    # uses their (velo octile x type x side) group's — see MIN_PITCHES_CELL_FREE.
    cell_n   = np.bincount(pk_id, minlength=len(pk_cats.categories))
    free     = cell_n >= MIN_PITCHES_CELL_FREE
    free_idx = np.flatnonzero(free).astype(np.int32)
    free_share = float(cell_n[free].sum()) / float(cell_n.sum())
    print(f"  Cells with own repertoire (>={MIN_PITCHES_CELL_FREE} pitches): "
          f"{free.sum():,} / {len(free):,}  covering {100 * free_share:.1f}% of pitches")

    knots, T, rank = build_density_basis()

    d = {
        "df": df, "r2": r2, "sigma": sigma,
        "pk_id": pk_id, "p_id": p_id, "vc_id": vc_id,
        "pk_to_gm": pk_to_gm,
        "nu_fixed": nu_fixed, "corr_vc": corr_vc,
        "PK": len(pk_cats.categories),
        "P": len(p_cats.categories), "GM": len(gm_cats.categories),
        "VC": VC,
        "pk_cats": list(pk_cats.categories), "p_cats": list(p_cats.categories),
        "free_idx": free_idx, "free_share": free_share,
        "knots": knots, "T": T, "rank": rank,
    }
    d["comp"] = density_logkernel(d)
    print(f"  Intent surface: {DENSITY_KNOTS[0]}x{DENSITY_KNOTS[1]} = {len(knots)} knots "
          f"at {2 * DENSITY_EXTENT_H / max(DENSITY_KNOTS[0] - 1, 1):.2f} ft spacing, "
          f"bump width {DENSITY_BW:.2f} ft, penalized rank {rank}")
    # What the data can resolve: intent spread implied by observed residual
    # spread minus the execution spread Stage 1 pinned.
    for j, dim in enumerate(("height", "side")):
        v = r2[:, j].var() - (sigma[:, j] ** 2).mean()
        print(f"    implied intent SD ({dim}): {np.sqrt(max(v, 0)):.3f} ft  "
              f"vs execution SD {sigma[:, j].mean():.3f} ft")
    return d


def _diff_matrix(m: int, order: int = 2) -> np.ndarray:
    D = np.eye(m)
    for _ in range(order):
        D = np.diff(D, axis=0)
    return D


def build_density_basis():
    """
    Penalized Gaussian-bump surface over the plate, in residual space.

    Returns (knots, T, rank) where `knots` is (M, 2) bump centres and `T` is
    (M, R) mapping R low-rank coefficients onto the M knot weights.

    The surface is f(x) proportional to sum_m exp(eta_m) * N(x | knot_m, bw^2)
    with eta = T @ beta. It is a continuous function of (height, side) —
    evaluable at any point on the plate, never binned. The knots are basis
    centres, exactly like spline knots; they are not targets and not zones.

    T comes from the eigendecomposition of a second-order difference penalty
    across the knot grid, scaled by d^-1/2 and truncated to the leading
    DENSITY_RANK directions. Consequences:
      - Smoothness is a prior, not a grid artefact. A 9x9 grid at 0.30 ft can
        express sharp structure where a cell has the pitches to support it and
        stays smooth where it does not.
      - The four null-space directions (constant and linear in each axis) are
        kept unpenalized, so the overall location and tilt of the surface are
        free while its wiggliness is shrunk.
      - Rank truncation keeps per-cell coefficients at DENSITY_RANK instead of
        81, which is what makes 1,300 free cells affordable.
    """
    mh, ms = DENSITY_KNOTS
    kh = np.linspace(-DENSITY_EXTENT_H, DENSITY_EXTENT_H, mh)
    ks = np.linspace(-DENSITY_EXTENT_S, DENSITY_EXTENT_S, ms)
    knots = np.array([(a, b) for a in kh for b in ks], dtype=float)

    Ph = _diff_matrix(mh).T @ _diff_matrix(mh)
    Ps = _diff_matrix(ms).T @ _diff_matrix(ms)
    S = np.kron(Ph, np.eye(ms)) + np.kron(np.eye(mh), Ps)

    d, U = np.linalg.eigh(S)
    tol = d.max() * 1e-10
    pen, null = d > tol, d <= tol
    # Penalised directions scaled by d^-1/2 so an i.i.d. prior on beta IS the
    # difference penalty; wiggly directions get small columns and are shrunk
    # hardest. Null space (constant + linear) left at unit scale.
    T_pen = U[:, pen] / np.sqrt(d[pen])[None, :]
    T_null = U[:, null]
    # Leading penalised directions are the smoothest ones (smallest eigenvalue).
    order = np.argsort(d[pen])
    keep = order[: max(DENSITY_RANK - T_null.shape[1], 1)]
    T = np.concatenate([T_null, T_pen[:, keep]], axis=1)
    return knots, np.ascontiguousarray(T), T.shape[1]


def density_logkernel(d: dict) -> np.ndarray:
    """
    (N, M) log N(y_i | knot_m, sigma_i^2 + bw^2) — parameter-free.

    This is the convolution of the intent surface's basis with execution noise,
    done analytically. It is why the basis is Gaussian: variances simply add,
    so the integral over the plate never has to be computed numerically. The
    whole matrix is constant, so the sampler evaluates the likelihood as
    logsumexp(log_w[cell] + this) and nothing else.
    """
    K     = d["knots"]
    r2    = d["r2"]
    sig   = d["sigma"]
    r     = d["corr_vc"][d["vc_id"]]
    bw2   = DENSITY_BW ** 2

    s_h = np.sqrt(sig[:, 0] ** 2 + bw2)
    s_s = np.sqrt(sig[:, 1] ** 2 + bw2)
    r_e = r * sig[:, 0] * sig[:, 1] / (s_h * s_s)
    one_m = np.clip(1.0 - r_e ** 2, 1e-9, None)
    norm = -np.log(2.0 * np.pi * s_h * s_s * np.sqrt(one_m))

    out = np.empty((len(r2), len(K)))
    for m, (k_h, k_s) in enumerate(K):
        z_h = (r2[:, 0] - k_h) / s_h
        z_s = (r2[:, 1] - k_s) / s_s
        out[:, m] = -0.5 * (z_h ** 2 - 2.0 * r_e * z_h * z_s + z_s ** 2) / one_m + norm
    return out


def _neg_log_post(theta: np.ndarray, d: dict, lam_g: float, lam_e: float,
                  rows: np.ndarray | None = None):
    """
    Penalized negative log likelihood and its exact gradient.

    The model is a one-layer network: a fixed penalized basis T, a per-cell
    embedding, and a softmax output.

        eta_c = (g[group(c)] + eps_c) @ T'          (81 knot logits)
        w_c   = softmax(eta_c)                      (intent surface)
        p(y_i)= sum_m w[c_i, m] * exp(COMP[i, m])   (surface * execution noise)

    COMP is constant, so the whole likelihood is a gather plus a logsumexp.

    Gradient is the standard mixture form: with a_im the responsibility of knot
    m for pitch i,

        d loglik / d eta[c, m] = sum_{i in c} (a_im - w[c, m])

    which is why this needs ~1,000 evaluations to converge where NUTS needed
    ~4,000,000 to characterise a posterior we are not using.
    """
    GM, R = d["GM"], d["rank"]
    n_free = len(d["free_idx"])
    T, COMP, pk_id = d["T"], d["comp"], d["pk_id"]
    PK = d["PK"]

    g = theta[: GM * R].reshape(GM, R)
    e = theta[GM * R:].reshape(n_free, R)

    beta = g[d["pk_to_gm"]].copy()
    beta[d["free_idx"]] += e
    eta = beta @ T.T                                     # (PK, 81)
    log_w = eta - logsumexp(eta, axis=1, keepdims=True)
    w = np.exp(log_w)

    if rows is None:
        idx, comp = pk_id, COMP
    else:
        idx, comp = pk_id[rows], COMP[rows]

    lp = log_w[idx] + comp
    ll_i = logsumexp(lp, axis=1)
    a = np.exp(lp - ll_i[:, None])                       # responsibilities

    # d loglik / d eta[c, m] = sum_{i in c} a_im  -  n_c * w[c, m]
    acc = np.zeros((PK, T.shape[0]))
    np.add.at(acc, idx, a)
    n_c = np.bincount(idx, minlength=PK).astype(float)
    d_eta = acc - n_c[:, None] * w

    d_beta = d_eta @ T                                   # (PK, R)
    d_g = np.zeros((GM, R))
    np.add.at(d_g, d["pk_to_gm"], d_beta)
    d_e = d_beta[d["free_idx"]]

    nll = -ll_i.sum() + 0.5 * lam_g * (g ** 2).sum() + 0.5 * lam_e * (e ** 2).sum()
    grad = np.concatenate([(-d_g + lam_g * g).ravel(),
                           (-d_e + lam_e * e).ravel()])
    return nll, grad


def fit_map(d: dict, lam_g: float = None, lam_e: float = None,
            holdout: float = 0.10, seed: int = 20260818, maxiter: int = 800):
    """
    Fit the intent surfaces by penalized maximum likelihood.

    Replaces NUTS. The parameter posterior is not used anywhere downstream —
    the output is the per-pitch intent distribution p(x | y_i), which is
    derived from the fitted surface, not from uncertainty about it. Sampling
    31,040 parameters to characterise a posterior nothing reads cost ~23 hours
    of projected runtime; this is the same model, same objective, optimised.

    What is genuinely lost: credible intervals on the surfaces themselves. If
    those are ever wanted, the model is unchanged and can be sampled.

    The penalty strength is chosen by held-out log likelihood on a random
    `holdout` share of pitches, which is also the falsifiability check on the
    whole idea: a richer intent surface only wins if it predicts pitches it was
    not fit on.
    """
    GM, R = d["GM"], d["rank"]
    n_free = len(d["free_idx"])
    n_par = GM * R + n_free * R

    rng = np.random.default_rng(seed)
    N = len(d["pk_id"])
    perm = rng.permutation(N)
    n_hold = int(holdout * N)
    hold, train = perm[:n_hold], perm[n_hold:]

    print(f"  Fitting by penalized MLE (L-BFGS): {n_par:,} params, "
          f"{len(train):,} train / {len(hold):,} held-out pitches")

    def run(lg, le, rows):
        res = minimize(_neg_log_post, np.zeros(n_par), args=(d, lg, le, rows),
                       jac=True, method="L-BFGS-B",
                       options={"maxiter": maxiter, "maxcor": 20})
        return res

    grid = ([lam_g] if lam_g is not None else [1.0, 10.0, 50.0, 200.0])
    best, best_ll = None, -np.inf
    print(f"    {'lambda':>8}  {'train ll/pitch':>15}  {'held-out ll/pitch':>18}")
    for lg in grid:
        le = lg if lam_e is None else lam_e
        res = run(lg, le, train)
        ll_tr = -_neg_log_post(res.x, d, 0.0, 0.0, train)[0] / len(train)
        ll_ho = -_neg_log_post(res.x, d, 0.0, 0.0, hold)[0] / len(hold)
        flag = ""
        if ll_ho > best_ll:
            best_ll, best, best_lam = ll_ho, res.x, lg
            flag = "  <- best"
        print(f"    {lg:8.1f}  {ll_tr:15.4f}  {ll_ho:18.4f}{flag}")

    # Baseline: one shared surface for everyone. If per-cell surfaces do not
    # beat this out of sample, the whole premise fails and should be reported
    # as failing rather than quietly kept.
    flat = np.zeros(n_par)
    ll_flat = -_neg_log_post(flat, d, 0.0, 0.0, hold)[0] / len(hold)
    print(f"    {'uniform':>8}  {'':>15}  {ll_flat:18.4f}  (no fitted intent)")
    print(f"    held-out gain over a uniform intent surface: "
          f"{best_ll - ll_flat:+.4f} nats/pitch")

    # Refit the chosen penalty on all pitches.
    res = run(best_lam, best_lam if lam_e is None else lam_e, None)
    print(f"    Converged: {res.success} ({res.message.strip()[:60]}), "
          f"{res.nit} iterations, {res.nfev} evaluations")

    g = res.x[: GM * R].reshape(GM, R)
    e = res.x[GM * R:].reshape(n_free, R)
    beta = g[d["pk_to_gm"]].copy()
    beta[d["free_idx"]] += e
    eta = beta @ d["T"].T
    log_w = eta - logsumexp(eta, axis=1, keepdims=True)

    return log_w, {"g": g, "eps": e, "lam": best_lam,
                   "ll_holdout": best_ll, "ll_uniform": ll_flat,
                   "success": bool(res.success), "nit": int(res.nit)}


def pitch_posterior(d: dict, log_w: np.ndarray) -> np.ndarray:
    """
    (N, M) posterior over WHERE he was aiming, for each individual pitch.

        p(x | y_i)  proportional to  f_c(x) * N(y_i | x, sigma_i)

    represented in the knot basis. This is the per-pitch intent distribution:
    the cell's surface reweighted by the evidence from that specific pitch. A
    pitch low-and-in shifts mass onto the low-and-in region of the surface even
    when that is a minor part of his repertoire; a pitch nothing explains falls
    back toward the cell's surface.

    V9 had none of this — its zone map used the cell's prior weights, identical
    for every pitch in the cell (measured within-cell SD of w: exactly 0.0).
    """
    lp = log_w[d["pk_id"]] + d["comp"]
    lp -= lp.max(axis=1, keepdims=True)
    p = np.exp(lp)
    return p / p.sum(axis=1, keepdims=True)


def evaluate_surface(weights: np.ndarray, knots: np.ndarray,
                     grid_h: np.ndarray, grid_s: np.ndarray) -> np.ndarray:
    """
    Evaluate the density at arbitrary plate coordinates.

    weights (M,) over knots -> density on the (len(grid_h), len(grid_s)) mesh.
    The surface is continuous; this is just sampling it for plotting or for
    writing an aggregated slice to disk.
    """
    bh = np.exp(-0.5 * ((grid_h[:, None] - knots[None, :, 0]) / DENSITY_BW) ** 2)
    bs = np.exp(-0.5 * ((grid_s[:, None] - knots[None, :, 1]) / DENSITY_BW) ** 2)
    dens = (bh * weights[None, :]) @ bs.T
    return dens / (dens.sum() + 1e-300)


def score(d: dict, post: np.ndarray) -> pd.DataFrame:
    """
    Per-pitch miss as the posterior-expected distance to where he was aiming.

        miss_i = integral || y_i - x || p(x | y_i) dx
               = sum_m p_im || y_i - knot_m ||

    An expectation over a continuous latent state, not a distance to a point.
    V9 used prior weights on two fitted targets, which charged every pitch the
    distance to targets it was not going for: a pitch landing perfectly on the
    primary target still scored (1-w) * separation, that floor spanned 0.297 ft
    of a 0.390 ft total range, and separation alone explained R^2 = 0.47 of
    average miss.
    """
    df = d["df"].copy()
    K  = d["knots"]
    r2, sig = d["r2"], d["sigma"]
    r = d["corr_vc"][d["vc_id"]]

    dh = r2[:, 0][:, None] - K[None, :, 0]
    ds = r2[:, 1][:, None] - K[None, :, 1]

    df["miss_dist_raw"] = (post * np.hypot(dh, ds)).sum(axis=1)

    z_h = dh / sig[:, 0][:, None]
    z_s = ds / sig[:, 1][:, None]
    denom = np.clip(1.0 - r ** 2, 1e-9, None)[:, None]
    d2 = (z_h ** 2 - 2.0 * r[:, None] * z_h * z_s + z_s ** 2) / denom
    df["miss_dist_std"] = (post * np.sqrt(np.clip(d2, 0.0, None))).sum(axis=1)

    # Spread of the intent posterior, in feet — how tightly the model can place
    # where he was aiming on this pitch. An honesty measure, reported beside the
    # metric rather than folded into it.
    mh = (post * K[None, :, 0]).sum(axis=1)
    ms = (post * K[None, :, 1]).sum(axis=1)
    vh = (post * (K[None, :, 0] - mh[:, None]) ** 2).sum(axis=1)
    vs = (post * (K[None, :, 1] - ms[:, None]) ** 2).sum(axis=1)
    df["aim_spread"] = np.sqrt(vh + vs)
    df["corr_hs"] = r
    # Back to true plate coordinates. Distances above are invariant under the
    # reflection and are left alone; anything with a left/right orientation is
    # flipped, or every left-handed pitcher reads as his own mirror image.
    sign = df["mirror_sign"].to_numpy()
    df["aim_h"] = df["mu_pop_h"].to_numpy() + df["omega_h"].to_numpy() + mh
    df["aim_s"] = unmirror_side(
        df["mu_pop_s"].to_numpy() + df["omega_s"].to_numpy() + ms, sign)

    g = df.groupby("PitcherId")
    return pd.DataFrame({
        "Pitcher":      g["Pitcher"].first(),
        "PitcherTeam":  g["PitcherTeam"].first(),
        "velo_bin":     g["velo_bin"].first(),
        "n_pitches":    g.size(),
        "avg_miss_raw": g["miss_dist_raw"].mean(),
        "avg_miss_std": g["miss_dist_std"].mean(),
        "corr_hs":      g["corr_hs"].mean(),
        "aim_spread":   g["aim_spread"].mean(),
    }).reset_index()


def aim_distribution(d: dict, post: np.ndarray, log_w: np.ndarray,
                     year: int) -> pd.DataFrame:
    """
    Per-pitch intent posterior, written as basis coefficients.

    One row per pitch with p_knot_0..M-1: the posterior over the plate in the
    knot basis. The density at any (height, side) is

        f_i(h, s) = sum_m p_knot_m * N(h | knot_m_h + base_h_i, bw^2)
                                   * N(s | knot_m_s + base_s_i, bw^2)

    so the output is a continuous surface, not a set of bin probabilities. No
    zone grid: binning a continuous quantity at the last step is exactly what
    this stage was rebuilt to stop doing. `utils.plate.density_at` evaluates it.

    A separate file writes the surface on a fine mesh for aggregated slices
    (pitcher x pitch type x batter side), which is what plots want.
    """
    df = d["df"]
    K  = d["knots"]
    M  = K.shape[0]
    base_h = df["mu_pop_h"].to_numpy() + df["omega_h"].to_numpy()
    base_s = df["mu_pop_s"].to_numpy() + df["omega_s"].to_numpy()

    out = pd.DataFrame({
        "PitcherId":  df["PitcherId"].values,
        "Pitcher":    df["Pitcher"].values,
        "GameID":     df["GameID"].values,
        "PitchNo":    df["PitchNo"].values,
        "pitch_type": df["pitch_type"].values,
        "BatterSide": df["BatterSide"].values,
        "count":      df["count"].values,
        "base_h":       base_h,
        # Mirrored frame, paired with mirror_sign. utils.plate applies the sign
        # when it evaluates the surface, so the density comes out in true plate
        # coordinates without the knot offsets being flipped twice.
        "base_s":       base_s,
        "mirror_sign":  df["mirror_sign"].to_numpy(),
        "matchup":      df["matchup"].to_numpy(),
    })
    for m in range(M):
        out[f"p_knot_{m}"] = post[:, m]

    mh = (post * K[None, :, 0]).sum(axis=1)
    ms = (post * K[None, :, 1]).sum(axis=1)
    out["aim_h"] = base_h + mh
    out["aim_s"] = unmirror_side(base_s + ms, out["mirror_sign"].to_numpy())

    # The cell's surface mean BEFORE this pitch reweights it — i.e. the single
    # point a one-target model would have used. Diagnostics contrasts the two:
    # the gap is exactly what collapsing a multimodal intent to its mean costs.
    prior = np.exp(log_w)[d["pk_id"]]
    out["prior_h"] = base_h + (prior * K[None, :, 0]).sum(axis=1)
    out["prior_s"] = unmirror_side(
        base_s + (prior * K[None, :, 1]).sum(axis=1),
        out["mirror_sign"].to_numpy())
    out["aim_spread"] = np.sqrt(
        (post * (K[None, :, 0] - mh[:, None]) ** 2).sum(axis=1)
        + (post * (K[None, :, 1] - ms[:, None]) ** 2).sum(axis=1)
    )

    path = OUTPUT_DIR / f"aim_distribution_{year}.parquet"
    out.to_parquet(path, index=False)

    # Aggregated surfaces on a fine mesh, for plotting.
    gh = np.linspace(-DENSITY_EXTENT_H - 0.6, DENSITY_EXTENT_H + 0.6, DENSITY_OUT_GRID[0])
    gs = np.linspace(-DENSITY_EXTENT_S - 0.6, DENSITY_EXTENT_S + 0.6, DENSITY_OUT_GRID[1])
    rows = []
    cats = d["pk_cats"]
    for c in range(d["PK"]):
        dens = evaluate_surface(np.exp(log_w[c]), K, gh, gs)
        rows.append({"cell": cats[c], "surface": dens.ravel().tolist()})
    surf = pd.DataFrame(rows)
    surf.attrs = {}
    spath = OUTPUT_DIR / f"intent_surface_{year}.parquet"
    pd.DataFrame({
        "cell":    surf["cell"],
        "surface": surf["surface"],
        "grid_h":  [gh.tolist()] * len(surf),
        "grid_s":  [gs.tolist()] * len(surf),
    }).to_parquet(spath, index=False)

    within = pd.DataFrame(post).groupby(pd.Series(d["pk_id"])).std().mean().mean()
    print(f"  Saved per-pitch intent posterior -> {path}  "
          f"({len(out):,} rows x {M} basis coefficients)")
    print(f"  Saved cell intent surfaces      -> {spath}  "
          f"({d['PK']:,} cells on a {DENSITY_OUT_GRID[0]}x{DENSITY_OUT_GRID[1]} mesh)")
    print(f"  Mean intent posterior spread: {out['aim_spread'].mean():.3f} ft")
    print(f"  Within-cell SD of the posterior: {within:.5f}  "
          f"(V9 was exactly 0.0 — its map ignored the individual pitch)")

    # How multimodal are the fitted surfaces, measured rather than assumed.
    ent = -(np.exp(log_w) * log_w).sum(axis=1)
    print(f"  Surface entropy: mean {ent.mean():.2f} nats "
          f"(uniform over {M} knots = {np.log(M):.2f})")
    return out


def main(year: int):
    print(f"Stage 3 | year={year}")
    df_all = load_inputs(year)
    print(f"  Total pitches loaded: {len(df_all):,}")

    d = build_design(df_all, year)
    log_w, info = fit_map(d)

    np.savez_compressed(
        OUTPUT_DIR / f"stage3_params_{year}.npz",
        g=info["g"], eps=info["eps"], log_w=log_w,
        knots=d["knots"], T=d["T"], pk_to_gm=d["pk_to_gm"],
        free_idx=d["free_idx"], cells=np.array(d["pk_cats"], dtype=object),
        lam=info["lam"], ll_holdout=info["ll_holdout"],
        ll_uniform=info["ll_uniform"],
    )
    print(f"  Saved fit -> {OUTPUT_DIR / f'stage3_params_{year}.npz'}")

    post = pitch_posterior(d, log_w)
    aim_distribution(d, post, log_w, year)

    results = score(d, post)
    print(f"\nScored {len(results)} pitchers.")

    results_q = results[results["n_pitches"] >= MIN_PITCHES_SCORED].copy()
    print(f"  On leaderboard (>={MIN_PITCHES_SCORED} scored pitches): "
          f"{len(results_q)} / {len(results)}")

    results_q["Command_Plus_Raw"] = command_plus(results_q["avg_miss_raw"].values)
    results_q["Command_Plus_Adj"] = command_plus(results_q["avg_miss_std"].values)
    results_q = results_q.sort_values("Command_Plus_Adj", ascending=False).reset_index(drop=True)

    out_lb  = OUTPUT_DIR / f"leaderboard_{year}.csv"
    out_all = OUTPUT_DIR / f"command_plus_{year}.parquet"
    results_q.to_csv(out_lb, index=False)
    results.to_parquet(out_all, index=False)

    print(f"\nSaved leaderboard -> {out_lb}")
    print(f"Saved all results  -> {out_all}")
    print(f"\nTop 10 (Command+ Adj):")
    print(results_q[[
        "Pitcher", "PitcherTeam", "velo_bin", "n_pitches",
        "avg_miss_raw", "avg_miss_std", "aim_spread",
        "Command_Plus_Raw", "Command_Plus_Adj",
    ]].head(10).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stage 3: per-cell intent density surface over the plate.",
    )
    parser.add_argument("--year", type=int, required=True,
                        help="season year (e.g. 2024)")
    # No sampler flags: Stage 3 is fit by optimisation, not MCMC. The flags are
    # still accepted and ignored so run_pipeline.py can forward its globals.
    add_sampler_args(parser, DEFAULT_SAMPLER)
    args = parser.parse_args()
    main(args.year)
