"""
Command+ V8 — cross-stage diagnostics.

One pass over all three stages, reading only what the pipeline already wrote.
Nothing is re-fit: the per-pitch aim centre, its posterior SD, and the 5x5 zone
grid all come straight out of `aim_distribution_<year>.parquet`, which Stage 3
builds from the same posterior means `score()` uses. Joining that back to the
clustered pitches on (GameID, PitchNo) recovers the actual plate location, the
batter, and everything else needed here.

What it reports
---------------
1. Freshness — whether each stage's outputs predate its inputs. Stage 3 reads
   Stage 1/2 from disk, so a stale Stage 3 silently scores against a previous
   pipeline's numbers and nothing in the run log would say so.
2. Posterior distributions — the scalar and variance parameters of all three
   stages, with divergences / R-hat / ESS on the same footing.
3. Miss distance three ways (see `miss_distances`).
4. An intended-zone heatmap for one well-identified pitcher/batter matchup.

Usage:
    uv run python diagnostics/diagnostics.py --year 2024
    uv run python diagnostics/diagnostics.py --year 2024 --pitcher "Last, First"
    uv run python diagnostics/diagnostics.py --year 2024 --no-plots
"""

from __future__ import annotations

import argparse
import pickle
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import arviz as az
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
from scipy.stats import norm

from utils import plate
from config import (
    DENSITY_BW,
    OUTPUT_DIR,
    ZONE_SIDE_EDGES, ZONE_HEIGHT_EDGES,
    MIN_PITCHES_SCORED,
)

CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
DIAG_DIR = OUTPUT_DIR / "diagnostics"

# Plate half-width + ball radius, and the nominal height zone. Same numbers the
# zone grid edges are built from, repeated here for the plot overlay.
ZONE_HALF_WIDTH = 0.83
ZONE_BOTTOM, ZONE_TOP = 1.5, 3.5

# A pitch type needs this many pitches in a matchup before its heatmap means
# anything; below it the panel is one or two pitches of noise.
MIN_CLUSTER_PITCHES = 5

# White -> yellow -> red. Red = "most likely he was aiming here", which is the
# one thing the heatmap has to communicate without a caption.
AIM_CMAP = LinearSegmentedColormap.from_list(
    "aim", ["#ffffff", "#ffeda0", "#feb24c", "#f03b20", "#8c0d0d"]
)


# ----------------------------------------------------------------------------
# reporting helpers
# ----------------------------------------------------------------------------

class Report:
    """Tee the report to stdout and a text file, so a run is archivable."""

    def __init__(self, path: Path):
        self.path = path
        self.lines: list[str] = []

    def __call__(self, text: str = ""):
        print(text, flush=True)
        self.lines.append(text)

    def rule(self, title: str):
        self(f"\n{'=' * 78}\n{title}\n{'=' * 78}")

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("\n".join(self.lines) + "\n", encoding="utf-8")


def flag(ok: bool, warn: str = "*** CHECK ***") -> str:
    return "OK" if ok else warn


# ----------------------------------------------------------------------------
# 1. freshness
# ----------------------------------------------------------------------------

def check_freshness(rep: Report, year: int) -> bool:
    """
    Verify each stage's outputs are newer than the inputs it consumed.

    Stages hand off through the filesystem, not through the process that
    produced them, so re-running Stage 1 without re-running Stage 3 leaves a
    leaderboard built on numbers that no longer exist anywhere. That is silent
    at run time; it is only visible in the mtimes.
    """
    rep.rule("1. OUTPUT FRESHNESS")

    def mtime(p: Path):
        return p.stat().st_mtime

    chain = [
        ("clustered", [OUTPUT_DIR / f"clustered_{year}.parquet"]),
        ("stage 1", [OUTPUT_DIR / f"stage1_{d}_{year}.pkl" for d in ("height", "side")]),
        ("stage 2", [OUTPUT_DIR / f"stage2_{d}_{year}.parquet" for d in ("height", "side")]),
        ("stage 3", [OUTPUT_DIR / f"aim_distribution_{year}.parquet",
                     OUTPUT_DIR / f"leaderboard_{year}.csv"]),
    ]

    missing = [p for _, paths in chain for p in paths if not p.exists()]
    if missing:
        rep("Missing required outputs:")
        for p in missing:
            rep(f"  {p}")
        raise SystemExit("Run the pipeline for this year before diagnosing it.")

    stamps = []
    for label, paths in chain:
        ts = [mtime(p) for p in paths]
        stamps.append((label, min(ts), max(ts)))
        rep(f"  {label:<10} {pd.Timestamp.fromtimestamp(min(ts)):%Y-%m-%d %H:%M:%S}"
            f"  ->  {pd.Timestamp.fromtimestamp(max(ts)):%Y-%m-%d %H:%M:%S}")

    stale = False
    for (prev_label, _, prev_new), (label, cur_old, _) in zip(stamps, stamps[1:]):
        if cur_old < prev_new:
            lag = (prev_new - cur_old) / 60.0
            rep(f"\n  *** STALE *** {label} predates {prev_label} by {lag:.0f} min.")
            rep(f"      {label} was built from an older {prev_label}; "
                f"re-run it before trusting anything below.")
            stale = True

    if not stale:
        rep("\n  Stage ordering consistent — every stage is newer than its inputs.  OK")
    return not stale


# ----------------------------------------------------------------------------
# 2. posterior distributions, all three stages
# ----------------------------------------------------------------------------

def load_trace(path: Path):
    if not path.exists():
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return az.from_netcdf(str(path))


def summarize(rep: Report, label: str, trace, var_names: list[str]):
    """
    Divergences / R-hat / ESS plus a per-parameter posterior table.

    var_names is kept deliberately narrow. Stage 2's `omega` is one parameter
    per batter x cluster and Stage 3's `delta` one per pitcher x cluster;
    summarizing either costs minutes and says nothing a spread summary doesn't.
    """
    if trace is None:
        rep(f"  [{label}] checkpoint missing — skipped")
        return None

    present = [v for v in var_names if v in trace.posterior]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        summary = az.summary(trace, var_names=present)

    n_div = int(trace.sample_stats["diverging"].sum().values)
    n_tot = int(trace.sample_stats["diverging"].size)
    max_rhat = float(summary["r_hat"].max())
    min_ess = float(summary["ess_bulk"].min())

    rep(f"\n  [{label}]  {n_tot:,} draws "
        f"({trace.posterior.sizes['chain']} chains x {trace.posterior.sizes['draw']})")
    rep(f"    Divergences : {n_div} ({n_div / n_tot:.2%})  {flag(n_div == 0)}")
    rep(f"    Max R-hat   : {max_rhat:.4f}  {flag(max_rhat < 1.01, '*** WARNING ***')}")
    rep(f"    Min ESS     : {min_ess:.0f}  {flag(min_ess > 400, '*** WARNING ***')}")

    cols = ["mean", "sd", "hdi_3%", "hdi_97%", "ess_bulk", "r_hat"]
    tbl = summary[cols].round(4)
    rep("\n" + "\n".join("    " + ln for ln in tbl.to_string().splitlines()))
    return summary


def stage_posteriors(rep: Report, year: int) -> dict:
    """Load every checkpoint and report the parameters that carry the model."""
    rep.rule("2. POSTERIOR DISTRIBUTIONS")

    traces = {}

    rep("\n--- Stage 1: population aim + execution spread " + "-" * 30)
    # tau_velo exists in the height trace only — the velo smooth is not fitted
    # for side (see stage1.py). Intersect rather than hard-coding per dimension.
    s1_vars = ["alpha", "nu", "tau_move", "tau_velo", "tau_tps", "tau_tc"]
    for dim in ("height", "side"):
        t = load_trace(CHECKPOINT_DIR / f"stage1_{dim}_{year}.nc")
        traces[f"stage1_{dim}"] = t
        present = ([v for v in s1_vars if v in t.posterior]
                   if t is not None else s1_vars)
        summarize(rep, f"stage1 {dim}", t, present)

    # The sigma submodel is a fixed input to Stages 2-3, so its coefficients are
    # worth printing in feet rather than left on the log-z scale they are fit on.
    for dim in ("height", "side"):
        pkl = OUTPUT_DIR / f"stage1_{dim}_{year}.pkl"
        if not pkl.exists():
            continue
        with open(pkl, "rb") as f:
            s1 = pickle.load(f)
        # V10: the sigma submodel is intercept + a movement spline surface +
        # pitch-type and count effects. The surface has no per-level names to
        # print, so its contribution is summarised by its spread; the
        # categorical parts still read as multipliers.
        base = float(np.exp(s1["sigma_intercept"]) * s1["y_sd"])
        rep(f"\n    [stage1 {dim}] execution sigma — intercept {base:.3f} ft; "
            f"multipliers exp(beta):")
        for key, label in (("sigma_typ", "pitch type"), ("sigma_cnt", "count")):
            mult = pd.Series({k: np.exp(v) for k, v in s1[key].items()}).sort_values()
            rep(f"      {label:<10} widest:   "
                + ", ".join(f"{k}={v:.3f}" for k, v in mult.tail(3).items()))
            rep(f"      {label:<10} tightest: "
                + ", ".join(f"{k}={v:.3f}" for k, v in mult.head(3).items()))
        bsm = np.asarray(s1["beta_sigma_move"])
        rep(f"      movement surface: {len(bsm)} coefficients, "
            f"|beta| max {np.abs(bsm).max():.3f}")
        rep(f"      nu = {s1['nu']:.2f}  (lower = heavier tails = more wild misses)")

    rep("\n--- Stage 2: batter x cluster location adjustment " + "-" * 26)
    for dim in ("height", "side"):
        t = load_trace(CHECKPOINT_DIR / f"stage2_{dim}_{year}.nc")
        traces[f"stage2_{dim}"] = t
        summarize(rep, f"stage2 {dim}", t, ["tau", "tau_bin"])

        omega_path = OUTPUT_DIR / f"stage2_{dim}_{year}.parquet"
        if omega_path.exists():
            om = pd.read_parquet(omega_path)[f"omega_{dim}"]
            rep(f"    omega over {len(om):,} batter x cluster cells: "
                f"sd={om.std():.4f} ft, "
                f"range=[{om.min():+.3f}, {om.max():+.3f}], "
                f"|omega|>0.10 ft: {(om.abs() > 0.10).mean():.1%}")

    rep("\n--- Stage 3: intent density surface " + "-" * 40)
    # Stage 3 is fit by optimisation, not sampling, so there is no trace to
    # summarise — the parameter posterior was never used downstream (the output
    # is p(intent | pitch), derived from the fitted surface) and sampling
    # 31,040 parameters for it projected to ~23 hours.
    npz = OUTPUT_DIR / f"stage3_params_{year}.npz"
    if npz.exists():
        z = np.load(npz, allow_pickle=True)
        rep(f"\n  [stage3] penalized MLE — lambda={float(z['lam']):.1f}")
        rep(f"    held-out log lik / pitch : {float(z['ll_holdout']):+.4f}")
        rep(f"    uniform-intent baseline  : {float(z['ll_uniform']):+.4f}")
        gain = float(z["ll_holdout"]) - float(z["ll_uniform"])
        rep(f"    gain from fitted intent  : {gain:+.4f} nats/pitch  "
            + flag(gain > 0, "*** NO GAIN — intent surfaces are not predictive ***"))
        rep("    A positive gain is the falsifiability check: the fitted intent")
        rep("    surfaces predict pitches they were not fit on. If this were ~0,")
        rep("    the multi-target premise would be unsupported by the data.")
    else:
        rep(f"\n  [stage3] {npz.name} missing — re-run Stage 3.")

    # V10 has no delta/gamma: a cell is a smooth density over the plate, not a
    # target location. What matters is how much a cell departs from its pooled
    # group surface, and how peaked the surfaces are.
    surf_path = OUTPUT_DIR / f"intent_surface_{year}.parquet"
    if surf_path.exists():
        sf = pd.read_parquet(surf_path)
        dens = np.stack(sf["surface"].apply(np.asarray).to_numpy())
        ent = -(dens * np.log(np.clip(dens, 1e-300, None))).sum(axis=1)
        n_cells = dens.shape[1]
        rep(f"\n    Intent surfaces over {len(sf):,} cells "
            f"({n_cells} mesh points each)")
        rep(f"      entropy: mean {ent.mean():.2f} nats, "
            f"range [{ent.min():.2f}, {ent.max():.2f}]  "
            f"(uniform = {np.log(n_cells):.2f})")
        # Effective area holding 50% of intent — a direct read on how spread a
        # pitcher's intended locations are, in mesh cells.
        srt = -np.sort(-dens, axis=1)
        half = (srt.cumsum(axis=1) < 0.5).sum(axis=1) + 1
        rep(f"      mesh points holding 50% of intent: median {np.median(half):.0f} "
            f"of {n_cells}  (smaller = more concentrated intent)")

    return traces


def plot_posteriors(traces: dict, year: int, outdir: Path) -> list[Path]:
    """One figure per stage: posterior densities for the parameters that matter."""
    specs = [
        ("stage1", ["alpha", "nu", "tau_move", "tau_velo", "tau_tps", "tau_tc"],
         ["stage1_height", "stage1_side"], "Stage 1 — population aim"),
        ("stage2", ["tau", "tau_bin"],
         ["stage2_height", "stage2_side"], "Stage 2 — batter x cluster adjustment"),
    ]

    paths: list[Path] = []
    for key, var_names, trace_keys, title in specs:
        available = [k for k in trace_keys if traces.get(k) is not None]
        if not available:
            continue
        vars_present = [
            v for v in var_names
            if any(v in traces[k].posterior for k in available)
        ]
        if not vars_present:
            continue
        n = len(vars_present)
        ncol = min(3, n)
        nrow = int(np.ceil(n / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.0 * nrow),
                                 squeeze=False)
        for ax in axes.ravel()[n:]:
            ax.axis("off")

        for i, v in enumerate(vars_present):
            ax = axes.ravel()[i]
            for k, color in zip(available, ("#1f77b4", "#d62728")):
                post = traces[k].posterior
                if v not in post:
                    continue
                ax.hist(post[v].values.ravel(), bins=60, density=True, alpha=0.45,
                        color=color, label=k.split("_")[-1])
            ax.set_title(v, fontsize=11)
            ax.tick_params(labelsize=8)
            ax.spines[["top", "right"]].set_visible(False)
            if i == 0:
                ax.legend(fontsize=8, frameon=False)

        fig.suptitle(f"{title} — {year}", fontsize=13)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        p = outdir / f"posteriors_{key}_{year}.png"
        fig.savefig(p, dpi=140)
        plt.close(fig)
        paths.append(p)

    # Stage 3: no trace to plot (fit by optimisation), so the figure is about
    # the fitted surfaces themselves — how concentrated each cell's intent is.
    surf_path = OUTPUT_DIR / f"intent_surface_{year}.parquet"
    if surf_path.exists():
        sf = pd.read_parquet(surf_path)
        dens = np.stack(sf["surface"].apply(np.asarray).to_numpy())
        srt = -np.sort(-dens, axis=1)
        half = (srt.cumsum(axis=1) < 0.5).sum(axis=1) + 1
        ent = -(dens * np.log(np.clip(dens, 1e-300, None))).sum(axis=1)

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        axes[0].hist(half, bins=50, color="#7a5195", alpha=0.75)
        axes[0].set_xlabel("mesh points holding 50% of intent")
        axes[0].set_ylabel("cells")
        axes[0].set_title("intent concentration per cell", fontsize=11)

        axes[1].hist(ent, bins=50, color="#2b8cbe", alpha=0.75)
        axes[1].axvline(np.log(dens.shape[1]), color="k", ls="--", lw=1,
                        label="uniform")
        axes[1].set_xlabel("surface entropy (nats)")
        axes[1].set_title("how spread each cell's intent is", fontsize=11)
        axes[1].legend(fontsize=8, frameon=False)
        for ax in axes:
            ax.spines[["top", "right"]].set_visible(False)

        fig.suptitle(f"Stage 3 — intent density surfaces — {year}", fontsize=13)
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        pth = outdir / f"posteriors_stage3_{year}.png"
        fig.savefig(pth, dpi=140)
        plt.close(fig)
        paths.append(pth)

    return paths


# ----------------------------------------------------------------------------
# 3. miss distance, weighted by the posterior over the intended location
# ----------------------------------------------------------------------------

def check_integrity(rep: Report, year: int):
    """
    Look for the same physical pitch stored more than once.

    TrackMan ships a game twice — `<game>.csv` and `<game>_unverified.csv` —
    and if the loader takes both, every pitch in that game enters the model
    twice under one PitchUID. Nothing downstream can detect that: the
    likelihood simply sees twice the evidence, so every posterior SD shrinks by
    roughly sqrt(2) and the model reports confidence it has not earned. It is
    invisible in the fit diagnostics, which is why it is checked here.
    """
    rep.rule("1b. DATA INTEGRITY — DUPLICATE PITCH RECORDS")

    cols = ["GameID", "PitchNo", "PitchUID", "source_file"]
    c = pd.read_parquet(OUTPUT_DIR / f"clustered_{year}.parquet", columns=cols)

    n_rows, n_uid = len(c), c["PitchUID"].nunique()
    dup_rows = n_rows - n_uid
    rep(f"  clustered rows      : {n_rows:,}")
    rep(f"  unique PitchUID     : {n_uid:,}")
    rep(f"  duplicate records   : {dup_rows:,} ({dup_rows / max(n_rows, 1):.1%})  "
        f"{flag(dup_rows == 0)}")

    if dup_rows:
        dup_uids = c.loc[c.duplicated("PitchUID", keep=False), "PitchUID"]
        by_src = (c[c["PitchUID"].isin(set(dup_uids))]
                  .assign(kind=lambda d: np.where(
                      d["source_file"].str.contains("unverified", na=False),
                      "unverified", "verified"))
                  .groupby("kind").size())
        rep("\n  Duplicated rows by source file:")
        for k, v in by_src.items():
            rep(f"    {k:<12} {v:,}")
        rep(f"\n  *** Every stage was fit on ~{n_rows / max(n_uid, 1):.2f}x the real "
            f"pitch count. ***")
        rep("      Posterior SDs are too small by roughly that factor's square "
            "root, so\n      the aim posteriors below are tighter than the data "
            "supports. Point\n      estimates are far less affected — a duplicated "
            "pitch pulls the mean\n      toward where it already was.")
        rep("      Fix upstream in pipeline/01_cluster_pitches.py (de-duplicate on\n"
            "      PitchUID, preferring the verified file), then re-run all stages.")

    aim = pd.read_parquet(OUTPUT_DIR / f"aim_distribution_{year}.parquet",
                          columns=["GameID", "PitchNo", "PitcherId"])
    n_aim = len(aim)
    n_key = len(aim.drop_duplicates())
    rep(f"\n  aim_distribution rows: {n_aim:,}   distinct "
        f"(GameID, PitchNo, PitcherId): {n_key:,}  {flag(n_aim == n_key)}")
    return dup_rows == 0


def load_pitches(rep: Report, year: int) -> pd.DataFrame:
    """
    Aim distribution joined to the actual plate locations.

    Only the columns needed are read off the clustered parquet — it is ~83 MB
    and all but a handful of its fields are irrelevant here.

    The join key is (GameID, PitchNo, PitcherId), which is the most specific
    key `aim_distribution` carries; Stage 3 does not write PitchUID. So the
    clustered side is collapsed to one row per PitchUID first, and the aim side
    to one row per key, before joining. Without that the duplicate records
    documented above turn a 1:1 join into a many-to-many one and silently
    multiply the row count.
    """
    aim = pd.read_parquet(OUTPUT_DIR / f"aim_distribution_{year}.parquet")
    n_aim = len(aim)
    aim = aim.drop_duplicates(subset=["GameID", "PitchNo", "PitcherId"])

    need = ["GameID", "PitchNo", "PitchUID", "PitcherId", "BatterId", "Batter",
            "PitcherThrows", "PlateLocHeight", "PlateLocSide"]
    act = pd.read_parquet(OUTPUT_DIR / f"clustered_{year}.parquet", columns=need)
    act = (act.drop_duplicates(subset=["PitchUID"])
              .drop_duplicates(subset=["GameID", "PitchNo", "PitcherId"])
              .drop(columns=["PitchUID"]))

    df = aim.merge(act, on=["GameID", "PitchNo", "PitcherId"], how="left")
    for col in ("PlateLocHeight", "PlateLocSide"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    unmatched = int(df["PlateLocHeight"].isna().sum())
    df = df.dropna(subset=["PlateLocHeight", "PlateLocSide"])

    rep(f"\n  Joined for scoring: {len(df):,} distinct pitches "
        f"(from {n_aim:,} aim rows; {unmatched:,} unmatched on the clustered side)")

    # --- aim with the batter effect removed ---------------------------------
    # aim = mu_pop + omega + delta. omega is the Stage 2 batter x cluster
    # adjustment, i.e. "everyone pitches this hitter differently". Subtracting
    # it leaves where THIS pitcher aims this pitch type irrespective of who is
    # standing in, which is the pitcher-level question.
    # Must match stage2.py / stage3.py exactly, platoon term included, or every
    # lookup misses and the batter effect silently reads as zero everywhere.
    key = (df["BatterId"].astype(str) + "_"
           + df["pitch_type"] + "_"
           + df["matchup"])
    for dim, col in (("height", "h"), ("side", "s")):
        om_path = OUTPUT_DIR / f"stage2_{dim}_{year}.parquet"
        om = pd.read_parquet(om_path)
        lut = dict(zip(om["batter_x_cluster"], om[f"omega_{dim}"]))
        # Unmatched -> 0.0, the same partial-pool-to-zero default Stage 3 uses
        # for a batter x cluster cell that never qualified.
        om_vals = key.map(lut).fillna(0.0).to_numpy()
        # omega_side is fit in the mirrored frame; aim_s / prior_s come out of
        # Stage 3 already un-mirrored. Flip omega to match before subtracting,
        # or the batter effect is removed with the wrong sign for every LHP.
        if col == "s" and "mirror_sign" in df.columns:
            om_vals = om_vals * df["mirror_sign"].to_numpy()
        df[f"omega_{col}"] = om_vals
        for pre in ("aim", "prior"):
            if f"{pre}_{col}" in df.columns:
                df[f"{pre}_{col}_nb"] = df[f"{pre}_{col}"] - df[f"omega_{col}"]

    rep(f"  Batter effect (omega) matched for {key.isin(lut).mean():.1%} of pitches; "
        f"the rest were already pooled to 0 by Stage 3.")
    return df


def _band_conditional_means(mu: np.ndarray, sd: np.ndarray,
                            edges: np.ndarray) -> np.ndarray:
    """
    E[X | X in band] for each band, under X ~ Normal(mu, sd).

    The zone grid's outer bands are unbounded, so they have no geometric centre
    to measure a miss against. The truncated-normal mean is the honest
    representative: it is where the aim posterior actually puts its mass inside
    that band, and it stays finite because the normal tail does.
    """
    a = (edges[None, :-1] - mu[:, None]) / sd[:, None]
    b = (edges[None, 1:] - mu[:, None]) / sd[:, None]
    Pa, Pb = norm.cdf(a), norm.cdf(b)
    # phi(+-inf) == 0, which is exactly the unbounded-band behaviour we want.
    pa = np.where(np.isfinite(a), norm.pdf(np.where(np.isfinite(a), a, 0.0)), 0.0)
    pb = np.where(np.isfinite(b), norm.pdf(np.where(np.isfinite(b), b, 0.0)), 0.0)
    mass = Pb - Pa
    safe = np.where(mass > 1e-12, mass, 1.0)
    out = mu[:, None] - sd[:, None] * (pb - pa) / safe
    # An empty band gets the nearest finite edge rather than a garbage value.
    fallback = np.where(np.isfinite(edges[None, :-1]), edges[None, :-1], edges[None, 1:])
    return np.where(mass > 1e-12, out, np.broadcast_to(fallback, out.shape))


def miss_distances(rep: Report, df: pd.DataFrame, year: int) -> pd.DataFrame:
    """
    Average miss three ways, per pitcher — the contrast that motivates V10.

    collapsed  — distance to the cell's surface MEAN, ignoring the individual
                 pitch. This is what a one-target model reports, and when a
                 pitcher works two spots it measures the distance to a point
                 between them that he never aims at.
    point      — distance to the per-pitch posterior mean. The pitch has
                 reweighted the surface, but the answer is still collapsed to
                 one coordinate.
    posterior  — sum_m p(knot m | pitch) * ||pitch - knot m||, the expectation
                 over the whole continuous intent posterior. This is what
                 Stage 3 scores on.

    collapsed - posterior is the headline: how much averaging over a multimodal
    intent inflates the apparent miss. posterior - point is Jensen's gap, i.e.
    what remains of the intent posterior's own spread.
    """
    rep.rule("3. AVERAGE MISS, WEIGHTED BY THE POSTERIOR OVER INTENDED LOCATION")

    knots = plate.knot_grid()
    W = plate.knot_weights(df)                       # (N, M)

    y_h = df["PlateLocHeight"].to_numpy()
    y_s = df["PlateLocSide"].to_numpy()
    base_h = df["base_h"].to_numpy()
    base_s = df["base_s"].to_numpy()

    df["miss_point"] = np.hypot(y_h - df["aim_h"].to_numpy(),
                                y_s - df["aim_s"].to_numpy())
    if "prior_h" in df.columns:
        df["miss_collapsed"] = np.hypot(y_h - df["prior_h"].to_numpy(),
                                        y_s - df["prior_s"].to_numpy())
    else:
        df["miss_collapsed"] = df["miss_point"]

    dh = (y_h - base_h)[:, None] - knots[None, :, 0]
    ds = (y_s - base_s)[:, None] - knots[None, :, 1]
    df["miss_posterior"] = (W * np.hypot(dh, ds)).sum(axis=1)

    # Spread of the intent posterior in feet — how tightly the model can place
    # where he was aiming on a given pitch.
    mh = (W * knots[None, :, 0]).sum(axis=1)
    ms = (W * knots[None, :, 1]).sum(axis=1)
    spread = np.sqrt((W * (knots[None, :, 0] - mh[:, None]) ** 2).sum(axis=1)
                     + (W * (knots[None, :, 1] - ms[:, None]) ** 2).sum(axis=1))
    df["aim_spread"] = spread

    kinds = ["miss_collapsed", "miss_point", "miss_posterior"]
    rep(f"\n  Over {len(df):,} pitches (feet):\n")
    rep("    measure            mean     sd     p25     p50     p75")
    for k in kinds:
        v = df[k]
        rep(f"    {k:<16} {v.mean():6.3f} {v.std():6.3f} "
            f"{v.quantile(.25):7.3f} {v.quantile(.50):7.3f} {v.quantile(.75):7.3f}")

    gap = df["miss_collapsed"].mean() - df["miss_posterior"].mean()
    rep(f"\n  Cost of collapsing intent to one point (collapsed - posterior): "
        f"{gap:+.4f} ft ({gap / max(df['miss_posterior'].mean(), 1e-9):+.1%})")
    rep("    Positive means pitchers really do work more than one spot, and a "
        "single-target\n    model was charging them for the distance to a point "
        "between those spots.")
    rep(f"  Jensen gap (posterior - point): "
        f"{df['miss_posterior'].mean() - df['miss_point'].mean():+.4f} ft")
    rep(f"  Mean intent posterior spread: {spread.mean():.3f} ft")

    # --- per pitcher ---------------------------------------------------------
    g = df.groupby("PitcherId")
    per = pd.DataFrame({
        "Pitcher": g["Pitcher"].first(),
        "n_pitches": g.size(),
        "avg_aim_spread": g["aim_spread"].mean(),
        **{k: g[k].mean() for k in kinds},
    }).reset_index()
    per["uncertainty_penalty"] = per["miss_posterior"] - per["miss_point"]
    per["multimodality_gain"] = per["miss_collapsed"] - per["miss_posterior"]

    scored = per[per["n_pitches"] >= MIN_PITCHES_SCORED].copy()
    rep(f"\n  Per pitcher (>= {MIN_PITCHES_SCORED} pitches): {len(scored)} of {len(per)}")

    if len(scored) > 2:
        rep(f"    corr(collapsed, posterior) across pitchers: "
            f"{scored['miss_collapsed'].corr(scored['miss_posterior']):.4f}")
        a = scored.sort_values("miss_collapsed").reset_index(drop=True)
        b = scored.sort_values("miss_posterior").reset_index(drop=True)
        rank_a = dict(zip(a["PitcherId"], a.index))
        rank_b = dict(zip(b["PitcherId"], b.index))
        shift = pd.Series({k: rank_b[k] - rank_a[k] for k in rank_a})
        rep(f"    Leaderboard rank change vs a one-target model: "
            f"mean |shift| = {shift.abs().mean():.1f} places, "
            f"max = {shift.abs().max()} places")

    lb_path = OUTPUT_DIR / f"leaderboard_{year}.csv"
    if lb_path.exists():
        lb = pd.read_csv(lb_path)
        keep = [c for c in ("PitcherId", "Command_Plus_Adj", "avg_miss_raw")
                if c in lb.columns]
        lb = lb[keep].assign(PitcherId=lb["PitcherId"].astype(str))
        merged = scored.assign(PitcherId=scored["PitcherId"].astype(str)) \
                       .merge(lb, on="PitcherId", how="inner")
        if len(merged) > 2 and "avg_miss_raw" in merged:
            rep(f"\n    Cross-check against leaderboard ({len(merged)} pitchers):")
            rep(f"      corr(miss_posterior, leaderboard avg_miss_raw) = "
                f"{merged['miss_posterior'].corr(merged['avg_miss_raw']):.4f}"
                f"   (expect ~1.000 — both are the same expectation)")
            if "Command_Plus_Adj" in merged:
                rep(f"      corr(Command_Plus_Adj, aim_spread) = "
                    f"{merged['Command_Plus_Adj'].corr(merged['avg_aim_spread']):+.4f}")
                rep("      A strong negative value would mean Command+ rewards the "
                    "pitchers\n      whose intent the model is least sure about — "
                    "the V9 failure mode,\n      where target separation explained "
                    "R^2 = 0.47 of average miss.")

    return per


def plot_miss(per: pd.DataFrame, year: int, outdir: Path) -> Path:
    scored = per[per["n_pitches"] >= MIN_PITCHES_SCORED]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))

    ax = axes[0]
    for k, c in (("miss_collapsed", "#2b8cbe"), ("miss_point", "#e34a33"),
                 ("miss_posterior", "#31a354")):
        ax.hist(scored[k], bins=30, alpha=0.45, color=c, label=k)
    ax.set_xlabel("average miss distance (ft)")
    ax.set_ylabel("pitchers")
    ax.set_title("Per-pitcher average miss")
    ax.legend(fontsize=8, frameon=False)

    # The headline contrast: what a one-target model charges vs what the full
    # intent posterior does. Points above the diagonal are pitchers whose
    # apparent miss was inflated by averaging over more than one intended spot.
    ax = axes[1]
    sc = ax.scatter(scored["miss_posterior"], scored["miss_collapsed"], s=18,
                    c=scored["avg_aim_spread"], cmap="viridis")
    lim = [min(scored["miss_posterior"].min(), scored["miss_collapsed"].min()) - .02,
           max(scored["miss_posterior"].max(), scored["miss_collapsed"].max()) + .02]
    ax.plot(lim, lim, "k--", lw=1, alpha=.6)
    ax.set_xlabel("posterior-weighted miss (ft)")
    ax.set_ylabel("one-target (collapsed) miss (ft)")
    ax.set_title("Cost of collapsing intent to a point")
    fig.colorbar(sc, ax=ax).set_label("mean intent spread (ft)", fontsize=8)

    ax = axes[2]
    ax.scatter(scored["avg_aim_spread"], scored["multimodality_gain"], s=18,
               color="#8c0d0d", alpha=.7)
    ax.set_xlabel("mean intent posterior spread (ft)")
    ax.set_ylabel("collapsed - posterior miss (ft)")
    ax.set_title("Who benefits from modelling multiple targets")

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(f"Miss distance under the intended-location posterior — {year}",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    p = outdir / f"miss_distance_{year}.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return p


def pick_pitcher(df: pd.DataFrame, pitcher: str | None) -> pd.DataFrame:
    """
    The highest-volume pitcher, whose aim cells carry the most evidence.

    Volume is the whole selection criterion. Picking on a tight aim posterior
    instead returns the most degenerate cell in the season — the posterior SD
    is smallest exactly where one cell was fit to many pitches, so "tightest"
    and "least interesting to look at" are the same pitcher.
    """
    if pitcher:
        sub = df[df["Pitcher"].str.contains(pitcher, case=False, na=False)]
        if sub.empty:
            raise SystemExit(f"No pitcher matching {pitcher!r}.")
        pid = sub.groupby("PitcherId").size().idxmax()
    else:
        pid = df.groupby("PitcherId").size().idxmax()
    return df[df["PitcherId"] == pid]


def _intent_density(rows: pd.DataFrame, S: np.ndarray, H: np.ndarray) -> np.ndarray:
    """
    The intent density over a plot mesh, averaged across a set of pitches.

    Each pitch carries a posterior over the plate in the knot basis, so this is
    a genuine continuous surface — evaluated, not binned. Averaging over his
    pitches answers "where does he try to throw this pitch type", with the
    spread coming from count, batter, and from his own multimodal repertoire.

    The batter effect is removed first (the `_nb` columns), so what is left is
    where THIS pitcher aims irrespective of who is standing in.
    """
    knots = plate.knot_grid()
    W = plate.knot_weights(rows)
    sign = (rows["mirror_sign"].to_numpy(float)
            if "mirror_sign" in rows.columns else np.ones(len(rows)))
    base_h = rows["base_h"].to_numpy()
    base_s = rows["base_s"].to_numpy()

    dens = np.zeros_like(S)
    for m in range(knots.shape[0]):
        w = W[:, m]
        if w.max() <= 1e-9:
            continue
        # Un-mirror the SUM, then remove the batter effect in true coordinates.
        mu_h = base_h + knots[m, 0] - rows["omega_h"].to_numpy()
        mu_s = sign * (base_s + knots[m, 1]) - rows["omega_s"].to_numpy()
        for r in np.flatnonzero(w > 1e-9):
            dens += w[r] * np.exp(
                -0.5 * (((H - mu_h[r]) / DENSITY_BW) ** 2
                        + ((S - mu_s[r]) / DENSITY_BW) ** 2)
            )
    total = dens.sum()
    return dens / total if total > 0 else dens


def _draw_zone(ax):
    ax.add_patch(Rectangle((-ZONE_HALF_WIDTH, ZONE_BOTTOM),
                           2 * ZONE_HALF_WIDTH, ZONE_TOP - ZONE_BOTTOM,
                           fill=False, ec="black", lw=1.8, zorder=5))
    for x in ZONE_SIDE_EDGES[2:4]:
        ax.plot([x, x], [ZONE_BOTTOM, ZONE_TOP], color="black", lw=0.7,
                alpha=0.5, zorder=5)
    for y in ZONE_HEIGHT_EDGES[2:4]:
        ax.plot([-ZONE_HALF_WIDTH, ZONE_HALF_WIDTH], [y, y], color="black",
                lw=0.7, alpha=0.5, zorder=5)


def plot_pitcher_intent(rep: Report, rows: pd.DataFrame, year: int, outdir: Path,
                        make_plot: bool) -> Path | None:
    """
    Intended-location heatmap: red where he was most likely trying to throw it.

    Three rows per pitch type:

      1. Full zone view — the aim posterior in context, with the actual plate
         locations overlaid. The distance from the red centre to the circles IS
         the miss the metric charges him for.
      2. Zoomed view — the same density at a scale where its shape is visible.
         The posterior is genuinely tight (aim SD is measured in hundredths of
         a foot, because a whole pitcher x cluster cell's pitches inform one
         delta), so at zone scale it is a few pixels and shows nothing. The
         zoom is what makes it a heatmap rather than a dot.
      3. The 5x5 grid the model actually reports, each square's probability
         printed.
    """
    pitcher = rows["Pitcher"].iloc[0]
    throws = rows["PitcherThrows"].iloc[0] if "PitcherThrows" in rows else "?"

    h_edges = np.asarray(ZONE_HEIGHT_EDGES, float)
    s_edges = np.asarray(ZONE_SIDE_EDGES, float)
    n_rows_g, n_cols_g = len(h_edges) - 1, len(s_edges) - 1

    rep.rule("4. WHERE IS HE AIMING? — PITCHER-LEVEL INTENDED-ZONE HEATMAP")
    rep(f"\n  Pitcher: {pitcher} ({throws}HP), {len(rows)} pitches, "
        f"all batters pooled (Stage 2 omega removed)")
    rep(f"  Mean spread of the per-pitch intent posterior: "
        f"{rows['aim_spread'].mean():.3f} ft")

    order = rows["pitch_type"].value_counts()
    order = order[order >= MIN_CLUSTER_PITCHES]
    if order.empty:                      # fall back rather than plot nothing
        order = rows["pitch_type"].value_counts()
    clusters = list(order.index[:3])
    rep("  Pitch types shown: " + ", ".join(f"{c} (n={order[c]})" for c in clusters))

    rep("\n  The density is continuous over the plate — evaluated, never binned.")
    rep("    Per pitch it is the cell's intent surface reweighted by where that")
    rep("    pitch actually went; averaged over his pitches it is where he tries")
    rep("    to throw the type, with spread from count, batter, and his own")
    rep("    repertoire of intended spots.")

    knots = plate.knot_grid()
    for cl in clusters:
        sub_ = rows[rows["pitch_type"] == cl]
        W = plate.knot_weights(sub_)
        avg = W.mean(axis=0)
        # Effective number of distinct intended spots, measured not assumed.
        ent = -(avg * np.log(np.clip(avg, 1e-300, None))).sum()
        srt = -np.sort(-avg)
        n90 = int((srt.cumsum() < 0.90).sum() + 1)
        top = int(avg.argmax())
        rep(f"\n    {cl}  (n={len(sub_)}, {sub_['count'].nunique()} distinct counts)")
        sgn = float(np.sign(sub_["mirror_sign"].mean())) or 1.0
        rep(f"      most likely intended offset: "
            f"({sgn * knots[top, 1]:+.2f} side, {knots[top, 0]:+.2f} height) ft "
            f"from population aim, p={avg[top]:.3f}   "
            f"(side shown in true plate coordinates)")
        rep(f"      effective intended spots: {np.exp(ent):.1f}  "
            f"(basis points holding 90% of intent: {n90} of {len(avg)})")
        rep(f"      aim centre ({sub_['aim_s_nb'].mean():+.2f} side, "
            f"{sub_['aim_h_nb'].mean():.2f} height) ft   "
            f"spread across pitches: h={sub_['aim_h_nb'].std():.3f} "
            f"s={sub_['aim_s_nb'].std():.3f} ft")

    if not make_plot:
        return None

    S, H = np.meshgrid(np.linspace(-2.0, 2.0, 320), np.linspace(0.5, 4.5, 320))

    ncol = len(clusters)
    fig, axes = plt.subplots(3, ncol, figsize=(4.6 * ncol, 13.2), squeeze=False)

    for j, cl in enumerate(clusters):
        sub = rows[rows["pitch_type"] == cl]

        # --- row 1: full zone, for context -----------------------------------
        ax = axes[0, j]
        ax.pcolormesh(S, H, _intent_density(sub, S, H), cmap=AIM_CMAP,
                      shading="auto")
        _draw_zone(ax)
        ax.scatter(sub["PlateLocSide"], sub["PlateLocHeight"], s=28,
                   facecolor="none", edgecolor="#111111", lw=1.1, zorder=6,
                   label="actual pitch")
        ax.set_xlim(-2.0, 2.0)
        ax.set_ylim(0.5, 4.5)
        ax.set_aspect("equal")
        ax.set_title(f"{cl}  (n={len(sub)})", fontsize=11)
        ax.set_xlabel("PlateLocSide (ft)")
        if j == 0:
            ax.set_ylabel("PlateLocHeight (ft)")
            ax.legend(fontsize=8, loc="upper left", frameon=True)

        # --- row 2: zoomed to the posterior, so its shape is readable --------
        ax = axes[1, j]
        cs, ch = sub["aim_s"].mean(), sub["aim_h"].mean()
        # Wide enough to hold the spread of the per-pitch centres as well as
        # each one's own SD; floored so a near-degenerate cell still gets a
        # window with visible extent.
        half = max(
            2.0 * max(sub["aim_spread"].max(), DENSITY_BW),
            2.0 * max(sub["aim_h"].std(ddof=0), sub["aim_s"].std(ddof=0), 0.0),
            0.15,
        )
        zs, zh = (np.linspace(cs - half, cs + half, 300),
                  np.linspace(ch - half, ch + half, 300))
        ZS, ZH = np.meshgrid(zs, zh)
        ax.pcolormesh(ZS, ZH, _intent_density(sub, ZS, ZH), cmap=AIM_CMAP,
                      shading="auto")
        _draw_zone(ax)
        ax.set_xlim(cs - half, cs + half)
        ax.set_ylim(ch - half, ch + half)
        ax.set_aspect("equal")
        ax.set_title(f"zoom — {2 * half:.2f} ft across", fontsize=10)
        ax.set_xlabel("PlateLocSide (ft)")
        if j == 0:
            ax.set_ylabel("PlateLocHeight (ft)")

        # --- row 3: the intent surface as a function of the two axes ---------
        # Marginals of the same continuous density. No zone grid: binning a
        # continuous quantity at the output step is what V10 removed.
        ax = axes[2, j]
        dens = _intent_density(sub, ZS, ZH)
        h_marg = dens.sum(axis=1)
        s_marg = dens.sum(axis=0)
        ax.plot(ZH[:, 0], h_marg / max(h_marg.max(), 1e-12), color="#8c0d0d",
                lw=1.8, label="height")
        ax.plot(ZS[0, :], s_marg / max(s_marg.max(), 1e-12), color="#2b8cbe",
                lw=1.8, label="side")
        ax.set_xlabel("feet")
        ax.set_ylabel("relative intent density")
        ax.set_title("marginal intent density", fontsize=10)
        ax.legend(fontsize=8, frameon=False)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        f"Where is he trying to throw it?\n"
        f"{pitcher} ({throws}HP) — {year} — {len(rows)} pitches, all batters pooled\n"
        f"red = higher probability that square is the intended target\n"
        f"catcher's view; open circles = where the pitch actually went",
        fontsize=11.5,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    p = outdir / f"intended_zone_{year}.png"
    fig.savefig(p, dpi=145)
    plt.close(fig)
    return p


# ----------------------------------------------------------------------------

def main(args):
    year = args.year
    outdir = Path(args.outdir) if args.outdir else DIAG_DIR
    outdir.mkdir(parents=True, exist_ok=True)

    rep = Report(outdir / f"diagnostics_{year}.txt")
    rep(f"Command+ V10 diagnostics — year {year}")
    rep(f"Generated: {pd.Timestamp.now():%Y-%m-%d %H:%M:%S}")

    fresh = check_freshness(rep, year)
    clean = check_integrity(rep, year)
    traces = stage_posteriors(rep, year)

    df = load_pitches(rep, year)
    per = miss_distances(rep, df, year)

    figures = []
    if not args.no_plots:
        figures += plot_posteriors(traces, year, outdir)
        figures.append(plot_miss(per, year, outdir))

    rows = pick_pitcher(df, args.pitcher)
    fig_path = plot_pitcher_intent(rep, rows, year, outdir,
                                   make_plot=not args.no_plots)
    if fig_path:
        figures.append(fig_path)

    rep.rule("SUMMARY")
    rep(f"  Freshness: {'OK' if fresh else '*** STALE — re-run the affected stages ***'}")
    rep(f"  Duplicate pitch records: "
        f"{'none' if clean else '*** PRESENT — see section 1b ***'}")
    rep(f"  Pitches diagnosed: {len(df):,}   pitchers: {df['PitcherId'].nunique()}")
    if figures:
        rep("  Figures:")
        for p in figures:
            rep(f"    {p}")
    rep(f"  Report: {rep.path}")

    rep.save()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Cross-stage diagnostics for the Command+ pipeline.",
    )
    ap.add_argument("--year", type=int, required=True, help="season year (e.g. 2024)")
    ap.add_argument("--pitcher", default=None,
                    help="pitcher for the intent heatmap, substring match "
                         "(default: whoever threw the most pitches)")
    ap.add_argument("--outdir", default=None,
                    help="where to write the report and figures "
                         "(default output/diagnostics/)")
    ap.add_argument("--no-plots", action="store_true", help="text report only")
    main(ap.parse_args())
