"""
Tensor-product B-spline bases for the Stage 1 movement surface.

Replaces the KMeans `pitch_cluster` index. The old design cut seven continuous
features into 80 hard labels and indexed `u_cxb` on them, which cost three
things:

  - Pitches whose cluster Stage 1 never fit (2.74% of 2024) fell through to the
    reference level. `compute_mu_pop` gave them the grand mean, which is
    defensible, but `compute_sigma_pop` silently handed them the alphabetically
    first cluster's execution spread, and sigma is both a likelihood weight and
    the Mahalanobis denominator in Command+ Adj.
  - `k` came from an elbow heuristic per (hand, pitch type), with no uncertainty
    attached to a choice that decides the resolution of the whole stage.
  - Cluster membership ranged from 1 to 86 pitchers. A cluster holding one
    pitcher means `u_cxb` for it IS that pitcher's aim, absorbed into the
    population term before Stage 3 ever sees him.

A smooth surface has no cells to be a singleton in, and an unseen pitch shape
interpolates from its neighbours instead of falling off a cliff.

Basis layout
------------
The movement plane is fit JOINTLY as a 2-D tensor product over
(HorzBreak, InducedVertBreak). Additive marginals would assert that a given
horizontal break means the same thing at every vertical break, which is false:
-8" horizontal is a sweeper at +2" IVB and a changeup at -6".

RelSpeed enters as a separate additive 1-D smooth rather than a third tensor
dimension. That is a deliberate size trade — 8x8 + 8 = 72 coefficients against
8x8x6 = 384 — and it is the one place this module assumes separability.

SpinRate and SpinAxis are dropped. Spin causes break, and break is measured
directly, so SpinRate is largely redundant. SpinAxis is CIRCULAR: 359 and 1 are
neighbours, and a plain B-spline basis (like the StandardScaler the old KMeans
path used) treats them as maximally distant. Re-adding it needs a periodic
basis, not this module.

Handedness
----------
HorzBreak is mirrored for left-handed pitchers so a lefty's slider lands in the
same movement region as a righty's and both share one surface, doubling the
effective data behind it. The RESPONSE is not mirrored — the plate is fixed and
pitchers of both hands aim at specific plate locations, not mirrored ones. The
systematic difference between hands is carried by the categorical block
(pitch_type x PitcherThrows x BatterSide) instead. That split is an assumption
worth validating, not a derivation.

Identifiability
---------------
B-spline bases sum to 1 at every x, so the raw column space contains the
constant vector and is collinear with the model intercept. Columns are centred
on their training means (stored in the spec, reapplied at scoring time) and the
last column is dropped, which leaves the block full rank and orthogonal to
alpha. Without this the intercept and the surface trade off freely and NUTS
crawls along the ridge.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.interpolate import BSpline

# Columns this module creates on the frame. Stage 1 and utils.scoring both call
# `add_spline_features` so the two share one definition of what gets fit — a
# mismatch would shift every pitch's aim by however much the mirroring differs.
FEAT_HB   = "sp_hb"      # HorzBreak, mirrored for LHP
FEAT_IVB  = "sp_ivb"     # InducedVertBreak
FEAT_VELO = "sp_velo"    # RelSpeed
FEAT_VALID = "sp_valid"  # all three present and finite

RAW_FEATURES = ["HorzBreak", "InducedVertBreak", "RelSpeed"]


def add_spline_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Attach the continuous predictors the Stage 1 surface is built on.

    Mirroring is NOT done here any more — it is done once, for every axis at
    once, by utils.matchup.add_matchup, which must have been called first. This
    module previously flipped HorzBreak on its own while PlateLocSide stayed
    unmirrored, which is coherent for the height model and incoherent for the
    side one. Reading the already-mirrored column makes it impossible for the
    two axes to disagree.

    Adds FEAT_VALID rather than dropping rows. Stage 1 fits on valid rows only,
    but Stages 2-3 must score every pitch, and a pitch with untracked movement
    should get the population grand mean (surface contribution 0) exactly as an
    unknown cluster used to — not be silently dropped from the leaderboard.
    """
    df = df.copy()
    for col in RAW_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "HorzBreak_m" not in df.columns:
        raise ValueError(
            "call utils.matchup.add_matchup(df) before add_spline_features — "
            "the movement surface is fit in the mirrored frame."
        )
    hb = df["HorzBreak_m"].to_numpy(float)

    df[FEAT_HB]   = hb
    df[FEAT_IVB]  = df["InducedVertBreak"].to_numpy(float)
    df[FEAT_VELO] = df["RelSpeed"].to_numpy(float)
    df[FEAT_VALID] = (
        np.isfinite(df[FEAT_HB]) & np.isfinite(df[FEAT_IVB]) & np.isfinite(df[FEAT_VELO])
    )
    return df


COMBO_KEY = ["PitcherId", "pitch_type"]


def combo_frame(df: pd.DataFrame):
    """
    Collapse to one row per (PitcherId, pitch_type) — the pitcher's typical
    shape for that pitch — and return (combo_df, combo_id).

    This is where the surface is evaluated, not per pitch, for two reasons.

    Faithfulness: the KMeans clusters it replaces were built from pitcher x
    pitch-type MEANS, so evaluating the surface at those same means is the
    exact smooth analogue of what V8 did. Evaluating per pitch would instead
    condition a pitcher's intended location on the realized break of the
    individual pitch, which is an outcome, not an input to his intent.

    Cost: movement is overwhelmingly a stable pitcher x type property
    (within/between SD ratio 0.10-0.34), so per-pitch evaluation buys almost
    no extra signal while turning a 1,904-row matrix product into a 41,199-row
    one inside every gradient. Measured: 2.69 -> 9.14 draws/s/chain, against
    V8's 7.8.
    """
    key = (df["PitcherId"].astype(str) + "|" + df["pitch_type"].astype(str))
    cats = pd.Categorical(key)
    cid = cats.codes.astype(np.int32)

    feats = [FEAT_HB, FEAT_IVB, FEAT_VELO]
    combo = (df.assign(_cid=cid).groupby("_cid")[feats].mean()
               .reindex(range(len(cats.categories))))
    combo["pitch_type"] = [c.split("|", 1)[1] for c in cats.categories]
    combo[FEAT_VALID] = np.isfinite(combo[feats].to_numpy(float)).all(axis=1)
    return combo.reset_index(drop=True), cid


def make_knots(values: np.ndarray, n_basis: int, degree: int = 3) -> np.ndarray:
    """
    Clamped knot vector with interior knots at data quantiles.

    A cubic basis with n_basis functions needs (n_basis - degree - 1) interior
    knots between clamped boundaries. Quantile placement puts resolution where
    the pitches actually are: fastball movement is dense and knuckle-curve
    movement is not, and evenly spaced knots would spend the same number of
    coefficients on both.
    """
    n_interior = n_basis - degree - 1
    if n_interior < 0:
        raise ValueError(
            f"n_basis={n_basis} is below the minimum {degree + 1} for degree {degree}"
        )
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        raise ValueError("no finite values to place knots on")

    lo, hi = float(v.min()), float(v.max())
    if hi <= lo:
        hi = lo + 1e-6

    if n_interior:
        qs = np.linspace(0.0, 1.0, n_interior + 2)[1:-1]
        interior = np.quantile(v, qs)
        # Ties collapse a knot when a feature is spiky; nudge them apart rather
        # than emitting a degenerate basis that silently loses a column.
        interior = np.maximum.accumulate(
            np.clip(interior, lo + 1e-6, hi - 1e-6)
        )
        for i in range(1, len(interior)):
            if interior[i] <= interior[i - 1]:
                interior[i] = np.nextafter(interior[i - 1], hi)
    else:
        interior = np.array([], dtype=float)

    return np.concatenate([
        np.repeat(lo, degree + 1), interior, np.repeat(hi, degree + 1)
    ])


def _basis_1d(x: np.ndarray, knots: np.ndarray, degree: int) -> np.ndarray:
    """
    Cubic B-spline design matrix, with values clipped into the knot span.

    Clipping means a pitch outside the training movement range is evaluated at
    the nearest boundary rather than extrapolated. Cubic extrapolation past the
    data is wild, and a 105 mph misread should get the fastest fitted aim, not a
    polynomial excursion.
    """
    lo, hi = knots[degree], knots[len(knots) - degree - 1]
    span = max(hi - lo, 1e-9)
    xc = np.clip(np.asarray(x, dtype=float), lo + 1e-9 * span, hi - 1e-9 * span)
    xc = np.nan_to_num(xc, nan=lo + 0.5 * span)
    return np.asarray(
        BSpline.design_matrix(xc, knots, degree, extrapolate=False).todense()
    )


def _row_kron(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise Kronecker product: (N, p) x (N, q) -> (N, p*q)."""
    return (a[:, :, None] * b[:, None, :]).reshape(a.shape[0], -1)


@dataclass
class SplineSpec:
    """
    A serializable basis. Everything needed to rebuild the exact same columns
    at scoring time lives here, so Stages 2-3 never re-derive knots from their
    own data — which would silently shift the basis and with it every aim.
    """
    name:     str
    features: list[str]
    knots:    list[np.ndarray]
    degree:   int = 3
    col_mean: np.ndarray | None = None
    drop_last: bool = True
    basis_scale: float = 1.0
    keep:     np.ndarray | None = None
    group_col:  str | None = None
    group_mean: dict = field(default_factory=dict)
    ortho_ref:  "SplineSpec | None" = None
    ortho_coef: np.ndarray | None = None
    rot:      np.ndarray | None = None
    _n_raw:   int = field(default=0)

    @property
    def n_cols(self) -> int:
        if self.rot is not None:
            return int(self.rot.shape[1])
        if self.keep is not None:
            return int(self.keep.size)
        return self._n_raw - 1 if self.drop_last else self._n_raw

    def _raw(self, df: pd.DataFrame) -> np.ndarray:
        mats = [
            _basis_1d(df[f].to_numpy(float), k, self.degree)
            for f, k in zip(self.features, self.knots)
        ]
        out = mats[0]
        for m in mats[1:]:
            out = _row_kron(out, m)
        return out

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Basis columns for every row of `df`, centred and reduced to full rank.

        Rows where FEAT_VALID is False are zeroed, so an untracked pitch
        contributes nothing to the surface and lands on the population mean.
        """
        X = self._raw(df)
        if self.col_mean is not None:
            X = X - self.col_mean[None, :]
        if self.drop_last:
            X = X[:, :-1]
        if self.keep is not None:
            X = X[:, self.keep]
        if self.group_col is not None:
            # Sweep out each group's mean so the surface is orthogonal to the
            # categorical block that carries the same grouping. An unseen group
            # subtracts nothing and therefore keeps the global centring, which
            # is the right fallback: no group mean was ever measured for it.
            g = df[self.group_col].to_numpy()
            zero = np.zeros(X.shape[1])
            X = X - np.stack([self.group_mean.get(k, zero) for k in g])
        if self.ortho_ref is not None:
            # Remove whatever the reference basis can already explain, so this
            # block carries only its own independent signal. See fit_spec.
            X = X - self.ortho_ref.transform(df) @ self.ortho_coef
        if self.rot is not None:
            X = X @ self.rot
        if FEAT_VALID in df.columns:
            X = X * df[FEAT_VALID].to_numpy(bool)[:, None]
        return np.ascontiguousarray(X, dtype=np.float64)


# A tensor column is kept only if it carries this share of the busiest
# column's mass. The movement plane is a diagonal cloud, not a rectangle, so a
# full 8x8 grid has corners (big arm-side run AND big drop) that essentially no
# pitch occupies. Their coefficients are unidentified and would just resample
# the prior every iteration — harmless to the fit, pure cost to the sampler.
MIN_REL_OCCUPANCY = 1e-3

# Singular directions below this fraction of the leading one are dropped in the
# rotation step. Purely numerical: a direction this small cannot move any
# fitted value, so keeping it only adds a parameter that resamples its prior.
SVD_TOL = 1e-6


def fit_spec(df: pd.DataFrame, name: str, features: list[str],
             n_basis: list[int], degree: int = 3,
             group_col: str | None = None,
             ortho_to: "SplineSpec | None" = None,
             min_rel_occupancy: float = MIN_REL_OCCUPANCY,
             svd_tol: float = SVD_TOL) -> SplineSpec:
    """
    Build a spec from training data: knots at its quantiles, centring at its
    column means, occupancy pruning, optional group centring, and the basis
    scale used to calibrate the coefficient prior.

    `group_col` is the identifiability fix that matters most in practice. Pitch
    type is very nearly a deterministic function of movement — a slider owns
    its own region of the plane — so a free movement surface and a categorical
    pitch-type effect are close to redundant. Fitting both left NUTS crawling
    along that ridge: every chain hit max treedepth and 60 draws took 615
    seconds. Sweeping each group's mean out of the basis makes the surface
    carry only WITHIN-type movement structure, orthogonal by construction to
    the between-type effect the categorical block estimates.

    `basis_scale` is sqrt(mean_i sum_j B_ij^2). Coefficients get an i.i.d.
    Normal(0, tau) prior, so the resulting function has pointwise SD
    approximately tau * basis_scale. Stage 1 divides the empirically derived
    TAU_CXB rates by this number, which keeps the prior on the SURFACE at the
    same width as the prior the old per-cluster `u_cxb` carried, instead of
    quietly changing how much movement structure the model is willing to see.
    """
    valid = df[df[FEAT_VALID]] if FEAT_VALID in df.columns else df
    knots = [make_knots(valid[f].to_numpy(float), n, degree)
             for f, n in zip(features, n_basis)]

    spec = SplineSpec(name=name, features=list(features), knots=knots,
                      degree=degree, col_mean=None, drop_last=True,
                      group_col=group_col)
    raw = spec._raw(valid)
    spec._n_raw = raw.shape[1]
    spec.col_mean = raw.mean(axis=0)

    centred = (raw - spec.col_mean[None, :])[:, :-1]
    mass = np.abs(raw[:, :-1]).sum(axis=0)
    spec.keep = np.flatnonzero(mass >= min_rel_occupancy * mass.max()).astype(int)
    centred = centred[:, spec.keep]

    if group_col is not None:
        g = valid[group_col].to_numpy()
        spec.group_mean = {
            k: centred[g == k].mean(axis=0) for k in pd.unique(g)
        }
        centred = centred - np.stack([spec.group_mean[k] for k in g])

    # --- make this block independent of another one --------------------------
    # Two bases that overlap leave the split of credit between them undefined,
    # and with a strong signal the sampler simply does not converge on one
    # answer. The movement surface and the velocity smooth overlap at canonical
    # correlation 0.45 — hard pitches break differently — and on the mirrored
    # 2024 side model that produced four chains reporting tau_velo at 0.077,
    # 0.080, 1.171 and 4.400, with R-hat 1.50 and ESS 7. tau_move was stable at
    # 0.450-0.452 in every chain, so only the shared direction was in dispute.
    #
    # Projecting this basis off the reference one sends that correlation to
    # exactly 0. It costs nothing: the two blocks TOGETHER still span the same
    # space (verified, max fitted difference 3.7e-14), and 94.5% of the velocity
    # basis scale survives, so velocity remains a real term rather than
    # collapsing the way tau_dev and u_count did.
    #
    # FOLLOW-UP: driving that correlation to 0 did NOT rescue the side model.
    # The chain values quoted above were measured with this orthogonalisation
    # already active, so movement-vs-velocity collinearity was not the whole
    # story: the residual non-identifiability is internal to the velocity basis
    # (the top three coefficients, over a region holding 1.3% of pitches) and
    # side simply has no velocity signal to pin them with. Stage 1 now fits the
    # velo smooth for height only. This block stays because height still uses
    # it, and the canonical correlation of 0.45 is real there too.
    if ortho_to is not None:
        ref = ortho_to.transform(valid)
        spec.ortho_ref = ortho_to
        spec.ortho_coef = np.linalg.lstsq(ref, centred, rcond=None)[0]
        centred = centred - ref @ spec.ortho_coef

    # --- rotate to the design's own principal axes ---------------------------
    # Neighbouring B-spline columns overlap heavily, so the raw basis is badly
    # conditioned and NUTS has to crawl along every correlated pair. V8's
    # `u_cxb` was a gather — one coefficient per row, no correlation — and ran
    # at 31 draws/s; the unrotated basis ran at 0.3-0.6.
    #
    # This is a pure reparameterisation, not a different model. With B = U S V'
    # and an i.i.d. Normal(0, tau) prior on w, setting v = V'w gives v the SAME
    # i.i.d. prior (V is orthogonal) and Bw = (BV)v, so likelihood and prior are
    # both unchanged while the design's columns become orthogonal. Row norms are
    # preserved too, so basis_scale — and therefore the prior calibration — is
    # identical either way.
    #
    # Directions below `svd_tol` of the leading singular value are dropped.
    # Those are numerically null: they contribute nothing to any fitted value
    # and would spend every iteration resampling their prior.
    _, sv, vt = np.linalg.svd(centred, full_matrices=False)
    n_keep = int(np.sum(sv >= svd_tol * sv.max())) if sv.size else 0
    spec.rot = np.ascontiguousarray(vt[:n_keep].T)
    centred = centred @ spec.rot

    spec.basis_scale = float(np.sqrt(np.mean((centred ** 2).sum(axis=1))))
    if not np.isfinite(spec.basis_scale) or spec.basis_scale <= 0:
        spec.basis_scale = 1.0
    return spec


def describe(spec: SplineSpec) -> str:
    rng = ", ".join(
        f"{f}[{k[spec.degree]:.1f},{k[len(k) - spec.degree - 1]:.1f}]"
        for f, k in zip(spec.features, spec.knots)
    )
    orth = f" _|_ {spec.ortho_ref.name}" if spec.ortho_ref is not None else ""
    return (f"{spec.name}: {spec.n_cols} cols over {rng}  "
            f"basis_scale={spec.basis_scale:.3f}{orth}")
