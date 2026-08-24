import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent

DB_PATH = Path("C:/Users/aasmi/Costal Carolina/initial results/trackman.db")

OUTPUT_DIR = PROJECT_ROOT / "output"
DATA_DIR   = PROJECT_ROOT / "data"
OUTPUT_DIR.mkdir(exist_ok=True)

# Qualifying thresholds
# Who enters the MODEL, not who reaches the leaderboard — that is
# MIN_PITCHES_SCORED. Deliberately 0: partial pooling exists so a 180-pitch
# reliever can be estimated by shrinking him toward his velocity group, and
# deleting him before fitting throws away both his estimate and his
# contribution to the group. At 700 this kept 44 of 498 pitchers and 40k of
# 122k pitches, and starved the octile groups that do the pooling.
# MIN_PITCHES_SCORED is the only gate that should exclude anyone, and it gates
# the LEADERBOARD, not the fit.
MIN_PITCHES_PITCHER         = 0
MIN_PITCHES_SCORED          = 300   # leaderboard minimum

# Per-cell minimums, both 1 = no pruning. Partial pooling is what handles a
# small sample: a pitcher with 8 sliders should get a delta shrunk nearly all
# the way to the global mean, not be deleted. Pruning also dropped his PITCHES,
# so they stopped informing tau and the octile means as well — pure loss.
#
# At the old thresholds Stage 2 saw 91.5% of pitches and Stage 3 89.2%. At 1
# both see 100%, cover all 520 batters and 496 pitchers, and Stage 2's octile
# groups are BETTER populated (median 22 members vs 14) because more batters
# qualify to fill them. Cost is roughly 2x the cell parameters per stage.
#
# One dependency: with no pruning, most cells are thin, so the empirical prior
# scale must be estimated from well-measured cells only or it picks up
# execution noise instead of between-player spread. See MIN_CELLS_FOR_RATE in
# stage2.py.
MIN_PITCHES_PITCHER_CLUSTER = 1     # pitcher x pitch-type x side
MIN_PITCHES_BATTER_CLUSTER  = 1     # batter  x pitch-type x side

# --- Stage 1 movement surface (V9: splines replace KMeans clusters) ------
# The old `u_cxb` was indexed on 80 hard KMeans labels. It is now a smooth
# tensor-product B-spline surface over the movement plane plus an additive velo
# smooth. See utils/splines.py for why the plane must be fit jointly and why
# SpinRate/SpinAxis are excluded.
#
# Sizes are the resolution knob. (8, 8) yields ~52 coefficients after empty
# corners of the tensor are pruned, against the 109 cluster x side levels the
# old design carried across two surfaces. Raise to (10, 10) (~80 columns) if
# the posterior surface looks under-resolved; cost is roughly linear.
SPLINE_DEGREE        = 3
SPLINE_N_MOVE        = (8, 8)   # HorzBreak x InducedVertBreak, joint
SPLINE_N_VELO        = 8        # RelSpeed, additive
SPLINE_N_SIGMA_MOVE  = (5, 5)   # coarser surface for the log-sigma submodel

# Mirroring is no longer optional and no longer partial. utils.matchup flips
# PlateLocSide, HorzBreak and RelSide together for every LHP, which is the only
# coherent choice: flipping the predictor but not the response (what this flag
# used to do) is fine for the height model, where height is invariant under the
# reflection, and wrong for the side model. See utils/matchup.py.

# Stage 1 sampling cap
# Raised from 30 with the move to splines, and the stratification cell changed
# with it. The old cell was cluster x batter x count (109 x 12) and 30 pitches
# each gave 33k rows, which was ample for 109 cell means but thin for a surface
# whose coefficients each draw on a region of the movement plane. The cell is
# now pitch_type x PitcherThrows x BatterSide x count (7 x 2 x 2 x 12 = 336),
# so 200 per cell gives roughly 60k rows with the hand and platoon balance the
# cap exists to enforce.
# Raised 200 -> 400 with the move to the matchup frame. The stratification
# cell went from pitch_type x hand x side x count (336) to
# pitch_type x matchup x count (168), so at 200 the cap bit twice as hard
# and the training sample fell from 41,199 rows to 25,649. At 400 the
# effective sample is back where it was, with the same platoon balance.
STAGE1_CAP_PER_CELL         = 400   # pitches per type x matchup x count cell

# MCMC
# Fixed seed so a run is reproducible: the 2024 Stage 2 side model lost a whole
# chain to a bad jitter draw, and without a seed that chain can't be recreated
# to confirm it was the init and not the geometry.
MCMC_SEED            = 20260818

STAGE1_CHAINS        = 4
# Raised 1000 -> 2000. At 1000 the 2024 height fit came back at r_hat 1.02 on
# tau_move (ESS 560) and u_tps_raw (ESS 690) — over the 1.01 threshold, but with
# ZERO divergences and all four chains agreeing on w_velo to three decimals.
# That is slow mixing through the non-centred scale terms, not a geometry
# problem, and the fix for slow mixing is draws: ESS/draw ran ~0.15, so 2000
# puts bulk ESS above 1,100 and r_hat under 1.01 without touching the model.
#
# Deliberately NOT raising STAGE1_ADAPT_DELTA to compensate. A higher
# target_accept buys smaller steps to escape divergences, and this fit has none
# — it would only cost wall time. Contrast Stage 2/3 below, where 0.95 is
# earned by an actual divergence history.
STAGE1_ITER_WARMUP   = 2000
STAGE1_ITER_SAMPLING = 2000
STAGE1_ADAPT_DELTA   = 0.95
STAGE1_INIT          = "adapt_diag"

STAGE2_CHAINS        = 4
STAGE2_ITER_WARMUP   = 1000
STAGE2_ITER_SAMPLING = 1000
STAGE2_ADAPT_DELTA   = 0.95   # was PyMC's 0.8 default; one chain adapted to a
                              # step size it couldn't sustain (accept 0.15) and
                              # produced all 744 divergences on its own.
STAGE2_INIT          = "adapt_diag"

STAGE3_CHAINS        = 4
STAGE3_ITER_WARMUP   = 500
STAGE3_ITER_SAMPLING = 500
STAGE3_ADAPT_DELTA   = 0.95   # matches Stage 2. The old 0.80 note claimed "no
                              # divergences", but that run was killed 7.9% into
                              # warmup (315/4,000 draws) and proved nothing. V8
                              # is a ~11k-parameter joint model with the same
                              # funnel that gave Stage 2 744 divergences at 0.80,
                              # so pay the step-size cost up front rather than
                              # discover it hours in.
STAGE3_INIT          = "adapt_diag"

# Unused since V8. Stage 3 is now a single joint model over every pitcher, so
# parallelism is at the chain level (--cores) exactly as in Stages 1 and 2;
# there is no pitcher-level worker pool left to size.
STAGE3_MAX_WORKERS   = None

# Empirically derived prior rates (from V4/V5)
#
# TAU_CXB now governs the movement SURFACE rather than per-cluster cell means,
# and TAU_CXCOUNT governs pitch_type x count rather than cluster x count. The
# numbers are unchanged on purpose: they were calibrated as the width of the
# prior on the FUNCTION, and Stage 1 rescales them by the basis scale (see
# utils.splines.fit_spec) so the surface carries the same prior width the cells
# did. What changed is the resolution of the thing being penalised, not how
# much structure the model is willing to believe in.
TAU_CXB_RATE_H     = 2.54
TAU_COUNT_RATE_H   = 9.80
TAU_CXCOUNT_RATE_H = 10.0

TAU_CXB_RATE_S     = 1.89
TAU_COUNT_RATE_S   = 10.75
TAU_CXCOUNT_RATE_S = 10.0

# Batter-side deviation surface: how differently a given pitch shape is aimed
# against a lefty vs a righty. Half the width of the shared surface, because it
# is a deviation from an already-fitted mean rather than the mean itself, and
# because the old cluster x batter_side design gave it no separate scale at all
# — it was folded into TAU_CXB and could absorb as much as the main effect.
TAU_SPLINE_DEV_RATE_H = 5.08   # 2x TAU_CXB_RATE_H
TAU_SPLINE_DEV_RATE_S = 3.78   # 2x TAU_CXB_RATE_S

# Additive velo smooth. Same width as the movement surface: velocity was one of
# the seven KMeans features, so its aim signal used to live inside u_cxb and
# there is no reason to believe it is smaller now that it has its own term.
TAU_VELO_RATE_H    = 2.54
TAU_VELO_RATE_S    = 1.89

# Categorical block: pitch_type x PitcherThrows x BatterSide (28 levels). This
# carries the platoon/handedness structure that the mirrored movement surface
# deliberately does not, so it is given room — a slider from a LHP to a LHB is
# aimed genuinely differently from the mirrored righty equivalent.
TAU_TPS_RATE_H     = 2.54
TAU_TPS_RATE_S     = 1.89

TAU_DELTA_RATE     = 3.0    # Stage 3 pitcher aim deviation (~0.33 ft mean)

# --- Stage 3 intent density SURFACE (V10) --------------------------------
# The pitcher's intent is a SURFACE over the plate, not a set of targets.
# Location is continuous in both axes, so each cell carries a smooth density
# f_c(height, side) and "targets" are simply wherever that surface has mass.
# Nothing is binned: the surface is defined and evaluable at every point.
#
# Basis is a fine grid of Gaussian bumps with a second-order difference penalty
# across the grid — a penalized RBF surface, same construction as the Stage 1
# movement surface. Gaussian rather than B-spline for one concrete reason: the
# likelihood needs the intent density convolved with execution noise, and
# Gaussian-on-Gaussian is closed form (variances add). A B-spline basis needs
# numerical quadrature over the plate for every pitch, ~200M ops per gradient.
#
# KNOTS is the surface RESOLUTION, not a claim about how many targets exist.
# 9x9 at ~0.30 ft spacing is far finer than the ~0.9 ft at which two modes
# become distinguishable, deliberately: the penalty should decide how smooth
# the surface is, not the grid. A coarse grid imposes smoothness whether or not
# the data wanted it — the same error the KMeans elbow heuristic made.
DENSITY_KNOTS     = (9, 9)    # (height, side) surface resolution
DENSITY_EXTENT_H  = 1.2       # ft from centre, height axis
DENSITY_EXTENT_S  = 1.2       # ft from centre, side axis
DENSITY_BW        = 0.30      # Gaussian bump width, ~= knot spacing

# Low-rank penalized representation: the 81 knot weights are reparameterized
# onto the eigenbasis of the difference penalty and truncated to the leading
# DENSITY_RANK components, so a cell carries this many coefficients while the
# surface still lives on the full fine grid. Standard reduced-rank smoother.
DENSITY_RANK      = 20

# Only cells with at least this many pitches get their own surface; thinner
# cells use the pooled surface for their (velo octile x type x side) group.
# Structural fix for what broke V9: one global scale was shared by all 3,851
# cells, so the ~56% holding under 20 pitches — which can say nothing about
# intent shape — held that scale up and imposed a spurious 0.7 ft separation
# on every cell.
MIN_PITCHES_CELL_FREE = 30

TAU_G_RATE        = 1.0   # spread of the pooled surface across groups
TAU_PI_RATE       = 2.0   # how far a cell may depart from its group

# Fine grid the density is written on for AGGREGATED slices. Per-pitch rows
# store coefficients instead — the surface is analytic, so a per-pitch grid
# would be redundant and enormous at 135k pitches.
DENSITY_OUT_GRID  = (41, 41)

# --- Stage 3 two-target mixture (V9 — REMOVED, kept for reference) --------
# Superseded by the anchor density above. The two-target model was measured
# and failed on its own terms: the mixture weight came back with posterior SD
# 0.1268 against a prior SD of 0.130 (no information), no cell's offset ever
# collapsed toward zero as the design required (minimum separation 0.285 ft),
# and target separation ended up explaining R^2 = 0.47 of average miss — so
# the leaderboard was ranking pitchers substantially on a prior-driven
# quantity. See git history / README for the full post-mortem.
# A pitcher does not have one intended location per pitch type. He works both
# sides: a fastball to a RHB is aimed up-and-in on some pitches and down-and-away
# on others, and averaging them puts his "target" at a spot he never throws to.
#
# Fitting a 2-component mixture to each well-sampled cell with the execution
# noise FIXED at Stage 1's sigma_pop, 33.6% of cells prefer two targets by BIC,
# separated by a median 1.41 ft with the secondary target taking ~34% of the
# pitches. (With a free covariance only 0.7% prefer two — the components absorb
# the spread by widening instead of separating, which hides the effect. The
# fixed-sigma test is the right one, because sigma really is known here.)
#
# Prior on the offset between the two targets. Mean 1.0 ft, slightly tighter
# than the 1.41 ft observed, so a cell needs evidence to claim a second target
# and a cell without evidence collapses to offset ~ 0, i.e. one target.
TAU_OFF_RATE       = 1.0

# The primary target always carries at least W_MIN of the pitches, which is what
# breaks label switching: without it the two components are exchangeable and
# every chain can order them differently. Capping at W_MAX keeps the secondary
# component from being starved of data, which would leave its location
# unidentified — a cell says "one target" through offset -> 0, not through
# weight -> 1.
STAGE3_W_MIN       = 0.50
STAGE3_W_MAX       = 0.95

# Stage 3 bivariate miss covariance (V7).
# LKJ concentration on the 2x2 height/side correlation. For d=2, LKJ(eta) is
# exactly Beta(eta, eta) on (corr + 1) / 2, so eta=2 puts a gentle hump at
# corr=0 — sparse clusters shrink toward "circular", and a pitcher needs real
# evidence to be credited with a tilted miss ellipse.
LKJ_ETA            = 2.0

# --- Aim distribution zone grid (Stage 3 output) -------------------------
# Per-pitch posterior probability that the pitcher's INTENDED target lies in
# each zone square. 5x5 grid: the inner 3x3 is the strike zone (thirds), the
# outer ring is unbounded so every draw lands somewhere and rows sum to 1.
# Side edges use the plate half-width + ball radius (0.83 ft); height edges
# use a nominal 1.5-3.5 ft zone.
ZONE_SIDE_EDGES   = [-np.inf, -0.83, -0.277, 0.277, 0.83, np.inf]
ZONE_HEIGHT_EDGES = [-np.inf, 1.5, 2.1667, 2.8333, 3.5, np.inf]

# Chunk size (in unique pitcher x cluster x game cells) for the posterior
# covariance pass. Caps peak memory: chunk x n_draws x 2 floats at a time.
AIM_DIST_CHUNK    = 2000
