# Command+ V6: Review, Field Survey, and Publication Roadmap

Written September 2026 against commit `a4213c2` (README, CODE_DOCUMENTATION,
config.py, stage1-3, run log `run_2024_20260823_124943.txt`).

---

## 1. What the report says (recap)

**Goal.** Measure pitcher command from 2024 college TrackMan data (SEC + Big 12
+ Coastal Carolina; 496 pitchers, 134,895 pitches) without assuming a single
target per pitch. Every pitch location is decomposed as

```
location = mu_pop (league tendency for this shape/count/matchup)
         + omega  (how this batter gets worked)
         + intent (where THIS pitcher was trying to go)
         + miss   (execution error)
```

and Command+ is the size of the last term.

**Three stages, fit sequentially, point estimates passed forward.**

| Stage | What it estimates | Method | Runtime |
|---|---|---|---|
| 1 | `mu_pop` and `log sigma_pop`: 2-D spline surface over (HB, IVB), velo smooth (height only), pitch type x matchup (14), pitch type x count (83). Student-t errors. Height and side are separate models, fit on a stratified 43k-row subsample. | PyMC NUTS, 4 x 2000 | 5h 35m |
| 2 | `omega`: batter x pitch-type effect, 3-level pooling (global -> exit-velo octile x type -> batter cell), non-centered. | PyMC NUTS, 4 x 1000 | 15m |
| 3 | Intent surface per pitcher x pitch type x batter side: softmax weights over 81 Gaussian bumps (9x9, 0.30 ft), low-rank (20) penalized basis, pooled toward (FB-velo octile x type x side). Execution noise convolved analytically. Cells < 30 pitches are fully pooled. | Penalized MLE, L-BFGS, lambda chosen on 10% held-out pitches | 2m |

**Metric.** Per-pitch miss is the posterior-expected distance to intent,
`sum_m p(knot_m | y_i) * ||y_i - knot_m||`, averaged per pitcher and scaled so
100 = league mean, 10 = 1 SD. Two versions: raw feet, and standardized by the
per-pitch execution ellipse. Leaderboard gate: 300 pitches (173 qualify).

**Key results reported.**
- Sampler health clean after removing the velo smooth from the side model
  (which had produced R-hat 1.50 / ESS 7 while both downstream stages still
  looked clean; the report makes a point of this).
- Variance decomposition in feet: `mu_pop` 0.07, `omega` 0.05, intent spread
  0.22-0.27, execution 0.83-0.90. Execution dwarfs everything.
- Held-out log-likelihood: +0.203 nats/pitch over a uniform intent surface;
  train and held-out within 0.014 nats at the chosen lambda.
- Command+ range 74.5-126.7; a Hagen Smith case study shows the surfaces are
  readable and that pooled cells are labelled honestly.

**Limitations the report states itself.** One season, so no year-over-year
test. No external validation against outcomes or any independent measure of
command. Stages pass point estimates, so a bad upstream fit is invisible
downstream. Mirroring assumes perfect L/R symmetry. Stage 3 is MAP, so there
are no credible intervals on surfaces or on Command+.

### Things I noticed that the report does not say

1. **No pitcher-level execution scale.** `sigma_pop` is a population quantity
   indexed by shape, type, count and matchup. A pitcher whose own execution
   noise is wider than his shape's `sigma_pop` can only be explained by a wider
   intent surface. The intent/execution split is therefore identified only by
   the smoothness penalty and by assuming every pitcher shares the population
   execution ellipse. This is the central identifiability question a referee
   will ask, and it is testable by simulation.
2. **The held-out split is by pitch, not by game or by pitcher.** It shows the
   surface generalizes to new pitches from the same cells, which is the right
   test for lambda, but a game-level split would be the sterner claim.
3. **The 30-pitch hard gate** for free cells sits on top of the `lam_e` ridge
   penalty, so shrinkage is discontinuous at n=30. A continuous rule (penalty
   scaled by n, or a proper hierarchical scale) removes the cliff.
4. **`Command_Plus_Adj` rewards hard-to-execute shapes.** Dividing by
   `sigma_pop` means a pitcher with high-variance pitch shapes is graded on a
   curve. Whether that is desirable depends on what the metric is for
   (skill vs. outcome prediction); the paper should test which version
   predicts outcomes and say so.
5. **`aim_spread` mixes two things** (posterior uncertainty and a genuinely
   diffuse repertoire), as the report admits. They can be separated: prior
   surface entropy per cell vs. posterior contraction per pitch.
6. **Stage 1 is fit on 32% of the data** (43,493 of 134,895 after the 400-per-cell
   cap). Fine for a paper, but it must be stated and the cap justified.
7. **Stale documentation in code.** The stage3.py module docstring still
   describes the V8 NUTS design; config.py carries removed V9 constants.
   Harmless now, but a released artifact should not contradict the README.
8. **The database almost certainly holds more than one season** (team pitch
   counts of ~55k and CCU at 196k are several seasons' worth, and
   `load_pitches_multi_year` exists). If so, the "one season" limitation is a
   choice, not a constraint, and year-over-year validation is available now.

---

## 2. Code review (whole repository)

Read in full: `config.py`, `run_pipeline.py`, `pipeline/01_cluster_pitches.py`,
`stage1/stage1.py`, `stage2/stage2.py`, `stage3/stage3.py`, every module in
`utils/`, `diagnostics/diagnostics.py`, `plots/intent_zone.py`,
`data/find_team_codes.py`, the three historical `.stan` files, and the
committed run log. Findings are ordered by how much they change the science.

### 2.1 Findings that change the results

**R1. `sigma_pop` is a Student-t scale, and Stage 3 uses it as a Gaussian
standard deviation.** Stage 1 fits `StudentT(nu, mu, sigma)` and stores the
posterior mean of `sigma`. Stage 3's `density_logkernel` is a bivariate
*normal* with that `sigma` as its SD. The SD of a Student-t with scale `s` is
`s * sqrt(nu / (nu - 2))`, so Stage 3's execution kernel is narrower than the
residuals it was fit to, by exactly the variance the intent surface then has
to absorb. The size of that gap, from the run log's own numbers:

| dim | nu | sigma (scale) | extra variance the t carries | sqrt | "implied intent SD" reported |
|---|---|---|---|---|---|
| height | 20.29 | 0.900 ft | 0.0886 ft² | **0.298 ft** | **0.269 ft** |
| side | 24.34 | 0.834 ft | 0.0623 ft² | **0.250 ft** | **0.217 ft** |

The intent spread the README presents as its substantive result ("intent
0.27, execution 0.90") is within the range this bookkeeping gap alone
produces. It is not proof the intent surfaces are empty, because a genuine
mixture of targets also shows up as heavy tails that a Student-t absorbs into
`nu`, but the decomposition as printed cannot be read as intent versus
execution until the scale is corrected. Two fixes, both cheap: multiply
`sigma` by `sqrt(nu/(nu-2))` before building the kernel, or drop the
Student-t in Stage 1 (posterior `nu` of 20 to 24 is near-Gaussian anyway; see
R9). Then re-read the "implied intent SD" line.

**R2. Stage 1 has no pitcher term, so `sigma_pop` already contains pitcher
intent.** The residual Stage 1 fits its scale to is `y - mu_pop`, which still
holds every pitcher's intent deviation and the batter effect. The "execution
spread" is therefore total residual spread given the population, and Stage 3
then treats it as pure execution noise. Combined with R1, the model's
intent/execution split is circular: the width it assigns to execution was
estimated with intent inside it. The clean fix is a pitcher x pitch-type
random intercept in Stage 1 (which also addresses R5), or fitting the sigma
submodel on residuals after a first pass at pitcher means.

**R3. The held-out baseline is a straw man.** `fit_map` compares against
`theta = 0`, which is a *uniform* intent surface over a 2.4 ft x 2.4 ft
square: per-axis SD 0.69 ft before convolution. No pitcher's intent is that
wide, so any concentrated surface wins by construction, and the "+0.203
nats/pitch" is mostly evidence that a 2.4 ft square is too wide. The null a
referee will ask for is *point intent*: all weight on the centre knot (the
model with zero intent spread), and a second baseline of one free Gaussian per
cell (four parameters). The paper's claim only stands if the 81-knot surface
beats those out of sample.

**R4. Each pitch is scored against a posterior that it helped fit.** A free
cell's surface is estimated from its own pitches, then each pitch's miss is
the expectation under `p(knot | y_i)`, which conditions on that same pitch. A
diffuse surface lets the posterior chase the pitch and is therefore
*charitable*: the metric is not identified from "how spread out is his
intent". `MIN_PITCHES_CELL_FREE = 30` bounds the influence of one pitch at
about 3% per cell, but that bound is loosest exactly at the cells that just
clear the gate. Cross-fitting fixes it: split by game into K folds, fit the
surfaces on K-1, score the held-out fold, rotate. The `diagnostics.py`
correlation between Command+ and `aim_spread` is the right check and should
be in the README.

**R5. The Stage 1 movement surface is evaluated at pitcher x pitch-type
mean movement, so it is a pitcher-level covariate.** `combo_frame` collapses
to 2,018 combos and the 56-column surface is fit to their means with no
pitcher effect alongside it. In sparse regions of the movement plane a
flexible surface can follow individual pitchers, moving their aim into
`mu_pop` before Stage 3 sees it. The stage1 docstring concedes this ("the
surface is still, functionally, partly indexed by the pitcher"). It is the
same omission as R2 and has the same fix.

**R6. The smoothing penalty grid did not bracket its optimum.** The grid is
`[1, 10, 50, 200]` and the held-out likelihood is monotone increasing as
lambda falls, with the best value at the boundary. Nothing below 1.0 was
tried. Extend to `[0.01, 0.03, 0.1, 0.3, 1, 3, 10]` and, ideally, separate
`lam_g` (group surfaces) from `lam_e` (cell deviations), which are currently
forced equal.

**R7. Serial correlation is acknowledged and then ignored in the split.** The
Stage 3 docstring reports a lag-1 residual correlation of about 0.45 within
games from the earlier design. If that holds, a random 10% pitch holdout
leaks through neighbouring pitches, the held-out gain is optimistic, and the
effective pitch count behind every cell is well under the nominal one. Split
by game, and report the autocorrelation in the paper either way; a 0.45
lag-1 correlation in *miss* is itself a finding (fatigue, sequencing, or
count structure the model does not carry).

### 2.2 Bugs and inconsistencies

**B1. Stage 3 computes `nu_fixed` and never uses it.** `build_design` loads
both Stage 1 `nu` values, averages them, prints "Fixed nu=22.32", and stores
it in the design dict. The likelihood is Gaussian. The docstrings, the README
("nu ... fixed downstream rather than re-estimated") and the log all say
Stage 3 is Student-t. Either use it or delete it and correct the prose.

**B2. `command_plus_{year}.parquet` never contains Command+.** In
`stage3.main`, the columns are added to `results_q` (a `.copy()` of the
qualifiers) and `results` is written without them. `plots/intent_zone.py::
command_plus()` looks for `Command_Plus_Adj` in that file and always returns
`None`, so the figure title never shows the score. The committed
`intent_zone_hagen_smith_2024.png` confirms it: no Command+ in the title,
while the README quotes 99.9 beside the figure. CODE_DOCUMENTATION also
describes the parquet as carrying every scored pitcher's Command+.

**B3. The "unpenalized null space" is penalized.** `build_density_basis` keeps
the four null-space columns (constant, linear in each axis, and their
product) at unit scale and says they are free; `_neg_log_post` then applies
`0.5 * lam * ||g||^2` to all 20 coefficients, null space included. The
constant column is also redundant under the softmax. Harmless at lambda = 1,
but the docstring is wrong and the effect grows with lambda.

**B4. Convergence is reported on five scalars.** `stage1.run_dim` summarises
`alpha, nu, tau_move, (tau_velo), tau_tps, tau_tc`; `stage2` summarises
`tau` only; `diagnostics.py` repeats the same lists. The vector parameters
(`w_move_raw`, `u_tps_raw`, `u_tc_raw`, `omega_raw`, `eta_raw`) are never
checked, and the Stage 2 side run printed PyMC's "effective sample size per
chain is smaller than 100 for some parameters" while the pipeline reported
min ESS 1,259 and "clean". The README's "unrounded maximum over all monitored
parameters" is true only in the narrow sense. Report max R-hat and min ESS
over *every* parameter, with `round_to="none"`, which also removes the
1.01-rounding artifact the README spends a paragraph on.

**B5. Miss is measured to knot centres, not to the intent.** The intent
density is a mixture of 0.30 ft bumps, but `score()` uses `||y - knot_m||`,
ignoring where inside the bump the intent sits. Under the model,
`E||y - x||` for `x ~ bump_m` is smaller than the distance to the centre and
shrinks toward `y`. The ranking is barely affected, but the reported feet
are biased upward and the paper should say "distance to the nearest 0.3 ft
basis centre, in expectation".

**B6. The prior on `nu` fights the data.** `Gamma(alpha=5, beta=0.9)` has
mean 5.6 and SD 2.5; the posteriors sit at 20 and 24, five to seven prior
SDs out. The posterior `nu` is therefore pulled low, which widens the
scale-to-SD gap in R1. Use a wide prior (`Gamma(2, 0.1)`) or a Gaussian
likelihood.

**B7. The sparse velocity bin is a pooling target for 87 of 496 pitchers.**
`pitcher_fb_bins` needs 25 fastballs to place a pitcher; 17.5% fail and are
pooled toward each other as if they were a skill group. `bins.py` warns that
this bin "must not be read as one", and Stage 3 reads it as one.

**B8. Stale documentation inside the code.** `stage3.py`'s module docstring
describes the V8 per-pitcher NUTS design with `delta`, `u_game` and AR(1).
`config.py` still carries `TAU_OFF_RATE`, `STAGE3_W_MIN/MAX`, `LKJ_ETA`,
`STAGE3_*` sampler settings, `AIM_DIST_CHUNK`, and comments about 28-level
`pitch_type x PitcherThrows x BatterSide`. `diagnostics.py` refers to a "5x5
zone grid" and to aim SDs "measured in hundredths of a foot" (the current
mean is 0.435 ft). `utils/scaling.build_sigma_design_matrix` is V8 code with
no caller. The `.stan` files are labelled historical, correctly.

**B9. Dead work in the pipeline.** Step 1 still runs the KMeans clustering
(about two minutes) that no stage reads. `nutpie` and `numba` are declared
dependencies and never used; `nutpie` would in fact be the cheapest way to
cut the 5.5-hour Stage 1. `stage3.py` accepts and ignores sampler flags.

**B10. No tests, no CI, and a machine-specific database path.** Nothing
checks that `SplineSpec.transform` reproduces the training columns, that
mirroring round-trips, or that `_neg_log_post`'s analytic gradient matches
finite differences. For a released artifact all three are one afternoon.

### 2.3 What is done well

Worth saying plainly, because a referee will notice these too. The mirrored
frame is applied to every flipping axis in one place and undone in one
place. Duplicate TrackMan records (verified and unverified files) are removed
and re-checked in diagnostics. Non-centred parameterisations are used
everywhere a scale is learned, and stored coefficients are `E[w * tau]`, not
`E[w] * E[tau]`, with the reason documented. The spline block is centred,
group-swept, orthogonalised against the velocity smooth and rotated to
principal axes, and the prior is calibrated on the function rather than the
coefficients. Pickles are version-tagged and refused on mismatch. The
orchestrator logs every resolved sampler setting. The plate-axis convention
is fixed from hit-by-pitch locations rather than assumed. The code explains
its decisions, including the ones that were tried and removed.

### 2.4 What this does to the roadmap

R1 to R4 mean the headline numbers may not survive contact with a referee, so
**a correctness pass comes before any validation work**:

1. Fix the scale (R1), drop or re-prior the Student-t (B6), and use or remove
   `nu_fixed` (B1).
2. Add a pitcher x pitch-type random intercept to Stage 1 so `sigma_pop` and
   the surface are estimated net of pitcher aim (R2, R5). Runtime will rise;
   this is where `nutpie` or NumPyro pays for itself.
3. Replace the uniform null with point-intent and one-Gaussian-per-cell
   baselines (R3), extend the lambda grid (R6), and split by game (R7).
4. Cross-fit the intent surfaces so no pitch is scored against itself (R4).
5. Report convergence over all parameters (B4), fix the Command+ parquet
   (B2), and correct the docstrings (B8).

Only then run the simulation study and the multi-season and outcome
validation in section 5. If the intent surfaces still beat point intent out
of sample after steps 1 to 4, the paper has its result. If they do not, the
honest paper is "location-only intent inference is weakly identified, and
here is what it takes to identify it", which is the release-angle paper (P2).

---

## 3. The field and where it stands

### Public command metrics and what each one actually measures

| Approach | Representative work | What "target" means | Known reliability |
|---|---|---|---|
| Outcome-value of location vs. league optimum | Location+ (FanGraphs), PitchingBot Command | No target; value of where the ball went | Location+ YoY R^2 ~0.39, r ~0.48 |
| Called-strike skill | BP CSAA (2017) | None; called strike prob. net of framing/umpire | Moderate |
| Catcher glove | COMMANDf/x (Sportvision, 2010s); OpenCommand (2024-26, YOLO on broadcast video + Statcast) | Glove position pre-pitch; avg miss ~13 in, median FB miss 7-10 in | OpenCommand claims season-level accuracy < 1 in; pitch-level poor |
| Release trajectory dispersion | Kirby Index (Bland, FanGraphs 2024); Max Bay's release-angle confidence ellipses | No target; tightness of (vertical, horizontal) release angle cluster | YoY R^2 ~0.50 for 4-seamers, beating Location+ |
| Inferred intent from location history | xCTRL (Ludwig, Brill, Wyner; Wharton, arXiv 2508.19184, Aug 2025) | K-component GMM per pitcher x hand x type via EM; posterior over components given the actual location; miss = distance to estimated intended location | "Strong stability", more predictive of runs than Location+/Stuff+; ~0.5 run/start between best and worst control |
| Distance to own per-type mean | CornBelters Command+ (2026 Prospect League) | Per-type centroid | Split-half r = 0.42 |
| **This project** | Command+ V6 | Continuous penalized density per cell, hierarchical pooling, population and batter terms removed, heteroscedastic execution noise | Not yet measured |

### Adjacent results that matter for the design

- **Release angles determine location almost deterministically.** Kusafuka et
  al. (J Sports Sci 2021; Frontiers 2020) show release projection angle has by
  far the largest effect on location: a 1-SD change (~0.7 deg) moves the pitch
  about half a plate width; spin rate moves it a tenth of a ball. FanGraphs'
  "It's Release Angles All The Way Down" reaches the same conclusion on
  Statcast data.
- **A DNN on 2M NCAA D1 pitches predicts final location from release metrics
  to ~15 cm** (Research Square, 2024). So the TrackMan release block (RelHeight,
  RelSide, Extension, VertRelAngle, HorzRelAngle, RelSpeed, spin) carries most
  of the location signal.
- **Biomechanics and command.** Driveline (Feb 2026) and SABR 2025 RP3 link
  consistent trunk position at foot plant and structured (not minimal)
  variability to lower miss distance. Driveline's OpenBiomechanics data is
  public.
- **College tracking is noisier than MLB.** Baseball America notes inconsistent
  TrackMan accuracy across college parks; CCU's database spans many venues.
- **Hierarchical Bayesian templates in baseball.** Deshpande & Wyner's pitch
  framing model (JQAS 2017) is the methodological reference point reviewers in
  this space expect to see cited.

### What is unexplored (the gaps)

1. **No intent-inference metric has been validated against an independent
   target signal.** OpenCommand now makes glove-derived targets public for
   2024-2026 MLB. Nobody has checked whether a location-only intent model
   recovers them.
2. **Release-side and plate-side evidence about intent have never been
   combined.** Kirby uses release angles only; xCTRL and this project use plate
   location only. TrackMan records both for every pitch.
3. **Nobody separates target selection from execution.** Location+ conflates
   them; xCTRL and this project remove the choice but do not value it. An
   intent surface times a run-value map is a "decision quality" number that
   has no public counterpart.
4. **Dynamics.** No public model of within-game or between-game drift in intent
   or execution (fatigue, times through the order, in-season development).
   This project explicitly dropped its game intercept and AR(1) term for
   runtime.
5. **College-specific measurement.** No published command model handles venue
   or unit calibration effects, which are far more serious in college than in
   MLB.
6. **Uncertainty on the metric.** None of the public metrics come with
   intervals. This project's MAP Stage 3 does not either, but the model is
   set up to produce them.
7. **Identifiability of intent vs. execution is asserted, not demonstrated,**
   in every location-only method including xCTRL.

---

## 4. Your background and what it buys you

From your public GitHub (Boston University data science coursework; repos on
offline RL and off-policy evaluation, A/B testing with peeking, linear mixed
models, computer-vision trackers, LLM fine-tuning, pitch sequencing, pitch
arsenal clustering):

- **Already have:** ML/DL fluency, mixed models (which is Stage 2 in another
  vocabulary), sequential decision-making and OPE, experiment design, CV.
- **Learnable in weeks and needed for the flagship paper:** HMC diagnostics
  and the non-centered/funnel vocabulary (Betancourt's "A Conceptual
  Introduction to HMC"; BDA3 ch. 5, 11-13); penalized splines and reduced-rank
  smoothers (Wood, *Generalized Additive Models*, ch. 5); mixture-model
  identifiability; simulation-based calibration (Talts et al. 2018).
- **Learnable and needed for the release-angle paper:** basic pitch-flight
  physics (Alan Nathan's trajectory calculator and papers), the TrackMan
  coordinate conventions.
- **Optional but high-leverage:** JAX/NumPyro to move Stage 1 off a 5.5-hour
  CPU NUTS run, which makes multi-season and ablation studies feasible.

---

## 5. Paper-level directions, ranked

### P1. Flagship: the model as it stands, with a validation package (recommended first)

**Claim.** A hierarchical latent-surface model recovers pitcher intent from
location alone, is stable across seasons, and predicts outcomes better than
per-type-centroid distance and than a K-component GMM (xCTRL-style) fit to the
same data.

**Work required.**
1. **Simulation study for identifiability.** Draw intent surfaces and execution
   ellipses (including pitcher-specific execution scales the model does not
   have), simulate seasons, refit, and report recovery of Command+ rank and of
   the intent/execution split. This answers the referee questions in section 2.
2. **Multi-season fits** (2022-2025 if the DB has them): year-over-year r and
   R^2 of Command+, split-half within season, and stabilization curve (how many
   pitches until r = 0.5). Compare against the same statistics for the
   centroid-distance baseline and Location+-style value.
3. **Outcome validation.** Regress next-season BB%, zone%, called-strike%,
   chase%, xwOBA-on-contact and runs/pitch on Command+ with Stuff-type
   covariates. Report both `Raw` and `Adj` and say which one wins.
4. **Baselines on identical data.** Per-type centroid; GMM with K chosen by
   BIC per cell; the intent surface without pooling; the surface without
   Stage 1/2 removal. Held-out log-likelihood and outcome prediction for each.
5. **Mirroring test.** Fit with and without the mirrored frame; report whether
   any asymmetry survives.
6. **Uncertainty.** Laplace approximation or a short NUTS run at the MAP for
   intervals on Command+, or a nonparametric bootstrap by game.
7. **Venue random effect in Stage 1** (college-specific, cheap, novel).

**Venue.** JQAS (fits the Deshpande & Wyner lineage) or Sloan/SABR analytics
research track; arXiv stat.AP preprint first.

**Risk.** Low. Everything is in the pipeline already; the cost is compute and
writing. The biggest scientific risk is the simulation showing that intent and
execution are only weakly separable, which is itself a publishable finding.

### P2. Two observations of intent: fusing release trajectory with plate location

**Claim.** Release angles are a second, nearly independent observation of where
a pitcher was trying to go. A joint model with a shared latent intent and two
measurement channels (release-side and plate-side) separates release error
from flight error and gives a command metric more reliable than either
channel alone.

**Why it is novel.** Kirby Index uses only release; xCTRL and this project use
only plate. Kusafuka et al. show release angle explains most of location, so
the residual after conditioning on release is "flight" variability (movement
and spin consistency), while release-angle dispersion around intent is
"delivery" variability. Two skills, currently reported as one.

**Work required.** Confirm the DB has VertRelAngle/HorzRelAngle (TrackMan v3
does). Fit a per-shape map from release parameters to expected plate location
(the DNN result says this is well-posed). Extend Stage 3 so each pitch
contributes two likelihood terms with a shared latent intent knot. Report a
decomposition table per pitcher: delivery miss, flight miss, total.

**Venue.** JQAS or Journal of Sports Sciences (if pitched at the physics); a
Sloan paper if pitched at the metric.

**Risk.** Medium. Release angles from college TrackMan units can be noisy and
unit-dependent, which is another argument for the venue effect in P1.

### P3. Target selection vs. execution: valuing the intent surface

**Claim.** Given a run-value (or called-strike / whiff probability) map over
the plate per pitch shape and count, the expected value of a pitcher's intent
surface is his "aim quality", and the drop from that to the value of his actual
locations is his "execution cost". Location+ is approximately the sum; this
splits it.

**Why this fits you.** The intent surface is a stochastic policy over targets.
Off-policy evaluation machinery (importance weighting, doubly robust) evaluates
counterfactual targeting policies for a given pitcher with his own execution
noise: "what if he aimed where the league aims" or "what is his optimal target
given his miss ellipse". This is the OPE benchmark repo pointed at baseball.

**Work required.** A value model over location (a GAM or GBM on outcomes by
shape/count/matchup, which the pipeline's Stage 1 features already support).
Then expectation of value under the intent surface, convolved with execution
noise. Report per-pitcher (aim quality, execution cost) and which one drives
outcomes.

**Venue.** Sloan (very much their taste), JQAS. Also a strong college-program
deliverable.

**Risk.** Medium-low. The value model must be fit on college outcomes and
college run environment is unusual; document that.

### P4. Command dynamics: drift, fatigue and development

**Claim.** Intent and execution move within games and across a season.
Reinstate the game-level intercept and AR(1) that V8 dropped, add pitch-count
and times-through-order covariates on the execution scale, and estimate an
in-season development curve for freshman-to-senior pitchers.

**Why this fits you.** Your pitch-sequencing work is the same data at the same
grain. A state-space or GP-over-time layer on the miss scale is a natural
extension; the execution scale is where fatigue should show up first.

**Work required.** Pitch-level ordering already exists (GameID, PitchNo).
Multi-season data for development curves. Fit as a fourth stage on Stage 3
residuals to keep runtime bounded.

**Venue.** JQAS, or a sports medicine/biomechanics journal if paired with
Driveline's OpenBiomechanics data.

**Risk.** Medium. Effects may be small relative to execution noise; power
analysis first.

### P5. Cross-league validation against catcher targets (booster for P1)

**Claim.** Port the model to public Statcast data and compare inferred intent
to OpenCommand's glove-derived targets for 2024-2025. If the location-only
model recovers glove targets at the season level, that is the first external
validation of any intent-inference metric, and it licenses applying the model
where no target data exist (college).

**Work required.** A Statcast loader (pybaseball), the same three stages,
OpenCommand join on pitch id. Report per-pitcher correlation between
model intent and glove target, and between Command+ and OpenCommand's median
miss. Note OpenCommand's own caveat that the glove is not the intent.

**Risk.** Medium. MLB pitchers are more homogeneous than college; the port
needs its own Stage 1 fit. Licence is CC BY-NC-SA, fine for a paper.

### P6. Methods paper: scalable inference for latent-surface mixtures

Only if P1 shows that full Bayes is needed. Amortized VI with a small network
mapping a cell's pitch cloud to surface coefficients; GPU NUTS in NumPyro for
Stage 1; simulation-based calibration for the whole pipeline. Venue:
Statistics and Computing or a Bayesian analysis workshop.

---

## 6. Recommended path

**Paper 1 = correctness pass (section 2.4) + P1 + P5 (+ venue effect).** It converts the current report into
a defensible metric paper with the one thing every competing metric lacks:
external validation. Rough order of work:

1. Confirm seasons available in `trackman.db`; extract 2022-2025 with the
   existing step 1. Confirm release-angle columns exist while you are there.
2. Simulation study (pure Python on the Stage 3 code; one week).
3. Multi-season fits. Move Stage 1 to NumPyro or reduce draws for the
   non-headline seasons if 5.5 h per season is a bottleneck.
4. Reliability and outcome tables against baselines.
5. Statcast port and OpenCommand comparison.
6. Write-up; arXiv stat.AP; submit to JQAS.

**Paper 2 = P2 or P3**, whichever the data supports first. P2 if the release
angles are clean; P3 if you want to lean on your OPE background and produce
something a college program can act on immediately.

**Housekeeping before any submission.** Refresh stage3.py and config.py
docstrings to the V6 design; remove the machine-specific DB path in favor of
an environment variable; add a data statement (conference filter, dedup rule,
the 32% Stage 1 subsample); tag a release.

---

## 7. Sources

- Ludwig, Brill, Wyner. *Separating Intent from Execution: A Probabilistic
  Approach to Pitch Location Accuracy.* arXiv:2508.19184 (2025);
  Wharton Sports Analytics summary "Introducing xCTRL".
- Bland. *Introducing the Kirby Index* and *Revisiting the Kirby Index*,
  FanGraphs (2024); *It's Release Angles All The Way Down*; *Release Angles
  and the Illusion of Waste*; *In Defense of Command*; *Why We Still Don't
  Have a Great Command Metric*.
- OpenCommand: github.com/tomdoyo/open-command (2024-2026 MLB glove targets).
- Baseball Prospectus, *Command and Control* (CSAA, 2017); *Adversarial
  Pitch Location* (2025).
- Kusafuka et al., *Impact of each release parameter on pitch location in
  baseball pitching*, J Sports Sci 39(10), 2021; *Influence of Release
  Parameters on Pitch Location in Skilled Baseball Pitching*, Front. Sports
  Act. Living, 2020.
- *A Context-Enhanced Deep Learning Approach to Predict Baseball Pitch Location
  from Ball Tracking Release Metrics*, Research Square (2024).
- Driveline Baseball, *The Interaction of Biomechanics and Command* (Feb 2026);
  SABR Analytics 2025 RP3, *The Interaction of Command and Biomechanics for
  Pitchers*.
- Deshpande & Wyner, *A Hierarchical Bayesian Model of Pitch Framing*, JQAS
  (2017).
- Normal CornBelters, *Creating a Command+ Metric for the 2026 Prospect League*.
- *Pitch-level college baseball data captured by optical tracking technology*,
  Data in Brief (2024).
- Baseball America, *College Baseball Run Environment Makes Pitcher Evaluation
  Harder Than Ever*.
