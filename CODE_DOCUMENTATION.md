# Code Layout and Setup

Everything needed to get the pipeline running from a clean machine, and a map of
which file does what. The modelling rationale lives in [README.md](README.md);
this document is the mechanics.

---

## Requirements

| | | |
|---|---|---|
| Python | 3.12 | pinned in `.python-version`; `pyproject.toml` allows ≥3.11 |
| Package manager | [uv](https://docs.astral.sh/uv/) | `uv.lock` is committed, so installs are reproducible |
| C++ compiler | g++ | **required in practice** — see below |
| Source data | `trackman.db` | 15.3 GB SQLite, not in the repo |
| Disk | ~1 GB free | for `output/`, per season |
| RAM | 16 GB comfortable | Stage 1 runs 4 chains in 4 processes |

### The compiler is not optional

PyMC compiles its log-density through PyTensor, which needs a C compiler. Without
one it silently falls back to a pure-Python backend — the model still runs and
still gives correct answers, but slowly enough that Stage 1 stops being viable.

`run_pipeline.py` handles this for you: it looks for `g++` on `PATH`, and if it
is missing, searches known WinGet mingw install locations and injects the
directory into `PATH` for that run. The run log prints which one it found:

```
Compiler: C:\Users\...\WinGet\Packages\...\mingw64\bin\g++.EXE
```

If it prints `WARNING: no g++ found`, stop and install one before fitting
anything. On Windows:

```powershell
winget install BrechtSanders.WinLibs.POSIX.UCRT
```

On Linux/macOS `build-essential` or Xcode command line tools are enough, and
`g++` will already be on `PATH`.

### A warning you can ignore

PyTensor prints this on every MCMC stage:

```
UserWarning: PyTensor could not link to a BLAS installation.
```

It is accurate but not important here. The models are dominated by elementwise
work and gathers rather than large matrix products, and the 2024 fit hits its
expected runtime with the warning present. Do not go chasing a conda BLAS install
on account of it.

---

## Setup

```bash
git clone <repo>
cd "Costal Carolina/command_plus_26"

# creates .venv from uv.lock and installs everything
uv sync

# confirm the environment resolves
uv run python -c "import pymc, arviz; print(pymc.__version__, arviz.__version__)"
```

`uv sync` reads `uv.lock`, so you get the exact versions the results were
produced with. `uv run <cmd>` executes inside that environment without needing
to activate it.

### Point it at the database

`config.py` line 6 carries an absolute path:

```python
DB_PATH = Path("C:/Users/aasmi/Costal Carolina/initial results/trackman.db")
```

**This is machine-specific and will not resolve on another box.** Edit it to
wherever your copy of `trackman.db` lives. The database is 15.3 GB and excluded
by the repo-root `.gitignore`, so it has to be obtained separately.

Note there is a 0-byte `trackman.db` at the repo root — a stub, not the database.
The real one is in `initial results/`.

**You may not need the database at all.** It is read only by step 1 (clustering),
and only on a cache miss. If `output/cache/raw_{year}.parquet` exists, step 1
loads that instead and never opens SQLite. If you have been handed a
`clustered_{year}.parquet`, you can skip step 1 entirely with `--steps 2-4` and
the DB path is irrelevant.

---

## Running it

Full pipeline, one season:

```bash
uv run python run_pipeline.py --year 2024
```

Skip clustering when `output/clustered_2024.parquet` already exists — this is the
common case when re-fitting:

```bash
uv run python run_pipeline.py --year 2024 --steps 2-4
```

Stages individually:

```bash
uv run python pipeline/01_cluster_pitches.py --year 2024
uv run python stage1/stage1.py --year 2024
uv run python stage2/stage2.py --year 2024
uv run python stage3/stage3.py --year 2024
```

Useful flags:

| Flag | Effect |
|---|---|
| `--dry-run` | print the command each step would run, then exit |
| `--steps 2-4` / `--steps 1,3` | run a subset; earlier outputs must already exist |
| `--no-cache` | force a fresh DB pull in step 1 |
| `--draws` / `--tune` / `--chains` / `--cores` | forwarded to every MCMC stage |
| `--target-accept` / `--max-treedepth` / `--init` / `--seed` | NUTS controls |
| `--stage1-args "..."` | per-stage override that wins over the global flags |

Omitted sampler flags leave each stage at its own `config.py` default — they
differ by stage, so a global `--draws` is usually the wrong tool. Stage 3 takes
no sampler flags at all: it is fit by optimization, not MCMC.

Every run writes a timestamped `run_{year}_{timestamp}.txt` at the project root
with full console output, plus a copy under `output/logs/`. One reference log is
committed; the rest are gitignored.

### Expect it to take a while

2024, on the machine these results came from:

| Step | Time | Share |
|---|---|---|
| 1 · Cluster pitches | ~2 min (cache hit) | — |
| 2 · Stage 1 | 5h 35m | 95% |
| 3 · Stage 2 | 15m 17s | 4% |
| 4 · Stage 3 | 1m 53s | <1% |
| **total** | **5h 52m** | |

Stage 1 dominates because it samples two models at 2,000 draws × 4 chains over a
capped 43,493 rows with a 56-column spline basis. For a smoke test that exercises
every code path in about two minutes:

```bash
uv run python stage1/stage1.py --year 2024 --draws 2 --tune 2 --chains 1
```

That writes real files with garbage numbers — it verifies the model builds and
saves, nothing more. Re-run properly afterwards.

---

## Code layout

```
command_plus_26/
├── config.py                  every threshold, prior rate and path
├── run_pipeline.py            orchestrator: 4 steps, logging, compiler detection
│
├── pipeline/
│   └── 01_cluster_pitches.py  DB -> filter to conferences -> pitch types -> parquet
│
├── stage1/stage1.py           population aim + execution-spread submodel (NUTS)
├── stage2/stage2.py           batter x pitch-type effect, 3-level pooling (NUTS)
├── stage3/stage3.py           pitcher intent surfaces + Command+ (penalized MLE)
│
├── utils/
│   ├── db.py                  SQLite reads
│   ├── teams.py               conference filter, reads data/conference_teams.csv
│   ├── clustering.py          pitch-type assignment
│   ├── splines.py             spline bases, orthogonalization, basis scaling
│   ├── matchup.py             the mirrored frame: mirror_sign, matchup, unmirror
│   ├── scoring.py             reconstruct mu_pop / sigma_pop; Command+ scaling
│   ├── plate.py               evaluate a Stage 3 intent density anywhere on the plate
│   ├── bins.py                exit-velo and fastball-velo octiles
│   ├── scaling.py             standardization helpers
│   ├── cli.py                 shared sampler-flag parsing
│   └── progress.py            live sampling progress, uniform diagnostic block
│
├── diagnostics/diagnostics.py post-fit report across all checkpoints
├── plots/intent_zone.py       intended-zone heat maps for one pitcher
└── data/conference_teams.csv  which teams are in scope
```

### What each stage reads and writes

| Stage | Reads | Writes |
|---|---|---|
| 1 · cluster | `trackman.db` (or `output/cache/raw_{yr}.parquet`) | `output/clustered_{yr}.parquet` |
| 2 · Stage 1 | `clustered_{yr}.parquet` | `stage1_{dim}_{yr}.pkl`, `checkpoints/stage1_{dim}_{yr}.nc` |
| 3 · Stage 2 | clustered + both Stage 1 pickles | `stage2_{dim}_{yr}.parquet`, `checkpoints/…nc` |
| 4 · Stage 3 | clustered + Stage 1 pickles + Stage 2 parquets | `stage3_params_{yr}.npz`, `aim_distribution_{yr}.parquet`, `intent_surface_{yr}.parquet`, `command_plus_{yr}.parquet`, `leaderboard_{yr}.csv` |

The `.pkl` files carry a version tag (`S1_VERSION`, currently
`v11-velo-height-only`). `utils/scoring.py` refuses any pickle that does not
match rather than falling back to lookups that no longer exist — a silent zero
there would zero the population aim for a whole season and still produce a
plausible-looking leaderboard.

`.nc` checkpoints are full NetCDF traces and are the largest thing on disk
(~320 MB each for Stage 2). They are not read by later stages — only by
`diagnostics/diagnostics.py` and any post-hoc analysis — so they can be deleted
to reclaim space without breaking the pipeline.

---

## Outputs

Everything lands in `output/`, which is gitignored apart from `figures/`.

| File | Size | What it is |
|---|---|---|
| `leaderboard_{yr}.csv` | 26 KB | qualified pitchers, ranked. The headline result. |
| `command_plus_{yr}.parquet` | 37 KB | every scored pitcher, including non-qualifiers |
| `aim_distribution_{yr}.parquet` | 119 MB | per-pitch intent posterior, 81 basis coefficients per row |
| `intent_surface_{yr}.parquet` | 52 MB | per-cell surfaces on a 41×41 mesh, for plotting |
| `stage3_params_{yr}.npz` | 2 MB | the fitted Stage 3 coefficients |
| `clustered_{yr}.parquet` | 67 MB | step 1 output, input to everything after |
| `checkpoints/*.nc` | ~660 MB | full posterior traces |
| `cache/raw_{yr}.parquet` | 103 MB | raw DB pull, lets you skip SQLite entirely |

Use `utils/plate.density_at()` to evaluate an intent density at arbitrary plate
coordinates rather than reading the basis coefficients directly — it handles the
mirrored frame, and getting the sign wrong puts every left-handed pitcher's
surface on the wrong side of the plate.

---

## Post-fit tools

**Diagnostics report** across every checkpoint — posterior summaries, R-hat/ESS
tables, density plots per stage:

```bash
uv run python diagnostics/diagnostics.py --year 2024
```

| Flag | Effect |
|---|---|
| `--year` | season (required) |
| `--pitcher` | pitcher for the intent heatmap, substring match (default: highest pitch count) |
| `--outdir` | where to write report and figures (default `output/diagnostics/`) |
| `--no-plots` | text report only |

**Intended-zone heat maps** for one pitcher, split by pitch type and platoon
matchup:

```bash
uv run python plots/intent_zone.py "Smith, Hagen"
uv run python plots/intent_zone.py "hagen smith"     # word order is free
uv run python plots/intent_zone.py "Smith" --list    # disambiguate a surname
```

| Flag | Effect |
|---|---|
| `--year` | season (default 2024) |
| `--out` | output path; extension picks the format (`.pdf`, `.png`) |
| `--combined` | pool all pitch types into one column |
| `--min-pitches N` | drop thin cells (default 0 — draw everything) |
| `--grid N` | render mesh resolution (default 130) |

An ambiguous name errors with the candidates listed rather than silently picking
one. Each run reads the full 119 MB `aim_distribution` parquet, so budget ~25s
per invocation; batching a whole roster would want a restructure to load once.

---

## Gotchas

**Windows shell.** `run_pipeline.py` and the stage scripts are plain Python and
run under either shell, but the repo is developed on Windows. Redirecting to
`/dev/null` from PowerShell creates a literal file called `nul` — `.gitignore`
covers it.

**Two `trackman.db` files exist.** The 0-byte one at the repo root is a stub. The
real 15.3 GB database is in `initial results/`.

**The `.stan` files are historical.** `stage1/stage1.stan`, `stage2/stage2.stan`
and `stage3/stage3.stan` document an older per-pitcher design. PyMC never reads
them; only the `.py` files are live. Do not treat them as the model.

**`MCMC_SEED` is pinned** (`config.py`, 20260818). A 2024 Stage 2 side fit once
lost an entire chain to an unlucky jitter draw, and without a fixed seed that
chain could not be recreated to confirm it was the init rather than the geometry.
Pass `--seed -1` for a nondeterministic run.

**Stage 3 has no sampler flags.** It is fit by L-BFGS. Passing `--draws` to it
does nothing.

---

Claude Code was used to assist with coding.
