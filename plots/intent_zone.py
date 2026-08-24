"""
Intended-zone heat maps for one pitcher, by pitch type and platoon matchup.

Renders the Stage 3 intent surface — where the pitcher was TRYING to throw,
with execution scatter removed — as a grid:

    columns   pitch type, ordered by usage
    top row   same-handed matchup   (LHP vs LHB, or RHP vs RHB)
    bottom    cross-handed matchup  (LHP vs RHB, or RHP vs LHB)

Averaged over every count. Each pitch carries its own base location (population
aim + batter adjustment, which is where count enters), so averaging the
per-pitch densities is a genuine marginal over counts rather than a fit to
pooled data.

Pitch types are kept separate on purpose. Averaging them together sums targets
that genuinely differ — a fastball and a slider are not aimed at the same place
— and the result is an elongated blob that looks like one diffuse target
instead of two sharp ones. Use --combined for the pooled view anyway.

EVERY pitch type is drawn, however rarely thrown. There is no minimum, because
a thin cell is not noise: Stage 3 pools it. A pitcher x type x side cell with
fewer than MIN_PITCHES_CELL_FREE (30) pitches carries no free deviation at all
(`beta[free_idx] += e` in stage3.py) — it IS its (velo octile x type x side)
group's surface, fully shrunk. Those panels are labelled "pooled", which is a
statement about WHOSE estimate is being shown, not about how noisy it is.

The surface is continuous. `utils.plate.density_at` evaluates the knot-basis
coefficients at arbitrary plate coordinates, so the mesh below is a rendering
choice and nothing is binned.

Axis convention, established from hit-by-pitch locations in the 2024 data
(LHB struck at PlateLocSide -1.84, RHB at +1.85): NEGATIVE PlateLocSide is the
left-handed batter's box, POSITIVE is the right-handed batter's box. Each row
labels the box so the orientation is never in doubt.

Usage:
    uv run python plots/intent_zone.py "Smith, Hagen"
    uv run python plots/intent_zone.py "hagen smith" --year 2024
    uv run python plots/intent_zone.py "Smith" --list
    uv run python plots/intent_zone.py "Smith, Hagen" --min-pitches 30
    uv run python plots/intent_zone.py "Smith, Hagen" --combined
"""

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

from config import (OUTPUT_DIR, DENSITY_EXTENT_H, DENSITY_EXTENT_S,
                    MIN_PITCHES_CELL_FREE)
from utils.plate import density_at

# Nominal zone, matching config.ZONE_*_EDGES.
ZONE_H = (1.5, 3.5)
ZONE_S = (-0.83, 0.83)

# Rendered view. The density is evaluated on a wider mesh than this so the
# tails are resolved; the axes then crop to the region worth looking at.
VIEW_H = (0.7, 4.5)
VIEW_S = (-2.3, 2.3)

# white = 0, blue in between, red = peak. Interpolating white->blue->red in RGB
# passes through magenta on the upper half, which reads as a natural "hotter"
# ramp and keeps the low end clearly empty.
CMAP = LinearSegmentedColormap.from_list(
    "white_blue_red",
    [(0.00, "#ffffff"), (0.18, "#dce6f5"), (0.42, "#2c6fbb"),
     (0.70, "#8b3fa8"), (1.00, "#d7191c")],
)

ROWS = (("same", "SAME-HANDED"), ("opp", "CROSS-HANDED"))


def normalize_name(s: str) -> str:
    """Lowercase, strip punctuation, sort tokens — so 'Hagen Smith' == 'Smith, Hagen'."""
    return " ".join(sorted(re.sub(r"[^a-z ]", " ", s.lower()).split()))


def resolve_pitcher(df: pd.DataFrame, query: str) -> tuple[str, str]:
    """
    Map a free-text name to exactly one (PitcherId, Pitcher).

    Tries token-set equality first so an exact name never loses to a substring,
    then falls back to substring. Ambiguity is an error with the candidates
    listed, never a silent pick of the first match.
    """
    names = df[["PitcherId", "Pitcher"]].drop_duplicates()
    q = normalize_name(query)

    exact = names[names["Pitcher"].map(normalize_name) == q]
    if len(exact) == 1:
        return exact.iloc[0]["PitcherId"], exact.iloc[0]["Pitcher"]

    hits = names[names["Pitcher"].str.lower().str.contains(query.lower().strip(), regex=False)]
    if len(hits) == 1:
        return hits.iloc[0]["PitcherId"], hits.iloc[0]["Pitcher"]
    if len(hits) == 0:
        raise SystemExit(f"No pitcher matching {query!r}. Try --list with a surname.")
    listing = "\n".join(f"    {r.Pitcher}  (id {r.PitcherId})" for r in hits.itertuples())
    raise SystemExit(f"{len(hits)} pitchers match {query!r} — be more specific:\n{listing}")


def pitcher_hand(pid: str, year: int) -> str | None:
    """'Left'/'Right' from the clustered file, or None if it cannot be read."""
    path = OUTPUT_DIR / f"clustered_{year}.parquet"
    if not path.exists():
        return None
    cl = pd.read_parquet(path, columns=["PitcherId", "PitcherThrows"])
    cl = cl[cl["PitcherId"].astype(str) == str(pid)]
    if cl.empty:
        return None
    return cl["PitcherThrows"].mode().iloc[0]


def command_plus(pid: str, year: int) -> float | None:
    path = OUTPUT_DIR / f"command_plus_{year}.parquet"
    if not path.exists():
        return None
    cp = pd.read_parquet(path)
    row = cp[cp["PitcherId"].astype(str) == str(pid)]
    if row.empty or "Command_Plus_Adj" not in row.columns:
        return None
    return float(row.iloc[0]["Command_Plus_Adj"])


def common_grid(rows: pd.DataFrame, n: int, pad: float = 0.55):
    """One mesh spanning every panel, so all surfaces are directly comparable."""
    sign = rows["mirror_sign"].to_numpy(float)
    real_s = sign * rows["base_s"].to_numpy(float)
    gh = np.linspace(rows["base_h"].min() - DENSITY_EXTENT_H - pad,
                     rows["base_h"].max() + DENSITY_EXTENT_H + pad, n)
    gs = np.linspace(real_s.min() - DENSITY_EXTENT_S - pad,
                     real_s.max() + DENSITY_EXTENT_S + pad, n)
    H, S = np.meshgrid(gh, gs, indexing="ij")
    return gh, gs, H, S


def draw(ax, gs, gh, dens, vmax, batter_side):
    """One heat map cell. Returns the mesh for the shared colorbar."""
    # gouraud shading: the surface is continuous, so render it continuously
    # rather than as cells. No pixel edges, no implied grid.
    mesh = ax.pcolormesh(gs, gh, dens, cmap=CMAP, vmin=0.0, vmax=vmax,
                         shading="gouraud", rasterized=True)
    # A few contours give the blob readable shape without imposing a lattice.
    # Levels are relative to THIS panel's peak so a lightly-used pitch still
    # shows its shape even when the shared colour scale leaves it pale.
    pk = float(dens.max())
    if pk > 0:
        ax.contour(gs, gh, dens, levels=np.array([0.25, 0.5, 0.75]) * pk,
                   colors="k", linewidths=0.45, alpha=0.28)

    ax.add_patch(Rectangle((ZONE_S[0], ZONE_H[0]),
                           ZONE_S[1] - ZONE_S[0], ZONE_H[1] - ZONE_H[0],
                           fill=False, ec="black", lw=1.3, zorder=5))

    # Batter box marker on the side the HBP test established for this hand.
    if batter_side in ("Left", "Right"):
        bx = ZONE_S[0] - 0.62 if batter_side == "Left" else ZONE_S[1] + 0.62
        ax.text(bx, np.mean(ZONE_H), f"{batter_side[0]}HB", rotation=90,
                ha="center", va="center", fontsize=7.5, color="#333333",
                fontweight="bold", alpha=0.75)
    return mesh


def frame(ax):
    ax.set_aspect("equal")
    ax.set_xlim(VIEW_S)
    ax.set_ylim(VIEW_H)
    ax.tick_params(labelsize=7)


def main():
    ap = argparse.ArgumentParser(
        description="Intended-zone heat maps for one pitcher, by pitch type and matchup.")
    ap.add_argument("pitcher", help='pitcher name, e.g. "Smith, Hagen" or "hagen smith"')
    ap.add_argument("--year", type=int, default=2024, help="season (default 2024)")
    ap.add_argument("--out", default=None,
                    help="output path (default output/intent_zone_<name>_<year>.pdf)")
    ap.add_argument("--grid", type=int, default=130,
                    help="render mesh resolution per axis (default 130)")
    ap.add_argument("--min-pitches", type=int, default=0,
                    help="drop type x matchup cells below this many pitches "
                         "(default 0 = draw everything; thin cells are pooled, not noisy)")
    ap.add_argument("--combined", action="store_true",
                    help="pool all pitch types into a single column")
    ap.add_argument("--list", action="store_true",
                    help="list every pitcher matching the name and exit")
    args = ap.parse_args()

    src = OUTPUT_DIR / f"aim_distribution_{args.year}.parquet"
    if not src.exists():
        raise SystemExit(f"{src} not found — run Stage 3 for {args.year} first.")

    ids = pd.read_parquet(src, columns=["PitcherId", "Pitcher"])

    if args.list:
        hits = ids[ids["Pitcher"].str.lower().str.contains(
            args.pitcher.lower().strip(), regex=False)]
        for r in hits[["PitcherId", "Pitcher"]].drop_duplicates().itertuples():
            print(f"{r.Pitcher}  (id {r.PitcherId})")
        return

    pid, name = resolve_pitcher(ids, args.pitcher)

    # Only now pull the wide frame (81 basis columns), and only this pitcher's rows.
    df = pd.read_parquet(src)
    rows = df[df["PitcherId"].astype(str) == str(pid)].copy()
    if rows.empty:
        raise SystemExit(f"{name} has no scored pitches in {args.year}.")

    hand = pitcher_hand(pid, args.year)
    cp = command_plus(pid, args.year)

    # matchup is same/opp in the mirrored frame; turn it into an actual side.
    if hand in ("Left", "Right"):
        side_of = {"same": hand, "opp": "Right" if hand == "Left" else "Left"}
    else:
        side_of = {"same": None, "opp": None}

    if args.combined:
        rows["_col"] = "All pitches"
        cols = ["All pitches"]
    else:
        rows["_col"] = rows["pitch_type"].astype(str)
        usage = rows["_col"].value_counts()
        if args.min_pitches > 0:
            per_cell = rows.groupby(["_col", "matchup"]).size()
            keep = {t for (t, _), n in per_cell.items() if n >= args.min_pitches}
            cols = [t for t in usage.index if t in keep]
            dropped = [(t, int(usage[t])) for t in usage.index if t not in keep]
            if dropped:
                print(f"  omitted (--min-pitches {args.min_pitches}): "
                      + ", ".join(f"{t} {n}" for t, n in dropped))
        else:
            cols = list(usage.index)
        if not cols:
            raise SystemExit(f"{name}: nothing to draw. Lower --min-pitches.")

    gh, gs, H, S = common_grid(rows, args.grid)

    # Compute every surface first: the colour scale is shared across the whole
    # figure, so red means the same density everywhere and a pitch he commands
    # tightly looks different from one he spreads around.
    # (density, n, pooled). `pooled` mirrors Stage 3's own free/shrunk split, so
    # the label reports the model's actual behaviour, not a plotting heuristic.
    surf: dict[tuple[str, str], tuple[np.ndarray, int, bool]] = {}
    for col in cols:
        for key, _ in ROWS:
            sub = rows[(rows["_col"] == col) & (rows["matchup"] == key)]
            if sub.empty:
                continue
            surf[(col, key)] = (density_at(sub, H, S, average=True), len(sub),
                                len(sub) < MIN_PITCHES_CELL_FREE)
    if not surf:
        raise SystemExit(f"{name}: nothing to draw.")
    vmax = max(float(d.max()) for d, _, _ in surf.values())

    ncol = len(cols)
    # Height follows the panels' fixed aspect (VIEW_H / VIEW_S) so the two rows
    # sit close together instead of floating in a slot sized for square axes.
    pw = 2.55
    ph = pw * (VIEW_H[1] - VIEW_H[0]) / (VIEW_S[1] - VIEW_S[0])
    fig, axes = plt.subplots(2, ncol, figsize=(pw * ncol + 1.2, 2 * ph + 1.9),
                             squeeze=False, sharex=True, sharey=True)

    mesh = None
    for j, col in enumerate(cols):
        for i, (key, rlabel) in enumerate(ROWS):
            ax = axes[i][j]
            frame(ax)
            got = surf.get((col, key))
            if got is None:
                ax.text(0.5, 0.5, "none thrown", transform=ax.transAxes,
                        ha="center", va="center", fontsize=8.5, color="#bbbbbb")
            else:
                dens, n, pooled = got
                mesh = draw(ax, gs, gh, dens, vmax, side_of[key])
                tag = f"n = {n:,}" + ("  ·  pooled" if pooled else "")
                ax.text(0.5, 0.985, tag, transform=ax.transAxes,
                        ha="center", va="top", fontsize=7.5,
                        color="#a06010" if pooled else "#555555",
                        style="italic" if pooled else "normal")
            if i == 0:
                ax.set_title(col, fontsize=10.5, fontweight="bold", pad=8)
            if j == 0:
                bs = side_of[key]
                vs = f"\nvs {bs[0]}HB" if bs else ""
                ax.set_ylabel(f"{rlabel}{vs}\n\nHeight above plate (ft)", fontsize=8.5)

    for ax in axes[1]:
        ax.set_xlabel("Plate side (ft)", fontsize=8)

    hand_txt = f"{hand[0]}HP" if hand in ("Left", "Right") else "?HP"
    cp_txt = f"   ·   Command+ {cp:.1f}" if cp is not None else ""
    fig.suptitle(
        f"{name}  ({hand_txt}){cp_txt}\n"
        f"Intended zone by pitch type, {args.year} — averaged over all counts"
        f"   ·   negative side = LHB box, positive = RHB box",
        fontsize=12, y=0.985)

    if mesh is not None:
        cbar = fig.colorbar(mesh, ax=axes, fraction=0.016, pad=0.015,
                            ticks=[0, vmax / 2, vmax])
        cbar.ax.set_yticklabels(["0", "0.5", "1"])
        cbar.set_label("Intent probability density (shared scale, peak = 1)", fontsize=8)
        cbar.ax.tick_params(labelsize=7)

    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    out = Path(args.out) if args.out else OUTPUT_DIR / f"intent_zone_{slug}_{args.year}.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)

    print(f"{name}  ({hand_txt}){cp_txt}")
    for col in cols:
        for key, rlabel in ROWS:
            got = surf.get((col, key))
            if got is None:
                continue
            dens, n, pooled = got
            i_, j_ = np.unravel_index(int(np.argmax(dens)), dens.shape)
            print(f"  {col:<10s} {rlabel:<13s} n={n:5,}  peak height {gh[i_]:.2f} ft, "
                  f"side {gs[j_]:+.2f} ft" + ("   [pooled]" if pooled else ""))
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
