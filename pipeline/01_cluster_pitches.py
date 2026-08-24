"""
Step 1: Load Trackman data, filter to target conferences, cluster pitches.

Outputs one parquet per year to output/clustered_{year}.parquet.
Run locally in RStudio-equivalent: uv run python pipeline/01_cluster_pitches.py --year 2024

These files are the input to Stage 1 (uploaded to SCC) and Stage 3 (used locally).
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from config import OUTPUT_DIR
from utils.clustering import assign_clusters, assign_pitch_types
from utils.db import load_pitches
from utils.teams import filter_to_target_conferences, get_target_codes

CACHE_DIR    = OUTPUT_DIR / "cache"
NUMERIC_COLS = ["PlateLocHeight", "PlateLocSide", "Balls", "Strikes"]


def load_raw_cached(year: int) -> pd.DataFrame:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"raw_{year}.parquet"
    if cache_path.exists():
        print(f"  Cache hit — loading from {cache_path}")
        return pd.read_parquet(cache_path)
    codes = get_target_codes()
    print(f"  Cache miss — querying DB ({len(codes)} teams)...")
    df = load_pitches(year, team_codes=codes)
    df.to_parquet(cache_path, index=False)
    print(f"  Cached to {cache_path}")
    return df


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep one row per physical pitch.

    Trackman publishes a game twice — `<game>.csv` and `<game>_unverified.csv` —
    and the DB holds both, so every pitch in such a game appears twice under a
    single PitchUID. Nothing downstream can detect that: the likelihood just
    sees twice the evidence, so every posterior SD shrinks by roughly sqrt(2)
    and the model reports precision it has not earned. The 2024 pull was 49.8%
    duplicate rows.

    The verified file wins. It is the corrected version of the same pitch, and
    when the two disagree on a tracked value it is the one to trust.
    """
    before = len(df)
    if "PitchUID" not in df.columns:
        return df

    df = df.copy()
    if "source_file" in df.columns:
        df["_unverified"] = (df["source_file"].fillna("")
                             .str.contains("unverified", case=False)
                             .astype(int))
    else:
        df["_unverified"] = 0

    # A blank PitchUID cannot be deduplicated on; fall back to the game/pitch
    # pair rather than collapsing every blank row into one.
    uid = df["PitchUID"].fillna("").astype(str)
    blank = uid.str.strip() == ""
    df["_key"] = uid.where(
        ~blank, df["GameID"].astype(str) + "|" + df["PitchNo"].astype(str)
    )

    df = (df.sort_values("_unverified", kind="stable")
            .drop_duplicates(subset="_key", keep="first")
            .drop(columns=["_unverified", "_key"]))

    removed = before - len(df)
    print(f"  Removed {removed:,} duplicate pitch records "
          f"({removed / max(before, 1):.1%}) -> {len(df):,} unique pitches")
    if blank.any():
        print(f"  ({int(blank.sum()):,} rows had no PitchUID; "
              f"deduplicated on GameID+PitchNo instead)")
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[
        df["PitcherTeam"].notna() & (df["PitcherTeam"] != "") &
        df["BatterTeam"].notna() & (df["BatterTeam"] != "") &
        (df["PitcherTeam"] != df["BatterTeam"]) &
        df["PlateLocHeight"].notna() &
        df["PlateLocSide"].notna() &
        df["PitcherThrows"].isin(["Left", "Right"]) &
        df["BatterSide"].isin(["Left", "Right"])
    ].copy()

    df["count"] = df["Balls"].astype(int).astype(str) + "-" + df["Strikes"].astype(int).astype(str)
    df["batter_left"] = (df["BatterSide"] == "Left").astype(int)
    return df


def main(year: int, no_cache: bool = False):
    if no_cache:
        cache_path = CACHE_DIR / f"raw_{year}.parquet"
        if cache_path.exists():
            cache_path.unlink()
            print(f"Cache cleared for {year}.")
    print(f"Loading {year} pitches...")
    raw = load_raw_cached(year)
    print(f"  Rows loaded: {len(raw):,}")

    print("Removing duplicate pitch records...")
    raw = deduplicate(raw)

    print("Verifying conference filter...")
    filtered = filter_to_target_conferences(raw)
    print(f"  After verification: {len(filtered):,}")

    print("Cleaning...")
    cleaned = clean(filtered)
    print(f"  After cleaning: {len(cleaned):,}")

    print("Clustering pitches by type (Stage 1 movement sub-clusters)...")
    clustered = assign_clusters(cleaned)
    has_cluster = clustered["pitch_cluster"].notna()
    print(f"  Rows with cluster assigned: {has_cluster.sum():,} / {len(clustered):,}"
          f"  ({clustered['pitch_cluster'].nunique()} clusters)")

    print("Labelling raw pitch types (Stages 2-3 grouping)...")
    clustered = assign_pitch_types(clustered)
    has_type = clustered["pitch_type"].notna()
    print(f"  Rows with pitch type assigned: {has_type.sum():,} / {len(clustered):,}"
          f"  ({clustered['pitch_type'].nunique()} types)")
    untyped = clustered.loc[~has_type, "TaggedPitchType"].value_counts()
    if len(untyped):
        print("  Untyped (dropped by Stages 2-3): "
              + ", ".join(f"{k}={v:,}" for k, v in untyped.items()))

    out_path = OUTPUT_DIR / f"clustered_{year}.parquet"
    clustered.to_parquet(out_path, index=False)
    print(f"Saved: {out_path}")

    print("\nCluster distribution:")
    print(clustered["pitch_cluster"].value_counts().head(20).to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True, help="Season year (e.g. 2024)")
    parser.add_argument("--no-cache", action="store_true", help="Force fresh DB pull")
    args = parser.parse_args()
    main(args.year, no_cache=args.no_cache)
