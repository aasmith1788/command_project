"""
Run this FIRST before anything else.
Queries the Trackman DB for every team code that appears as PitcherTeam,
prints them sorted by pitch count, and helps you verify conference_teams.csv.

Usage:
    uv run python data/find_team_codes.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from config import DATA_DIR
from utils.db import get_all_team_codes

TEAMS_CSV = DATA_DIR / "conference_teams.csv"
TARGET_SCHOOLS = [
    "alabama", "arkansas", "auburn", "florida", "georgia", "kentucky",
    "lsu", "mississippi", "missouri", "ole miss", "oklahoma", "south carolina",
    "tennessee", "texas", "vanderbilt",
    "arizona", "baylor", "byu", "cincinnati", "colorado", "houston",
    "iowa state", "kansas", "oklahoma state", "tcu", "texas tech", "ucf",
    "utah", "west virginia",
    "coastal carolina",
]


def main():
    print("Querying all team codes from Trackman DB...\n")
    all_codes = get_all_team_codes()
    print(f"Found {len(all_codes)} unique PitcherTeam codes.\n")
    print(all_codes.to_string(index=False))

    print("\n\n--- MATCHING CANDIDATES ---")
    print("Codes that likely correspond to target schools (fuzzy match on school names):\n")

    known = pd.read_csv(TEAMS_CSV)
    guessed = known["trackman_code"].str.lower()

    for _, row in all_codes.iterrows():
        code = row["PitcherTeam"].lower()
        for school in TARGET_SCHOOLS:
            school_parts = school.replace(" ", "_").split("_")
            if any(part in code for part in school_parts):
                print(f"  {row['PitcherTeam']:15s}  n={row['n_pitches']:>7,}   (matched: {school})")
                break

    print("\nUpdate conference_teams.csv with the correct trackman_code values,")
    print("then set verified=true for each confirmed entry.")


if __name__ == "__main__":
    main()
