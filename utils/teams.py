from pathlib import Path

import pandas as pd

from config import DATA_DIR

_TEAMS_CSV = DATA_DIR / "conference_teams.csv"


def load_team_map() -> pd.DataFrame:
    return pd.read_csv(_TEAMS_CSV)


def get_target_codes() -> set[str]:
    """Return the set of Trackman PitcherTeam codes for SEC + Big 12 + CCU."""
    df = load_team_map()
    verified = df[df["verified"].astype(bool)]
    return set(verified["trackman_code"].dropna().str.strip())


def filter_to_target_conferences(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only rows where BOTH PitcherTeam and BatterTeam are in the target conference set.
    This ensures we model pitchers facing the same competitive pool.
    """
    codes = get_target_codes()
    mask = df["PitcherTeam"].isin(codes) & df["BatterTeam"].isin(codes)
    return df[mask].copy()


def add_team_info(df: pd.DataFrame) -> pd.DataFrame:
    """Join school_name and conference onto df via PitcherTeam."""
    team_map = load_team_map()[["trackman_code", "school_name", "conference"]]
    team_map = team_map.rename(columns={
        "trackman_code": "PitcherTeam",
        "school_name": "PitcherSchool",
        "conference": "PitcherConference",
    })
    return df.merge(team_map, on="PitcherTeam", how="left")
