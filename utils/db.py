import sqlite3
from pathlib import Path

import pandas as pd

from config import DB_PATH


def _team_sql_filter(codes: set[str]) -> str:
    """Build SQL IN clause string from a set of team codes."""
    quoted = ", ".join(f"'{c}'" for c in sorted(codes))
    return (
        f"AND PitcherTeam IN ({quoted}) "
        f"AND BatterTeam  IN ({quoted})"
    )


def load_pitches(
    year: int,
    db_path: Path = DB_PATH,
    team_codes: set[str] | None = None,
) -> pd.DataFrame:
    team_filter = _team_sql_filter(team_codes) if team_codes else ""
    con = sqlite3.connect(db_path)
    df = pd.read_sql_query(
        f"SELECT * FROM pitches WHERE Date LIKE '{year}%' {team_filter}", con
    )
    con.close()
    return df


def load_pitches_multi_year(
    years: list[int],
    db_path: Path = DB_PATH,
    team_codes: set[str] | None = None,
) -> pd.DataFrame:
    date_where = " OR ".join(f"Date LIKE '{y}%'" for y in years)
    team_filter = _team_sql_filter(team_codes) if team_codes else ""
    con = sqlite3.connect(db_path)
    df = pd.read_sql_query(
        f"SELECT * FROM pitches WHERE ({date_where}) {team_filter}", con
    )
    con.close()
    return df


def get_all_team_codes(db_path: Path = DB_PATH) -> pd.DataFrame:
    """Return all unique PitcherTeam codes and their pitch counts."""
    con = sqlite3.connect(db_path)
    df = pd.read_sql_query(
        """
        SELECT PitcherTeam, COUNT(*) as n_pitches
        FROM pitches
        WHERE PitcherTeam IS NOT NULL AND PitcherTeam != ''
        GROUP BY PitcherTeam
        ORDER BY n_pitches DESC
        """,
        con,
    )
    con.close()
    return df
