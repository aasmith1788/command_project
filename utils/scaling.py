import numpy as np
import pandas as pd


def z_score(series: pd.Series) -> tuple[pd.Series, float, float]:
    """Return (z-scored series, mean, std)."""
    mu = series.mean()
    sd = series.std()
    return (series - mu) / sd, float(mu), float(sd)


def unscale(z: np.ndarray, mu: float, sd: float) -> np.ndarray:
    return z * sd + mu


def build_sigma_design_matrix(
    df: pd.DataFrame,
    cluster_col: str = "pitch_cluster",
    count_col: str = "count",
) -> tuple[np.ndarray, list[str]]:
    """
    Build treatment-coded design matrix for the sigma log-linear submodel.
    Returns (matrix, column_names).
    First column is the intercept.
    Reference level for each factor is the first alphabetically (drop_first=True).
    """
    cluster_dummies = pd.get_dummies(df[cluster_col], prefix="c", drop_first=True, dtype=float)
    count_dummies = pd.get_dummies(df[count_col], prefix="n", drop_first=True, dtype=float)
    intercept = pd.Series(np.ones(len(df)), name="intercept")
    X = pd.concat([intercept, cluster_dummies, count_dummies], axis=1)
    return X.values.astype(np.float64), list(X.columns)
