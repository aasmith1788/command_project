import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Raw Trackman tags, collapsed to the pitch types a pitcher actually thinks in.
# The fastball variants are one pitch: a tagger's choice between "Fastball" and
# "FourSeamFastBall" is not information about intent. Sinker stays separate
# because it is aimed differently, not just labelled differently.
PITCH_TYPE_MAP = {
    "Fastball":         "Fastball",
    "FourSeamFastBall": "Fastball",
    "TwoSeamFastBall":  "Fastball",
    "OneSeamFastBall":  "Fastball",
    "Sinker":           "Sinker",
    "Slider":           "Slider",
    "Curveball":        "Curveball",
    "ChangeUp":         "ChangeUp",
    "Cutter":           "Cutter",
    "Splitter":         "Splitter",
    # Undefined / Other / Knuckleball deliberately absent -> NaN -> dropped.
    # ~3.7% of pitches. A pitch nobody could name has no aim to model.
}


def assign_pitch_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add `pitch_type`: the raw Trackman tag, normalized.

    This does NOT replace `pitch_cluster`. The two coexist and are used at
    different stages, because each stage models what the previous one left over
    and they need different pitch resolution:

      Stage 1  uses `pitch_cluster` (KMeans movement sub-clusters). The
               population aim and execution spread genuinely depend on the
               shape of the pitch, so the fine split earns its keep there.

      Stages 2 uses `pitch_type`. They model the residual after Stage 1 has
      and 3  already removed the shape-specific location tendency, so the
               sub-clusters buy nothing and cost everything: 109 cluster x
               side levels crossed with the octiles left 77% of Stage 3's
               groups holding one pitcher, which is why the group level could
               not be identified. 14 levels instead of 109.
    """
    df = df.copy()
    df["pitch_type"] = df["TaggedPitchType"].map(PITCH_TYPE_MAP)
    return df


CLUSTER_FEATURES = [
    "HorzBreak", "InducedVertBreak", "RelSpeed",
    "SpinRate", "SpinAxis", "RelHeight", "RelSide",
]
MIN_ROWS_FOR_CLUSTER = 50
MAX_K = 20
MIN_K = 2


def _find_elbow(k_values: np.ndarray, wss_values: np.ndarray) -> int:
    k_norm = (k_values - k_values.min()) / (k_values.max() - k_values.min() + 1e-9)
    w_norm = (wss_values - wss_values.min()) / (wss_values.max() - wss_values.min() + 1e-9)
    p1, p2 = np.array([k_norm[0], w_norm[0]]), np.array([k_norm[-1], w_norm[-1]])
    d = p2 - p1
    dists = np.abs(
        d[1] * k_norm - d[0] * w_norm + p2[0] * p1[1] - p2[1] * p1[0]
    ) / (np.linalg.norm(d) + 1e-9)
    return int(k_values[np.argmax(dists)])


def assign_clusters(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """
    Cluster pitches by type per (PitcherThrows, TaggedPitchType).
    Logic mirrors V5 R pipeline: summarise per pitcher, cluster those summaries,
    join labels back to pitch level.
    Unclusterable rows get pitch_cluster = NaN.
    """
    df = df.copy()
    for col in CLUSTER_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # One row per pitcher x hand x pitch type — same as R groupby
    summary = (
        df.dropna(subset=CLUSTER_FEATURES)
        .groupby(["PitcherId", "PitcherThrows", "TaggedPitchType"])[CLUSTER_FEATURES]
        .mean()
        .reset_index()
    )
    summary["pitch_cluster"] = pd.NA

    scaler = StandardScaler()
    combos = summary[["PitcherThrows", "TaggedPitchType"]].drop_duplicates()

    for _, row in combos.iterrows():
        hand, pitch = row["PitcherThrows"], row["TaggedPitchType"]
        if hand not in ("Left", "Right"):
            continue

        mask = (summary["PitcherThrows"] == hand) & (summary["TaggedPitchType"] == pitch)
        subset_idx = summary.index[mask & summary[CLUSTER_FEATURES].notna().all(axis=1)]

        if len(subset_idx) < MIN_ROWS_FOR_CLUSTER:
            continue

        data = summary.loc[subset_idx, CLUSTER_FEATURES].values
        scaled = scaler.fit_transform(data)

        max_k = min(MAX_K, len(subset_idx) // 10)
        if max_k < MIN_K:
            continue

        rng = np.arange(MIN_K, max_k + 1)
        wss = np.array([
            KMeans(n_clusters=k, n_init=10, max_iter=50, random_state=seed)
            .fit(scaled).inertia_
            for k in rng
        ])
        optimal_k = _find_elbow(rng, wss)

        km = KMeans(n_clusters=optimal_k, n_init=25, max_iter=100, random_state=seed)
        labels = km.fit_predict(scaled)

        hand_abbrev = "R" if hand == "Right" else "L"
        summary.loc[subset_idx, "pitch_cluster"] = [
            f"{pitch}_{hand_abbrev}_{c + 1}" for c in labels
        ]

    df = df.merge(
        summary[["PitcherId", "PitcherThrows", "TaggedPitchType", "pitch_cluster"]],
        on=["PitcherId", "PitcherThrows", "TaggedPitchType"],
        how="left",
    )
    return df
