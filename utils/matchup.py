"""
Handedness mirroring and the two-matchup frame.

Reflect the world across the plate's vertical centreline and a LHP becomes a
RHP, a LHB becomes a RHB, and PlateLocSide / HorzBreak / RelSide all flip sign
while height, InducedVertBreak and RelSpeed are unchanged. Under that
reflection

    LHP vs LHB  ==  RHP vs RHB      (same-handed)
    LHP vs RHB  ==  RHP vs LHB      (opposite-handed)

so the four handedness cells are really two. Working in the mirrored frame

  - doubles the data behind every cell and every pooling group,
  - turns the side axis into "arm side vs glove side", which is the axis a
    pitcher actually aims along, rather than "toward first base",
  - and makes the model say the physically true thing, that a lefty's slider to
    a lefty is the mirror image of a righty's slider to a righty, instead of
    estimating the two independently from half the data each.

This is a symmetry ASSUMPTION, not a derivation. It is directly testable: fit
with and without and compare Stage 3's held-out log likelihood, which it now
reports natively.

IMPORTANT — mirroring is all-or-nothing per axis. An earlier version flipped
HorzBreak for LHP but left PlateLocSide alone, on the reasoning that "the plate
is fixed". That is coherent for the HEIGHT model, because height is invariant
under the reflection, and incoherent for the SIDE model: it fed the surface
mirrored movement and asked it to predict unmirrored location, so a lefty's and
a righty's sweeper landed on the same basis coordinates with opposite-signed
targets and the surface could only fit their average. Everything that flips
must flip together, which is why it is defined once, here.

Outputs are in the mirrored frame and MUST be sent back through
`unmirror_side` before being plotted or reported, or every left-handed
pitcher's chart is a mirror image of the truth.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SAME = "same"
OPP  = "opp"

# Columns that flip sign under the reflection. Height, InducedVertBreak,
# RelSpeed and SpinRate do not appear here because they are invariant.
MIRRORED = {
    "PlateLocSide": "PlateLocSide_m",
    "HorzBreak":    "HorzBreak_m",
    "RelSide":      "RelSide_m",
}


def add_matchup(df: pd.DataFrame) -> pd.DataFrame:
    """
    Attach the mirrored coordinates and the matchup label.

    Adds:
      mirror_sign     +1 for RHP, -1 for LHP — the reflection to undo later
      matchup         "same" or "opp"
      PlateLocSide_m  mirrored plate side  (the side model's response)
      HorzBreak_m     mirrored horizontal break
      RelSide_m       mirrored release side

    The raw columns are left untouched, so anything that needs true plate
    coordinates still has them and nothing silently reads a mirrored value
    thinking it is real.
    """
    df = df.copy()
    throws = df["PitcherThrows"].to_numpy()
    sign = np.where(throws == "Left", -1.0, 1.0)
    df["mirror_sign"] = sign

    # A switch hitter's BatterSide is recorded per pitch, so this is the stance
    # actually taken rather than a fixed attribute — which is what matters.
    df["matchup"] = np.where(throws == df["BatterSide"].to_numpy(), SAME, OPP)

    for raw, mir in MIRRORED.items():
        df[mir] = pd.to_numeric(df[raw], errors="coerce").to_numpy() * sign
    return df


def unmirror_side(values: np.ndarray, mirror_sign: np.ndarray) -> np.ndarray:
    """
    Send a side-axis quantity back to true plate coordinates.

    Apply to anything with a left/right orientation before it reaches a human:
    aim, intent surfaces, omega. Do NOT apply to distances or spreads — those
    are invariant under reflection and flipping them is a no-op at best.
    """
    return np.asarray(values) * np.asarray(mirror_sign)


def type_x_matchup(df: pd.DataFrame) -> pd.Series:
    """
    The categorical resolution every stage models at: pitch type crossed with
    matchup. 14 levels, against 28 for pitch type x pitcher hand x batter side,
    with twice the pitches behind each.
    """
    return df["pitch_type"].astype(str) + "_" + df["matchup"].astype(str)
