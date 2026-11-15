"""Ball-player features for the tactical ATD, in the same columns as the external training data."""

import numpy as np
import pandas as pd

from possession.features import team_shape
from possession.features.normalize import rank_normalize
from possession.features.ball_player import COLUMNS, frame_features

SPEED_LAG = 3  # frames: ball speed is measured over 0.12 s so it resembles the 10 fps external tracking


def build(players, ball, teams, length_m, width_m, fps=25, home_sign=1.0, second_half_start=110808, reference=None):
    """One feature row per frame (ball-player and team-shape columns). Coordinates are converted from 0-1 to metres
    with the pitch centre at (0, 0); `home_sign` is the home team's first-half attack direction (+1 = towards +x) and
    flips from `second_half_start` on. The team-shape columns are rank-normalised within the match over the rows of
    `reference` (the graded frames)."""
    scale = np.array([length_m, width_m], dtype=float)
    ball_m = (ball.astype(float) - 0.5) * scale
    players_m = (players.astype(float) - 0.5) * scale
    rows = []
    for t in range(len(ball)):
        b = None if np.isnan(ball_m[t, 0]) else tuple(ball_m[t])
        prev = None
        if t >= SPEED_LAG and not np.isnan(ball_m[t - SPEED_LAG, 0]):
            prev = tuple(ball_m[t - SPEED_LAG])
        seen = ~np.isnan(players_m[t, :, 0])
        xy, tm = players_m[t][seen], teams[seen]
        f1 = frame_features(b, xy, tm, prev, SPEED_LAG / fps)
        disp = (xy - players_m[t - fps][seen]) if t >= fps else np.full_like(xy, np.nan)
        sign = home_sign if t < second_half_start else -home_sign
        rows.append(f1 + team_shape.frame_shape_features(xy, tm, disp, sign))
    out = pd.DataFrame(rows, columns=COLUMNS + team_shape.COLUMNS)
    return rank_normalize(out, [c for c in team_shape.COLUMNS if c != "sh_ok"], reference)
