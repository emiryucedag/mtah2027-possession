"""Temporal context features: short centred windows over the per-frame features of one match.

A stoppage or a possession shows over seconds, not in one frame, and the whole match is available (offline), so windows
centred on each frame are allowed. All windows are NaN-aware (frames without a value are skipped; a window needs at least
30% of its frames).
"""

import numpy as np
import pandas as pd

BALL_CONTEXT = ("ball_speed_1s", "ball_still_1s", "d_diff_1s", "d_diff_3s", "nearest_H_1s", "line_dist_1s")
SHAPE_CONTEXT = ("sp_H_2s", "sp_A_2s", "still05_2s", "still15_2s", "still05_5s", "still15_5s")
CONTEXT_COLUMNS = BALL_CONTEXT + SHAPE_CONTEXT


def _roll(series, seconds, how, fps):
    window = int(2 * seconds * fps) + 1
    r = series.rolling(window, center=True, min_periods=max(1, int(0.3 * window)))
    return getattr(r, how)()


def add_context(df, fps=25):
    """Return a copy of `df` with the context columns appended."""
    out = df.copy()
    speed = df["ball_speed"]
    still = (speed < 1.0).astype(float).where(speed.notna())
    nearest = (df["d_H"] <= df["d_A"]).astype(float).where(df["d_H"].notna() & df["d_A"].notna())
    out["ball_speed_1s"] = _roll(speed, 1, "median", fps)
    out["ball_still_1s"] = _roll(still, 1, "mean", fps)
    out["d_diff_1s"] = _roll(df["d_diff"], 1, "mean", fps)
    out["d_diff_3s"] = _roll(df["d_diff"], 3, "mean", fps)
    out["nearest_H_1s"] = _roll(nearest, 1, "mean", fps)
    out["line_dist_1s"] = _roll(df["line_dist"], 1, "min", fps)
    for name, secs in (("sp_H", 2), ("sp_A", 2), ("still05", 2), ("still15", 2)):
        out["%s_%ds" % (name, secs)] = _roll(df[name], secs, "mean", fps)
    out["still05_5s"] = _roll(df["still05"], 5, "mean", fps)
    out["still15_5s"] = _roll(df["still15"], 5, "mean", fps)
    return out


def add_context_by_stretch(df, stretches, fps=25):
    """Context columns computed separately inside each stretch (inclusive frame ranges), NaN elsewhere.

    The tactical recording contains stretches that are not play (before the match, half-time); windows must not reach across them.
    """
    out = df.copy()
    for col in CONTEXT_COLUMNS:
        out[col] = np.nan
    for a, b in stretches:
        ctx = add_context(df.iloc[a:b + 1].reset_index(drop=True), fps)
        out.loc[out.index[a:b + 1], list(CONTEXT_COLUMNS)] = ctx[list(CONTEXT_COLUMNS)].to_numpy()
    return out
