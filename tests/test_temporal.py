import numpy as np
import pandas as pd

from possession.features import ball_player, team_shape
from possession.features.temporal import CONTEXT_COLUMNS, add_context, add_context_by_stretch


def _table(n=400, speed=0.5):
    cols = ball_player.COLUMNS + team_shape.COLUMNS
    df = pd.DataFrame(np.nan, index=range(n), columns=cols)
    df["ball_ok"] = 1.0
    df["ball_speed"] = speed
    df["d_H"], df["d_A"], df["d_diff"], df["line_dist"] = 2.0, 5.0, 3.0, 10.0
    for c in ("sp_H", "sp_A", "still05", "still15"):
        df[c] = 0.5
    return df


def test_windows_of_a_constant_signal_equal_the_constant():
    out = add_context(_table())
    assert set(CONTEXT_COLUMNS) <= set(out.columns)
    mid = out.iloc[200]
    assert mid["ball_speed_1s"] == 0.5 and mid["ball_still_1s"] == 1.0 and mid["nearest_H_1s"] == 1.0
    assert mid["d_diff_3s"] == 3.0 and mid["still05_5s"] == 0.5


def test_a_window_skips_missing_frames_and_needs_enough_of_them():
    df = _table()
    df.loc[150:250, "ball_speed"] = np.nan
    out = add_context(df)
    assert np.isnan(out.loc[200, "ball_speed_1s"])               # the whole 1 s window is empty
    assert out.loc[100, "ball_speed_1s"] == 0.5                  # far from the gap


def test_stretches_do_not_leak_into_each_other():
    df = _table()
    df.loc[:199, "still05"] = 0.0
    df.loc[200:, "still05"] = 1.0
    out = add_context_by_stretch(df, [(0, 199), (200, 399)])
    assert out.loc[199, "still05_5s"] == 0.0 and out.loc[200, "still05_5s"] == 1.0   # a plain window would mix them
    assert add_context(df).loc[199, "still05_5s"] > 0.0
    gap = add_context_by_stretch(df, [(0, 99)])
    assert gap.loc[300, CONTEXT_COLUMNS[0]] != gap.loc[300, CONTEXT_COLUMNS[0]]      # outside the stretches: NaN
