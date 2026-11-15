import numpy as np
import pandas as pd

from possession.features.ball_player import COLUMNS, frame_features
from possession.model import baseline


def test_no_ball_gives_nan_features_and_ball_ok_zero():
    f = frame_features(None, np.zeros((0, 2)), np.array([], dtype=object))
    assert f[0] == 0.0 and np.isnan(f[1:]).all()


def test_nearest_team_distances():
    players = np.array([[1.0, 0.0], [10.0, 0.0], [-3.0, 0.0]])
    teams = np.array(["H", "A", "A"], dtype=object)
    f = dict(zip(COLUMNS, frame_features((0.0, 0.0), players, teams)))
    assert f["ball_ok"] == 1.0 and f["d_H"] == 1.0 and f["d_A"] == 3.0 and f["d_diff"] == 2.0
    assert f["n_H_5"] == 1 and f["n_A_5"] == 1 and f["n_A_10"] == 1


def test_smooth_majority_and_dead_without_ball():
    codes = np.array(["H", "H", "A", "H", "H", "", "", ""], dtype=object)
    out = baseline.smooth(codes, 1)
    assert list(out[:5]) == ["H", "H", "H", "H", "H"] and out[7] == "D"


def test_nearest_team_needs_the_ball():
    df = pd.DataFrame({"ball_ok": [1.0, 0.0], "d_H": [1.0, np.nan], "d_A": [2.0, np.nan]})
    assert list(baseline.nearest_team(df)) == ["H", ""]
