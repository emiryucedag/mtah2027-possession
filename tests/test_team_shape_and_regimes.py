import numpy as np
import pandas as pd

from possession.external.masks import gap_mask
from possession.features import team_shape
from possession.model.classifier import RegimeEmission


def _players(n_each=5):
    xy = np.vstack([np.c_[np.linspace(-20, 20, n_each), np.zeros(n_each)], np.c_[np.linspace(-20, 20, n_each), np.full(n_each, 10.0)]])
    teams = np.array(["H"] * n_each + ["A"] * n_each, dtype=object)
    return xy, teams


def test_standing_players_are_still_and_orientation_flips_the_centroid():
    xy, teams = _players()
    xy[:, 0] += 10.0
    still = np.zeros_like(xy)
    f = dict(zip(team_shape.COLUMNS, team_shape.frame_shape_features(xy, teams, still, 1.0)))
    g = dict(zip(team_shape.COLUMNS, team_shape.frame_shape_features(xy, teams, still, -1.0)))
    assert f["sh_ok"] == 1.0 and f["still05"] == 1.0 and f["sp_H"] == 0.0
    assert f["cx_H"] == 10.0 and g["cx_H"] == -10.0


def test_a_jump_is_treated_as_an_identity_swap_and_too_few_players_give_nan():
    xy, teams = _players()
    disp = np.zeros_like(xy)
    disp[:, 0] = 50.0                      # 50 m in one second: not a real run
    f = dict(zip(team_shape.COLUMNS, team_shape.frame_shape_features(xy, teams, disp, 1.0)))
    assert np.isnan(f["sp_H"]) and np.isnan(f["still05"])
    few = dict(zip(team_shape.COLUMNS, team_shape.frame_shape_features(xy[:2], teams[:2], disp[:2], 1.0)))
    assert few["sh_ok"] == 0.0 and np.isnan(few["cx_H"])


def test_gap_mask_hides_roughly_the_target_share():
    shares = [gap_mask(150000, np.random.default_rng(s)).mean() for s in range(20)]
    assert 0.3 < float(np.mean(shares)) < 0.5


class _Fixed:
    def __init__(self, value, column):
        self.value, self.column = value, column

    def log_ratio(self, df):
        out = np.zeros((len(df), 3))
        out[df[self.column].to_numpy() == 1] = self.value
        return out


def test_regimes_use_the_ball_model_where_the_ball_is_seen_and_the_scaled_shape_model_elsewhere():
    df = pd.DataFrame({"ball_ok": [1.0, 0.0, 0.0], "sh_ok": [1.0, 1.0, 0.0]})
    emission = RegimeEmission(_Fixed(2.0, "ball_ok"), _Fixed(1.0, "sh_ok"), beta=0.5)
    out = emission.log_ratio(df)
    assert out[0, 0] == 2.0            # ball regime
    assert out[1, 0] == 0.5            # no ball, shape model scaled by beta
    assert out[2, 0] == 0.0            # nothing available: the decoder carries the state
