import numpy as np
import pandas as pd

from possession.features.ball_player import COLUMNS
from possession.model.classifier import FrameClassifier
from possession.model import durations
from possession.pipeline.make_submission import decode_match


def _cfg(n=300):
    return {"model": {"p_stay": 0.97, "leave_to_other_team": 0.59, "leave_to_dead": 0.41, "alpha": 0.2},
            "match": {"n_frames": n, "halves": [[20, 99], [150, 249]], "second_half_dead_prefix": 30},
            "time": {"block_frames": 5}}


class _Silent(FrameClassifier):
    def __init__(self):
        pass

    def log_ratio(self, df):
        return np.zeros((len(df), 3))


def test_graded_layout_and_the_second_half_dead_prefix():
    cfg = _cfg()
    dur = durations.min_geometric(60, 5, 100) + durations.min_geometric(100, 10, 100)
    out = decode_match(cfg, _Silent(), dur, pd.DataFrame(np.zeros((300, len(COLUMNS))), columns=COLUMNS))
    assert (out[:20] == "X").all() and (out[100:150] == "X").all() and (out[250:] == "X").all()
    assert (out[150:180] == "D").all()
    assert set(out[20:100]) <= {"H", "A", "D"} and set(out[180:250]) <= {"H", "A", "D"}
    assert len(out) == 300
