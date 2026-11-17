"""SkillCorner Open Data (MIT, https://github.com/SkillCorner/opendata): labels and features at 25 Hz.

Per match the loader returns
  label     'H' / 'A' / 'D' per 25 Hz frame, 'X' outside the two periods. Inside a phase of play it is the phase's
            team in possession (H = home), inside a period but outside any phase it is D (phases are only defined
            while the ball is in play).
  group     the tracking `possession.group` field as 'H' / 'A' / '' (a second, independent view of possession)
  features  DataFrame: the ball-player columns (`features.ball_player`) and the ball-free team-shape columns
            (`features.team_shape`), one row per 25 Hz frame
The 10 fps source is mapped to 25 Hz by taking the previous source frame.
"""

import json
import os
from collections import deque

import numpy as np
import pandas as pd

from possession.features import ball_player, team_shape
from possession.features.normalize import rank_normalize

SRC_FPS, OUT_FPS = 10, 25
LAG = SRC_FPS  # one second, for displacements
SIDE = {"home team": "H", "away team": "A"}
FEATURE_COLUMNS = ball_player.COLUMNS + team_shape.COLUMNS
SHAPE_NORMALIZED = tuple(c for c in team_shape.COLUMNS if c != "sh_ok")


def _orientation(meta, period):
    """+1 if the home team attacks towards +x in this period, else -1 (None outside the periods)."""
    if period not in (1, 2):
        return None
    side = meta["home_team_side"][period - 1]
    return 1.0 if side == "left_to_right" else -1.0


def _read_tracking(path, team_of, meta):
    period, group, feats = [], [], []
    prev_ball = None
    history = deque(maxlen=LAG + 1)
    nan_shape = [0.0] + [np.nan] * (len(team_shape.COLUMNS) - 1)
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            per = r["period"] or 0
            period.append(per)
            group.append(SIDE.get(r["possession"]["group"], ""))
            b = r["ball_data"]
            ball = (b["x"], b["y"]) if b["is_detected"] and b["x"] is not None else None
            cur = {p["player_id"]: (p["x"], p["y"]) for p in r["player_data"] if p["is_detected"] and team_of.get(p["player_id"])}
            pids = list(cur)
            xy = np.array([cur[i] for i in pids], dtype=float).reshape(-1, 2)
            teams = np.array([team_of[i] for i in pids], dtype=object)
            f1 = ball_player.frame_features(ball, xy, teams, prev_ball, 1.0 / SRC_FPS)
            sign = _orientation(meta, per)
            if sign is None:
                f2 = nan_shape
            else:
                old = history[0] if len(history) == LAG + 1 else {}
                disp = np.array([(cur[i][0] - old[i][0], cur[i][1] - old[i][1]) if i in old else (np.nan, np.nan) for i in pids],
                                dtype=float).reshape(-1, 2)
                f2 = team_shape.frame_shape_features(xy, teams, disp / (LAG / SRC_FPS), sign)
            feats.append(f1 + f2)
            history.append(cur)
            prev_ball = ball
    return np.array(period), np.array(group, dtype=object), np.array(feats, dtype=float)


def load_match(match_dir):
    match_id = os.path.basename(os.path.normpath(match_dir))
    with open(os.path.join(match_dir, match_id + "_match.json")) as f:
        meta = json.load(f)
    home_id = meta["home_team"]["id"]
    team_of = {p["id"]: ("H" if p["team_id"] == home_id else "A") for p in meta["players"]}
    period, group, feats = _read_tracking(os.path.join(match_dir, match_id + "_tracking_extrapolated.jsonl"), team_of, meta)
    n = len(period)

    phases = pd.read_csv(os.path.join(match_dir, match_id + "_phases_of_play.csv"))
    phase_team = np.full(n, "", dtype=object)
    for p in phases.itertuples():
        phase_team[p.frame_start:p.frame_end] = "H" if p.team_in_possession_id == home_id else "A"
    label = np.where(period > 0, np.where(phase_team != "", phase_team, "D"), "X").astype(object)

    n25 = int(np.ceil(n * OUT_FPS / SRC_FPS))
    idx = np.minimum(n - 1, np.arange(n25) * SRC_FPS // OUT_FPS)
    label25 = label[idx]
    features = pd.DataFrame(feats[idx], columns=FEATURE_COLUMNS)
    features = rank_normalize(features, SHAPE_NORMALIZED, reference=label25 != "X")
    return {
        "match_id": match_id,
        "label": label25,
        "group": group[idx],
        "features": features,
    }
