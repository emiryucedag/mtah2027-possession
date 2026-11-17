"""DFL open data ("IDSSE", https://doi.org/10.6084/m9.figshare.28196177), 7 Bundesliga matches 2022/23.

Licence CC BY 4.0, data owner Deutsche Fussball Liga (DFL). Credit: Bassek et al. 2025, "An integrated dataset of
spatiotemporal and event data in elite soccer", Scientific Data 12:195, doi 10.1038/s41597-025-04505-y.

The tracking is full view, 25 Hz, in metres with the pitch centre at (0, 0), and the ball frames carry the provider's own
`BallStatus` (1 alive, 0 dead) and `BallPossession` (1 home, 2 away) on every frame. This loader turns a match into

  label     'D' when the ball is dead, else 'H' / 'A' by `BallPossession` (halves are concatenated, the gap is dropped)
  features  the ball-player and team-shape columns of the tactical-tracking feature builder
  period    1 / 2

`BallPossession` changes on every contested touch (median block 3.8 s), much finer than the possession blocks of the
challenge, so these labels are meant for the dead / alive split, not for the possession targets.
"""

import glob
import os
import re

import numpy as np
import pandas as pd

from possession.features import atd_features

FPS = 25
PITCH = (105.0, 68.0)


def read_positions(path):
    """{half: {'ball': (n, x, y, possession, status), 'persons': {person_id: (team_id, n, x, y)}}}."""
    halves, cur, ball = {}, None, False
    with open(path) as f:
        for line in f:
            if line.startswith("<FrameSet"):
                half = re.search(r'GameSection="(\w+)"', line).group(1)
                team = re.search(r'TeamId="([^"]+)"', line).group(1)
                person = re.search(r'PersonId="([^"]+)"', line).group(1)
                ball = team == "BALL"
                entry = halves.setdefault(half, {"ball": [[], [], [], [], []], "persons": {}})
                cur = entry["ball"] if ball else entry["persons"].setdefault(person, (team, [], [], []))
            elif line.startswith("<Frame") and cur is not None:
                p = line.split('"')
                if ball:
                    for col, value in zip(cur, (int(p[1]), float(p[5]), float(p[7]), int(p[-4]), int(p[-2]))):
                        col.append(value)
                else:
                    cur[1].append(int(p[1])); cur[2].append(float(p[5])); cur[3].append(float(p[7]))
    out = {}
    for half, entry in halves.items():
        out[half] = {"ball": tuple(np.array(c) for c in entry["ball"]),
                     "persons": {k: (v[0], np.array(v[1]), np.array(v[2]), np.array(v[3])) for k, v in entry["persons"].items()}}
    return out


def home_team_id(matchinfo_path):
    with open(matchinfo_path) as f:
        for line in f:
            m = re.search(r'<Team TeamId="([^"]+)"[^>]*Role="home"', line)
            if m:
                return m.group(1)
    raise ValueError("no home team in " + matchinfo_path)


def _half_arrays(half, home):
    n, bx, by, poss, status = half["ball"]
    if not (np.diff(n) == 1).all():
        raise ValueError("ball frames are not contiguous")
    persons = sorted(half["persons"].items(), key=lambda kv: (kv[1][0] != home, kv[0]))
    players = np.full((len(n), len(persons), 2), np.nan, dtype=np.float32)
    teams = np.empty(len(persons), dtype=object)
    for slot, (_, (team, pn, px, py)) in enumerate(persons):
        rows = pn - n[0]
        ok = (rows >= 0) & (rows < len(n))
        players[rows[ok], slot, 0], players[rows[ok], slot, 1] = px[ok], py[ok]
        teams[slot] = "H" if team == home else "A"
    return players, np.c_[bx, by].astype(np.float32), teams, poss, status


def load_match(folder, match_id):
    """`folder` holds the three DFL xml files of the match; `match_id` looks like 'DFL-MAT-J03WOH'."""
    pos = glob.glob(os.path.join(folder, "*positions_raw_observed*%s.xml" % match_id))[0]
    info = glob.glob(os.path.join(folder, "*matchinformation*%s.xml" % match_id))[0]
    home = home_team_id(info)
    halves = read_positions(pos)
    parts = [_half_arrays(halves[h], home) for h in ("firstHalf", "secondHalf")]
    n_first = len(parts[0][1])
    # the slots of the two halves differ (substitutions), so each half is turned into features with its own slot layout
    status = np.concatenate([p[4] for p in parts])
    poss = np.concatenate([p[3] for p in parts])
    label = np.where(status == 0, "D", np.where(poss == 1, "H", "A")).astype(object)
    frames = []
    for half_i, (pl, ball, tm, _, _) in enumerate(parts):
        norm = lambda a: np.where(np.isnan(a), np.nan, a / np.array(PITCH, dtype=np.float32) + 0.5)
        first = np.nanmean(pl[:FPS, tm == "H", 0])
        sign = 1.0 if first < 0 else -1.0                       # home defends the left side at the kick-off -> attacks +x
        frames.append(atd_features.build(norm(pl), norm(ball), tm, PITCH[0], PITCH[1], FPS, sign, 10 ** 9,
                                         np.ones(len(ball), dtype=bool)))
    features = pd.concat(frames, ignore_index=True)
    return {"match_id": match_id, "label": label, "group": np.full(len(label), "", dtype=object), "features": features,
            "period": np.r_[np.ones(n_first, dtype=int), np.full(len(label) - n_first, 2)]}
