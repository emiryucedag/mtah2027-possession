"""Ball-player features for one frame: who is near the ball, how close, how fast the ball moves.

Inputs are in metres with the pitch centre at (0, 0), x along the length. Frames without a ball give NaN
for every ball-dependent column; `ball_ok` says whether the ball was seen.
"""

import numpy as np

COLUMNS = ("ball_ok", "d_H", "d_A", "d_diff", "n_H_5", "n_A_5", "n_H_10", "n_A_10", "ball_speed", "line_dist")
HALF_LENGTH, HALF_WIDTH = 52.65, 34.0
FAR = 60.0  # distance used when a team has no tracked player in the frame


def frame_features(ball, players, teams, prev_ball=None, dt=None):
    """`ball` = (x, y) or None; `players` = array (n, 2); `teams` = array of 'H'/'A' (n,)."""
    nan = float("nan")
    if ball is None:
        return [0.0] + [nan] * (len(COLUMNS) - 1)
    bx, by = ball
    if len(players):
        d = np.hypot(players[:, 0] - bx, players[:, 1] - by)
        home, away = d[teams == "H"], d[teams == "A"]
    else:
        home = away = np.array([])
    d_h = home.min() if len(home) else FAR
    d_a = away.min() if len(away) else FAR
    speed = nan
    if prev_ball is not None and dt:
        speed = float(np.hypot(bx - prev_ball[0], by - prev_ball[1]) / dt)
    line = min(HALF_LENGTH - abs(bx), HALF_WIDTH - abs(by))
    return [1.0, d_h, d_a, d_a - d_h, float((home < 5).sum()), float((away < 5).sum()),
            float((home < 10).sum()), float((away < 10).sum()), speed, line]
