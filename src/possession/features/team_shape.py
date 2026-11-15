"""Ball-free team-shape and motion features for one frame.

These need only the positions of the visible players and each player's displacement over the previous second, so
they exist whether or not the ball was seen. Coordinates are metres with the pitch centre at (0, 0); `sign` (+1 or -1)
orients the x axis so that the home team attacks towards +x.

Players are matched across time by tracker id (external data) or by slot (tactical tracking); a displacement above
12 m in one second is treated as an identity swap and dropped.
"""

import numpy as np

COLUMNS = ("sh_ok", "cx_H", "cx_A", "cx_all", "spread_H", "spread_A", "cdist", "sp_H", "sp_A", "still05", "still15",
           "vx_H", "vx_A")
MAX_SPEED = 12.0  # m/s over one second
MIN_TEAM = 3


def _mean(values, mask, minimum=MIN_TEAM):
    return float(values[mask].mean()) if mask.sum() >= minimum else np.nan


def frame_shape_features(xy, teams, disp, sign):
    """`xy` (n, 2) visible players; `teams` (n,) of 'H' / 'A'; `disp` (n, 2) displacement over 1 s (NaN if unmatched)."""
    nan = np.nan
    if len(xy) == 0:
        return [0.0] + [nan] * (len(COLUMNS) - 1)
    xo = xy[:, 0] * sign
    home, away = teams == "H", teams == "A"
    sp = np.hypot(disp[:, 0], disp[:, 1])
    sp[sp > MAX_SPEED] = nan
    fin = np.isfinite(sp)
    vx = disp[:, 0] * sign
    vx = np.where(fin, vx, nan)

    cx_h, cx_a = _mean(xo, home), _mean(xo, away)
    spread = []
    for m in (home, away):
        if m.sum() >= MIN_TEAM:
            c = xy[m].mean(axis=0)
            spread.append(float(np.hypot(xy[m, 0] - c[0], xy[m, 1] - c[1]).mean()))
        else:
            spread.append(nan)
    cdist = nan
    if home.sum() >= MIN_TEAM and away.sum() >= MIN_TEAM:
        cdist = float(np.hypot(xy[home].mean(axis=0)[0] - xy[away].mean(axis=0)[0], xy[home].mean(axis=0)[1] - xy[away].mean(axis=0)[1]))
    ok = 1.0 if (home.sum() >= MIN_TEAM and away.sum() >= MIN_TEAM) else 0.0
    still05 = float((sp[fin] < 0.5).mean()) if fin.sum() >= 2 * MIN_TEAM else nan
    still15 = float((sp[fin] < 1.5).mean()) if fin.sum() >= 2 * MIN_TEAM else nan
    return [ok, cx_h, cx_a, float(xo.mean()) if len(xo) >= 2 * MIN_TEAM else nan, spread[0], spread[1], cdist,
            _mean(sp, home & fin), _mean(sp, away & fin), still05, still15, _mean(vx, home & fin), _mean(vx, away & fin)]
