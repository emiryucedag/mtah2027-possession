"""The simplest reference: the team of the player closest to the ball, then a centred majority vote.

This mirrors the idea of the organisers' published baseline (nearest tracked player to the ball; D where the ball
is not seen; majority vote over +-`window` frames) so that results on external data can be compared with it.
"""

import numpy as np


def nearest_team(features):
    """'H' / 'A' where the ball is seen (the nearer team), '' elsewhere."""
    ok = features["ball_ok"].to_numpy() == 1
    out = np.full(len(features), "", dtype=object)
    out[ok] = np.where(features["d_H"].to_numpy()[ok] <= features["d_A"].to_numpy()[ok], "H", "A")
    return out


def smooth(codes, window):
    """Majority vote over +-window frames; D where nobody had the ball in the window. Ties go to H."""
    h = (codes == "H").astype(int)
    a = (codes == "A").astype(int)
    if window <= 0:
        return np.where(h == 1, "H", np.where(a == 1, "A", "D")).astype(object)
    hc, ac = np.r_[0, np.cumsum(h)], np.r_[0, np.cumsum(a)]
    idx = np.arange(len(codes))
    lo, hi = np.maximum(0, idx - window), np.minimum(len(codes), idx + window + 1)
    hh, aa = hc[hi] - hc[lo], ac[hi] - ac[lo]
    out = np.full(len(codes), "D", dtype=object)
    seen = (hh + aa) > 0
    out[seen] = np.where(hh[seen] >= aa[seen], "H", "A")
    return out
