"""Per-match rank normalisation of the team-shape features.

The external matches (broadcast-derived tracking, partial view of the pitch, noisier speeds) and the tactical tracking
(all players, smoother speeds) have clearly different distributions of speed, stillness and team spread. Replacing
each such feature by its rank within the match (0-1) removes that scale difference. Only the match's own
unlabelled feature values are used, never labels or the ground truth.
"""

import numpy as np


def rank_normalize(df, columns, reference=None):
    """Return a copy where each of `columns` is replaced by its empirical CDF value computed on the rows of `reference`
    (a boolean array; all rows by default). NaN stays NaN."""
    out = df.copy()
    ref = np.ones(len(df), dtype=bool) if reference is None else np.asarray(reference, dtype=bool)
    for c in columns:
        values = df[c].to_numpy(dtype=float)
        pool = np.sort(values[ref & ~np.isnan(values)])
        if len(pool) == 0:
            continue
        out[c] = np.where(np.isnan(values), np.nan, np.searchsorted(pool, values, side="right") / len(pool))
    return out
