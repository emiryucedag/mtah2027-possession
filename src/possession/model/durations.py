"""Duration models for the explicit-duration decoder, in blocks of `block` frames.

Two forms are provided:
  min_geometric  zero below a minimum duration, geometric tail above it with a given mean (priors only, no fitting)
  lognormal      log-normal fitted to observed segment lengths, with a minimum duration imposed
Each returns log-probabilities indexed by the duration d in blocks (d = 0 is unused) and the matching log-survival
function, which the decoder uses for a segment cut off by the end of the sequence.
"""

import numpy as np

LIVE = ("H", "A")


def segment_lengths(label_arrays, kind, block):
    """Lengths in blocks of the maximal runs of `kind` ('live' = H or A, 'D') in the given label arrays."""
    out = []
    for lab in label_arrays:
        wanted = np.isin(lab, LIVE) if kind == "live" else (lab == "D")
        edge = np.diff(np.r_[0, wanted.astype(np.int8), 0])
        starts, stops = np.flatnonzero(edge == 1), np.flatnonzero(edge == -1)
        for a, b in zip(starts, stops):
            # a live run is split where the team changes
            if kind == "live":
                cuts = np.flatnonzero(lab[a + 1:b] != lab[a:b - 1]) + 1
                bounds = np.r_[0, cuts, b - a]
                out.extend(np.diff(bounds))
            else:
                out.append(b - a)
    return np.maximum(1, np.round(np.array(out, dtype=float) / block)).astype(int)


def _finish(pmf, dmax):
    pmf = np.asarray(pmf, dtype=float)
    pmf = pmf / pmf.sum()
    logp = np.full(dmax + 1, -np.inf)
    logp[1:] = np.log(np.clip(pmf, 1e-300, None))
    surv = np.full(dmax + 1, -np.inf)
    surv[1:] = np.log(np.clip(np.cumsum(pmf[::-1])[::-1], 1e-300, None))
    return logp, surv


def min_geometric(mean_blocks, min_blocks, dmax):
    d = np.arange(1, dmax + 1)
    p = 1.0 / max(1.0, mean_blocks - min_blocks + 1)
    pmf = np.where(d >= min_blocks, (1 - p) ** np.maximum(d - min_blocks, 0) * p, 0.0)
    return _finish(pmf, dmax)


def lognormal(lengths_blocks, min_blocks, dmax):
    x = np.log(np.maximum(lengths_blocks[lengths_blocks >= min_blocks], 1))
    mu, sigma = x.mean(), max(x.std(), 0.1)
    d = np.arange(1, dmax + 1)
    pdf = np.exp(-(np.log(d) - mu) ** 2 / (2 * sigma ** 2)) / (d * sigma * np.sqrt(2 * np.pi))
    pmf = np.where(d >= min_blocks, pdf, 0.0)
    return _finish(pmf, dmax)


LIVE_MIN_S, DEAD_MIN_S = 2.0, 4.0
LIVE_MAX_S, DEAD_MAX_S = 120.0, 180.0


def build(train_labels, kind, block, fps):
    """(live_logdur, live_logsurv, dead_logdur, dead_logsurv) fitted on training label arrays.

    kind = 'min_geometric' (minimum duration + geometric tail with the observed mean) or 'lognormal'.
    """
    per_s = fps / block
    live = segment_lengths(train_labels, "live", block)
    dead = segment_lengths(train_labels, "D", block)
    mins = (int(LIVE_MIN_S * per_s), int(DEAD_MIN_S * per_s))
    maxs = (int(LIVE_MAX_S * per_s), int(DEAD_MAX_S * per_s))
    if kind == "min_geometric":
        return min_geometric(live.mean(), mins[0], maxs[0]) + min_geometric(dead.mean(), mins[1], maxs[1])
    return lognormal(live, mins[0], maxs[0]) + lognormal(dead, mins[1], maxs[1])


def team_lengths(label_arrays, team, block):
    """Lengths in blocks of the possession blocks of one team ('H' or 'A')."""
    out = []
    for lab in label_arrays:
        wanted = lab == team
        edge = np.diff(np.r_[0, wanted.astype(np.int8), 0])
        out.extend(np.flatnonzero(edge == -1) - np.flatnonzero(edge == 1))
    return np.maximum(1, np.round(np.array(out, dtype=float) / block)).astype(int)


def fit_asymmetry_slope(label_arrays, block):
    """Slope of log(team mean possession duration / overall mean) against (2 * team share of possession time - 1).

    One point per team per match, fitted through the origin: a team that has more of the ball has longer possessions.
    """
    xs, ys = [], []
    for lab in label_arrays:
        lens = {t: team_lengths([lab], t, block) for t in ("H", "A")}
        if min(len(v) for v in lens.values()) == 0:
            continue
        total = {t: lens[t].sum() for t in lens}
        overall = np.concatenate(list(lens.values())).mean()
        for t in lens:
            xs.append(2.0 * total[t] / (total["H"] + total["A"]) - 1.0)
            ys.append(np.log(lens[t].mean() / overall))
    xs, ys = np.array(xs), np.array(ys)
    return float((xs * ys).sum() / (xs * xs).sum())


def build_teams(train_labels, block, fps, share_h, kappa):
    """Six arrays (H and A duration models with their own means, then the dead-ball model).

    `share_h` is the home team's share of possession time (0-1); the means follow log(mean_team / mean) = kappa * (2 * share_team - 1).
    """
    per_s = fps / block
    live = segment_lengths(train_labels, "live", block)
    dead = segment_lengths(train_labels, "D", block)
    mins = (int(LIVE_MIN_S * per_s), int(DEAD_MIN_S * per_s))
    maxs = (int(LIVE_MAX_S * per_s), int(DEAD_MAX_S * per_s))
    models = []
    for share in (share_h, 1.0 - share_h):
        mean = live.mean() * np.exp(kappa * (2.0 * share - 1.0))
        models.extend(min_geometric(max(mean, mins[0] + 1), mins[0], maxs[0]))
    return tuple(models) + min_geometric(dead.mean(), mins[1], maxs[1])
