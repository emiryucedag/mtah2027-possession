"""Explicit-duration (hidden semi-Markov) Viterbi decoder for the three states H, A, D.

A path is a sequence of segments; a segment of state s covering blocks [a, b) scores
    log T(previous -> s) + log P_dur(s, b - a) + sum of the emission scores of s over the blocks,
where the first segment uses the initial distribution instead of T. A segment cut off by the end of the sequence
uses the survival function of its duration instead of the probability mass. Consecutive segments of the same state
are allowed only with a small probability (`same_state`), which lets the decoder split a possession that is longer
than the longest modelled duration.
"""

import numpy as np

from possession.model.hmm import STATES, block_sum


def transition_log(to_other_team, to_dead, same_state=1e-3):
    """Log probabilities of moving between segment states (rows: from, columns: to)."""
    T = np.array([[same_state, (1 - same_state) * to_other_team, (1 - same_state) * to_dead],
                  [(1 - same_state) * to_other_team, same_state, (1 - same_state) * to_dead],
                  [(1 - same_state) * 0.5, (1 - same_state) * 0.5, same_state]])
    return np.log(T / T.sum(axis=1, keepdims=True))


def viterbi(emission, log_t, log_init, logdur, logsurv):
    """Segments [(start, stop, state_index)] of the best path. `emission` is (T, 3) block scores.

    `logdur` and `logsurv` are lists with one array per state, indexed by duration in blocks (index 0 unused).
    """
    T = len(emission)
    E = np.vstack([np.zeros(3), np.cumsum(emission, axis=0)])
    V = np.full((T + 1, 3), -np.inf)
    BP = np.full((T + 1, 3), -np.inf)            # best score of entering state s at boundary t from a previous segment
    BPs = np.full((T + 1, 3), -1, dtype=np.int8)  # the previous segment's state achieving it
    BP[0] = log_init
    back_d = np.zeros((T + 1, 3), dtype=np.int32)
    back_s = np.zeros((T + 1, 3), dtype=np.int8)
    for t in range(1, T + 1):
        for s in range(3):
            dur = logsurv[s] if t == T else logdur[s]
            dmax = min(len(dur) - 1, t)
            d = np.arange(1, dmax + 1)
            idx = t - d
            cand = BP[idx, s] + dur[d] - E[idx, s]
            j = int(cand.argmax())
            V[t, s] = cand[j] + E[t, s]
            back_d[t, s] = d[j]
            back_s[t, s] = BPs[idx[j], s]
        M = V[t][:, None] + log_t
        BPs[t] = M.argmax(axis=0)
        BP[t] = M.max(axis=0)
    s, t, segments = int(V[T].argmax()), T, []
    while t > 0:
        d = int(back_d[t, s])
        segments.append((t - d, t, s))
        s, t = int(back_s[t, s]), t - d
    return segments[::-1]


def decode(frame_log_lik, durations, to_other_team, to_dead, alpha, block=5, init=(0.5, 0.5, 0.0)):
    """Frame-level labels from per-frame log-likelihood ratios of shape (n, 3).

    `durations` is either (live_logdur, live_logsurv, dead_logdur, dead_logsurv), where the live model applies to both
    H and A, or (H_logdur, H_logsurv, A_logdur, A_logsurv, dead_logdur, dead_logsurv) for team-specific durations.
    """
    n = len(frame_log_lik)
    emission = alpha * block_sum(frame_log_lik, block)
    if len(durations) == 4:
        live_p, live_s, dead_p, dead_s = durations
        dur_p, dur_s = [live_p, live_p, dead_p], [live_s, live_s, dead_s]
    else:
        dur_p, dur_s = [durations[0], durations[2], durations[4]], [durations[1], durations[3], durations[5]]
    init = np.clip(np.array(init, dtype=float), 1e-6, None)
    segments = viterbi(emission, transition_log(to_other_team, to_dead), np.log(init / init.sum()),
                       dur_p, dur_s)
    path = np.zeros(len(emission), dtype=np.int8)
    for a, b, s in segments:
        path[a:b] = s
    return np.array(STATES, dtype=object)[np.repeat(path, block)[:n]]
