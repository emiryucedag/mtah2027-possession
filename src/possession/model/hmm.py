"""A three-state (H, A, D) hidden Markov decoder working on blocks of frames.

Per-frame evidence is summed over a block (default 5 frames = 0.2 s), tempered by `alpha` (consecutive frames are
strongly correlated, so adding their log-likelihoods one by one would be overconfident), and decoded with Viterbi.
The geometric stay probability is the simple stage; an explicit-duration decoder will replace it.
"""

import numpy as np

STATES = ("H", "A", "D")


def transition_matrix(p_stay, to_other_team, to_dead):
    """From a live state: stay with p_stay, else move to the other team / to D; from D: to H or A equally."""
    leave = 1.0 - p_stay
    T = np.zeros((3, 3))
    T[0] = [p_stay, leave * to_other_team, leave * to_dead]
    T[1] = [leave * to_other_team, p_stay, leave * to_dead]
    T[2] = [leave * 0.5, leave * 0.5, p_stay]
    return T


def block_sum(frame_scores, block):
    """Sum an (n, 3) array over consecutive blocks of `block` rows (the last block may be shorter)."""
    n = len(frame_scores)
    pad = (-n) % block
    padded = np.vstack([frame_scores, np.zeros((pad, 3))]) if pad else frame_scores
    return padded.reshape(-1, block, 3).sum(axis=1)


def viterbi(emission, trans, init):
    """Most likely state path. `emission` (T, 3) log-scores, `trans` (3, 3) probabilities, `init` (3,) probabilities."""
    T = len(emission)
    log_t = np.log(trans)
    score = np.log(init) + emission[0]
    back = np.zeros((T, 3), dtype=np.int8)
    for t in range(1, T):
        cand = score[:, None] + log_t
        back[t] = cand.argmax(axis=0)
        score = cand.max(axis=0) + emission[t]
    path = np.zeros(T, dtype=np.int8)
    path[-1] = score.argmax()
    for t in range(T - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def decode(frame_log_lik, p_stay, to_other_team, to_dead, alpha, block=5, init=(0.5, 0.5, 0.0)):
    """Frame-level labels ('H'/'A'/'D') from per-frame log-likelihood ratios of shape (n, 3)."""
    n = len(frame_log_lik)
    emission = alpha * block_sum(frame_log_lik, block)
    init = np.clip(np.array(init, dtype=float), 1e-6, None)
    path = viterbi(emission, transition_matrix(p_stay, to_other_team, to_dead), init / init.sum())
    return np.array(STATES, dtype=object)[np.repeat(path, block)[:n]]
