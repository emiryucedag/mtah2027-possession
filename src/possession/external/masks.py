"""Generic ball-gap masks, used to simulate the missing ball of the tactical tracking on external matches.

The masks follow no pattern measured on the challenge match: alternating hidden / visible stretches with
log-uniform hidden lengths, tuned only so that about `target` of the frames are hidden.
"""

import numpy as np


def gap_mask(n, rng, target=0.4, hidden_s=(1.0, 60.0), fps=25):
    """Boolean array, True where the ball is hidden."""
    mask = np.zeros(n, dtype=bool)
    t, hidden = 0, bool(rng.random() < 0.5)
    low, high = np.log(hidden_s[0]), np.log(hidden_s[1])
    while t < n:
        length = int(np.exp(rng.uniform(low, high)) * fps)
        if hidden:
            mask[t:t + length] = True
        else:
            length = int(length * (1 - target) / target * rng.uniform(0.5, 1.5))
        t += max(length, 1)
        hidden = not hidden
    return mask
