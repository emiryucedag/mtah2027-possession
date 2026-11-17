"""Training weights for the external labels, and the alternative 'tracking field' label.

The two automatic views of possession (the phase team and the tracking `possession.group` field) disagree on
~16% of frames, mostly within a couple of seconds of a phase edge. Frames near an edge or in conflict are less
trustworthy, so they get a lower weight when a classifier is fitted.
"""

import numpy as np

LIVE = ("H", "A")


def edge_distance(label):
    """Frames from each frame to the nearest change of label."""
    n = len(label)
    edges = np.flatnonzero(label[1:] != label[:-1]) + 1
    if len(edges) == 0:
        return np.full(n, 10 ** 6)
    pos = np.arange(n)
    k = np.searchsorted(edges, pos)
    left = np.where(k > 0, pos - edges[np.maximum(k - 1, 0)], 10 ** 6)
    right = np.where(k < len(edges), edges[np.minimum(k, len(edges) - 1)] - pos, 10 ** 6)
    return np.minimum(left, right)


def conflict_mask(label, group):
    return np.isin(label, LIVE) & (group != "") & (group != label)


def training_weights(label, group, fps=25):
    """1.0 more than 2 s from an edge, 0.5 for 1-2 s, 0 for conflicting frames or <1 s from an edge."""
    dist = edge_distance(label)
    w = np.where(dist >= 2 * fps, 1.0, np.where(dist >= fps, 0.5, 0.0))
    w[conflict_mask(label, group)] = 0.0
    w[label == "X"] = 0.0
    return w


def tracking_field_label(label, group):
    """The phase label with the tracking possession field substituted where both exist (a second 'truth')."""
    out = label.copy()
    m = np.isin(label, LIVE) & (group != "")
    out[m] = group[m]
    return out


def dead_weights(label, fps=25):
    """Weights for a dead / live classifier: 1.0 more than 2 s from a change between dead and live, 0.5 for 1-2 s, 0 closer.

    The home / away flips inside live play do not matter for this split, so only dead <-> live changes count as edges.
    """
    dead = np.where(label == "D", "D", np.where(label == "X", "X", "L"))
    dist = edge_distance(dead)
    w = np.where(dist >= 2 * fps, 1.0, np.where(dist >= fps, 0.5, 0.0))
    w[label == "X"] = 0.0
    return w
