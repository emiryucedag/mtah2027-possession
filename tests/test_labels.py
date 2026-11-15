import numpy as np

from possession.external import labels


def test_edge_distance():
    lab = np.array(list("HHHHAAAA"), dtype=object)
    d = labels.edge_distance(lab)
    assert d[4] == 0 and d[3] == 1 and d[0] == 4 and d[7] == 3   # an edge sits on the first frame of a new label


def test_training_weights_follow_distance_and_conflict():
    lab = np.array(["H"] * 200 + ["A"] * 200 + ["X"] * 10, dtype=object)
    grp = np.array([""] * len(lab), dtype=object)
    grp[100] = "A"                        # a conflict in the middle of an H block
    w = labels.training_weights(lab, grp)
    assert w[0] == 1.0                    # far from every edge
    assert w[100] == 0.0                  # conflicting frame
    assert w[199] == 0.0 and w[200] == 0.0   # right at the H/A edge
    assert w[400:].sum() == 0.0           # unlabelled frames never count


def test_tracking_field_label_substitutes_only_live_frames():
    lab = np.array(["H", "H", "D", "A"], dtype=object)
    grp = np.array(["A", "", "H", "H"], dtype=object)
    out = labels.tracking_field_label(lab, grp)
    assert list(out) == ["A", "H", "D", "H"]
