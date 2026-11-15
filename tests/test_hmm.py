import numpy as np

from possession.model import hmm


def test_transition_rows_sum_to_one():
    T = hmm.transition_matrix(0.97, 0.59, 0.41)
    assert np.allclose(T.sum(axis=1), 1.0)


def test_block_sum_handles_a_short_last_block():
    s = np.ones((7, 3))
    out = hmm.block_sum(s, 5)
    assert out.shape == (2, 3) and out[0, 0] == 5 and out[1, 0] == 2


def test_decoder_removes_a_flicker_but_keeps_a_real_change():
    n = 1000
    llr = np.zeros((n, 3))
    llr[:500, 0] = 1.0           # evidence for H in the first half ...
    llr[500:, 1] = 1.0           # ... and for A in the second
    llr[200:210, 1] = 2.0        # a short contrary blip
    pred = hmm.decode(llr, 0.97, 0.59, 0.41, alpha=0.1, block=5)
    assert (pred[:200] == "H").all() and (pred[200:210] == "H").all()
    assert pred[-1] == "A" and set(pred) <= {"H", "A", "D"}


def test_gap_without_evidence_carries_the_state():
    llr = np.zeros((600, 3))
    llr[:100, 0] = 1.0
    pred = hmm.decode(llr, 0.97, 0.59, 0.41, alpha=0.5, block=5)
    assert (pred == "H").all()
