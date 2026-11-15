import numpy as np

from possession.model import durations, hsmm


def _durs(live_min=10, dead_min=20, dmax=300):
    lp, ls = durations.min_geometric(60, live_min, dmax)
    dp, ds = durations.min_geometric(100, dead_min, dmax)
    return lp, ls, dp, ds


def test_segment_lengths_split_live_runs_at_team_changes():
    lab = np.array(list("HHHHHAAAADDDD"), dtype=object)
    assert sorted(durations.segment_lengths([lab], "live", 1)) == [4, 5]
    assert list(durations.segment_lengths([lab], "D", 1)) == [4]


def test_duration_models_are_proper_distributions():
    for lp, ls in (durations.min_geometric(60, 10, 300), durations.lognormal(np.array([20, 40, 80, 120, 60]), 10, 300)):
        assert abs(np.exp(lp[1:]).sum() - 1.0) < 1e-6
        assert np.isneginf(lp[0]) and lp[5] < -600              # (practically) impossible below the minimum
        assert abs(ls[10] - 0.0) < 1e-6                         # survival from the minimum is 1


def test_clean_evidence_is_recovered_with_correct_boundaries():
    rng = np.random.default_rng(0)
    truth = np.r_[np.zeros(200), np.ones(150), np.full(100, 2), np.zeros(250)].astype(int)  # blocks: H, A, D, H
    llr = np.zeros((len(truth) * 5, 3))
    for t, s in enumerate(truth):
        llr[t * 5:(t + 1) * 5, s] = 1.0 + 0.3 * rng.standard_normal(5)
    pred = hsmm.decode(llr, _durs(), 0.59, 0.41, alpha=0.5, block=5)
    got = np.array([{"H": 0, "A": 1, "D": 2}[p] for p in pred[::5]])
    assert (got == truth).mean() > 0.99


def test_a_short_weak_blip_is_removed_and_a_real_segment_is_kept():
    llr = np.zeros((1000, 3))
    llr[:, 0] = 0.3                      # mild evidence for H everywhere
    llr[480:500, 1] = 0.6                # 4 blocks of weak evidence for A: shorter than the 10-block minimum
    pred = hsmm.decode(llr, _durs(live_min=10), 0.59, 0.41, alpha=1.0, block=5)
    assert (pred == "H").all()
    llr[300:375, 1] = 3.0                # 15 blocks of strong evidence for A: a real possession
    pred = hsmm.decode(llr, _durs(live_min=10), 0.59, 0.41, alpha=1.0, block=5)
    assert abs((pred == "A").sum() - 75) <= 10 and (pred[320:360] == "A").all()


def test_end_of_sequence_segment_may_be_shorter_than_the_minimum():
    llr = np.zeros((400, 3))
    llr[:300, 0] = 1.0
    llr[300:, 1] = 3.0                   # the final stretch (20 blocks) is fine, but try a stretch shorter than the minimum
    pred = hsmm.decode(llr[:340], _durs(live_min=20), 0.59, 0.41, alpha=1.0, block=5)
    assert pred[-1] == "A"               # censored last segment is scored with the survival function


def test_runs_in_reasonable_time_on_a_half_match():
    import time
    llr = np.zeros((20000, 3))
    t0 = time.time()
    hsmm.decode(llr, _durs(dmax=600), 0.59, 0.41, alpha=0.1, block=5)
    assert time.time() - t0 < 30


def test_team_specific_durations_make_the_longer_team_stickier():
    from possession.model import durations as dur
    labels = []
    rng = np.random.default_rng(1)
    for _ in range(6):                    # synthetic matches in which H has most of the ball and longer possessions
        seq = []
        for _ in range(40):
            seq += ["H"] * int(rng.integers(100, 300)) + ["A"] * int(rng.integers(20, 120))
        labels.append(np.array(seq, dtype=object))
    kappa = dur.fit_asymmetry_slope(labels, 5)
    assert kappa > 0
    six = dur.build_teams(labels, 5, 25, 0.7, kappa)
    assert len(six) == 6
    p_h, p_a = np.exp(six[0]), np.exp(six[2])
    d = np.arange(len(p_h))
    assert (p_h[1:] * d[1:]).sum() > (p_a[1:] * d[1:]).sum()      # H's mean duration is longer than A's
    llr = np.zeros((600, 3))
    pred = hsmm.decode(llr, six, 0.59, 0.41, alpha=1.0, block=5)
    assert set(pred) <= {"H", "A", "D"}
