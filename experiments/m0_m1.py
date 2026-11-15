"""Stages M0 and M1 on external matches, with leave-matches-out cross-validation and the official scorer.

M0  nearest player to the ball + majority vote over +-window frames (the published baseline's idea)
M1  per-frame logistic classifier on ball-player features + a geometric hidden Markov decoder

The classifier of each fold is fitted on the other folds' matches only. The decoder's two settings (evidence
temperature `alpha`, per-block stay probability) are swept as a grid and reported as a table; nothing is selected
on the match that is scored.

    MTAH_EXTERNAL_DIR=/path/to/external MTAH_SCORER_DIR=/path/to/evaluation python experiments/m0_m1.py
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from possession import config  # noqa: E402
from possession.eval.scorer import score  # noqa: E402
from possession.external import labels as lab  # noqa: E402
from possession.external.cache import load_all  # noqa: E402
from possession.model import baseline, hmm  # noqa: E402
from possession.model.classifier import FrameClassifier  # noqa: E402

ALPHAS = (0.02, 0.05, 0.1, 0.2, 0.5, 1.0)
STAYS = (0.97, 0.98, 0.99, 0.995)
WINDOWS = (0, 25, 50, 125)
FOLDS = 5


if __name__ == "__main__":
    cfg = config.load()
    data = load_all(cfg)
    ids = sorted(data)
    m = cfg["model"]
    rows = []
    for fold in range(FOLDS):
        test = ids[fold::FOLDS]
        train = [i for i in ids if i not in test]
        clf = FrameClassifier().fit(
            [data[i]["features"] for i in train], [data[i]["label"] for i in train],
            [lab.training_weights(data[i]["label"], data[i]["group"]) for i in train])
        for mid in test:
            d = data[mid]
            gt_phase = d["label"]
            gt_track = lab.tracking_field_label(gt_phase, d["group"])
            nearest = baseline.nearest_team(d["features"])
            for w in WINDOWS:
                pred = np.where(gt_phase == "X", "X", baseline.smooth(nearest, w))
                for truth, gt in (("phase", gt_phase), ("tracking", gt_track)):
                    s, f, c = score(gt, pred)
                    rows.append({"match": mid, "model": "M0 window %d" % w, "alpha": np.nan, "stay": np.nan,
                                 "truth": truth, "spellF1": s, "macroF1": f, "count_err": c})
            llr = clf.log_ratio(d["features"])
            for alpha in ALPHAS:
                for stay in STAYS:
                    pred = hmm.decode(llr, stay, m["leave_to_other_team"], m["leave_to_dead"], alpha, cfg["time"]["block_frames"])
                    pred = np.where(gt_phase == "X", "X", pred)
                    for truth, gt in (("phase", gt_phase), ("tracking", gt_track)):
                        s, f, c = score(gt, pred)
                        rows.append({"match": mid, "model": "M1", "alpha": alpha, "stay": stay,
                                     "truth": truth, "spellF1": s, "macroF1": f, "count_err": c})
            print("scored", mid, flush=True)

    res = pd.DataFrame(rows)
    os.makedirs(cfg["paths"]["output_dir"], exist_ok=True)
    res.to_csv(os.path.join(cfg["paths"]["output_dir"], "m0_m1_results.csv"), index=False)
    phase = res[res.truth == "phase"]
    print("\nM0 (nearest player + majority window), 20 matches, spellF1 / macroF1 against the phase label")
    print(phase[phase.model.str.startswith("M0")].groupby("model")[["spellF1", "macroF1", "count_err"]].mean().round(3).to_string())
    m1 = phase[phase.model == "M1"]
    print("\nM1 spellF1 (phase label), mean over matches, rows = alpha, columns = stay probability per 0.2 s block")
    print(m1.pivot_table(index="alpha", columns="stay", values="spellF1", aggfunc="mean").round(3).to_string())
    best = m1.groupby(["alpha", "stay"]).spellF1.mean().idxmax()
    print("\nbest grid cell by mean spellF1:", best)
    for truth in ("phase", "tracking"):
        sub = res[(res.truth == truth) & (res.alpha == best[0]) & (res.stay == best[1])]
        print("M1 at the best cell vs %s label: spellF1 %.3f (sd across matches %.3f), macroF1 %.3f, count error %.0f" % (
            truth, sub.spellF1.mean(), sub.spellF1.std(), sub.macroF1.mean(), sub.count_err.mean()))
    base = res[(res.truth == "tracking") & (res.model == "M0 window 50")]
    print("M0 window 50 vs tracking label: spellF1 %.3f" % base.spellF1.mean())
