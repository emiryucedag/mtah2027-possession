"""Stages M2 and M3: explicit durations, compared with the geometric decoder (M1) on external matches.

M1  geometric hidden Markov decoder (as in m0_m1.py), stay probability fixed
M2  explicit-duration decoder, minimum duration + geometric tail (mean taken from the training matches)
M3  explicit-duration decoder, log-normal durations fitted on the training matches (minimum duration imposed)

Leave-matches-out cross-validation; the classifier and the duration models of each fold are fitted on the other folds.
The evidence temperature `alpha` is swept and reported as a table; nothing is selected on the scored match.

    MTAH_EXTERNAL_DIR=/path/to/external MTAH_SCORER_DIR=/path/to/evaluation python experiments/m2_m3.py
"""

import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from possession import config  # noqa: E402
from possession.eval.scorer import score  # noqa: E402
from possession.external import labels as lab  # noqa: E402
from possession.external.cache import load_all  # noqa: E402
from possession.model import durations, hmm, hsmm  # noqa: E402
from possession.model.classifier import FrameClassifier  # noqa: E402

ALPHAS = (0.05, 0.1, 0.2, 0.4)
FOLDS = 5


if __name__ == "__main__":
    cfg = config.load()
    block, fps = cfg["time"]["block_frames"], cfg["time"]["fps"]
    m = cfg["model"]
    data = load_all(cfg, verbose=False)
    ids = sorted(data)
    rows = []
    for fold in range(FOLDS):
        test = ids[fold::FOLDS]
        train = [i for i in ids if i not in test]
        train_labels = [data[i]["label"] for i in train]
        clf = FrameClassifier().fit([data[i]["features"] for i in train], train_labels,
                                    [lab.training_weights(data[i]["label"], data[i]["group"]) for i in train])
        dur = {k: durations.build(train_labels, k, block, fps) for k in ("min_geometric", "lognormal")}
        for mid in test:
            d = data[mid]
            gt_phase = d["label"]
            gt_track = lab.tracking_field_label(gt_phase, d["group"])
            llr = clf.log_ratio(d["features"])
            t0 = time.time()
            for alpha in ALPHAS:
                variants = {"M1": hmm.decode(llr, m["p_stay"], m["leave_to_other_team"], m["leave_to_dead"], alpha, block)}
                for name, key in (("M2", "min_geometric"), ("M3", "lognormal")):
                    variants[name] = hsmm.decode(llr, dur[key], m["leave_to_other_team"], m["leave_to_dead"], alpha, block)
                for name, pred in variants.items():
                    pred = np.where(gt_phase == "X", "X", pred)
                    for truth, gt in (("phase", gt_phase), ("tracking", gt_track)):
                        s, f, c = score(gt, pred)
                        rows.append({"match": mid, "model": name, "alpha": alpha, "truth": truth,
                                     "spellF1": s, "macroF1": f, "count_err": c})
            print("scored %s in %.0f s" % (mid, time.time() - t0), flush=True)

    res = pd.DataFrame(rows)
    os.makedirs(cfg["paths"]["output_dir"], exist_ok=True)
    res.to_csv(os.path.join(cfg["paths"]["output_dir"], "m2_m3_results.csv"), index=False)
    for truth in ("phase", "tracking"):
        sub = res[res.truth == truth]
        print("\nspellF1 against the %s label (mean over 20 matches), rows = model, columns = alpha" % truth)
        print(sub.pivot_table(index="model", columns="alpha", values="spellF1", aggfunc="mean").round(3).to_string())
    sub = res[res.truth == "phase"]
    print("\nmacroF1 (phase label)")
    print(sub.pivot_table(index="model", columns="alpha", values="macroF1", aggfunc="mean").round(3).to_string())
    print("\npossession-count error, predicted minus true (phase label)")
    print(sub.pivot_table(index="model", columns="alpha", values="count_err", aggfunc="mean").round(0).to_string())
