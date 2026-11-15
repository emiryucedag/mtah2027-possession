"""Stage M4: a ball-missing regime built on ball-free team-shape features, tested on external matches.

The tactical tracking has no ball on ~41% of the graded frames, but the external matches have the ball on ~93% of the
frames inside play. To test the missing-ball regime the ball is hidden on external frames with generic gap masks
(`external/masks.py`, ~40% hidden), and compared with the ball-player-only model, which can only carry the state.

    variant        evidence used on frames without a ball
    M2 carry       none (the decoder carries the state through the gap)
    F2 only        team-shape classifier on every frame, ball ignored
    M4 beta=b      ball-player + shape classifier where the ball is seen, shape classifier x b where it is missing

    MTAH_EXTERNAL_DIR=/path/to/external MTAH_SCORER_DIR=/path/to/evaluation python experiments/m4.py
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
from possession.external.masks import gap_mask  # noqa: E402
from possession.model import durations, hsmm  # noqa: E402
from possession.model.classifier import FrameClassifier, RegimeEmission  # noqa: E402

ALPHAS = (0.1, 0.2)
BETAS = (0.25, 0.5, 1.0)
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
        tl = [data[i]["label"] for i in train]
        tt = [data[i]["features"] for i in train]
        tw = [lab.training_weights(data[i]["label"], data[i]["group"]) for i in train]
        f1 = FrameClassifier().fit(tt, tl, tw)
        regime = RegimeEmission.fit(tt, tl, tw)
        dur = durations.build(tl, "min_geometric", block, fps)
        for mid in test:
            d = data[mid]
            gt_phase = d["label"]
            gt_track = lab.tracking_field_label(gt_phase, d["group"])
            t0 = time.time()
            for scenario in ("natural", "hidden40"):
                df = d["features"].copy()
                if scenario == "hidden40":
                    df.loc[gap_mask(len(df), np.random.default_rng(int(mid))), "ball_ok"] = 0.0
                evidence = {"M2 carry": f1.log_ratio(df), "F2 only": regime.free_clf.log_ratio(df)}
                for b in BETAS:
                    regime.beta = b
                    evidence["M4 beta=%.2f" % b] = regime.log_ratio(df)
                for name, llr in evidence.items():
                    for alpha in ALPHAS:
                        pred = hsmm.decode(llr, dur, m["leave_to_other_team"], m["leave_to_dead"], alpha, block)
                        pred = np.where(gt_phase == "X", "X", pred)
                        for truth, gt in (("phase", gt_phase), ("tracking", gt_track)):
                            s, f, c = score(gt, pred)
                            rows.append({"match": mid, "scenario": scenario, "variant": name, "alpha": alpha, "truth": truth,
                                         "spellF1": s, "macroF1": f, "count_err": c})
            print("scored %s in %.0f s" % (mid, time.time() - t0), flush=True)

    res = pd.DataFrame(rows)
    os.makedirs(cfg["paths"]["output_dir"], exist_ok=True)
    res.to_csv(os.path.join(cfg["paths"]["output_dir"], "m4_results.csv"), index=False)
    for scenario in ("natural", "hidden40"):
        for truth in ("phase", "tracking"):
            sub = res[(res.scenario == scenario) & (res.truth == truth)]
            print("\nspellF1 [%s ball, %s label], rows = variant, columns = alpha" % (scenario, truth))
            print(sub.pivot_table(index="variant", columns="alpha", values="spellF1", aggfunc="mean").round(3).to_string())
    sub = res[(res.scenario == "hidden40") & (res.truth == "phase")]
    print("\nmacroF1 [hidden40, phase label]")
    print(sub.pivot_table(index="variant", columns="alpha", values="macroF1", aggfunc="mean").round(3).to_string())
    print("\npossession-count error [hidden40, phase label]")
    print(sub.pivot_table(index="variant", columns="alpha", values="count_err", aggfunc="mean").round(0).to_string())
