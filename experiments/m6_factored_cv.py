"""Stage M6: does the factored emission (dead / live model trained with the DFL matches) hold up on SkillCorner possession?

Leave-matches-out cross-validation over the 20 SkillCorner matches. The three-class reference (M4) is trained on the training
folds only. The factored model trains its home / away part on the training folds and its dead / live part on the training
folds plus all seven DFL matches (weighted `DFL_WEIGHT` times per frame). Scored with the organisers' scorer.

    MTAH_EXTERNAL_DIR=... MTAH_SCORER_DIR=... python experiments/m6_factored_cv.py
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from possession import config  # noqa: E402
from possession.eval.scorer import score  # noqa: E402
from possession.external import labels as lab  # noqa: E402
from possession.external.cache import load_all, load_idsse  # noqa: E402
from possession.external.masks import gap_mask  # noqa: E402
from possession.model import durations, hsmm  # noqa: E402
from possession.model.classifier import FactoredEmission, RegimeEmission  # noqa: E402

DFL_WEIGHT = 3.0

if __name__ == "__main__":
    cfg = config.load()
    block, fps, m = cfg["time"]["block_frames"], cfg["time"]["fps"], cfg["model"]
    sk, dfl = load_all(cfg, verbose=False), load_idsse(cfg, verbose=False)
    ids = sorted(sk)
    dt = [dfl[i]["features"] for i in sorted(dfl)]
    dl = [dfl[i]["label"] for i in sorted(dfl)]
    dw = [DFL_WEIGHT * lab.dead_weights(l) for l in dl]
    rows = []
    for fold in range(5):
        test = ids[fold::5]
        train = [i for i in ids if i not in test]
        t, l = [sk[i]["features"] for i in train], [sk[i]["label"] for i in train]
        pw = [lab.training_weights(sk[i]["label"], sk[i]["group"]) for i in train]
        dead_w = [lab.dead_weights(x) for x in l]
        dur = durations.build(l, "min_geometric", block, fps)
        models = {"M4 three-class (SkillCorner)": RegimeEmission.fit(t, l, pw, m["beta"]),
                  "factored, dead/live with SkillCorner + DFL": FactoredEmission.fit(t + dt, l + dl, dead_w + dw, t, l, pw, m["beta"]),
                  "factored, dead/live with SkillCorner only": FactoredEmission.fit(t, l, dead_w, t, l, pw, m["beta"])}
        for mid in test:
            d = sk[mid]
            gt_phase, gt_track = d["label"], lab.tracking_field_label(d["label"], d["group"])
            for scenario in ("natural", "hidden40"):
                df = d["features"].copy()
                if scenario == "hidden40":
                    df.loc[gap_mask(len(df), np.random.default_rng(int(mid))), "ball_ok"] = 0.0
                for name, model in models.items():
                    pred = hsmm.decode(model.log_ratio(df), dur, m["leave_to_other_team"], m["leave_to_dead"], m["alpha"], block)
                    pred = np.where(gt_phase == "X", "X", pred)
                    for truth, gt in (("phase", gt_phase), ("tracking", gt_track)):
                        s, f, c = score(gt, pred)
                        rows.append({"match": mid, "scenario": scenario, "model": name, "truth": truth, "spellF1": s, "macroF1": f, "count_err": c})
        print("fold", fold, "done", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(cfg["paths"]["output_dir"], "m6_results.csv"), index=False)
    for scenario in ("natural", "hidden40"):
        sub = res[res.scenario == scenario]
        print("\n[%s ball] mean over 20 matches" % scenario)
        print(sub.pivot_table(index="model", columns="truth", values=["spellF1", "macroF1"], aggfunc="mean").round(3).to_string())
