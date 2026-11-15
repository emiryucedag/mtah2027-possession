"""Dead-ball detection on full-view tracking with the provider's own ball status (DFL open data, 7 matches).

Reference: the three-class model trained on SkillCorner only (the pipeline before this stage).
Factored models: a dead / live classifier and a home / away classifier; the home / away part is always trained on
SkillCorner possession phases, the dead / live part on different mixes of data:

    V1  SkillCorner only                       V3  SkillCorner + the other 6 DFL matches
    V2  the other 6 DFL matches only           V4  as V3 with the DFL frames weighted 3x

Leave-one-DFL-match-out: the held-out match is never in the training data. Scored on the held-out match's ball status
(D = dead): frame precision / recall / F1 of D and the block-level F1 of the dead-ball blocks (the organisers' possession-spell
scorer applied to the dead blocks). Possession (H / A) is not scored here: the provider's possession flips on every contested
touch and is finer than the challenge's.

    MTAH_EXTERNAL_DIR=... MTAH_SCORER_DIR=... python experiments/idsse_dead.py
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


def dead_metrics(true, pred):
    """Frame-level and block-level scores of the dead ball (D) between two label arrays."""
    t, p = true == "D", pred == "D"
    tp = (t & p).sum()
    prec, rec = tp / max(p.sum(), 1), tp / max(t.sum(), 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    as_blocks = lambda a: np.where(a == "D", "H", "A").astype(object)   # dead blocks become the "possessions" of the scorer
    block_f1, _, count_err = score(as_blocks(true), as_blocks(pred))
    return prec, rec, f1, block_f1, count_err, float(p.mean())


if __name__ == "__main__":
    cfg = config.load()
    block, fps, m = cfg["time"]["block_frames"], cfg["time"]["fps"], cfg["model"]
    sk, dfl = load_all(cfg, verbose=False), load_idsse(cfg, verbose=False)
    sk_ids, dfl_ids = sorted(sk), sorted(dfl)
    sk_tables = [sk[i]["features"] for i in sk_ids]
    sk_labels = [sk[i]["label"] for i in sk_ids]
    sk_phase_w = [lab.training_weights(sk[i]["label"], sk[i]["group"]) for i in sk_ids]
    sk_dead_w = [lab.dead_weights(sk[i]["label"]) for i in sk_ids]
    dur = durations.build(sk_labels, "min_geometric", block, fps)
    reference = RegimeEmission.fit(sk_tables, sk_labels, sk_phase_w, m["beta"])
    rows = []
    for held in dfl_ids:
        rest = [i for i in dfl_ids if i != held]
        d_tables = [dfl[i]["features"] for i in rest]
        d_labels = [dfl[i]["label"] for i in rest]
        d_w = [lab.dead_weights(dfl[i]["label"]) for i in rest]
        mixes = {
            "V1 SkillCorner only": (sk_tables, sk_labels, sk_dead_w),
            "V2 DFL only": (d_tables, d_labels, d_w),
            "V3 SkillCorner + DFL": (sk_tables + d_tables, sk_labels + d_labels, sk_dead_w + d_w),
            "V4 SkillCorner + DFL x%d" % DFL_WEIGHT: (sk_tables + d_tables, sk_labels + d_labels, sk_dead_w + [DFL_WEIGHT * w for w in d_w]),
        }
        models = {"reference (3-class, SkillCorner)": reference}
        for name, (t, l, w) in mixes.items():
            models[name] = FactoredEmission.fit(t, l, w, sk_tables, sk_labels, sk_phase_w, m["beta"])
        d = dfl[held]
        for scenario in ("natural", "hidden40"):
            df = d["features"].copy()
            if scenario == "hidden40":
                df.loc[gap_mask(len(df), np.random.default_rng(abs(hash(held)) % 2 ** 31)), "ball_ok"] = 0.0
            for name, model in models.items():
                pred = hsmm.decode(model.log_ratio(df), dur, m["leave_to_other_team"], m["leave_to_dead"], m["alpha"], block)
                prec, rec, f1, bf1, cerr, share = dead_metrics(d["label"], pred)
                rows.append({"held": held, "scenario": scenario, "model": name, "D precision": prec, "D recall": rec, "D F1": f1,
                             "D block F1": bf1, "D block count error": cerr, "predicted D share": share,
                             "true D share": float((d["label"] == "D").mean())})
        print("done", held, flush=True)
    res = pd.DataFrame(rows)
    os.makedirs(cfg["paths"]["output_dir"], exist_ok=True)
    res.to_csv(os.path.join(cfg["paths"]["output_dir"], "idsse_dead_results.csv"), index=False)
    for scenario in ("natural", "hidden40"):
        sub = res[res.scenario == scenario]
        print("\n[%s ball] mean over the 7 held-out DFL matches (true dead share %.3f)" % (scenario, sub["true D share"].mean()))
        print(sub.groupby("model")[["D precision", "D recall", "D F1", "D block F1", "D block count error", "predicted D share"]].mean().round(3).to_string())
