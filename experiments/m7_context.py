"""Stage M7: do temporal context features (short centred windows) help, on SkillCorner possession and DFL dead ball?

Variants (dead / live and home / away classifiers both get the extra columns):
  frozen          single-frame features only (the frozen model)
  + speed ctx     + windowed speed / stillness of the players (2 s and 5 s)
  + both          + windowed speed / stillness and windowed ball features (ball speed, ball stillness, nearest team, distance to line)

Part 1  SkillCorner leave-matches-out cross-validation, organisers' scorer (phase label), ball as recorded and hidden on ~40%.
Part 2  DFL leave-one-match-out, dead-ball frame F1 and dead-block F1 from the provider's ball status.

    MTAH_EXTERNAL_DIR=... MTAH_SCORER_DIR=... python experiments/m7_context.py
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
from possession.features.temporal import BALL_CONTEXT, SHAPE_CONTEXT, add_context  # noqa: E402
from possession.model import durations, hsmm  # noqa: E402
from possession.model.classifier import FactoredEmission  # noqa: E402

DFL_WEIGHT = 3.0
VARIANTS = {"frozen": ((), ()), "+ speed ctx": ((), SHAPE_CONTEXT), "+ both": (BALL_CONTEXT, SHAPE_CONTEXT)}


def dead_metrics(true, pred):
    t, p = true == "D", pred == "D"
    tp = (t & p).sum()
    prec, rec = tp / max(p.sum(), 1), tp / max(t.sum(), 1)
    blocks = lambda a: np.where(a == "D", "H", "A").astype(object)
    bf1, _, cerr = score(blocks(true), blocks(pred))
    return 2 * prec * rec / max(prec + rec, 1e-9), bf1


if __name__ == "__main__":
    cfg = config.load()
    block, fps, m = cfg["time"]["block_frames"], cfg["time"]["fps"], cfg["model"]
    sk, dfl = load_all(cfg, verbose=False), load_idsse(cfg, verbose=False)
    for d in list(sk.values()) + list(dfl.values()):
        d["features"] = add_context(d["features"], fps)
    ids, dfl_ids = sorted(sk), sorted(dfl)
    dt = [dfl[i]["features"] for i in dfl_ids]
    dl = [dfl[i]["label"] for i in dfl_ids]
    dw = [DFL_WEIGHT * lab.dead_weights(x) for x in dl]

    rows = []
    for fold in range(5):
        test = ids[fold::5]
        train = [i for i in ids if i not in test]
        t, l = [sk[i]["features"] for i in train], [sk[i]["label"] for i in train]
        pw = [lab.training_weights(sk[i]["label"], sk[i]["group"]) for i in train]
        sdw = [lab.dead_weights(x) for x in l]
        dur = durations.build(l, "min_geometric", block, fps)
        models = {name: FactoredEmission.fit(t + dt, l + dl, sdw + dw, t, l, pw, m["beta"], be, se) for name, (be, se) in VARIANTS.items()}
        for mid in test:
            d = sk[mid]
            gt_phase, gt_track = d["label"], lab.tracking_field_label(d["label"], d["group"])
            for scenario in ("natural", "hidden40"):
                df = d["features"].copy()
                if scenario == "hidden40":
                    df.loc[gap_mask(len(df), np.random.default_rng(int(mid))), "ball_ok"] = 0.0
                for name, model in models.items():
                    pred = np.where(gt_phase == "X", "X", hsmm.decode(model.log_ratio(df), dur, m["leave_to_other_team"], m["leave_to_dead"], m["alpha"], block))
                    for truth, gt in (("phase", gt_phase), ("tracking", gt_track)):
                        s, f, c = score(gt, pred)
                        rows.append({"match": mid, "scenario": scenario, "variant": name, "truth": truth, "spellF1": s, "macroF1": f, "count_err": c})
        print("SkillCorner fold", fold, "done", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(cfg["paths"]["output_dir"], "m7_skillcorner.csv"), index=False)
    for scenario in ("natural", "hidden40"):
        print("\nPART 1 SkillCorner [%s ball], mean over 20 matches" % scenario)
        print(res[res.scenario == scenario].pivot_table(index="variant", columns="truth", values=["spellF1", "macroF1"], aggfunc="mean").round(3).to_string())
    print("possession-count error (natural):", res[(res.scenario == "natural") & (res.truth == "phase")].groupby("variant").count_err.mean().round(0).to_dict())

    dres = []
    skt, skl = [sk[i]["features"] for i in ids], [sk[i]["label"] for i in ids]
    spw = [lab.training_weights(sk[i]["label"], sk[i]["group"]) for i in ids]
    ssw = [lab.dead_weights(x) for x in skl]
    dur = durations.build(skl, "min_geometric", block, fps)
    for held in dfl_ids:
        rest = [i for i in dfl_ids if i != held]
        rt, rl = [dfl[i]["features"] for i in rest], [dfl[i]["label"] for i in rest]
        rw = [DFL_WEIGHT * lab.dead_weights(x) for x in rl]
        for name, (be, se) in VARIANTS.items():
            model = FactoredEmission.fit(skt + rt, skl + rl, ssw + rw, skt, skl, spw, m["beta"], be, se)
            d = dfl[held]
            for scenario in ("natural", "hidden40"):
                df = d["features"].copy()
                if scenario == "hidden40":
                    df.loc[gap_mask(len(df), np.random.default_rng(abs(hash(held)) % 2 ** 31)), "ball_ok"] = 0.0
                pred = hsmm.decode(model.log_ratio(df), dur, m["leave_to_other_team"], m["leave_to_dead"], m["alpha"], block)
                f1, bf1 = dead_metrics(d["label"], pred)
                dres.append({"held": held, "scenario": scenario, "variant": name, "D F1": f1, "D block F1": bf1})
        print("DFL", held, "done", flush=True)
    dres = pd.DataFrame(dres)
    dres.to_csv(os.path.join(cfg["paths"]["output_dir"], "m7_dfl.csv"), index=False)
    print("\nPART 2 DFL dead ball, mean over 7 held-out matches")
    print(dres.pivot_table(index="variant", columns="scenario", values=["D F1", "D block F1"], aggfunc="mean").round(3).to_string())
