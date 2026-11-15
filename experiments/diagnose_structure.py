"""Label-free structure of the predictions: external matches (held out) versus the tactical tracking.

For each prediction (no ground truth involved for the tactical tracking) it reports the number of possessions, their mean
duration, the dead-ball share, and the rate of state changes per minute split by whether the ball was seen. On the
external matches the same numbers of the true labels are shown for reference.

    MTAH_EXTERNAL_DIR=... MTAH_DATA_DIR=... python experiments/diagnose_structure.py
"""

import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from possession import config  # noqa: E402
from possession.external import labels as lab  # noqa: E402
from possession.external.cache import load_all  # noqa: E402
from possession.external.masks import gap_mask  # noqa: E402
from possession.features import atd_features  # noqa: E402
from possession.io import atd  # noqa: E402
from possession.model import durations, hsmm  # noqa: E402
from possession.model.classifier import RegimeEmission  # noqa: E402
from possession.pipeline.make_submission import decode_match  # noqa: E402

FPS = 25


def structure(pred, ball_ok, graded):
    """Statistics of a label array restricted to the graded frames (a boolean mask)."""
    out = {}
    live = np.isin(pred, ["H", "A"]) & graded
    edge = np.diff(np.r_[0, live.astype(np.int8), 0])
    # possessions: maximal runs of one team inside graded stretches
    runs = 0
    total = 0
    idx = np.flatnonzero(graded)
    seg_start = np.r_[0, np.flatnonzero(np.diff(idx) > 1) + 1]
    seg_stop = np.r_[seg_start[1:], len(idx)]
    for a, b in zip(seg_start, seg_stop):
        sl = pred[idx[a]:idx[b - 1] + 1]
        for team in ("H", "A"):
            e = np.diff(np.r_[0, (sl == team).astype(np.int8), 0])
            lens = np.flatnonzero(e == -1) - np.flatnonzero(e == 1)
            runs += len(lens)
            total += lens.sum()
    minutes = graded.sum() / FPS / 60
    out["possessions"] = runs
    out["mean_dur_s"] = total / FPS / max(runs, 1)
    out["dead_share"] = float((pred[graded] == "D").mean())
    change = np.r_[False, pred[1:] != pred[:-1]] & graded
    for name, regime in (("ball seen", ball_ok & graded), ("ball missing", ~ball_ok & graded)):
        out["changes/min " + name] = change[regime].sum() / (regime.sum() / FPS / 60) if regime.sum() else np.nan
    out["share ball missing"] = float((~ball_ok & graded).sum() / graded.sum())
    return out


if __name__ == "__main__":
    cfg = config.load()
    block, fps = cfg["time"]["block_frames"], cfg["time"]["fps"]
    m = cfg["model"]
    data = load_all(cfg, verbose=False)
    ids = sorted(data)
    rows = []
    for fold in range(5):
        test = ids[fold::5]
        train = [i for i in ids if i not in test]
        tl = [data[i]["label"] for i in train]
        regime = RegimeEmission.fit([data[i]["features"] for i in train], tl,
                                    [lab.training_weights(data[i]["label"], data[i]["group"]) for i in train], m["beta"])
        dur = durations.build(tl, "min_geometric", block, fps)
        for mid in test:
            d = data[mid]
            graded = d["label"] != "X"
            truth = structure(d["label"], d["features"]["ball_ok"].to_numpy() == 1, graded)
            rows.append({"source": "external true labels", **truth})
            for scenario in ("natural", "hidden40"):
                df = d["features"].copy()
                if scenario == "hidden40":
                    df.loc[gap_mask(len(df), np.random.default_rng(int(mid))), "ball_ok"] = 0.0
                pred = hsmm.decode(regime.log_ratio(df), dur, m["leave_to_other_team"], m["leave_to_dead"], m["alpha"], block)
                rows.append({"source": "external predicted, ball %s" % scenario,
                             **structure(pred, df["ball_ok"].to_numpy() == 1, graded)})

    tac = os.path.join(cfg["paths"]["match_data_dir"], "tactical")
    meta = glob.glob(os.path.join(tac, "*_tactical_atd.xml"))[0]
    raw = glob.glob(os.path.join(tac, "*_tactical_atd.txt"))[0]
    from possession.pipeline.make_submission import fit_models
    clf, dur = fit_models(cfg)
    players, ball = atd.read_atd(raw, cfg["match"]["n_frames"])
    graded = np.zeros(cfg["match"]["n_frames"], dtype=bool)
    for a, b in cfg["match"]["halves"]:
        graded[a:b + 1] = True
    feats = atd_features.build(players, ball, atd.slot_teams(meta), cfg["pitch"]["length_m"], cfg["pitch"]["width_m"], fps,
                               atd.home_attack_sign(meta), cfg["match"]["halves"][1][0], graded)
    pred = decode_match(cfg, clf, dur, feats)
    # the first 2,934 frames of the second half are dead by rule: leave them out of the structure statistics
    graded_stat = graded.copy()
    graded_stat[cfg["match"]["halves"][1][0]:cfg["match"]["halves"][1][0] + cfg["match"]["second_half_dead_prefix"]] = False
    rows.append({"source": "tactical predicted (no labels)", **structure(pred, feats["ball_ok"].to_numpy() == 1, graded_stat)})

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(cfg["paths"]["output_dir"], "structure_diagnosis.csv"), index=False)
    print(res.groupby("source").mean(numeric_only=True).round(3).T.to_string())
    print("\nexternal matches: sd of the true possession count %.0f, of the mean duration %.1f s, of the dead share %.3f" % (
        res[res.source == "external true labels"].possessions.std(), res[res.source == "external true labels"].mean_dur_s.std(),
        res[res.source == "external true labels"].dead_share.std()))
