"""Produce the `frame,label` submission for the tactical recording.

    python -m possession.pipeline.make_submission

Steps: fit the emission classifier and the duration model on the external matches -> read the tactical ATD -> ball-player features ->
decode each graded half -> write `submission/submission.csv` (199,319 rows). Frames outside the graded halves are
written as `X`; the first 2,934 graded frames of the second half (before the kick-off) are dead ball by rule.
"""

import glob
import os

import numpy as np

from possession import config
from possession.external import cache, labels
from possession.features import atd_features
from possession.features.temporal import BALL_CONTEXT, SHAPE_CONTEXT, add_context, add_context_by_stretch
from possession.io import atd
from possession.model import durations, hsmm
from possession.model.classifier import FactoredEmission


def fit_models(cfg):
    """Emission model and duration model, all fitted on external matches only.

    The home / away part is trained on the SkillCorner possession phases. The dead / live part is trained on the SkillCorner
    matches plus the DFL open-data matches (full-view tracking with the provider's own ball status), the latter weighted
    `dfl_weight` times per frame. Each part has a ball regime and a ball-missing regime. With `context` the features also
    include short centred windows over the per-frame values.
    """
    sk = cache.load_all(cfg)
    dfl = cache.load_idsse(cfg)
    ids, dfl_ids = sorted(sk), sorted(dfl)
    t, l = [sk[i]["features"] for i in ids], [sk[i]["label"] for i in ids]
    ha_w = [labels.training_weights(sk[i]["label"], sk[i]["group"]) for i in ids]
    dead_w = [labels.dead_weights(x) for x in l]
    dt, dl = [dfl[i]["features"] for i in dfl_ids], [dfl[i]["label"] for i in dfl_ids]
    dw = [cfg["model"]["dfl_weight"] * labels.dead_weights(x) for x in dl]
    ball_extra = shape_extra = ()
    if cfg["model"].get("context"):
        t, dt = [add_context(x, cfg["time"]["fps"]) for x in t], [add_context(x, cfg["time"]["fps"]) for x in dt]
        ball_extra, shape_extra = BALL_CONTEXT, SHAPE_CONTEXT
    clf = FactoredEmission.fit(t + dt, l + dl, dead_w + dw, t, l, ha_w, cfg["model"]["beta"], ball_extra, shape_extra)
    dur = durations.build(l, cfg["model"]["durations"], cfg["time"]["block_frames"], cfg["time"]["fps"])
    return clf, dur


def decode_match(cfg, clf, dur, features):
    m, n = cfg["model"], cfg["match"]["n_frames"]
    out = np.full(n, "X", dtype=object)
    for half, (a, b) in enumerate(cfg["match"]["halves"]):
        start = a
        if half == 1:
            start = a + cfg["match"]["second_half_dead_prefix"]
            out[a:start] = "D"
        llr = clf.log_ratio(features.iloc[start:b + 1])
        out[start:b + 1] = hsmm.decode(llr, dur, m["leave_to_other_team"], m["leave_to_dead"], m["alpha"],
                                       cfg["time"]["block_frames"])
    return out


def main():
    cfg = config.load()
    tactical = os.path.join(cfg["paths"]["match_data_dir"], "tactical")
    meta = glob.glob(os.path.join(tactical, "*_tactical_atd.xml"))[0]
    raw = glob.glob(os.path.join(tactical, "*_tactical_atd.txt"))[0]
    clf, dur = fit_models(cfg)
    players, ball = atd.read_atd(raw, cfg["match"]["n_frames"])
    graded = np.zeros(cfg["match"]["n_frames"], dtype=bool)
    for a, b in cfg["match"]["halves"]:
        graded[a:b + 1] = True
    features = atd_features.build(players, ball, atd.slot_teams(meta), cfg["pitch"]["length_m"], cfg["pitch"]["width_m"],
                                  cfg["time"]["fps"], atd.home_attack_sign(meta), cfg["match"]["halves"][1][0], graded)
    if cfg["model"].get("context"):
        features = add_context_by_stretch(features, [tuple(h) for h in cfg["match"]["halves"]], cfg["time"]["fps"])
    pred = decode_match(cfg, clf, dur, features)
    os.makedirs(cfg["paths"]["output_dir"], exist_ok=True)
    path = os.path.join(cfg["paths"]["output_dir"], "submission.csv")
    with open(path, "w") as f:
        f.write("frame,label\n")
        f.writelines("%d,%s\n" % (i, lab) for i, lab in enumerate(pred))
    print("wrote", path)


if __name__ == "__main__":
    main()
