"""Stage M5: does letting the two teams have different possession lengths help?

A team that has more of the ball tends to have longer possessions (log of the mean duration grows roughly linearly with
the team's share of possession time; the slope `kappa` is fitted on the training matches). Options compared on external
matches (leave-matches-out cross-validation, evidence from the three-regime emission of stage M4):

  symmetric        both teams share one duration model (the current pipeline)
  unsupervised     decode once symmetrically, read each team's possession share from that first decode, then decode
                   again with team-specific durations (shrunk towards equal by `lam`); no ground truth involved
  oracle           the true possession share of the scored match is given (an upper bound, not available in practice)
  reversed oracle  the true shares, swapped (what a wrong assumption would cost)
  fixed prior      the home team is assumed to hold `PRIOR_SHARE` of the possession in every match, true or not
  prior shift ...  instead of changing durations, add log(2 * share) to the home evidence and log(2 * (1 - share)) to
                   the away evidence of every frame (a team prior in the emission), with the true share (oracle), the
                   first-pass estimate, or the fixed prior

Reported overall and split by whether the home team really had the larger share.

    MTAH_EXTERNAL_DIR=... MTAH_SCORER_DIR=... python experiments/m5.py
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
from possession.model.classifier import RegimeEmission  # noqa: E402

ALPHA, BETA, FOLDS = 0.2, 0.5, 5
PRIOR_SHARE = 0.63          # an assumed possession share for a dominant home team
LAMBDAS = (0.5, 1.0)


def share_home(labels):
    h, a = (labels == "H").sum(), (labels == "A").sum()
    return h / max(1, h + a)


if __name__ == "__main__":
    cfg = config.load()
    block, fps = cfg["time"]["block_frames"], cfg["time"]["fps"]
    m = cfg["model"]
    data = load_all(cfg, verbose=False)
    ids = sorted(data)
    rows, slopes = [], []
    for fold in range(FOLDS):
        test = ids[fold::FOLDS]
        train = [i for i in ids if i not in test]
        tl = [data[i]["label"] for i in train]
        regime = RegimeEmission.fit([data[i]["features"] for i in train], tl,
                                    [lab.training_weights(data[i]["label"], data[i]["group"]) for i in train], BETA)
        sym = durations.build(tl, "min_geometric", block, fps)
        kappa = durations.fit_asymmetry_slope(tl, block)
        slopes.append(kappa)
        for mid in test:
            d = data[mid]
            gt_phase = d["label"]
            gt_track = lab.tracking_field_label(gt_phase, d["group"])
            true_share = share_home(gt_phase)
            t0 = time.time()
            for scenario in ("natural", "hidden40"):
                df = d["features"].copy()
                if scenario == "hidden40":
                    df.loc[gap_mask(len(df), np.random.default_rng(int(mid))), "ball_ok"] = 0.0
                llr = regime.log_ratio(df)

                def run(durs, evidence=llr):
                    return hsmm.decode(evidence, durs, m["leave_to_other_team"], m["leave_to_dead"], ALPHA, block)

                def shifted(share):
                    return llr + np.array([np.log(2 * share), np.log(2 * (1 - share)), 0.0])

                first = run(sym)
                est = share_home(first)
                variants = {"symmetric": first,
                            "oracle": run(durations.build_teams(tl, block, fps, true_share, kappa)),
                            "reversed oracle": run(durations.build_teams(tl, block, fps, 1 - true_share, kappa)),
                            "fixed prior (home %.2f)" % PRIOR_SHARE: run(durations.build_teams(tl, block, fps, PRIOR_SHARE, kappa))}
                variants["prior shift oracle"] = run(sym, shifted(true_share))
                variants["prior shift unsupervised"] = run(sym, shifted(np.clip(est, 0.3, 0.7)))
                variants["prior shift fixed (home %.2f)" % PRIOR_SHARE] = run(sym, shifted(PRIOR_SHARE))
                for lam in LAMBDAS:
                    variants["unsupervised lam %.1f" % lam] = run(durations.build_teams(tl, block, fps, 0.5 + lam * (est - 0.5), kappa))
                for name, pred in variants.items():
                    pred = np.where(gt_phase == "X", "X", pred)
                    for truth, gt in (("phase", gt_phase), ("tracking", gt_track)):
                        s, f, c = score(gt, pred)
                        rows.append({"match": mid, "scenario": scenario, "variant": name, "truth": truth, "spellF1": s,
                                     "macroF1": f, "count_err": c, "true_share_home": true_share, "est_share_home": est})
            print("scored %s in %.0f s" % (mid, time.time() - t0), flush=True)

    res = pd.DataFrame(rows)
    os.makedirs(cfg["paths"]["output_dir"], exist_ok=True)
    res.to_csv(os.path.join(cfg["paths"]["output_dir"], "m5_results.csv"), index=False)
    print("\nfitted asymmetry slope kappa per fold:", np.round(slopes, 2))
    base = res[(res.truth == "phase") & (res.variant == "symmetric")].groupby("match").agg(t=("true_share_home", "first"), e=("est_share_home", "mean"))
    print("home possession share: true vs first-pass estimate, correlation %.2f; mean absolute error %.3f" % (
        base.t.corr(base.e), (base.t - base.e).abs().mean()))
    for scenario in ("natural", "hidden40"):
        for truth in ("phase", "tracking"):
            sub = res[(res.scenario == scenario) & (res.truth == truth)]
            print("\nspellF1 [%s ball, %s label], mean over 20 matches" % (scenario, truth))
            print(sub.groupby("variant").spellF1.mean().round(3).to_string())
    sub = res[(res.scenario == "natural") & (res.truth == "phase")].copy()
    sub["home"] = np.where(sub.true_share_home > 0.55, "home >55%", np.where(sub.true_share_home < 0.45, "home <45%", "balanced"))
    print("\nspellF1 [natural ball, phase label] by the home team's true share (matches per group: %s)" % sub.groupby("home").match.nunique().to_dict())
    print(sub.pivot_table(index="variant", columns="home", values="spellF1", aggfunc="mean").round(3).to_string())
