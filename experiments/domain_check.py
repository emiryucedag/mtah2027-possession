"""Compare the distribution of the team-shape features on the external matches with the tactical tracking.

Only feature values are compared (no labels, nothing is tuned from this). Large differences point to a domain shift
between the broadcast-derived external tracking and the tactical tracking.

    MTAH_EXTERNAL_DIR=... MTAH_DATA_DIR=... python experiments/domain_check.py
"""

import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from possession import config  # noqa: E402
from possession.external.cache import load_all  # noqa: E402
from possession.features import atd_features, team_shape  # noqa: E402
from possession.io import atd  # noqa: E402

if __name__ == "__main__":
    cfg = config.load()
    data = load_all(cfg, verbose=False)
    ext = pd.concat([d["features"][np.isin(d["label"], ["H", "A", "D"])] for d in data.values()], ignore_index=True)
    tac = os.path.join(cfg["paths"]["match_data_dir"], "tactical")
    meta = glob.glob(os.path.join(tac, "*_tactical_atd.xml"))[0]
    raw = glob.glob(os.path.join(tac, "*_tactical_atd.txt"))[0]
    players, ball = atd.read_atd(raw, cfg["match"]["n_frames"])
    graded = np.zeros(cfg["match"]["n_frames"], bool)
    for a, b in cfg["match"]["halves"]:
        graded[a:b + 1] = True
    feats = atd_features.build(players, ball, atd.slot_teams(meta), cfg["pitch"]["length_m"], cfg["pitch"]["width_m"],
                               cfg["time"]["fps"], atd.home_attack_sign(meta), cfg["match"]["halves"][1][0], graded)
    tac_f = feats[graded]
    rows = []
    for c in ("ball_ok",) + team_shape.COLUMNS:
        rows.append({"column": c,
                     "external median": ext[c].median(), "external p10": ext[c].quantile(0.1), "external p90": ext[c].quantile(0.9),
                     "tactical median": tac_f[c].median(), "tactical p10": tac_f[c].quantile(0.1), "tactical p90": tac_f[c].quantile(0.9),
                     "external nan%": 100 * ext[c].isna().mean(), "tactical nan%": 100 * tac_f[c].isna().mean()})
    print(pd.DataFrame(rows).set_index("column").round(2).to_string())
    h1, h2 = feats.iloc[cfg["match"]["halves"][0][0]:cfg["match"]["halves"][0][1]], feats.iloc[cfg["match"]["halves"][1][0] + cfg["match"]["second_half_dead_prefix"]:cfg["match"]["halves"][1][1]]
    print("\norientation check, mean oriented home-team centroid x (rank-normalised: 0.5 = the match median; the check is that both halves agree):")
    print("  tactical 1st half %.2f, 2nd half %.2f" % (h1.cx_H.mean(), h2.cx_H.mean()))
    print("  external pooled %.2f" % ext.cx_H.mean())
