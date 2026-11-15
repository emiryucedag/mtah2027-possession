"""Thin wrapper around the organisers' scorer (evaluation/possession_metrics.py of the challenge repository).

The scorer is not copied into this repository; `scripts/get_scorer.py` fetches it (or point MTAH_SCORER_DIR at a
local clone).
"""

import sys

from possession import config

_pm = None


def _scorer():
    global _pm
    if _pm is None:
        sys.path.insert(0, config.load()["paths"]["scorer_dir"])
        import possession_metrics as pm  # noqa: E402
        _pm = pm
    return _pm


def score(gt, pred):
    """(spellF1 mean over IoU, macroF1, predicted-minus-true possession count) of `pred` against `gt`.

    Both are string arrays of 'H' / 'A' / 'D' / 'X'; frames where `gt` is 'X' are not scored.
    """
    pm = _scorer()
    g, p = pm.label_codes(gt), pm.label_codes(pred)
    runs = pm.valid_runs(g)
    filled = pm.fill_abstentions(p, runs)
    spell = pm.spell_scores(g, filled, runs)
    cm3 = pm.scored_confusion(pm.confusion(g, p))
    return spell["mean_f1"], pm.macro_f1(cm3), spell["n_pred"] - spell["n_gt"]
