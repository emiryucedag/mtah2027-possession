"""Parse the downloaded external matches once and keep the derived tables in the cache folder."""

import os
import pickle

from possession.external import idsse
from possession.external.skillcorner import load_match


def load_all(cfg, verbose=True):
    root = os.path.join(cfg["paths"]["external_dir"], "skillcorner")
    os.makedirs(cfg["paths"]["cache_dir"], exist_ok=True)
    data = {}
    for mid in sorted(x for x in os.listdir(root) if x.isdigit()):
        cache = os.path.join(cfg["paths"]["cache_dir"], "skillcorner_%s.pkl" % mid)
        if os.path.exists(cache):
            with open(cache, "rb") as f:
                data[mid] = pickle.load(f)
        else:
            data[mid] = load_match(os.path.join(root, mid))
            with open(cache, "wb") as f:
                pickle.dump(data[mid], f)
        if verbose:
            print("loaded", mid, flush=True)
    return data


def load_idsse(cfg, verbose=True):
    """The seven DFL matches of the IDSSE open data, parsed once and cached."""
    root = os.path.join(cfg["paths"]["external_dir"], "idsse")
    os.makedirs(cfg["paths"]["cache_dir"], exist_ok=True)
    data = {}
    for pos in sorted(f for f in os.listdir(root) if "positions_raw_observed" in f):
        mid = pos.rsplit("_", 1)[1][:-4]
        cache = os.path.join(cfg["paths"]["cache_dir"], "idsse_%s.pkl" % mid)
        if os.path.exists(cache):
            with open(cache, "rb") as f:
                data[mid] = pickle.load(f)
        else:
            data[mid] = idsse.load_match(root, mid)
            with open(cache, "wb") as f:
                pickle.dump(data[mid], f)
        if verbose:
            print("loaded", mid, flush=True)
    return data
