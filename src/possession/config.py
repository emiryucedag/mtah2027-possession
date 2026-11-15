"""Configuration: configs/default.yaml with environment-variable overrides for the paths."""

import os

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ENV = {
    "match_data_dir": "MTAH_DATA_DIR",
    "external_dir": "MTAH_EXTERNAL_DIR",
    "cache_dir": "MTAH_CACHE_DIR",
    "scorer_dir": "MTAH_SCORER_DIR",
}


def load(path=None):
    with open(path or os.path.join(ROOT, "configs", "default.yaml")) as f:
        cfg = yaml.safe_load(f)
    for key, var in ENV.items():
        value = os.environ.get(var, cfg["paths"][key])
        cfg["paths"][key] = value if os.path.isabs(value) else os.path.join(ROOT, value)
    cfg["paths"]["output_dir"] = os.path.join(ROOT, cfg["paths"]["output_dir"])
    return cfg
