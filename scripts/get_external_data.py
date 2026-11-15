"""Download the external training data: SkillCorner Open Data (MIT), pinned to one commit.

    python scripts/get_external_data.py                    # SkillCorner (all matches) and the DFL open data
    python scripts/get_external_data.py --limit 3          # a few SkillCorner matches, for a quick check
    python scripts/get_external_data.py --source dfl       # only the DFL open data (about 2.2 GB)
    python scripts/get_external_data.py --source skillcorner

Files go to <external_dir>/skillcorner/<match_id>/. The tracking file is stored with Git LFS, so it is fetched from
media.githubusercontent.com and its size is checked against the LFS pointer in the repository.
Credit: SkillCorner (https://github.com/SkillCorner/opendata), MIT licence.

The DFL open data ("IDSSE", https://doi.org/10.6084/m9.figshare.28196177, CC BY 4.0, data owner Deutsche Fussball Liga) is
downloaded from Figshare into <external_dir>/idsse/ (the match information and the tracking files; sizes and md5 are checked).
Credit: Bassek et al. 2025, Scientific Data 12:195, doi 10.1038/s41597-025-04505-y.
"""

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from possession import config  # noqa: E402

COMMIT = "4340d274572876239c154c90bc507a9b3250a656"
RAW = "https://raw.githubusercontent.com/SkillCorner/opendata/%s/" % COMMIT
MEDIA = "https://media.githubusercontent.com/media/SkillCorner/opendata/%s/" % COMMIT
FILES = ("match.json", "phases_of_play.csv", "dynamic_events.csv")


def curl(url, out=None):
    cmd = ["curl", "-sSfL"] + (["-o", out] if out else []) + [url]
    res = subprocess.run(cmd, check=True, capture_output=out is None)
    return None if out else res.stdout.decode()


def match_ids():
    listing = json.loads(curl(RAW + "data/matches.json"))
    return sorted(str(m["id"]) for m in listing)


def fetch_match(mid, dest):
    folder = os.path.join(dest, mid)
    os.makedirs(folder, exist_ok=True)
    for name in FILES:
        path = os.path.join(folder, "%s_%s" % (mid, name))
        if not os.path.exists(path):
            curl(RAW + "data/matches/%s/%s_%s" % (mid, mid, name), path)
    track = "%s_tracking_extrapolated.jsonl" % mid
    path = os.path.join(folder, track)
    pointer = curl(RAW + "data/matches/%s/%s" % (mid, track))
    size = int(re.search(r"size (\d+)", pointer).group(1))
    if not os.path.exists(path) or os.path.getsize(path) != size:
        curl(MEDIA + "data/matches/%s/%s" % (mid, track), path)
    if os.path.getsize(path) != size:
        raise RuntimeError("size mismatch for %s: %d != %d" % (track, os.path.getsize(path), size))


FIGSHARE = "https://api.figshare.com/v2/articles/28196177"
DFL_PARTS = ("matchinformation", "positions_raw_observed")


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_dfl(dest, dry_run=False):
    files = [f for f in json.loads(curl(FIGSHARE))["files"] if any(part in f["name"] for part in DFL_PARTS)]
    os.makedirs(dest, exist_ok=True)
    for i, f in enumerate(sorted(files, key=lambda f: f["name"]), 1):
        path = os.path.join(dest, f["name"])
        if not (os.path.exists(path) and os.path.getsize(path) == f["size"] and md5(path) == f["computed_md5"]):
            if dry_run:
                print("missing or different:", f["name"])
                continue
            curl(f["download_url"], path)
            if os.path.getsize(path) != f["size"] or md5(path) != f["computed_md5"]:
                raise RuntimeError("size or md5 mismatch for " + f["name"])
        print("%d/%d %s" % (i, len(files), f["name"]), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=("all", "skillcorner", "dfl"), default="all")
    ap.add_argument("--limit", type=int, default=0, help="SkillCorner: only the first N matches")
    ap.add_argument("--dry-run", action="store_true", help="DFL: only report files that are missing or different")
    args = ap.parse_args()
    external = config.load()["paths"]["external_dir"]
    if args.source in ("all", "skillcorner"):
        dest = os.path.join(external, "skillcorner")
        ids = match_ids()[:args.limit or None]
        for i, mid in enumerate(ids, 1):
            fetch_match(mid, dest)
            print("%d/%d %s" % (i, len(ids), mid), flush=True)
    if args.source in ("all", "dfl"):
        fetch_dfl(os.path.join(external, "idsse"), args.dry_run)
