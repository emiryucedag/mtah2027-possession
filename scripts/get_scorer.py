"""Fetch the organisers' public repository (it contains the scorer) at a pinned commit into third_party/.

    python scripts/get_scorer.py
"""

import os
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
URL = "https://github.com/metrica-sports/more-than-a-hack-2027.git"
COMMIT = "f8f2629ff1e0985f81d7e59d8413678514ee1530"
DEST = os.path.join(ROOT, "third_party", "more-than-a-hack-2027")

if __name__ == "__main__":
    if not os.path.isdir(DEST):
        os.makedirs(os.path.dirname(DEST), exist_ok=True)
        subprocess.run(["git", "clone", "-q", URL, DEST], check=True)
    subprocess.run(["git", "-C", DEST, "checkout", "-q", COMMIT], check=True)
    print("scorer ready:", os.path.join(DEST, "evaluation"))
