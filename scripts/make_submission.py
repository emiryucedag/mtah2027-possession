"""Produce the `frame,label` submission for the tactical recording.

    python scripts/make_submission.py

Equivalent to `python -m possession.pipeline.make_submission` with the source folder on the path; the paths come from
configs/default.yaml and the MTAH_* environment variables (see README).
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from possession.pipeline.make_submission import main  # noqa: E402

if __name__ == "__main__":
    main()
