# Frame-level ball possession: More than a Hack 2027

For every frame of the tactical recording (199,319 frames, 25 fps) this code labels who has the ball: `H` (home),
`A` (away) or `D` (dead ball), inside the two graded halves, and `X` elsewhere. The output is the `frame,label` CSV the
challenge asks for. The method, its validation and its limits are described in `docs/TECHNICAL_REPORT.md`.

**Status: submission version.** The model is frozen (parameters in `configs/default.yaml`). A run from a clean copy of this
repository (new virtual environment, pinned requirements, unit tests passing) reproduces the submission file bit for bit.

## Approach in one paragraph
Possession is a sequence of long blocks, so a per-frame decision flickers and is heavily penalised by the ranking metric
(spellF1). The model therefore has two parts: an emission model that turns per-frame evidence into scores for `H` / `A` / `D`,
and a hidden semi-Markov decoder with explicit durations that combines the scores over time so the labels form coherent
blocks. When the ball is not seen the emission relies on player speed evidence only. All parameters are learned or chosen on
**external data**, never on the challenge match, its ground truth or its video.

## Model
- Features per frame (`src/possession/features/`): with the ball, the distance from the ball to the nearest home and away
  player, the number of players of each team within 5 and 10 m, the ball speed and the distance of the ball to the pitch
  boundary; with or without the ball, the mean speed of each team and the share of players that stand still (rank-normalised
  within the match, so the different speed scales of tracking systems do not matter). Optional short centred windows over
  those values exist (`features/temporal.py`, switch `context` in the config) but are off in the submission.
- Emission, factored into a dead / live classifier and a home / away classifier, each with a ball regime and a ball-missing
  regime (`src/possession/model/classifier.py`). The home / away part is fitted on 20 external matches (SkillCorner Open
  Data, possession phases); the dead / live part on the same matches plus 7 matches of the DFL open data (full-view
  tracking with the provider's own ball status, CC BY 4.0, weighted 3x per frame). Samples are weighted down near label
  edges and where two automatic views of possession disagree.
- Decoder: three-state explicit-duration Viterbi on 0.2 s blocks with a tempered evidence term
  (`src/possession/model/hsmm.py`). A possession lasts at least 2 s and a dead ball at least 4 s, with a geometric tail
  fitted on the external matches (`src/possession/model/durations.py`).
- Rules from the challenge documentation: only the two graded halves are labelled (`X` elsewhere); the first 2,934 graded
  frames of the second half (before the kick-off) are dead ball.
- Position-based team-shape features (centroids, spread) are computed but not used: in the broadcast-derived training data
  the visible players cluster around the ball, so they act as a view-dependent proxy that does not transfer to a full-view
  tracking (`features/team_shape.py`, `model/classifier.py`).

## Validation on external matches
Leave-matches-out cross-validation on the 20 SkillCorner matches, organisers' scorer, phase-based label:

| Method | spellF1 | macroF1 |
|---|---|---|
| Nearest player to the ball + majority vote over +-2 s (the baseline idea) | 0.345 | 0.733 |
| Per-frame classifier + geometric HMM decoder | 0.445 | 0.701 |
| Per-frame classifier + explicit-duration decoder | 0.476 | 0.784 |
| **Final model: factored emission, dead / live also trained on the DFL matches** | **0.469** | **0.747** |
| Same with temporal context features (evaluated, not used) | 0.506 | 0.755 |

With the ball hidden on top of what is already missing, on ~40% of the frames (generic gap masks, `experiments/m4.py`), the
ball-player-only model drops from 0.474 to 0.341 spellF1; the speed regime brings the final model to
0.39. On the 7 DFL matches (held out one at a time) the dead ball is found with a frame F1 of 0.97 (0.92 with the ball hidden
on ~40%, `experiments/m7_context.py`). The temporal context features gained on the external data but were not used: see
`docs/TECHNICAL_REPORT.md`.

These are numbers on external matches (broadcast-derived or optical tracking), not on the challenge match, and the external
labels are themselves automatic, so they are not predictions of the challenge score.

## Layout
```
configs/default.yaml        paths, frame layout, model settings
scripts/get_external_data.py   download the external training data (SkillCorner, DFL; sizes and md5 checked)
scripts/get_scorer.py          fetch the organisers' scorer (pinned commit) into third_party/
scripts/make_submission.py     end-to-end run: train, read the tracking, decode, write the CSV
src/possession/
  io/atd.py                    reader for the tactical tracking data
  features/                    ball-player and speed features, per-match rank normalisation
  external/                    external data: readers, labels, training weights, caching, gap masks
  model/                       classifiers, explicit-duration decoder, duration models, simple baseline
  eval/scorer.py               wrapper around the organisers' scorer
  pipeline/make_submission.py  the pipeline behind scripts/make_submission.py
experiments/                cross-validated comparisons on the external matches (m7_context.py: also the context-feature comparison)
tests/                      unit tests
docs/                       TECHNICAL_REPORT.md, DATA_AND_LICENSES.md
```

## Reproduce
Python 3.14 was used; dependencies are pinned in `requirements.txt`. The steps below take roughly 15 minutes plus the
downloads (about 4 GB).
```bash
pip install -r requirements.txt
python scripts/get_scorer.py                         # organisers' scorer -> third_party/
python scripts/get_external_data.py                  # SkillCorner and DFL data -> external/
export MTAH_DATA_DIR=/path/to/the/provided/match     # folder that holds tactical/ (provided to participants)
python scripts/make_submission.py                    # -> submission/submission.csv
python third_party/more-than-a-hack-2027/evaluation/validate_submission.py submission/submission.csv
python -m pytest                                     # unit tests
python experiments/m7_context.py                      # external cross-validation, with and without context features
```
Paths can be moved with `MTAH_DATA_DIR`, `MTAH_EXTERNAL_DIR`, `MTAH_CACHE_DIR` and `MTAH_SCORER_DIR`. A successful run
validates as 199,319 rows with about 29% `H`, 21% `A`, 29% `D` and 20% `X`.
The challenge data (video, tracking, tagging, homography) is provided by the organisers to registered participants and
is not part of this repository. External data and derived tables are downloaded or built locally and are not committed.

## Data and licences
The code is under the MIT licence (`LICENSE`). Data sources and their licences: `docs/DATA_AND_LICENSES.md`.
