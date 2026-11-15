# Technical report: frame-level ball possession, More than a Hack 2027

## 1. Summary
The submission labels each of the 199,319 frames of the tactical recording with `H`, `A` or `D` inside the two graded halves, and `X` elsewhere. The model has two parts. A per-frame emission model turns ball-player evidence (and, where the ball is missing, player speed evidence) into scores for the three labels. A hidden semi-Markov decoder with explicit durations combines those scores over time so that the output consists of coherent possession blocks rather than a flickering sequence. Every learned or chosen parameter comes from external data (SkillCorner Open Data and the DFL open data); nothing was fitted to the challenge match, its ground truth or its video.

## 2. How the task and the metric shaped the design
- The ranking metric, spellF1, matches possession blocks (maximal runs of `H` or of `A`) by temporal overlap. Measured with the organisers' scorer on external matches, one spurious split per minute lowers spellF1 from 1.00 to 0.81, three per minute to 0.57, while deleting every possession shorter than 2 s only lowers it to 0.90. Flicker is far more costly than missing a short possession, so the decoder is built to prefer long, coherent blocks.
- Random timing errors of up to 5 frames cost under 0.005 spellF1 and of up to 1 s about 0.1; a systematic 1 s lag costs about 0.24 on the external matches (0.35 in the organisers' own table). The labels of the training data themselves disagree by 0.5 to 1 s about when possession changes, so frame-exact decisions are not pursued.
- A predicted `X` is carried forward by the scorer, so the model never abstains inside the graded frames.

## 3. Data
**Provided and used:** the tactical tracking data (ATD) with its metadata (team of each track slot, attack direction of the first half), and the frame layout of the graded halves from the challenge documentation.
**Provided and not used:** the video, the homographies, the Smart Tagging, the broadcast recording. Smart Tagging was inspected on the public first five minutes: its set-piece intervals start inside the stoppage rather than at the restart and it misses stoppages, so it does not give dead-ball boundaries.
**External, used:**
- SkillCorner Open Data (MIT), 20 A-League matches: tracking derived from broadcast video with per-frame possession phases. Labels: inside a phase of play the team in possession, outside any phase (ball out of play) dead ball. Provides the home / away training signal and the duration statistics.
- DFL open data (CC BY 4.0), 7 Bundesliga matches: full-view 25 Hz tracking with the provider's own ball status on every frame. Used for the dead / live part only: its possession flag changes on every contested touch (median block 3.8 s), much finer than the possession blocks of this challenge.
- Cross-domain checks (not training): the three Metrica sample games, whose tracking is full view and video-derived. They have no licence file; they are credited here and were used only to test how a model trained on the other data behaves on a full-view tracking. The scripts for these checks are not part of the repository.
**The public ground truth (first five minutes)** was used twice, only as a check that the pipeline is wired correctly: once on an early version (team accuracy 0.89 where both teams are live, no home / away swap) and once on the next version (spellF1 0.52, macroF1 0.83, frame accuracy 0.86, team accuracy 0.92; 19 predicted possessions against 17 true; dead ball found with precision 0.96 and recall 0.61). A later variant with temporal context features (described in section 7) was also scored on the window by the team; its dead-ball F1 fell from 0.75 to 0.38 while its spellF1 was unchanged within noise (0.529 against 0.522), and it was **not used** for that reason. The window therefore did influence which of two versions was submitted; it was not used to tune any parameter. The window holds only 17 possessions, so one missed possession moves spellF1 by about 0.06 and these figures are not a performance estimate.

### Training data in detail: source, who labelled it, how much
| Dataset | Source | Who produced the labels | How much |
|---|---|---|---|
| SkillCorner Open Data | github.com/SkillCorner/opendata (MIT) | The provider's own pipeline: tracking from broadcast video by computer vision and machine learning, and its phase-of-play annotation. We added no labels; our rule only reads the provider's fields (team in possession during a phase, no phase = ball out of play). | 20 matches, about 33 hours of play |
| DFL open data (IDSSE) | doi 10.6084/m9.figshare.28196177 (CC BY 4.0) | The data owner (Deutsche Fussball Liga): optical tracking and the ball-status flag supplied with it. We added no labels. | 7 matches, about 11 hours (137,000 frames per match at 25 Hz) |
| Metrica sample games (cross-domain checks only, not used for training) | github.com/metrica-sports/sample-data | The provider's tracking and event data; our rule derives a label from the events with a script. | 3 games |
No labelled data comes from the provided game. Every label used in training is derived by a script from fields supplied by the data owners; nobody labelled frames by hand for this project.

## 4. Method
**Tracking preparation.** Track slots of the ATD are tracklets (they appear, disappear and swap), so only per-frame sets of players and team aggregates are used, never per-slot histories beyond one second. Coordinates are converted to metres with a 105.3 x 68.0 m pitch. Players' speeds are displacements of the same slot over one second; a displacement above 12 m/s is treated as an identity swap and dropped.
**Features.** With the ball: distance from the ball to the nearest home and nearest away player and their difference, the number of players of each team within 5 m and 10 m of the ball, the ball speed over 0.12 s, the distance of the ball to the nearest pitch boundary. Without the ball, and also combined with it: the mean speed of each team and the share of players moving slower than 0.5 m/s and 1.5 m/s. These speed features are rank-normalised within the match, so differences of scale between tracking systems do not matter.
**Emission model.** Four logistic regressions: a dead / live classifier and a home / away classifier, each for two regimes (ball present: ball-player and speed features; ball missing: speed features only, weighted by 0.5). P(D | x) comes from the dead / live model and P(H | x) = (1 - P(D | x)) P(home | live, x), likewise for A; dividing by the class priors gives a scaled likelihood. The home / away models are trained on the 20 SkillCorner matches; the dead / live models on the SkillCorner matches plus the 7 DFL matches (DFL frames weighted 3x). Training frames within 1 s of a label edge get weight 0, within 1 to 2 s weight 0.5, and frames where the tracking's own possession field disagrees with the phase label get weight 0.
**Decoder.** A three-state explicit-duration Viterbi decoder on 0.2 s blocks. Per-frame evidence is summed over a block and tempered by 0.2 because consecutive frames are strongly correlated. Durations: a possession lasts at least 2 s and a dead ball at least 4 s, with a geometric tail whose mean is the mean of the training labels; the maximum is 120 s and 180 s. Transitions out of a live state go to the other team with probability 0.59 and to dead ball with 0.41 (counted on the external labels), out of dead ball equally to either team. The last segment of a half is scored with the survival function of its duration.
**Rules of the task.** Only the two graded halves are labelled (`X` elsewhere); the first 2,934 graded frames of the second half, before the kick-off, are dead ball by rule; each half is decoded separately. The home team is the first team in the metadata (Barcelona) and the attack direction of the second half is the reverse of the first.

## 4b. Possession rules implemented
- **A possession** is a maximal run of one team (`H` or `A`) with the ball in play. It ends when the other team gets the ball or when the ball goes out of play. A pass inside a team does not end it.
- **Dead ball (`D`)** is any stretch in which the ball is out of play, from the moment it leaves play until play restarts: throw-ins, goal kicks, corners, free kicks, goals and kick-offs, and other stoppages. In the training data this is every stretch outside a phase of play, and 94% of such stretches could be tied to one of those restart types.
- **Restarts:** after a dead ball the team that restarts is not fixed by a rule; the evidence of the first live frames decides, and the decoder allows either team with equal prior.
- **Loose balls, duels and challenges:** there is no separate state. A contested ball belongs to whichever team the evidence favours; in the training data the two automatic views of possession disagree most in such moments (43% of frames in the contested phases), so the labels there are the least reliable. No possession shorter than 2 s is produced, which means a very brief change of team inside a duel is absorbed into the neighbouring possession.
- **Kick-off and half-time:** only the two graded halves are labelled; the first 2,934 graded frames of the second half, before the kick-off, are dead ball.

## 5. Missing and noisy tracking
The ball is present on 59.3% of the graded frames and its gaps are long (150 gaps, median 10.7 s; 0.3% of the missing frames lie in gaps of 5 s or less), so interpolation does not help. During gaps the speed and stillness of the players supplies weaker evidence about dead ball and possession; the decoder's durations carry the state through what the evidence does not settle. The ATD sometimes contains more than 11 tracked players for a team (12.9% of the graded frames); the features use robust statistics (nearest distances, counts, shares) rather than means over slots.

## 6. Validation (external data only)
All numbers are on external matches, scored with the organisers' scorer, and are not predictions of the challenge score: the external labels are themselves automatic (two views of possession from the same provider agree at spell level only about 0.5 spellF1) and the external tracking differs from the tactical tracking.
| Check | Result |
|---|---|
| SkillCorner, leave-matches-out, vs phase label, spellF1 / macroF1 | nearest player + 2 s vote 0.345 / 0.733; geometric HMM 0.445 / 0.701; explicit durations 0.476 / 0.784; **final model (factored emission) 0.469 / 0.747**; with temporal context (not used) 0.506 / 0.755 |
| same, ball hidden on a further ~40% of the frames | final 0.388 / 0.706 (with temporal context, not used, 0.411 / 0.704; ball-player evidence only 0.341) |
| DFL, leave-one-match-out, dead-ball frame F1 / dead-block F1 | final 0.971 / 0.917 (with temporal context, not used, 0.978 / 0.937); ball hidden on ~40%: 0.919 / 0.795 (0.924 / 0.823) |
| Metrica games 1-3 (full view, event-based labels; no Metrica in training), spellF1 / macroF1 | final 0.476 / 0.716 (with temporal context, not used, 0.514 / 0.729; position-based shape features 0.273); dead-frame recall 0.70 |
The external numbers are internal consistency of a model trained and tested on automatic labels; the final model predicts 304 possessions on the Metrica games against about 277 in the event labels (with context: 257).
Metrica labels are derived from events, so their dead-ball share (0.28 to 0.38) is a lower bound.

## 7. Decisions and negative results
- **Position-based team-shape features were removed.** In the broadcast-derived training data the visible players are those near the camera, so their mean position stands in for the ball position. A model that used them merged possessions on full-view tracking (Metrica spellF1 0.27 against 0.45 without them). Only speed and stillness features, which carry over, were kept.
- **Team-specific durations and a team prior (for a dominant team) gave no gain** on external matches, even with the true possession share given; the model is symmetric and contains nothing specific to a team.
- **A ball position estimated from the players was not built.** Perfect ball estimation could recover at most about 0.10 spellF1, but a gradient-boosting estimator had a median error of 10 to 12 m and the team nearest to its estimate agreed with the team nearest to the real ball no better than chance.
- **The dead ball is trained with full-view data.** The DFL matches improved dead-ball recall on a video-derived full-view tracking (0.58 to 0.69) at a small cost on the broadcast-derived data (spellF1 0.477 to 0.469).

- **Temporal context features were tried and not used.** A rule written before the experiment accepted them on the external data (Metrica 0.476 to 0.514, SkillCorner 0.469 to 0.506, DFL 0.971 to 0.978). On the public five-minute window the team then saw a large fall in dead-ball F1 (0.75 to 0.38). A plausible cause, not proven, is that the context features rely on a nearly still ball as a dead-ball cue: in the DFL data the ball is exactly still in dead ball (99.7% of dead frames below 1 m/s), while in the challenge tracking, as in the SkillCorner and Metrica data, only about 10% of all frames are below 1 m/s. As this could not be settled on external data in time, the simpler model without context features is submitted. The code and the switch (`context` in the config) remain in the repository.

## 8. Limitations
- The labels of the training data are automatic and noisy (timing uncertainty 0.5 to 1 s, concentrated at transitions and in contested phases); the human ground truth may follow either view of possession.
- Whether the challenge's definition of dead ball equals the one in the training data is assumed, not verified; short stoppages may be treated differently (the cost for spellF1 of skipping dead blocks under 10 s was about 0.06 on external matches).
- The tactical tracking is a different system from all training data; the real test is the challenge tracking, which can only be inspected through label-free statistics.
- The external validation numbers rest on 20 broadcast-derived matches, 7 optical matches and 3 video-derived matches.
- Several choices (alpha 0.2, beta 0.5, DFL weight 3, minimum durations of 2 s and 4 s, the window lengths) came from coarse grids or single comparisons over external data.
- The optional temporal windows (not used in the submission) use frames on both sides, which is only possible because the whole recording is processed offline.
- The absolute position of the ball on the pitch was not tested as a feature.

## 9. Reproduction
See the README. In short: install the pinned requirements, fetch the scorer and the external data with the scripts, point `MTAH_DATA_DIR` at the provided tactical data, and run `python scripts/make_submission.py`. The run is deterministic on one machine (two runs give the same file). All parameters are in `configs/default.yaml`.

## 10. Credits
SkillCorner Open Data (SkillCorner, MIT licence). DFL open data: Deutsche Fussball Liga (DFL); Bassek, M., Rein, R., Weber, H., Memmert, D. et al. (2025), "An integrated dataset of spatiotemporal and event data in elite soccer", Scientific Data 12:195, doi 10.1038/s41597-025-04505-y (CC BY 4.0). Metrica Sports sample data (cross-domain checks). The scorer is the organisers' own code.
