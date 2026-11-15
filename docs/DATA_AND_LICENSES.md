# Data and licences

## Used by the code in this repository
| Source | Use | Licence | Pinned version |
|---|---|---|---|
| SkillCorner Open Data, https://github.com/SkillCorner/opendata | training and validation data: tracking, phases of play, possession field | MIT ("The MIT License, Copyright (c) 2020 SkillCorner. https://skillcorner.com"). The README asks: "if you use the data, we kindly ask that you credit SkillCorner" | commit `4340d274572876239c154c90bc507a9b3250a656` |
| DFL open data ("IDSSE"), https://doi.org/10.6084/m9.figshare.28196177 | training and validation of the dead-ball (dead / live) part: full-view tracking with the provider's ball status | CC BY 4.0, data owner Deutsche Fussball Liga (DFL) | Figshare record, files verified by size and md5 |
| Organisers' repository, https://github.com/metrica-sports/more-than-a-hack-2027 | the scoring code (`evaluation/possession_metrics.py`), used unchanged and fetched at run time | see the licence file in that repository | commit `f8f2629ff1e0985f81d7e59d8413678514ee1530` |

Credit: tracking and phase data courtesy of SkillCorner. Tracking and ball-status data of the DFL matches: Deutsche Fussball Liga (DFL); Bassek, M., Rein, R., Weber, H., Memmert, D. et al. (2025), "An integrated dataset of spatiotemporal and event data in elite soccer", Scientific Data 12:195, doi 10.1038/s41597-025-04505-y.

## Provided by the organisers (not redistributed)
The video, tracking data (ATD), Smart Tagging and homography of the challenge match are provided to registered
participants under the terms of data use and are not part of this repository.

## How the data is used
- All learned parameters and all decoder settings are chosen on the external matches above (SkillCorner and the DFL matches).
- The challenge match's ground truth (the public first five minutes included) and video are not used to train or tune
  anything. The public five minutes were scored for pipeline checks and, once, to compare two finished versions (see
  `TECHNICAL_REPORT.md`, section 3). The tracking data of the challenge match is only an input to the finished pipeline.
- Downloaded external data and derived tables live in `external/` and `cache/`, which are not committed.
