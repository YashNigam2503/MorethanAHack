# More Than A Hack 2027 — possession segmentation (BAR-ATH)

Frame-level H/A/D/X possession labels for the 199,319 frames of the tactical video, produced by a
hand-written, untrained decoder over the provided automatic tracking (ATD) and Smart Tagging.
The method, its rules and its limitations are described in [TECHNICAL.md](TECHNICAL.md).

No match data, caches or predictions are in this repository. Everything generated goes to
`local_cache/`, which is Git-ignored.

## Setup

Python 3.11 (tested on 3.11.18).

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows;  source .venv/bin/activate on macOS/Linux
pip install -r requirements.txt
```

Put the competition data bundle in `./data` (the layout in the organisers' `DATA.md`:
`data/tactical/BAR-ATH_tactical_atd.xml`, `..._atd.txt`, `..._smart_tagging.xml`), or point
`MTAH_DATA_DIR` at it.

## Regenerate the submitted file

Run from the repository root, in this order:

```bash
python stage2_features.py                              # validate tracking, cache geometry + player speeds
python possession_evidence.py                          # per-frame geometry and tag evidence
python possession_model.py --dead-from-missing --viterbi
python evaluation/validate_submission.py local_cache/submission_v4_viterbi.csv
```

The submitted file is `local_cache/submission_v4_viterbi.csv`, uploaded as `<team_name>.csv`.
Every step is deterministic, so the same data produces a byte-identical file; the exact
configuration of each run is written next to it as `submission_v4_viterbi_config.json`.

If `possession_evidence.py` reports Smart Tagging team names that do not match the tracking
metadata, it stops and asks for explicit aliases (`--home-alias` / `--away-alias`); pass the same
aliases every time you regenerate.

## Other decoder modes

All are off by default and write to their own files, so none overwrites another.

| command | output |
|---|---|
| `python possession_model.py` | `submission_v1.csv` — greedy causal decoder |
| `... --backdate-possession --backdate-dead-entry --backdate-restart` | `submission_v2.csv` |
| `... --viterbi` | `submission_v3.csv` |
| `... --dead-from-missing` | `submission_v4.csv` |
| `... --dead-from-missing --viterbi` | `submission_v4_viterbi.csv` (submitted) |

## Tests

Synthetic fixtures only; no match data is needed.

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

Passing tests show that the code does what it documents. They are not evidence of predictive
accuracy.

## Repository layout

```
stage2_features.py        tracking validation and per-frame geometry / player-speed cache
possession_evidence.py    geometry and Smart Tagging evidence arrays
possession_model.py       greedy and Viterbi decoders, dead-ball rules, CLI
games.py                  per-game constants and data paths            (organisers, MIT)
starter/                  Smart Tagging loader and baseline helpers     (organisers, MIT)
evaluation/               public submission validator and metrics       (organisers, MIT)
tests/                    synthetic unit, contract and end-to-end tests
```

## Licence

`games.py`, `starter/` and `evaluation/` are copied unchanged from the organisers' repository
(metrica-sports/more-than-a-hack-2027) under the MIT Licence in
[LICENSE-METRICA.txt](LICENSE-METRICA.txt). The match data is not covered by any licence in this
repository and is not included.
