# Career Readiness Datathon (Analytrix TCET, PS3) — Design

## Goal
Predict `Career_Readiness_Level` (Low/Medium/High) from a deliberately corrupted
50k-row dataset, and present cleaning, insights and model so judges follow it in 5 min.

## Corruption catalogue (measured on raw data)
| Type | Example | Fix |
|---|---|---|
| Missing tokens | nan, NULL, MISSING, ?, "" | -> NaN |
| JSON wrapper | {"v": 59} | extract v |
| Range | 51-56 | midpoint |
| Base64 categorical | UmV0YWls | decode |
| Reversed string | tsylanA ataD | reverse, match vocab |
| Currency suffix | 118274 USD | strip |
| Hex ID | 0x1 | int(x,16) |
| Corrupted label | muideM / TWVkaXVt / missing | same fixes; drop unrecoverable |

## Components
- src/preprocess.py: one pure function per corruption + clean(df) -> (clean_df, report)
- src/train.py: clean -> stratified split -> LogReg + HistGradientBoosting; run with and
  without Readiness_Score (leakage audit); save pipeline to models/
- notebooks/exploration.ipynb + reports/*.png: story charts
- app/app.py: Streamlit demo (upload dirty CSV or sample, corruption report, predictions)
- tests/test_preprocess.py: assert-based tests per cleaner
- README.md, slides.md

## Data flow
raw CSV -> clean() -> features -> model. Same clean() in training and app.

## Error handling
Unparseable -> NaN -> imputed. Unrecoverable label rows dropped and counted.
App rejects files missing required columns.

## Housekeeping
Trim requirements.txt; dataset at data/raw/ps3.csv (committed).
