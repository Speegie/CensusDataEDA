# CensusDataEDA

## Project
Census Insurance EDA — exploratory analysis of ACS 5-year data focused on insurance variables at the tract/county level. Originally developed in Codex, continued in Claude Code.

## GitHub
- Repo: https://github.com/Speegie/CensusDataEDA (private)
- Owner: Speegie (Ian VanArsdall, speegie.ian@gmail.com)
- Branch: main
- Git identity: name="Ian VanArsdall", email="speegie.ian@gmail.com"

## Project Structure
- `config/acs_table_bundle.md` — ACS variable bundle and derived metrics
- `src/census_insurance/tables.py` — variable registry
- `scripts/build_acs_insurance_dataset.py` — end-to-end ACS pull + feature engineering
- `scripts/plot_insurance_scores.py` — visualization script
- `fetch_acs_option_a.py` — alternate Census API fetch approach
- `acs_tract_insurance_starter.csv` — starter dataset (committed)
- `plots/` — output charts (committed)
- `data/` — raw/processed data (excluded by .gitignore)
- `.venv/` — excluded by .gitignore

## Environment Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run Example
```bash
PYTHONPATH=src python scripts/build_acs_insurance_dataset.py \
  --year 2024 --geography tract --state 17 --county 031
```

## Git / GitHub Auth
GitHub requires a Personal Access Token (PAT) — password auth is not supported.
To avoid password prompts when pushing:
```bash
export GITHUB_TOKEN=your_pat_here
git remote set-url origin https://$GITHUB_TOKEN@github.com/Speegie/CensusDataEDA.git
```
PAT needs `repo` scope (classic token).

## Suggested Next Steps
- Add ACS margins of error (MOE) and propagate uncertainty for ratio features
- Join external risk layers (NOAA, FEMA, NHTSA) by GEOID/spatial crosswalk
- Score tracts with a composite risk-opportunity index for underwriting and growth
