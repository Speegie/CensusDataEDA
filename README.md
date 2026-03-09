# Census Insurance EDA Starter

A practical Phase 1 starter for insurance-focused exploratory analysis using ACS 5-year data.

## What this includes

- Insurance-oriented ACS variable bundle
- GEOID-based join strategy (state/county/tract)
- Script to pull from Census API and produce:
  - raw joined dataset
  - feature-enriched dataset

## Project structure

- `config/acs_table_bundle.md`: table bundle and derived metrics
- `src/census_insurance/tables.py`: variable registry used by the pull script
- `scripts/build_acs_insurance_dataset.py`: end-to-end ACS pull + feature engineering

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

Example: tract-level pull for Cook County, IL (`state=17`, `county=031`):

```bash
PYTHONPATH=src python scripts/build_acs_insurance_dataset.py \
  --year 2024 \
  --geography tract \
  --state 17 \
  --county 031
```

County-level for all counties in a state:

```bash
PYTHONPATH=src python scripts/build_acs_insurance_dataset.py \
  --year 2024 \
  --geography county \
  --state 17
```

Outputs are written to `data/processed/` as CSV and Parquet.

## Suggested next phase

- Add ACS margins of error (MOE) and propagate uncertainty for ratio features
- Join external risk layers (NOAA, FEMA, NHTSA) by GEOID/spatial crosswalk
- Score tracts with a composite risk-opportunity index for underwriting and growth
