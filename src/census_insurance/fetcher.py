"""ACS tract-level data fetcher for insurance EDA."""

from __future__ import annotations

import os
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
import requests

from census_insurance.tables import VARIABLES

BASE_URL_TEMPLATE = "https://api.census.gov/data/{year}/acs/acs5"
# Census API hard limit is 50 variables per request (excluding geo/NAME).
# We fetch estimates + MOE pairs, so max 24 pairs per request = 48 vars + NAME + ~4 geo = 53.
# Use 22 pairs per chunk to stay safely under the limit.
_CHUNK_SIZE = 22


def _chunked(items: List[str], size: int) -> Iterable[List[str]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def _safe_divide(num: pd.Series, den: pd.Series) -> pd.Series:
    return num / den.replace({0: np.nan})


def _min_max_scale(s: pd.Series) -> pd.Series:
    lo, hi = s.min(skipna=True), s.max(skipna=True)
    if pd.isna(lo) or pd.isna(hi) or lo == hi:
        return pd.Series(0.0, index=s.index)
    return (s - lo) / (hi - lo)


def _fetch_variables(
    year: int,
    var_codes: List[str],
    state: str,
    county: str,
    api_key: str | None,
) -> pd.DataFrame:
    """Fetch a list of ACS variable codes for all tracts in a county."""
    params: Dict[str, str] = {
        "get": ",".join(["NAME", *var_codes]),
        "for": "tract:*",
        "in": f"state:{state} county:{county}",
    }
    if api_key:
        params["key"] = api_key

    response = requests.get(
        BASE_URL_TEMPLATE.format(year=year), params=params, timeout=60
    )
    response.raise_for_status()

    try:
        payload = response.json()
    except requests.exceptions.JSONDecodeError as exc:
        raise RuntimeError(
            f"Census API returned non-JSON.\nURL: {response.url}\n"
            f"Status: {response.status_code}\nBody: {response.text[:500]}"
        ) from exc

    if isinstance(payload, list) and payload and payload[0] == "error":
        raise RuntimeError(
            f"Census API error.\nURL: {response.url}\n"
            f"Error: {payload[1] if len(payload) > 1 else payload}"
        )

    df = pd.DataFrame(payload[1:], columns=payload[0])
    # Drop any duplicate columns the API occasionally returns
    return df.loc[:, ~df.columns.duplicated()].copy()


def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["homeownership_rate"] = _safe_divide(
        out["owner_occupied_units"], out["occupied_units_total"]
    )
    out["old_housing_share"] = _safe_divide(
        out["housing_built_1979_earlier"] + out["housing_built_1980_1989"],
        out["housing_built_total"],
    )
    out["long_commute_share"] = _safe_divide(
        out["workers_commute_60_89"] + out["workers_commute_90_plus"],
        out["workers_commute_total"],
    )
    out["poverty_rate"] = _safe_divide(out["poverty_count"], out["poverty_universe"])
    out["bachelors_plus_share"] = _safe_divide(
        out["education_bachelors"]
        + out["education_masters"]
        + out["education_professional"]
        + out["education_doctorate"],
        out["education_total_25_plus"],
    )
    out["no_vehicle_share"] = _safe_divide(
        out["households_no_vehicle"], out["households_vehicle_total"]
    )

    # MOE quality flag for income
    if "median_household_income_moe" in out.columns:
        out["income_moe_ratio"] = _safe_divide(
            out["median_household_income_moe"].abs(),
            out["median_household_income"].abs(),
        )
        out["poor_quality_income"] = out["income_moe_ratio"] > 0.2

    # --- General scores (0–1) ---
    # Opportunity: tracts with high income, high homeownership, low poverty
    out["opportunity_score"] = (
        _min_max_scale(out["median_household_income"])
        + _min_max_scale(out["homeownership_rate"])
        + (1 - _min_max_scale(out["poverty_rate"]))
    ) / 3

    # Risk: tracts with high poverty, long commutes, old housing, no vehicles
    out["risk_score"] = (
        _min_max_scale(out["poverty_rate"])
        + _min_max_scale(out["long_commute_share"])
        + _min_max_scale(out["old_housing_share"])
        + _min_max_scale(out["no_vehicle_share"])
    ) / 4

    # --- Homeowners / P&C scores ---
    # Market opportunity: high homeownership, high home value, low poverty, educated pop
    out["homeowners_opportunity"] = (
        _min_max_scale(out["homeownership_rate"])
        + _min_max_scale(out["median_home_value"])
        + (1 - _min_max_scale(out["poverty_rate"]))
        + _min_max_scale(out["bachelors_plus_share"])
    ) / 4

    # Property risk: old housing, low income, high poverty, economic vulnerability
    out["homeowners_risk"] = (
        _min_max_scale(out["old_housing_share"])
        + (1 - _min_max_scale(out["median_household_income"]))
        + _min_max_scale(out["poverty_rate"])
        + _min_max_scale(out["no_vehicle_share"])
    ) / 4

    return out


def fetch_tracts(
    state: str,
    county: str,
    year: int = 2024,
    api_key: str | None = None,
) -> pd.DataFrame:
    """Fetch ACS 5-year tract-level data and return an enriched DataFrame.

    Parameters
    ----------
    state:   2-digit FIPS state code, e.g. "17"
    county:  3-digit FIPS county code, e.g. "031"
    year:    ACS release year (default 2024)
    api_key: Optional Census API key (falls back to CENSUS_API_KEY env var)
    """
    if api_key is None:
        api_key = os.getenv("CENSUS_API_KEY")

    estimate_codes = list(VARIABLES.values())
    # ACS codes end in 'E' for estimates, 'M' for MOE — swap last character
    moe_codes = [c[:-1] + "M" for c in estimate_codes]

    # Pair estimates with their MOEs and chunk together
    paired = [code for est, moe in zip(estimate_codes, moe_codes) for code in (est, moe)]

    id_cols = ["NAME", "state", "county", "tract"]
    chunks = []
    for var_chunk in _chunked(paired, _CHUNK_SIZE * 2):
        chunk_df = _fetch_variables(year, var_chunk, state, county, api_key)
        chunks.append(chunk_df)

    # Merge all chunks on geography identity columns
    merged = chunks[0]
    for chunk_df in chunks[1:]:
        merge_on = [c for c in id_cols if c in merged.columns and c in chunk_df.columns]
        merged = merged.merge(chunk_df, on=merge_on, how="inner")

    # Build GEOID
    merged["GEOID"] = merged["state"] + merged["county"] + merged["tract"]

    # Rename: raw ACS codes → human-readable names (estimates and MOEs)
    estimate_rename = {v: k for k, v in VARIABLES.items()}
    moe_rename = {v[:-1] + "M": f"{k}_moe" for k, v in VARIABLES.items()}
    merged = merged.rename(columns={**estimate_rename, **moe_rename})

    # Drop any leftover raw ACS code columns (B#####_###E or _M pattern)
    raw_cols = [
        c for c in merged.columns
        if c not in {**estimate_rename, **moe_rename}  # already renamed
        and len(c) >= 10 and c[0] == "B" and "_" in c
    ]
    merged = merged.drop(columns=raw_cols, errors="ignore")

    # Cast all non-identity columns to numeric
    id_set = {"GEOID", "NAME", "state", "county", "tract"}
    for col in merged.columns:
        if col not in id_set:
            merged[col] = pd.to_numeric(merged[col], errors="coerce")

    return _build_features(merged)
