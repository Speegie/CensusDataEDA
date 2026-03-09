import os

import pandas as pd
import requests

YEAR = 2024
DATASET = "acs/acs5"
BASE_URL = f"https://api.census.gov/data/{YEAR}/{DATASET}"

# Start with one county for fast testing: Cook County, IL
STATE_FIPS = "17"
COUNTY_FIPS = "031"

# Insurance-relevant variables (ACS estimate fields end with _E)
VARIABLES = {
    "population_total": "B01001_001E",
    "median_household_income": "B19013_001E",
    "housing_units_total": "B25001_001E",
    "occupied_units_total": "B25003_001E",
    "owner_occupied_units": "B25003_002E",
    "median_home_value": "B25077_001E",
    "housing_built_total": "B25034_001E",
    "housing_built_1980_1989": "B25034_006E",
    "housing_built_1979_earlier": "B25034_011E",
    "workers_commute_total": "B08303_001E",
    "workers_commute_60_89": "B08303_012E",
    "workers_commute_90_plus": "B08303_013E",
    "households_vehicle_total": "B08201_001E",
    "households_no_vehicle": "B08201_002E",
    "poverty_universe": "B17001_001E",
    "poverty_count": "B17001_002E",
}


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.replace({0: pd.NA})


def min_max_scale(series: pd.Series) -> pd.Series:
    min_val = series.min(skipna=True)
    max_val = series.max(skipna=True)
    if pd.isna(min_val) or pd.isna(max_val) or min_val == max_val:
        return pd.Series(0.0, index=series.index)
    return (series - min_val) / (max_val - min_val)


def fetch_acs_tracts(state_fips: str, county_fips: str) -> pd.DataFrame:
    moe_variables = {
        f"{name}_moe": code.replace("_E", "_M") for name, code in VARIABLES.items()
    }
    request_variables = [*VARIABLES.values(), *moe_variables.values()]

    params = {
        "get": ",".join(["NAME", *request_variables]),
        "for": "tract:*",
        "in": f"state:{state_fips} county:{county_fips}",
    }

    # Optional API key (recommended for larger pulls)
    api_key = os.getenv("CENSUS_API_KEY")
    if api_key:
        params["key"] = api_key

    response = requests.get(BASE_URL, params=params, timeout=60)
    response.raise_for_status()

    try:
        payload = response.json()
    except requests.exceptions.JSONDecodeError as exc:
        snippet = response.text[:500].strip()
        raise RuntimeError(
            "Census API returned non-JSON content.\n"
            f"URL: {response.url}\n"
            f"Status: {response.status_code}\n"
            f"Body preview: {snippet}"
        ) from exc

    if isinstance(payload, list) and payload and payload[0] == "error":
        raise RuntimeError(
            "Census API returned an error.\n"
            f"URL: {response.url}\n"
            f"Error: {payload[1] if len(payload) > 1 else payload}"
        )

    df = pd.DataFrame(payload[1:], columns=payload[0])
    df = df.loc[:, ~df.columns.duplicated()].copy()

    # Build canonical readable columns from raw ACS codes.
    for name, code in VARIABLES.items():
        if code in df.columns:
            df[name] = df[code]
        moe_code = code.replace("_E", "_M")
        if moe_code in df.columns:
            df[f"{name}_moe"] = df[moe_code]

    # Build GEOID for joins
    df["GEOID"] = df["state"] + df["county"] + df["tract"]

    # Convert numeric columns
    id_cols = {"NAME", "state", "county", "tract", "GEOID"}
    for column in list(df.columns):
        if column not in id_cols:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    required = {
        "owner_occupied_units",
        "occupied_units_total",
        "workers_commute_60_89",
        "workers_commute_90_plus",
        "workers_commute_total",
        "households_no_vehicle",
        "households_vehicle_total",
        "housing_built_1979_earlier",
        "housing_built_1980_1989",
        "housing_built_total",
        "median_household_income",
        "housing_units_total",
        "poverty_count",
        "poverty_universe",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(
            "Missing expected columns after API fetch. "
            f"Missing: {missing}. "
            f"Returned columns: {list(df.columns)}"
        )

    # Example derived features
    df["homeownership_rate"] = safe_divide(
        df["owner_occupied_units"], df["occupied_units_total"]
    )
    df["long_commute_share"] = safe_divide(
        df["workers_commute_60_89"] + df["workers_commute_90_plus"], df["workers_commute_total"]
    )
    df["poverty_rate"] = safe_divide(df["poverty_count"], df["poverty_universe"])
    df["no_vehicle_share"] = safe_divide(
        df["households_no_vehicle"], df["households_vehicle_total"]
    )
    df["old_housing_share"] = safe_divide(
        df["housing_built_1980_1989"] + df["housing_built_1979_earlier"],
        df["housing_built_total"],
    )
    df["income_per_housing_unit"] = safe_divide(
        df["median_household_income"], df["housing_units_total"]
    )
    df["commute_stress_index"] = df["long_commute_share"] * df["no_vehicle_share"]
    df["housing_risk_proxy"] = df["old_housing_share"] * (1 - df["homeownership_rate"])

    # MOE quality signals (example for income)
    if "median_household_income_moe" in df.columns:
        df["income_moe_ratio"] = safe_divide(
            df["median_household_income_moe"].abs(), df["median_household_income"].abs()
        )
        df["poor_quality_income"] = df["income_moe_ratio"] > 0.2
    else:
        df["income_moe_ratio"] = pd.NA
        df["poor_quality_income"] = pd.NA

    # Simple normalized scores for tract ranking
    opportunity_parts = [
        min_max_scale(df["median_household_income"]),
        min_max_scale(df["homeownership_rate"]),
        1 - min_max_scale(df["poverty_rate"]),
    ]
    risk_parts = [
        min_max_scale(df["poverty_rate"]),
        min_max_scale(df["long_commute_share"]),
        min_max_scale(df["old_housing_share"]),
        min_max_scale(df["no_vehicle_share"]),
    ]
    df["opportunity_score"] = sum(opportunity_parts) / len(opportunity_parts)
    df["risk_score"] = sum(risk_parts) / len(risk_parts)

    return df


if __name__ == "__main__":
    dataframe = fetch_acs_tracts(STATE_FIPS, COUNTY_FIPS)
    print(dataframe.head(5))
    print(f"\nRows: {len(dataframe)}")

    output_file = "acs_tract_insurance_starter.csv"
    dataframe.to_csv(output_file, index=False)
    print(f"Wrote {output_file}")
