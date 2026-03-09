#!/usr/bin/env python3
"""Build an insurance-focused ACS dataset joined by GEOID."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd
import requests

from census_insurance.tables import VARIABLES

BASE_URL_TEMPLATE = "https://api.census.gov/data/{year}/acs/acs5"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=2024, help="ACS year, e.g. 2024")
    parser.add_argument(
        "--geography",
        choices=["state", "county", "tract"],
        default="tract",
        help="Target geography level",
    )
    parser.add_argument("--state", default="17", help="2-digit FIPS state code")
    parser.add_argument("--county", default="031", help="3-digit FIPS county code")
    parser.add_argument(
        "--output-dir",
        default="data/processed",
        help="Directory for output CSV/Parquet",
    )
    parser.add_argument("--api-key", default=None, help="Optional Census API key")
    return parser.parse_args()


def chunked(items: List[str], size: int) -> Iterable[List[str]]:
    for idx in range(0, len(items), size):
        yield items[idx : idx + size]


def geography_params(geography: str, state: str, county: str) -> Dict[str, str]:
    if geography == "state":
        return {"for": "state:*"}
    if geography == "county":
        return {"for": "county:*", "in": f"state:{state}"}
    return {"for": "tract:*", "in": f"state:{state} county:{county}"}


def fetch_chunk(
    year: int,
    variables: List[str],
    geography: str,
    state: str,
    county: str,
    api_key: str | None,
) -> pd.DataFrame:
    params: Dict[str, str] = {
        "get": ",".join(["NAME", *variables]),
        **geography_params(geography, state, county),
    }
    if api_key:
        params["key"] = api_key

    response = requests.get(BASE_URL_TEMPLATE.format(year=year), params=params, timeout=60)
    response.raise_for_status()
    payload = response.json()
    return pd.DataFrame(payload[1:], columns=payload[0])


def add_geoid(df: pd.DataFrame) -> pd.DataFrame:
    geo_cols = [col for col in ["state", "county", "tract"] if col in df.columns]
    df["GEOID"] = df[geo_cols].agg("".join, axis=1)
    return df


def cast_numeric(df: pd.DataFrame, keep: List[str]) -> pd.DataFrame:
    numeric_cols = [col for col in df.columns if col not in keep]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def safe_divide(num: pd.Series, den: pd.Series) -> pd.Series:
    ratio = num / den.replace({0: pd.NA})
    return ratio.astype("float64")


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["homeownership_rate"] = safe_divide(out["owner_occupied_units"], out["occupied_units_total"])
    out["old_housing_share"] = safe_divide(
        out["housing_built_1979_earlier"] + out["housing_built_1980_1989"],
        out["housing_built_total"],
    )
    out["long_commute_share"] = safe_divide(
        out["workers_commute_60_89"] + out["workers_commute_90_plus"],
        out["workers_commute_total"],
    )
    out["poverty_rate"] = safe_divide(out["poverty_count"], out["poverty_universe"])
    out["bachelors_plus_share"] = safe_divide(
        out["education_bachelors"]
        + out["education_masters"]
        + out["education_professional"]
        + out["education_doctorate"],
        out["education_total_25_plus"],
    )
    out["no_vehicle_share"] = safe_divide(out["households_no_vehicle"], out["households_vehicle_total"])
    return out


def main() -> None:
    args = parse_args()

    reverse_map = {v: k for k, v in VARIABLES.items()}
    variable_codes = list(VARIABLES.values())

    chunks = []
    for variable_chunk in chunked(variable_codes, size=20):
        chunk_df = fetch_chunk(
            year=args.year,
            variables=variable_chunk,
            geography=args.geography,
            state=args.state,
            county=args.county,
            api_key=args.api_key,
        )
        chunks.append(chunk_df)

    merged = chunks[0]
    join_cols = [c for c in ["NAME", "state", "county", "tract"] if c in merged.columns]
    for chunk_df in chunks[1:]:
        merged = merged.merge(chunk_df, on=join_cols, how="inner")

    renamed = merged.rename(columns=reverse_map)
    renamed = add_geoid(renamed)

    keep_cols = [c for c in ["GEOID", "NAME", "state", "county", "tract"] if c in renamed.columns]
    renamed = cast_numeric(renamed, keep=keep_cols)

    features = build_features(renamed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_name = f"acs5_{args.year}_{args.geography}"
    raw_csv = out_dir / f"{base_name}_raw.csv"
    feat_csv = out_dir / f"{base_name}_features.csv"
    renamed.to_csv(raw_csv, index=False)
    features.to_csv(feat_csv, index=False)

    try:
        renamed.to_parquet(out_dir / f"{base_name}_raw.parquet", index=False)
        features.to_parquet(out_dir / f"{base_name}_features.parquet", index=False)
    except Exception as exc:  # pragma: no cover
        print(f"Parquet export skipped: {exc}")

    print(f"Wrote {raw_csv}")
    print(f"Wrote {feat_csv}")


if __name__ == "__main__":
    main()
