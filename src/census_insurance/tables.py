"""ACS table/variable bundles for insurance-focused EDA."""

VARIABLES = {
    "population_total": "B01001_001E",
    "median_household_income": "B19013_001E",
    "housing_units_total": "B25001_001E",
    "owner_occupied_units": "B25003_002E",
    "occupied_units_total": "B25003_001E",
    "median_home_value": "B25077_001E",
    "housing_built_1979_earlier": "B25034_011E",
    "housing_built_1980_1989": "B25034_006E",
    "housing_built_total": "B25034_001E",
    "workers_commute_total": "B08303_001E",
    "workers_commute_60_89": "B08303_012E",
    "workers_commute_90_plus": "B08303_013E",
    "poverty_universe": "B17001_001E",
    "poverty_count": "B17001_002E",
    "education_total_25_plus": "B15003_001E",
    "education_bachelors": "B15003_022E",
    "education_masters": "B15003_023E",
    "education_professional": "B15003_024E",
    "education_doctorate": "B15003_025E",
    "households_vehicle_total": "B08201_001E",
    "households_no_vehicle": "B08201_002E",
}

FEATURE_DESCRIPTIONS = {
    "homeownership_rate": "Owner-occupied units / occupied units",
    "old_housing_share": "(1980-1989 + 1979 and earlier) / all housing structures",
    "long_commute_share": "Workers with commute >= 60 min / all workers",
    "poverty_rate": "Population in poverty / poverty universe",
    "bachelors_plus_share": "Population 25+ with bachelor's or higher / population 25+",
    "no_vehicle_share": "Households with no vehicle / all households",
}
