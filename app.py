"""Census Insurance EDA — Streamlit Dashboard."""

from __future__ import annotations

import io
import os
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

# Allow running from repo root without installing the package
sys.path.insert(0, str(Path(__file__).parent / "src"))
from census_insurance.fetcher import fetch_national_overview, fetch_state_overview, fetch_tracts

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Census Insurance EDA",
    page_icon="📊",
    layout="wide",
)

# ---------------------------------------------------------------------------
# FIPS reference data (top states + all IL counties for demo)
# ---------------------------------------------------------------------------
STATES = {
    "01": "Alabama", "02": "Alaska", "04": "Arizona", "05": "Arkansas",
    "06": "California", "08": "Colorado", "09": "Connecticut", "10": "Delaware",
    "11": "District of Columbia", "12": "Florida", "13": "Georgia", "15": "Hawaii",
    "16": "Idaho", "17": "Illinois", "18": "Indiana", "19": "Iowa",
    "20": "Kansas", "21": "Kentucky", "22": "Louisiana", "23": "Maine",
    "24": "Maryland", "25": "Massachusetts", "26": "Michigan", "27": "Minnesota",
    "28": "Mississippi", "29": "Missouri", "30": "Montana", "31": "Nebraska",
    "32": "Nevada", "33": "New Hampshire", "34": "New Jersey", "35": "New Mexico",
    "36": "New York", "37": "North Carolina", "38": "North Dakota", "39": "Ohio",
    "40": "Oklahoma", "41": "Oregon", "42": "Pennsylvania", "44": "Rhode Island",
    "45": "South Carolina", "46": "South Dakota", "47": "Tennessee", "48": "Texas",
    "49": "Utah", "50": "Vermont", "51": "Virginia", "53": "Washington",
    "54": "West Virginia", "55": "Wisconsin", "56": "Wyoming",
}


@st.cache_data(show_spinner="Fetching tract boundaries from TIGER…")
def fetch_tract_geojson(state_fips: str, county_fips: str) -> dict:
    """Fetch census tract GeoJSON from Census TIGER REST API (auto-paginates)."""
    base = (
        "https://tigerweb.geo.census.gov/arcgis/rest/services"
        "/TIGERweb/Tracts_Blocks/MapServer/0/query"
    )
    features: list = []
    offset = 0
    while True:
        params = {
            "where": f"STATE='{state_fips}' AND COUNTY='{county_fips}'",
            "outFields": "GEOID,NAME,STATE,COUNTY,TRACT",
            "returnGeometry": "true",
            "outSR": "4326",
            "f": "geojson",
            "resultOffset": offset,
            "resultRecordCount": 500,
        }
        resp = requests.get(base, params=params, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        batch = data.get("features", [])
        features.extend(batch)
        if not data.get("exceededTransferLimit", False) or not batch:
            break
        offset += len(batch)
    return {"type": "FeatureCollection", "features": features}


# FIPS → 2-letter abbreviation (used for Plotly's built-in USA-states locationmode)
_FIPS_TO_ABBR = {
    "01": "AL", "02": "AK", "04": "AZ", "05": "AR", "06": "CA",
    "08": "CO", "09": "CT", "10": "DE", "11": "DC", "12": "FL",
    "13": "GA", "15": "HI", "16": "ID", "17": "IL", "18": "IN",
    "19": "IA", "20": "KS", "21": "KY", "22": "LA", "23": "ME",
    "24": "MD", "25": "MA", "26": "MI", "27": "MN", "28": "MS",
    "29": "MO", "30": "MT", "31": "NE", "32": "NV", "33": "NH",
    "34": "NJ", "35": "NM", "36": "NY", "37": "NC", "38": "ND",
    "39": "OH", "40": "OK", "41": "OR", "42": "PA", "44": "RI",
    "45": "SC", "46": "SD", "47": "TN", "48": "TX", "49": "UT",
    "50": "VT", "51": "VA", "53": "WA", "54": "WV", "55": "WI",
    "56": "WY",
}


@st.cache_data(show_spinner="Fetching county boundaries from TIGER…")
def fetch_county_geojson(state_fips: str) -> dict:
    """Fetch county boundary GeoJSON for one state, filtered from Plotly's dataset."""
    url = "https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json"
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    all_data = resp.json()
    features = [f for f in all_data["features"] if f["id"].startswith(state_fips)]
    return {"type": "FeatureCollection", "features": features}


@st.cache_data(show_spinner="Loading counties…")
def fetch_counties(state_fips: str) -> dict[str, str]:
    """Return {county_fips: county_name} for all counties in a state."""
    params = {
        "get": "NAME",
        "for": "county:*",
        "in": f"state:{state_fips}",
    }
    resp = requests.get(
        "https://api.census.gov/data/2024/acs/acs5",
        params=params,
        timeout=15,
    )
    resp.raise_for_status()
    payload = resp.json()
    # payload[0] = ['NAME', 'state', 'county'], payload[1:] = data rows
    return {row[2]: row[0] for row in payload[1:]}

# ---------------------------------------------------------------------------
# Metric metadata — used for labels, tooltips, and explanations
# ---------------------------------------------------------------------------
METRIC_META = {
    "opportunity_score": {
        "label": "Opportunity Score",
        "color": "Blues",
        "desc": (
            "Composite score (0–1) measuring how attractive a tract is for writing "
            "new insurance business. Higher = more desirable market. "
            "Built from: median household income (normalized), homeownership rate, "
            "and inverse poverty rate — equally weighted."
        ),
    },
    "risk_score": {
        "label": "Risk Score",
        "color": "Reds",
        "desc": (
            "Composite score (0–1) measuring the overall socioeconomic risk profile "
            "of a tract. Higher = more risk indicators present. "
            "Built from: poverty rate, long-commute share (≥60 min), old housing share "
            "(pre-1990), and no-vehicle share — equally weighted."
        ),
    },
    "homeowners_opportunity": {
        "label": "Homeowners Opportunity",
        "color": "Teal",
        "desc": (
            "Homeowners/P&C-specific market opportunity score (0–1). "
            "Combines: homeownership rate (direct market size), median home value "
            "(coverage amount proxy), inverse poverty rate (policy retention signal), "
            "and bachelor's-degree-plus share (risk behavior proxy) — equally weighted."
        ),
    },
    "homeowners_risk": {
        "label": "Homeowners Risk",
        "color": "Oranges",
        "desc": (
            "Homeowners/P&C-specific property risk score (0–1). "
            "Combines: old housing share (pre-1990 structures — outdated wiring, plumbing, "
            "roofing drive higher claim frequency), inverse income (economic stress on "
            "maintenance), poverty rate (underinsurance and moral hazard), and no-vehicle "
            "share (economic vulnerability proxy) — equally weighted."
        ),
    },
    "homeownership_rate": {
        "label": "Homeownership Rate",
        "color": "Greens",
        "desc": "Owner-occupied units ÷ total occupied units. Direct measure of the addressable homeowners insurance market in a tract.",
    },
    "old_housing_share": {
        "label": "Old Housing Share",
        "color": "Oranges",
        "desc": "Structures built before 1990 ÷ total structures. Older homes have higher claim frequency due to aging systems (electrical, plumbing, HVAC, roofing).",
    },
    "long_commute_share": {
        "label": "Long Commute Share",
        "color": "Purples",
        "desc": "Workers with commute ≥60 min ÷ total workers. Proxy for auto insurance exposure — longer commutes correlate with more miles driven and higher accident risk.",
    },
    "poverty_rate": {
        "label": "Poverty Rate",
        "color": "Reds",
        "desc": "Population below poverty line ÷ poverty universe. Correlates with underinsurance, policy lapses, and moral hazard across insurance lines.",
    },
    "bachelors_plus_share": {
        "label": "Bachelor's+ Share",
        "color": "Blues",
        "desc": "Population 25+ with bachelor's degree or higher ÷ population 25+. Education level correlates positively with policy retention, claims behavior, and income stability.",
    },
    "no_vehicle_share": {
        "label": "No Vehicle Share",
        "color": "Greys",
        "desc": "Households with no vehicle ÷ total households. Indicates economic vulnerability. Also inversely related to auto insurance market size.",
    },
    "median_household_income": {
        "label": "Median Household Income",
        "color": "Greens",
        "desc": "ACS estimate of median household income for the tract. Key driver of insurance affordability, coverage limits, and premium volume.",
    },
    "median_home_value": {
        "label": "Median Home Value",
        "color": "Teal",
        "desc": "ACS estimate of median owner-occupied home value. Directly sets the coverage amount (and therefore premium) for homeowners policies.",
    },
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _geojson_center(geojson: dict) -> tuple[float, float]:
    """Return (lat, lon) centroid of a GeoJSON FeatureCollection."""
    lons: list[float] = []
    lats: list[float] = []
    for f in geojson.get("features", []):
        geom = f.get("geometry", {})
        coords = geom.get("coordinates", [])
        if geom.get("type") == "Polygon":
            for ring in coords:
                lons += [p[0] for p in ring]
                lats += [p[1] for p in ring]
        elif geom.get("type") == "MultiPolygon":
            for poly in coords:
                for ring in poly:
                    lons += [p[0] for p in ring]
                    lats += [p[1] for p in ring]
    if not lats:
        return 39.0, -98.0
    return sum(lats) / len(lats), sum(lons) / len(lons)


_DOLLAR_METRICS = {"median_household_income", "median_home_value"}
_PCT_METRICS = {
    "homeownership_rate", "poverty_rate", "old_housing_share",
    "long_commute_share", "no_vehicle_share", "bachelors_plus_share",
}


def _fmt_metric(val: float, key: str) -> str:
    if key in _DOLLAR_METRICS:
        return f"${val:,.0f}"
    if key in _PCT_METRICS:
        return f"{val:.1%}"
    return f"{val:.3f}"


def short_name(name: str) -> str:
    """Extract the Census tract label from the full NAME field."""
    return name.split(";")[0].strip() if name and ";" in name else name


def quadrant_label(opp: float, risk: float, opp_mid: float, risk_mid: float) -> str:
    high_opp = opp >= opp_mid
    high_risk = risk >= risk_mid
    if high_opp and not high_risk:
        return "Grow Aggressively"
    if high_opp and high_risk:
        return "Write Carefully"
    if not high_opp and not high_risk:
        return "Deprioritize"
    return "Avoid"


QUADRANT_COLORS = {
    "Grow Aggressively": "#2ecc71",
    "Write Carefully": "#f39c12",
    "Deprioritize": "#95a5a6",
    "Avoid": "#e74c3c",
}

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Fetching national state-level data from Census API…")
def load_national(year: int) -> pd.DataFrame:
    return fetch_national_overview(year)


@st.cache_data(show_spinner="Fetching tract data from Census API…")
def load_live(state: str, county: str, year: int) -> pd.DataFrame:
    return fetch_tracts(state, county, year)


@st.cache_data(show_spinner="Fetching county overview from Census API…")
def load_state_overview(state: str, year: int) -> pd.DataFrame:
    return fetch_state_overview(state, year)


def load_csv(uploaded) -> pd.DataFrame:
    return pd.read_csv(uploaded)


@st.cache_data(show_spinner="Loading full US counties GeoJSON…")
def load_full_counties_geojson() -> dict:
    """Load full Plotly US counties GeoJSON (all 50 states)."""
    url = "https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json"
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    return resp.json()


@st.cache_data(show_spinner="Loading ZIP → county crosswalk…")
def load_zip_crosswalk() -> pd.DataFrame:
    """Download Census 2020 ZCTA-to-county relationship file.

    Returns DataFrame with columns: zip (5-digit ZCTA), county_fips (5-digit FIPS).
    For ZCTAs spanning multiple counties, keeps the county with the largest land overlap.
    """
    url = (
        "https://www2.census.gov/geo/docs/maps-data/data/rel2020/zcta520/"
        "tab20_zcta520_county20_natl.txt"
    )
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    xwalk = pd.read_csv(io.StringIO(resp.text), sep="|", dtype=str)
    xwalk.columns = [c.strip().upper() for c in xwalk.columns]
    zcta_col = next((c for c in xwalk.columns if "ZCTA5" in c and "GEOID" in c), None)
    county_col = next((c for c in xwalk.columns if "COUNTY" in c and "GEOID" in c), None)
    area_col = next((c for c in xwalk.columns if "AREALAND_PART" in c), None)
    if not zcta_col or not county_col:
        raise ValueError(f"Unexpected crosswalk columns: {list(xwalk.columns)}")
    if area_col:
        xwalk[area_col] = pd.to_numeric(xwalk[area_col], errors="coerce").fillna(0)
        xwalk = xwalk.sort_values(area_col, ascending=False)
    xwalk = xwalk.drop_duplicates(subset=zcta_col, keep="first")
    return xwalk[[zcta_col, county_col]].rename(
        columns={zcta_col: "zip", county_col: "county_fips"}
    )

# ---------------------------------------------------------------------------
# Sidebar — data source & geography
# ---------------------------------------------------------------------------

st.sidebar.title("Census Insurance EDA")
st.sidebar.markdown("---")

data_source = st.sidebar.radio(
    "Data source",
    ["Live — Census API", "Upload CSV"],
    index=0,
)

df: pd.DataFrame | None = None

if data_source == "Live — Census API":
    state_label = st.sidebar.selectbox(
        "State",
        options=list(STATES.keys()),
        format_func=lambda k: f"{STATES[k]} ({k})",
        index=list(STATES.keys()).index("17"),
    )
    counties = fetch_counties(state_label)
    county_options = sorted(counties.keys(), key=lambda k: counties[k])
    county_input = st.sidebar.selectbox(
        "County",
        options=county_options,
        format_func=lambda k: counties[k],
    )
    county_fips = county_input
    year = st.sidebar.selectbox("ACS Year", [2024, 2023, 2022, 2021, 2020], index=0)

    col_btn1, col_btn2 = st.sidebar.columns(2)
    if col_btn1.button("State Overview", use_container_width=True):
        state_df = load_state_overview(state_label, year)
        st.session_state["state_df"] = state_df
        st.session_state["state_fips"] = state_label
        st.session_state["overview_year"] = year
        # Clear any previous tract-level data so the overview tab is shown first
        st.session_state.pop("df", None)

    if col_btn2.button("County Detail", type="primary", use_container_width=True):
        df = load_live(state_label, county_fips, year)
        st.session_state["df"] = df
        st.session_state["source_label"] = (
            f"{STATES[state_label]} — {counties.get(county_fips, county_fips)} ({year} ACS 5-yr)"
        )

    if "df" in st.session_state:
        df = st.session_state["df"]

else:
    uploaded = st.sidebar.file_uploader("Upload a CSV", type=["csv"])
    if uploaded:
        df = load_csv(uploaded)
        st.session_state["df"] = df
        st.session_state["source_label"] = f"Uploaded: {uploaded.name}"
        if "df" in st.session_state:
            df = st.session_state["df"]

# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------

st.title("Census Insurance EDA")

if df is not None:
    source_label = st.session_state.get("source_label", "")
    if source_label:
        st.caption(f"County data loaded: {source_label}")
    csv_bytes = df.to_csv(index=False).encode()
    st.sidebar.download_button(
        "⬇ Download current data (CSV)",
        data=csv_bytes,
        file_name="census_insurance_tracts.csv",
        mime="text/csv",
    )

tab_drilldown, tab_state, tab_map, tab_general, tab_homeowners, tab_portfolio = st.tabs(
    ["🌎 Drill-Down Map", "🗺️ State Overview", "🗺️ Map View", "📊 General Overview", "🏠 Homeowners / P&C", "📋 Client Portfolio"]
)

# ===========================================================================
# TAB 0 — DRILL-DOWN MAP  (Phase 1: US state-level choropleth)
# ===========================================================================
with tab_drilldown:
    st.header("National Drill-Down Map")
    st.markdown(
        "National view of all 50 states scored by the selected metric. "
        "Select a state below the national map to drill down to county level."
    )

    dd_col1, dd_col2 = st.columns([2, 1])
    with dd_col1:
        dd_metric = st.selectbox(
            "Color states by",
            list(METRIC_META.keys()),
            format_func=lambda k: METRIC_META[k]["label"],
            key="dd_metric",
        )
    with dd_col2:
        dd_year = st.selectbox("ACS Year", [2024, 2023, 2022, 2021, 2020], key="dd_year")

    dd_meta = METRIC_META[dd_metric]
    st.info(f"**{dd_meta['label']}** — {dd_meta['desc']}")

    try:
        nat_df = load_national(dd_year)
    except Exception as exc:
        st.error(f"Could not load national ACS data: {exc}")
        nat_df = None

    if nat_df is not None:
        # Map state FIPS → abbreviation for Plotly's built-in USA-states renderer
        nat_df = nat_df.copy()
        nat_df["state_abbr"] = nat_df["GEOID"].map(_FIPS_TO_ABBR)
        nat_df = nat_df.dropna(subset=["state_abbr"])

        c_lo = float(nat_df[dd_metric].quantile(0.05))
        c_hi = float(nat_df[dd_metric].quantile(0.95))

        _NAT_HOVER: dict[str, str] = {
            "opportunity_score": ":.3f",
            "risk_score": ":.3f",
            "median_household_income": ":$,.0f",
            "homeownership_rate": ":.1%",
            "poverty_rate": ":.1%",
        }
        nat_hover = {
            k: v for k, v in _NAT_HOVER.items()
            if k in nat_df.columns and k != dd_metric
        }
        nat_labels = {
            k: METRIC_META.get(k, {}).get("label", k)
            for k in list(nat_hover) + [dd_metric]
        }

        fig_nat = px.choropleth(
            nat_df,
            locations="state_abbr",
            locationmode="USA-states",
            color=dd_metric,
            color_continuous_scale=dd_meta["color"],
            range_color=[c_lo, c_hi],
            hover_name="state_name",
            hover_data=nat_hover,
            labels=nat_labels,
            scope="usa",
            height=520,
        )
        fig_nat.update_layout(margin={"r": 0, "t": 30, "l": 0, "b": 0})
        st.plotly_chart(fig_nat, use_container_width=True)
        st.caption("Alaska and Hawaii shown as insets. Color range: 5th–95th percentile.")

        s = nat_df[dd_metric].dropna()
        kc1, kc2, kc3, kc4, kc5 = st.columns(5)
        kc1.metric("States", len(s))
        kc2.metric("Min", _fmt_metric(s.min(), dd_metric))
        kc3.metric("Median", _fmt_metric(s.median(), dd_metric))
        kc4.metric("Mean", _fmt_metric(s.mean(), dd_metric))
        kc5.metric("Max", _fmt_metric(s.max(), dd_metric))

        with st.expander("📊 State Rankings", expanded=False):
            rank_df = (
                nat_df[["state_name", dd_metric]]
                .sort_values(dd_metric, ascending=False)
                .copy()
            )
            rank_df.index = range(1, len(rank_df) + 1)
            st.dataframe(
                rank_df.rename(columns={"state_name": "State", dd_metric: dd_meta["label"]}),
                use_container_width=True,
                height=420,
            )

    # -----------------------------------------------------------------------
    # Phase 2 — County drill-down
    # -----------------------------------------------------------------------
    st.markdown("---")

    dd_state_col1, _dd_state_col2 = st.columns([3, 1])
    with dd_state_col1:
        dd_drill_state = st.selectbox(
            "🔍 Drill down — select a state to view county map",
            options=[""] + sorted(STATES.keys(), key=lambda k: STATES[k]),
            format_func=lambda k: "— select a state —" if k == "" else STATES[k],
            key="dd_drill_state",
        )

    if not dd_drill_state:
        st.caption("Select a state above to load its county-level choropleth.")
    else:
        _state_name = STATES[dd_drill_state]

        try:
            _county_df = load_state_overview(dd_drill_state, dd_year)
        except Exception as _exc:
            st.error(f"Could not load county data for {_state_name}: {_exc}")
            _county_df = None

        if _county_df is not None:
            try:
                _county_geojson = fetch_county_geojson(dd_drill_state)
            except Exception as _exc:
                st.error(f"Could not load county boundaries: {_exc}")
                _county_geojson = None

            if _county_geojson and _county_df[dd_metric].notna().any():
                _c_lo = float(_county_df[dd_metric].quantile(0.05))
                _c_hi = float(_county_df[dd_metric].quantile(0.95))

                _map_lat, _map_lon = _geojson_center(_county_geojson)

                _COUNTY_HOVER: dict[str, str] = {
                    "opportunity_score": ":.3f",
                    "risk_score": ":.3f",
                    "median_household_income": ":$,.0f",
                    "homeownership_rate": ":.1%",
                    "poverty_rate": ":.1%",
                    "old_housing_share": ":.1%",
                    "no_vehicle_share": ":.1%",
                }
                _county_hover = {
                    k: v for k, v in _COUNTY_HOVER.items()
                    if k in _county_df.columns and k != dd_metric
                }
                _county_labels = {
                    k: METRIC_META.get(k, {}).get("label", k)
                    for k in list(_county_hover) + [dd_metric]
                }

                fig_county = px.choropleth_mapbox(
                    _county_df,
                    geojson=_county_geojson,
                    locations="GEOID",
                    featureidkey="id",
                    color=dd_metric,
                    color_continuous_scale=dd_meta["color"],
                    range_color=[_c_lo, _c_hi],
                    hover_name="county_name",
                    hover_data=_county_hover,
                    labels=_county_labels,
                    mapbox_style="carto-positron",
                    zoom=5,
                    center={"lat": _map_lat, "lon": _map_lon},
                    opacity=0.75,
                    height=540,
                )
                fig_county.update_layout(margin={"r": 0, "t": 30, "l": 0, "b": 0})
                st.plotly_chart(fig_county, use_container_width=True)

                # Compact stats row right below the map
                _s = _county_df[dd_metric].dropna()
                _sc1, _sc2, _sc3, _sc4, _sc5, _sc6 = st.columns(6)
                _sc1.caption("5th–95th pct color range")
                _sc2.metric("Counties", len(_s))
                _sc3.metric("Min", _fmt_metric(_s.min(), dd_metric))
                _sc4.metric("Median", _fmt_metric(_s.median(), dd_metric))
                _sc5.metric("Mean", _fmt_metric(_s.mean(), dd_metric))
                _sc6.metric("Max", _fmt_metric(_s.max(), dd_metric))

                with st.expander(f"📊 County Rankings — {_state_name}", expanded=False):
                    _rank_cols = ["county_name", dd_metric] + [
                        k for k in [
                            "opportunity_score", "risk_score",
                            "median_household_income", "homeownership_rate",
                            "poverty_rate", "old_housing_share", "no_vehicle_share",
                        ]
                        if k in _county_df.columns and k != dd_metric
                    ]
                    _rank_county = (
                        _county_df[_rank_cols]
                        .sort_values(dd_metric, ascending=False)
                        .copy()
                    )
                    _rank_county.index = range(1, len(_rank_county) + 1)
                    _rank_county = _rank_county.rename(columns={
                        "county_name": "County",
                        **{k: METRIC_META[k]["label"] for k in _rank_county.columns if k in METRIC_META},
                    })
                    st.dataframe(_rank_county, use_container_width=True, height=420)

            # -------------------------------------------------------------------
            # Phase 3 — Tract drill-down
            # -------------------------------------------------------------------
            st.markdown("---")

            _county_opts = (
                _county_df[["county", "county_name"]]
                .sort_values("county_name")
                .drop_duplicates("county")
            )
            _county_fips_map = dict(zip(_county_opts["county"], _county_opts["county_name"]))

            _tract_col1, _tract_col2 = st.columns([3, 1])
            with _tract_col1:
                dd_drill_county = st.selectbox(
                    "🔍 Drill down further — select a county to view tract map",
                    options=[""] + _county_opts["county"].tolist(),
                    format_func=lambda k: "— select a county —" if k == "" else _county_fips_map.get(k, k),
                    key="dd_drill_county",
                )

            if not dd_drill_county:
                st.caption("Select a county above to load its tract-level choropleth.")
            else:
                _sel_county_name = _county_fips_map.get(dd_drill_county, dd_drill_county)

                try:
                    _tract_df = load_live(dd_drill_state, dd_drill_county, dd_year)
                except Exception as _exc:
                    st.error(f"Could not load tract data for {_sel_county_name}: {_exc}")
                    _tract_df = None

                if _tract_df is not None:
                    try:
                        _tract_geojson = fetch_tract_geojson(dd_drill_state, dd_drill_county)
                    except Exception as _exc:
                        st.error(f"Could not load tract boundaries: {_exc}")
                        _tract_geojson = None

                    if _tract_geojson and _tract_df[dd_metric].notna().any():
                        _t_lo = float(_tract_df[dd_metric].quantile(0.02))
                        _t_hi = float(_tract_df[dd_metric].quantile(0.98))

                        _tract_lat, _tract_lon = _geojson_center(_tract_geojson)

                        _TRACT_HOVER: dict[str, str] = {
                            "opportunity_score": ":.3f",
                            "risk_score": ":.3f",
                            "homeowners_opportunity": ":.3f",
                            "homeowners_risk": ":.3f",
                            "median_household_income": ":$,.0f",
                            "homeownership_rate": ":.1%",
                            "poverty_rate": ":.1%",
                            "old_housing_share": ":.1%",
                            "no_vehicle_share": ":.1%",
                        }
                        _tract_hover = {
                            k: v for k, v in _TRACT_HOVER.items()
                            if k in _tract_df.columns and k != dd_metric
                        }
                        _tract_labels = {
                            k: METRIC_META.get(k, {}).get("label", k)
                            for k in list(_tract_hover) + [dd_metric]
                        }

                        fig_tract = px.choropleth_mapbox(
                            _tract_df,
                            geojson=_tract_geojson,
                            locations="GEOID",
                            featureidkey="properties.GEOID",
                            color=dd_metric,
                            color_continuous_scale=dd_meta["color"],
                            range_color=[_t_lo, _t_hi],
                            hover_name=_tract_df["NAME"].apply(short_name),
                            hover_data=_tract_hover,
                            labels=_tract_labels,
                            mapbox_style="carto-positron",
                            zoom=9,
                            center={"lat": _tract_lat, "lon": _tract_lon},
                            opacity=0.75,
                            height=560,
                        )
                        fig_tract.update_layout(margin={"r": 0, "t": 30, "l": 0, "b": 0})
                        st.plotly_chart(fig_tract, use_container_width=True)

                        # Compact stats row
                        _t = _tract_df[dd_metric].dropna()
                        _tc1, _tc2, _tc3, _tc4, _tc5, _tc6 = st.columns(6)
                        _tc1.caption("2nd–98th pct color range")
                        _tc2.metric("Tracts", len(_t))
                        _tc3.metric("Min", _fmt_metric(_t.min(), dd_metric))
                        _tc4.metric("Median", _fmt_metric(_t.median(), dd_metric))
                        _tc5.metric("Mean", _fmt_metric(_t.mean(), dd_metric))
                        _tc6.metric("Max", _fmt_metric(_t.max(), dd_metric))

                        with st.expander(f"📊 Tract Rankings — {_sel_county_name}", expanded=False):
                            _tract_rank_cols = ["NAME", dd_metric] + [
                                k for k in [
                                    "opportunity_score", "risk_score",
                                    "homeowners_opportunity", "homeowners_risk",
                                    "median_household_income", "homeownership_rate",
                                    "poverty_rate", "old_housing_share", "no_vehicle_share",
                                ]
                                if k in _tract_df.columns and k != dd_metric
                            ]
                            _tract_rank = (
                                _tract_df[_tract_rank_cols]
                                .sort_values(dd_metric, ascending=False)
                                .copy()
                            )
                            _tract_rank["NAME"] = _tract_rank["NAME"].apply(short_name)
                            _tract_rank.index = range(1, len(_tract_rank) + 1)
                            _tract_rank = _tract_rank.rename(columns={
                                "NAME": "Tract",
                                **{k: METRIC_META[k]["label"] for k in _tract_rank.columns if k in METRIC_META},
                            })
                            st.dataframe(_tract_rank, use_container_width=True, height=420)


# ===========================================================================
# TAB 1 — STATE OVERVIEW
# ===========================================================================
with tab_state:
    state_df: pd.DataFrame | None = st.session_state.get("state_df")

    if state_df is None:
        st.header("State Overview")
        st.info(
            "👈  Select a state and click **State Overview** to load county-level data "
            "for all counties in the state."
        )
    else:
        state_fips = st.session_state.get("state_fips", "")
        state_name = STATES.get(state_fips, state_fips)
        overview_year = st.session_state.get("overview_year", 2024)

        st.header(f"State Overview — {state_name} ({overview_year} ACS 5-yr)")
        st.markdown(
            f"All **{len(state_df)}** counties in {state_name}, scored using the same "
            "methodology as the tract-level views. Use this to identify which counties "
            "to drill into, then click **County Detail** in the sidebar."
        )

        # --- KPI cards ---
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Counties", len(state_df))
        c2.metric("Avg Opportunity", f"{state_df['opportunity_score'].mean():.2f}")
        c3.metric("Avg Risk", f"{state_df['risk_score'].mean():.2f}")
        c4.metric(
            "Median HH Income",
            f"${state_df['median_household_income'].median():,.0f}",
        )
        c5.metric(
            "Avg Homeownership",
            f"{state_df['homeownership_rate'].mean():.1%}",
        )

        st.markdown("---")

        # --- Sortable county table ---
        st.subheader("County Rankings")

        sort_col = st.selectbox(
            "Sort by",
            ["opportunity_score", "risk_score", "homeowners_opportunity",
             "homeowners_risk", "median_household_income", "median_home_value",
             "homeownership_rate", "poverty_rate"],
            format_func=lambda k: METRIC_META.get(k, {}).get("label", k),
            key="state_sort_col",
        )
        sort_asc = st.checkbox("Sort ascending", value=False, key="state_sort_asc")

        table_cols = [
            "county_name", "county", "opportunity_score", "risk_score",
            "homeowners_opportunity", "homeowners_risk",
            "median_household_income", "median_home_value",
            "homeownership_rate", "poverty_rate", "old_housing_share",
        ]
        available_table_cols = [c for c in table_cols if c in state_df.columns]
        display_df = (
            state_df[available_table_cols]
            .sort_values(sort_col, ascending=sort_asc)
            .reset_index(drop=True)
        )
        display_df.index += 1  # 1-based rank

        state_fmt = {
            "opportunity_score": "{:.3f}",
            "risk_score": "{:.3f}",
            "homeowners_opportunity": "{:.3f}",
            "homeowners_risk": "{:.3f}",
            "median_household_income": "${:,.0f}",
            "median_home_value": "${:,.0f}",
            "homeownership_rate": "{:.1%}",
            "poverty_rate": "{:.1%}",
            "old_housing_share": "{:.1%}",
        }
        st.dataframe(
            display_df.style.format(
                {k: v for k, v in state_fmt.items() if k in display_df.columns}
            ),
            use_container_width=True,
            height=480,
        )

        st.markdown("---")

        # --- Bar chart ---
        st.subheader("Top Counties by Score")

        bar_metric = st.selectbox(
            "Metric to chart",
            ["opportunity_score", "risk_score", "homeowners_opportunity",
             "homeowners_risk", "median_household_income", "homeownership_rate"],
            format_func=lambda k: METRIC_META.get(k, {}).get("label", k),
            key="state_bar_metric",
        )
        top_n_state = st.slider("Counties to show", 5, len(state_df), min(20, len(state_df)), key="state_top_n")

        bar_df = state_df.nlargest(top_n_state, bar_metric)[["county_name", bar_metric]].copy()
        is_dollar_metric = bar_metric in ("median_household_income", "median_home_value")

        fig_state_bar = px.bar(
            bar_df,
            x=bar_metric,
            y="county_name",
            orientation="h",
            title=f"Top {top_n_state} Counties — {METRIC_META.get(bar_metric, {}).get('label', bar_metric)}",
            labels={
                bar_metric: METRIC_META.get(bar_metric, {}).get("label", bar_metric),
                "county_name": "",
            },
            color_discrete_sequence=["#3498db"],
        )
        if is_dollar_metric:
            fig_state_bar.update_xaxes(tickformat="$,.0f")
        fig_state_bar.update_layout(height=max(400, top_n_state * 26), yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig_state_bar, use_container_width=True)

        st.markdown("---")

        # --- Drill-down selector ---
        st.subheader("Drill Down to Tract Level")
        st.markdown(
            "Select a county below, then click **County Detail** in the sidebar "
            "to load all census tracts for that county in the General Overview and "
            "Homeowners tabs."
        )

        county_options_state = sorted(
            state_df["county"].tolist(),
            key=lambda k: state_df.loc[state_df["county"] == k, "county_name"].iloc[0],
        )
        county_names_map = dict(zip(state_df["county"], state_df["county_name"]))

        drill_county = st.selectbox(
            "County to explore",
            options=county_options_state,
            format_func=lambda k: county_names_map.get(k, k),
            key="drill_county",
        )

        # Sync the sidebar county selector to the drilled county
        if drill_county and drill_county in st.session_state.get("_county_options", [county_options_state]):
            st.session_state["_drill_county"] = drill_county

        row = state_df[state_df["county"] == drill_county].iloc[0]
        drill_c1, drill_c2, drill_c3, drill_c4 = st.columns(4)
        drill_c1.metric("Opportunity Score", f"{row['opportunity_score']:.3f}")
        drill_c2.metric("Risk Score", f"{row['risk_score']:.3f}")
        drill_c3.metric("Homeowners Opportunity", f"{row['homeowners_opportunity']:.3f}")
        drill_c4.metric("Homeowners Risk", f"{row['homeowners_risk']:.3f}")

        st.info(
            f"**To load tract-level data for {county_names_map.get(drill_county, drill_county)}:** "
            f"Select this county in the sidebar County dropdown, then click **County Detail**."
        )


# Tabs below here require county-level tract data (df).
# If not yet loaded, stop rendering — tabs 2-4 will appear but be empty.
if df is None:
    st.stop()


# ===========================================================================
# TAB 2 — MAP VIEW  (choropleth of tract-level data)
# ===========================================================================
with tab_map:
    st.header("Geographic Map — Census Tract View")
    st.markdown(
        "Choropleth map of all census tracts in the selected county, colored by your "
        "chosen metric. Hover over any tract for details."
    )

    # Map tab requires live API data (state/county columns present in df)
    if "state" not in df.columns or "county" not in df.columns:
        st.info(
            "Map view requires live Census API data. Upload CSV mode does not include "
            "boundary information. Click **County Detail** in the sidebar to load data."
        )
    else:
        map_state = df["state"].iloc[0]
        map_county = df["county"].iloc[0]

        # --- Controls row ---
        ctrl_col1, ctrl_col2 = st.columns([2, 1])
        with ctrl_col1:
            map_metric = st.selectbox(
                "Color tracts by",
                list(METRIC_META.keys()),
                format_func=lambda k: METRIC_META[k]["label"],
                key="map_metric",
            )
        with ctrl_col2:
            map_opacity = st.slider("Opacity", 0.3, 1.0, 0.75, 0.05, key="map_opacity")

        meta = METRIC_META[map_metric]
        st.info(f"**{meta['label']}** — {meta['desc']}")

        # --- Fetch boundaries ---
        try:
            geojson = fetch_tract_geojson(map_state, map_county)
        except Exception as exc:
            st.error(f"Could not load tract boundaries from TIGER REST API: {exc}")
            geojson = None

        if geojson and not geojson["features"]:
            st.warning("No boundary data returned for this county from TIGER.")
            geojson = None

        if geojson:
            center_lat, center_lon = _geojson_center(geojson)

            # Build hover_data: fixed set of key metrics, hide the selected one
            _MAP_HOVER: dict[str, str] = {
                "opportunity_score": ":.3f",
                "risk_score": ":.3f",
                "homeowners_opportunity": ":.3f",
                "homeowners_risk": ":.3f",
                "median_household_income": ":$,.0f",
                "homeownership_rate": ":.1%",
                "poverty_rate": ":.1%",
                "old_housing_share": ":.1%",
            }
            hover_data = {
                k: v
                for k, v in _MAP_HOVER.items()
                if k in df.columns and k != map_metric
            }
            hover_labels = {
                k: METRIC_META.get(k, {}).get("label", k)
                for k in list(hover_data) + [map_metric]
            }

            # Clip color range to 2nd–98th percentile (reduces outlier distortion)
            c_lo = float(df[map_metric].quantile(0.02))
            c_hi = float(df[map_metric].quantile(0.98))

            fig_map = px.choropleth_mapbox(
                df,
                geojson=geojson,
                locations="GEOID",
                featureidkey="properties.GEOID",
                color=map_metric,
                color_continuous_scale=meta["color"],
                range_color=[c_lo, c_hi],
                hover_name=df["NAME"].apply(short_name),
                hover_data=hover_data,
                labels=hover_labels,
                mapbox_style="carto-positron",
                zoom=9,
                center={"lat": center_lat, "lon": center_lon},
                opacity=map_opacity,
                height=600,
            )
            fig_map.update_layout(margin={"r": 0, "t": 30, "l": 0, "b": 0})
            st.plotly_chart(fig_map, use_container_width=True)

            st.caption(
                "Color range clipped to 2nd–98th percentile to reduce outlier distortion. "
                "Boundary data: Census TIGER/Line (2020)."
            )

            st.markdown("---")

            # --- Summary stats for selected metric ---
            st.subheader(f"{meta['label']} — Summary")
            s = df[map_metric].dropna()
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Tracts", f"{len(s):,}")
            m2.metric("Min", _fmt_metric(s.min(), map_metric))
            m3.metric("Median", _fmt_metric(s.median(), map_metric))
            m4.metric("Mean", _fmt_metric(s.mean(), map_metric))
            m5.metric("Max", _fmt_metric(s.max(), map_metric))

            st.markdown("---")

            # --- Top / Bottom tracts for selected metric ---
            map_top_n = st.slider("Tracts to show in rankings", 5, 25, 10, key="map_top_n")
            col_top_map, col_bot_map = st.columns(2)

            with col_top_map:
                top_map = df.nlargest(map_top_n, map_metric)[["NAME", map_metric]].copy()
                top_map["NAME"] = top_map["NAME"].apply(short_name)
                fig_top_map = px.bar(
                    top_map, x=map_metric, y="NAME", orientation="h",
                    title=f"Top {map_top_n} Tracts",
                    labels={map_metric: meta["label"], "NAME": ""},
                    color_discrete_sequence=["#3498db"],
                )
                if map_metric in _DOLLAR_METRICS:
                    fig_top_map.update_xaxes(tickformat="$,.0f")
                elif map_metric in _PCT_METRICS:
                    fig_top_map.update_xaxes(tickformat=".0%")
                fig_top_map.update_layout(height=360, yaxis={"categoryorder": "total ascending"})
                st.plotly_chart(fig_top_map, use_container_width=True)

            with col_bot_map:
                bot_map = df.nsmallest(map_top_n, map_metric)[["NAME", map_metric]].copy()
                bot_map["NAME"] = bot_map["NAME"].apply(short_name)
                fig_bot_map = px.bar(
                    bot_map, x=map_metric, y="NAME", orientation="h",
                    title=f"Bottom {map_top_n} Tracts",
                    labels={map_metric: meta["label"], "NAME": ""},
                    color_discrete_sequence=["#e74c3c"],
                )
                if map_metric in _DOLLAR_METRICS:
                    fig_bot_map.update_xaxes(tickformat="$,.0f")
                elif map_metric in _PCT_METRICS:
                    fig_bot_map.update_xaxes(tickformat=".0%")
                fig_bot_map.update_layout(height=360, yaxis={"categoryorder": "total descending"})
                st.plotly_chart(fig_bot_map, use_container_width=True)


# ===========================================================================
# TAB 3 — GENERAL OVERVIEW  (tract level)
# ===========================================================================
with tab_general:
    st.header("General Insurance Market Overview")
    st.markdown(
        "This tab provides a broad view of the census tracts in the selected geography, "
        "scored on general **market opportunity** and **socioeconomic risk** — applicable "
        "across insurance lines."
    )

    # --- KPI cards ---
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Tracts", f"{len(df):,}")
    col2.metric(
        "Avg Opportunity Score",
        f"{df['opportunity_score'].mean():.2f}",
        help=METRIC_META['opportunity_score']['desc'],
    )
    col3.metric(
        "Avg Risk Score",
        f"{df['risk_score'].mean():.2f}",
        help=METRIC_META['risk_score']['desc'],
    )
    col4.metric(
        "Median Household Income",
        f"${df['median_household_income'].median():,.0f}",
        help=METRIC_META['median_household_income']['desc'],
    )

    st.markdown("---")

    # --- Quadrant scatter ---
    st.subheader("Opportunity vs Risk by Tract")

    with st.expander("ℹ️ How to read this chart"):
        st.markdown(
            """
The four quadrants tell you how to prioritize each tract:

| Quadrant | Opportunity | Risk | Action |
|---|---|---|---|
| **Grow Aggressively** | High | Low | Favorable market — focus sales and distribution here |
| **Write Carefully** | High | High | Opportunity exists but underwrite selectively; price for risk |
| **Deprioritize** | Low | Low | Limited market size — not worth active investment |
| **Avoid** | Low | High | High risk with low opportunity — unfavorable economics |

*Midpoint lines are drawn at the median of each score across all tracts.*
"""
        )

    opp_mid = df["opportunity_score"].median()
    risk_mid = df["risk_score"].median()
    df["quadrant"] = df.apply(
        lambda r: quadrant_label(r["opportunity_score"], r["risk_score"], opp_mid, risk_mid),
        axis=1,
    )

    fig_scatter = px.scatter(
        df,
        x="opportunity_score",
        y="risk_score",
        color="quadrant",
        color_discrete_map=QUADRANT_COLORS,
        hover_name=df["NAME"].apply(short_name),
        hover_data={
            "opportunity_score": ":.3f",
            "risk_score": ":.3f",
            "median_household_income": ":$,.0f",
            "homeownership_rate": ":.1%",
            "poverty_rate": ":.1%",
            "quadrant": False,
        },
        labels={
            "opportunity_score": "Opportunity Score",
            "risk_score": "Risk Score",
            "quadrant": "Strategy",
        },
        opacity=0.7,
        height=520,
    )
    fig_scatter.add_vline(x=opp_mid, line_dash="dash", line_color="grey", opacity=0.5)
    fig_scatter.add_hline(y=risk_mid, line_dash="dash", line_color="grey", opacity=0.5)
    fig_scatter.update_layout(legend_title_text="Strategy")
    st.plotly_chart(fig_scatter, use_container_width=True)

    # Quadrant summary
    q_counts = df["quadrant"].value_counts()
    cols = st.columns(4)
    for i, q in enumerate(["Grow Aggressively", "Write Carefully", "Deprioritize", "Avoid"]):
        count = q_counts.get(q, 0)
        pct = count / len(df) * 100
        cols[i].metric(q, f"{count} tracts ({pct:.0f}%)")

    st.markdown("---")

    # --- Score distributions ---
    st.subheader("Score Distributions")

    col_a, col_b = st.columns(2)

    with col_a:
        fig_opp = px.histogram(
            df, x="opportunity_score", nbins=40,
            title="Opportunity Score Distribution",
            labels={"opportunity_score": "Opportunity Score"},
            color_discrete_sequence=["#3498db"],
        )
        fig_opp.update_layout(showlegend=False, height=320)
        st.plotly_chart(fig_opp, use_container_width=True)

    with col_b:
        fig_risk = px.histogram(
            df, x="risk_score", nbins=40,
            title="Risk Score Distribution",
            labels={"risk_score": "Risk Score"},
            color_discrete_sequence=["#e74c3c"],
        )
        fig_risk.update_layout(showlegend=False, height=320)
        st.plotly_chart(fig_risk, use_container_width=True)

    st.markdown("---")

    # --- Top/bottom tracts ---
    st.subheader("Top and Bottom Tracts")

    top_n = st.slider("Number of tracts to show", 5, 30, 15)
    rank_by = st.selectbox(
        "Rank by",
        ["opportunity_score", "risk_score", "median_household_income", "homeownership_rate", "poverty_rate"],
        format_func=lambda k: METRIC_META.get(k, {}).get("label", k),
    )

    col_top, col_bot = st.columns(2)

    with col_top:
        top_df = df.nlargest(top_n, rank_by)[["NAME", rank_by]].copy()
        top_df["NAME"] = top_df["NAME"].apply(short_name)
        fig_top = px.bar(
            top_df, x=rank_by, y="NAME", orientation="h",
            title=f"Top {top_n} Tracts by {METRIC_META.get(rank_by, {}).get('label', rank_by)}",
            labels={rank_by: METRIC_META.get(rank_by, {}).get("label", rank_by), "NAME": ""},
            color_discrete_sequence=["#2ecc71"],
        )
        fig_top.update_layout(height=420, yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig_top, use_container_width=True)

    with col_bot:
        bot_df = df.nsmallest(top_n, rank_by)[["NAME", rank_by]].copy()
        bot_df["NAME"] = bot_df["NAME"].apply(short_name)
        fig_bot = px.bar(
            bot_df, x=rank_by, y="NAME", orientation="h",
            title=f"Bottom {top_n} Tracts by {METRIC_META.get(rank_by, {}).get('label', rank_by)}",
            labels={rank_by: METRIC_META.get(rank_by, {}).get("label", rank_by), "NAME": ""},
            color_discrete_sequence=["#e74c3c"],
        )
        fig_bot.update_layout(height=420, yaxis={"categoryorder": "total descending"})
        st.plotly_chart(fig_bot, use_container_width=True)

    st.markdown("---")

    # --- Metric explanations ---
    st.subheader("Metric Definitions")
    with st.expander("Show all metric definitions"):
        for key, meta in METRIC_META.items():
            st.markdown(f"**{meta['label']}** — {meta['desc']}")

    st.markdown("---")

    # --- Filterable data table ---
    st.subheader("Tract-Level Data Table")

    display_cols = [
        "NAME", "GEOID", "opportunity_score", "risk_score",
        "median_household_income", "homeownership_rate", "poverty_rate",
        "old_housing_share", "long_commute_share", "no_vehicle_share",
        "bachelors_plus_share", "population_total", "quadrant",
    ]
    available_cols = [c for c in display_cols if c in df.columns]

    with st.expander("Filter tracts", expanded=False):
        f1, f2 = st.columns(2)
        opp_min, opp_max = float(df["opportunity_score"].min()), float(df["opportunity_score"].max())
        risk_min, risk_max = float(df["risk_score"].min()), float(df["risk_score"].max())
        opp_range = f1.slider(
            "Opportunity Score range",
            opp_min, opp_max, (opp_min, opp_max), step=0.01,
        )
        risk_range = f2.slider(
            "Risk Score range",
            risk_min, risk_max, (risk_min, risk_max), step=0.01,
        )
        quadrant_filter = st.multiselect(
            "Strategy quadrant",
            options=["Grow Aggressively", "Write Carefully", "Deprioritize", "Avoid"],
            default=["Grow Aggressively", "Write Carefully", "Deprioritize", "Avoid"],
        )

    mask = (
        df["opportunity_score"].between(*opp_range)
        & df["risk_score"].between(*risk_range)
        & df["quadrant"].isin(quadrant_filter)
    )
    filtered = df.loc[mask, available_cols].copy()
    filtered["NAME"] = filtered["NAME"].apply(short_name)

    # Format for display
    fmt = {
        "opportunity_score": "{:.3f}",
        "risk_score": "{:.3f}",
        "median_household_income": "${:,.0f}",
        "homeownership_rate": "{:.1%}",
        "poverty_rate": "{:.1%}",
        "old_housing_share": "{:.1%}",
        "long_commute_share": "{:.1%}",
        "no_vehicle_share": "{:.1%}",
        "bachelors_plus_share": "{:.1%}",
        "population_total": "{:,.0f}",
    }
    st.write(f"Showing {len(filtered):,} of {len(df):,} tracts")
    st.dataframe(
        filtered.style.format({k: v for k, v in fmt.items() if k in filtered.columns}),
        use_container_width=True,
        height=400,
    )


# ===========================================================================
# TAB 4 — HOMEOWNERS / P&C
# ===========================================================================
with tab_homeowners:
    st.header("Homeowners / Property & Casualty Deep Dive")
    st.markdown(
        "This tab focuses on metrics most relevant to **homeowners and property insurance**. "
        "Two specialized scores — Homeowners Opportunity and Property Risk — are built from "
        "variables that directly drive insurance market size, claims frequency, and policy behavior."
    )

    # --- Score methodology expander ---
    with st.expander("📐 Score Methodology — How are these scores built?", expanded=True):
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.markdown("#### Homeowners Opportunity Score")
            st.markdown(
                """
Measures how attractive a tract is for writing homeowners policies. Range 0–1, higher = better market.

| Component | Weight | Why it matters |
|---|---|---|
| Homeownership rate | 25% | Directly determines the size of the insurable market — only homeowners buy homeowners insurance |
| Median home value | 25% | Sets coverage amount and drives premium volume — higher value homes = larger policies |
| Inverse poverty rate | 25% | Low poverty correlates with policy retention, timely premium payment, and lower moral hazard |
| Bachelor's+ share | 25% | Education level correlates with insurance literacy, retention, and favorable claims behavior |

*All components are min-max normalized to 0–1 across tracts before averaging.*
"""
            )
        with col_m2:
            st.markdown("#### Property Risk Score")
            st.markdown(
                """
Measures structural and economic risk factors that drive homeowners claim frequency and severity. Range 0–1, higher = more risk.

| Component | Weight | Why it matters |
|---|---|---|
| Old housing share | 25% | Pre-1990 structures have aging electrical, plumbing, HVAC, and roofing — leading drivers of water damage, fire, and weather claims |
| Inverse income | 25% | Lower income households defer maintenance, increasing the likelihood of preventable claims |
| Poverty rate | 25% | High poverty areas see more underinsurance, policy lapses, and claims disputes |
| No-vehicle share | 25% | Proxy for economic vulnerability and concentrated distress — correlates with difficult claim environments |

*All components are min-max normalized to 0–1 across tracts before averaging.*
"""
            )

    st.markdown("---")

    # --- KPIs ---
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Tracts", f"{len(df):,}")
    col2.metric(
        "Avg Homeowners Opportunity",
        f"{df['homeowners_opportunity'].mean():.2f}",
        help=METRIC_META['homeowners_opportunity']['desc'],
    )
    col3.metric(
        "Avg Property Risk",
        f"{df['homeowners_risk'].mean():.2f}",
        help=METRIC_META['homeowners_risk']['desc'],
    )
    col4.metric(
        "Median Home Value",
        f"${df['median_home_value'].median():,.0f}",
        help=METRIC_META['median_home_value']['desc'],
    )
    col5.metric(
        "Avg Homeownership Rate",
        f"{df['homeownership_rate'].mean():.1%}",
        help=METRIC_META['homeownership_rate']['desc'],
    )

    st.markdown("---")

    # --- Homeowners quadrant scatter ---
    st.subheader("Homeowners Opportunity vs Property Risk")

    with st.expander("ℹ️ How to read this chart"):
        st.markdown(
            """
Same quadrant logic as the general tab, but scored specifically for homeowners insurance:

| Quadrant | Interpretation |
|---|---|
| **Grow Aggressively** | High homeownership, high home values, low poverty, low old-housing — ideal market |
| **Write Carefully** | Strong market but elevated property risk — apply stricter underwriting, consider inspection requirements |
| **Deprioritize** | Low homeownership / low home values — limited premium volume opportunity |
| **Avoid** | High risk with limited market — adverse selection concentration |
"""
        )

    ho_mid = df["homeowners_opportunity"].median()
    hr_mid = df["homeowners_risk"].median()
    df["hw_quadrant"] = df.apply(
        lambda r: quadrant_label(r["homeowners_opportunity"], r["homeowners_risk"], ho_mid, hr_mid),
        axis=1,
    )

    fig_hw = px.scatter(
        df,
        x="homeowners_opportunity",
        y="homeowners_risk",
        color="hw_quadrant",
        color_discrete_map=QUADRANT_COLORS,
        hover_name=df["NAME"].apply(short_name),
        hover_data={
            "homeowners_opportunity": ":.3f",
            "homeowners_risk": ":.3f",
            "median_home_value": ":$,.0f",
            "homeownership_rate": ":.1%",
            "old_housing_share": ":.1%",
            "poverty_rate": ":.1%",
            "hw_quadrant": False,
        },
        labels={
            "homeowners_opportunity": "Homeowners Opportunity Score",
            "homeowners_risk": "Property Risk Score",
            "hw_quadrant": "Strategy",
        },
        opacity=0.7,
        height=520,
    )
    fig_hw.add_vline(x=ho_mid, line_dash="dash", line_color="grey", opacity=0.5)
    fig_hw.add_hline(y=hr_mid, line_dash="dash", line_color="grey", opacity=0.5)
    st.plotly_chart(fig_hw, use_container_width=True)

    q_counts_hw = df["hw_quadrant"].value_counts()
    cols_hw = st.columns(4)
    for i, q in enumerate(["Grow Aggressively", "Write Carefully", "Deprioritize", "Avoid"]):
        count = q_counts_hw.get(q, 0)
        pct = count / len(df) * 100
        cols_hw[i].metric(q, f"{count} tracts ({pct:.0f}%)")

    st.markdown("---")

    # --- Component breakdown ---
    st.subheader("Component Metric Distributions")
    st.markdown("Explore the raw distributions of each metric that feeds into the homeowners scores.")

    hw_metrics = [
        "homeownership_rate", "median_home_value", "poverty_rate",
        "bachelors_plus_share", "old_housing_share", "median_household_income", "no_vehicle_share",
    ]
    available_hw = [m for m in hw_metrics if m in df.columns]

    metric_choice = st.selectbox(
        "Select metric to explore",
        available_hw,
        format_func=lambda k: METRIC_META.get(k, {}).get("label", k),
    )

    meta = METRIC_META[metric_choice]
    st.info(f"**{meta['label']}** — {meta['desc']}")

    col_dist, col_box = st.columns([2, 1])

    with col_dist:
        is_dollar = metric_choice in ("median_household_income", "median_home_value")
        is_pct = not is_dollar

        fig_hist = px.histogram(
            df, x=metric_choice, nbins=50,
            title=f"Distribution: {meta['label']}",
            labels={metric_choice: meta["label"]},
            color_discrete_sequence=[px.colors.sequential.Blues[4]],
        )
        if is_pct:
            fig_hist.update_xaxes(tickformat=".0%")
        else:
            fig_hist.update_xaxes(tickformat="$,.0f")
        fig_hist.update_layout(showlegend=False, height=360)
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_box:
        stats = df[metric_choice].describe()
        fmt_fn = (lambda v: f"${v:,.0f}") if is_dollar else (lambda v: f"{v:.1%}")
        st.markdown("**Summary Statistics**")
        for label, key in [("Min", "min"), ("25th pct", "25%"), ("Median", "50%"), ("75th pct", "75%"), ("Max", "max"), ("Mean", "mean")]:
            st.markdown(f"- **{label}:** {fmt_fn(stats[key])}")

    st.markdown("---")

    # --- Scatter: home value vs old housing ---
    st.subheader("Home Value vs Old Housing Share")
    st.markdown(
        "A key underwriting tension: high home value is attractive for premium volume, but "
        "old housing increases expected claims. Tracts in the upper-left are ideal for homeowners — "
        "valuable properties with newer construction."
    )

    fig_val_age = px.scatter(
        df,
        x="old_housing_share",
        y="median_home_value",
        color="homeownership_rate",
        color_continuous_scale="Teal",
        hover_name=df["NAME"].apply(short_name),
        hover_data={
            "old_housing_share": ":.1%",
            "median_home_value": ":$,.0f",
            "homeownership_rate": ":.1%",
            "poverty_rate": ":.1%",
        },
        labels={
            "old_housing_share": "Old Housing Share (pre-1990)",
            "median_home_value": "Median Home Value",
            "homeownership_rate": "Homeownership Rate",
        },
        opacity=0.7,
        height=480,
    )
    fig_val_age.update_yaxes(tickformat="$,.0f")
    fig_val_age.update_xaxes(tickformat=".0%")
    st.plotly_chart(fig_val_age, use_container_width=True)

    st.markdown("---")

    # --- Top tracts table ---
    st.subheader("Top Homeowners Opportunity Tracts")

    top_hw_n = st.slider("Show top N tracts", 5, 50, 20, key="hw_top_n")
    sort_hw = st.selectbox(
        "Sort by",
        ["homeowners_opportunity", "homeowners_risk", "median_home_value", "homeownership_rate"],
        format_func=lambda k: METRIC_META.get(k, {}).get("label", k),
        key="hw_sort",
    )

    hw_table_cols = [
        "NAME", "GEOID", "homeowners_opportunity", "homeowners_risk",
        "homeownership_rate", "median_home_value", "old_housing_share",
        "poverty_rate", "bachelors_plus_share", "hw_quadrant",
    ]
    available_hw_cols = [c for c in hw_table_cols if c in df.columns]
    top_hw = df.nlargest(top_hw_n, sort_hw)[available_hw_cols].copy()
    top_hw["NAME"] = top_hw["NAME"].apply(short_name)

    hw_fmt = {
        "homeowners_opportunity": "{:.3f}",
        "homeowners_risk": "{:.3f}",
        "homeownership_rate": "{:.1%}",
        "median_home_value": "${:,.0f}",
        "old_housing_share": "{:.1%}",
        "poverty_rate": "{:.1%}",
        "bachelors_plus_share": "{:.1%}",
    }
    st.dataframe(
        top_hw.style.format({k: v for k, v in hw_fmt.items() if k in top_hw.columns}),
        use_container_width=True,
        height=420,
    )

# ===========================================================================
# TAB 5 — CLIENT PORTFOLIO
# ===========================================================================
with tab_portfolio:
    st.header("Client Portfolio Analysis")
    st.markdown(
        "Upload a CSV of customer or prospect ZIP codes to map your portfolio "
        "against market opportunity and identify coverage gaps."
    )

    port_file = st.file_uploader(
        "Upload portfolio CSV (must contain a 5-digit ZIP code column)",
        type=["csv"],
        key="port_file",
    )

    if port_file is None:
        st.info(
            "Upload a CSV to get started. Your file should have a column of 5-digit ZIP codes, "
            "and optionally a numeric volume column (# policies, customers, premiums, etc.)."
        )
        st.markdown("**Example format:**")
        st.dataframe(
            pd.DataFrame({"zip_code": ["60601", "60602", "60614"], "policies": [45, 32, 18]}),
            use_container_width=False,
            hide_index=True,
        )
    else:
        port_raw = pd.read_csv(port_file, dtype=str)
        st.success(f"Loaded **{len(port_raw):,} rows** × {len(port_raw.columns)} columns")

        # --- Column selectors ---
        _port_cols = list(port_raw.columns)
        _zip_candidates = [
            c for c in _port_cols
            if port_raw[c].astype(str).str.match(r"^\s*\d{5}\s*$").mean() > 0.7
        ]
        _default_zip_idx = _port_cols.index(_zip_candidates[0]) if _zip_candidates else 0

        _pc1, _pc2, _pc3 = st.columns(3)
        with _pc1:
            port_zip_col = st.selectbox(
                "ZIP code column",
                options=_port_cols,
                index=_default_zip_idx,
                key="port_zip_col",
            )
        with _pc2:
            _vol_opts = ["— count rows (each row = 1) —"] + [c for c in _port_cols if c != port_zip_col]
            port_vol_col = st.selectbox(
                "Volume column (optional)",
                options=_vol_opts,
                key="port_vol_col",
            )
        with _pc3:
            port_year = st.selectbox("ACS Year", [2024, 2023, 2022, 2021, 2020], key="port_year")

        # --- Aggregate ZIP → volume ---
        _port_zips = port_raw[port_zip_col].astype(str).str.strip().str.zfill(5)
        if port_vol_col == _vol_opts[0]:
            _port_vols = pd.Series(1.0, index=port_raw.index)
        else:
            _port_vols = pd.to_numeric(port_raw[port_vol_col], errors="coerce").fillna(1.0)

        _zip_vol_agg = (
            pd.DataFrame({"zip": _port_zips, "volume": _port_vols})
            .groupby("zip")["volume"].sum()
            .reset_index()
        )

        # --- ZIP → county FIPS crosswalk ---
        try:
            _crosswalk = load_zip_crosswalk()
        except Exception as _xw_err:
            st.error(f"Failed to load ZIP crosswalk: {_xw_err}")
            st.stop()

        _zip_county = _zip_vol_agg.merge(_crosswalk, on="zip", how="left")
        _n_unmatched = int(_zip_county["county_fips"].isna().sum())
        _n_matched = int(_zip_county["county_fips"].notna().sum())

        _county_vol = (
            _zip_county.dropna(subset=["county_fips"])
            .groupby("county_fips")["volume"].sum()
            .reset_index()
        )
        _county_vol["state_fips"] = _county_vol["county_fips"].str[:2]

        if _n_unmatched > 0:
            st.caption(
                f"⚠ {_n_matched:,} ZIPs matched · {_n_unmatched:,} unmatched "
                "(invalid ZIPs, PO box-only ZCTAs, or territories outside the 50 states)"
            )

        # --- Load ACS county data for all represented states ---
        _port_states = sorted(_county_vol["state_fips"].unique().tolist())
        _acs_frames = []
        with st.spinner(f"Loading ACS county data for {len(_port_states)} state(s)…"):
            for _sfips in _port_states:
                try:
                    _acs_frames.append(load_state_overview(_sfips, port_year))
                except Exception:
                    pass

        if not _acs_frames:
            st.error("Could not load ACS data for any state in the portfolio. Check ZIP column selection.")
            st.stop()

        _acs_all = pd.concat(_acs_frames, ignore_index=True)

        # Merge: ACS counties (left) ← portfolio volume
        _portfolio = _acs_all.merge(
            _county_vol[["county_fips", "volume"]].rename(columns={"county_fips": "GEOID"}),
            on="GEOID",
            how="left",
        )
        _portfolio["volume"] = _portfolio["volume"].fillna(0.0)
        _port_covered = _portfolio[_portfolio["volume"] > 0].copy()

        # --- KPI CARDS ---
        st.markdown("---")
        _k1, _k2, _k3, _k4, _k5 = st.columns(5)
        _k1.metric("ZIPs uploaded", f"{len(_zip_vol_agg):,}")
        _k2.metric("Counties covered", f"{(_portfolio['volume'] > 0).sum():,}")
        _k3.metric("States covered", f"{len(_port_states):,}")
        _total_vol = _port_covered["volume"].sum()
        _high_opp_vol = _port_covered.loc[_port_covered["opportunity_score"] > 0.5, "volume"].sum()
        _k4.metric(
            "% Vol in High-Opp Counties",
            f"{_high_opp_vol / _total_vol:.1%}" if _total_vol > 0 else "—",
        )
        _wavg_opp = (
            (_port_covered["volume"] * _port_covered["opportunity_score"]).sum() / _total_vol
            if _total_vol > 0 else float("nan")
        )
        _k5.metric(
            "Wtd Avg Opp Score",
            f"{_wavg_opp:.3f}" if not pd.isna(_wavg_opp) else "—",
        )

        # --- COVERAGE MAP ---
        st.markdown("---")
        st.subheader("Portfolio Coverage Map")
        st.caption(
            "Counties shaded by total portfolio volume. "
            "Only counties represented in the uploaded states are shown."
        )
        try:
            _full_geojson = load_full_counties_geojson()
            _map_port_df = _port_covered.copy()
            if not _map_port_df.empty:
                fig_port = px.choropleth_mapbox(
                    _map_port_df,
                    geojson=_full_geojson,
                    locations="GEOID",
                    featureidkey="id",
                    color="volume",
                    color_continuous_scale="Blues",
                    mapbox_style="carto-positron",
                    zoom=3,
                    center={"lat": 39.0, "lon": -96.0},
                    opacity=0.8,
                    height=520,
                    labels={"volume": "Volume"},
                    hover_data={
                        "county_name": True,
                        "opportunity_score": ":.3f",
                        "risk_score": ":.3f",
                        "volume": ":,.0f",
                    },
                )
                fig_port.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
                st.plotly_chart(fig_port, use_container_width=True)
            else:
                st.warning("No counties matched — check your ZIP column selection.")
        except Exception as _map_err:
            st.error(f"Map error: {_map_err}")

        # --- OPPORTUNITY GAP TABLE ---
        st.markdown("---")
        st.subheader("Opportunity Gap Analysis")
        st.caption(
            "Counties in your states with **high opportunity but low/no portfolio coverage**. "
            "Gap Score = Opportunity Score × (1 − Coverage Index). "
            "Coverage Index = county volume ÷ max county volume."
        )

        _max_vol = _portfolio["volume"].max()
        _portfolio["coverage_index"] = (
            _portfolio["volume"] / _max_vol if _max_vol > 0 else 0.0
        )
        _portfolio["gap_score"] = (
            _portfolio["opportunity_score"] * (1.0 - _portfolio["coverage_index"])
        )

        _gap_cols = [
            "county_name", "GEOID", "volume", "gap_score",
            "opportunity_score", "risk_score",
            "median_household_income", "homeownership_rate", "poverty_rate",
        ]
        _avail_gap = [c for c in _gap_cols if c in _portfolio.columns]
        _gap_table = (
            _portfolio[_avail_gap]
            .sort_values("gap_score", ascending=False)
            .head(50)
            .copy()
        )
        _gap_table.index = range(1, len(_gap_table) + 1)

        _gap_fmt = {
            "volume": "{:,.0f}",
            "gap_score": "{:.3f}",
            "opportunity_score": "{:.3f}",
            "risk_score": "{:.3f}",
            "median_household_income": "${:,.0f}",
            "homeownership_rate": "{:.1%}",
            "poverty_rate": "{:.1%}",
        }
        st.dataframe(
            _gap_table.style.format({k: v for k, v in _gap_fmt.items() if k in _gap_table.columns}),
            use_container_width=True,
            height=480,
        )

        st.download_button(
            "⬇ Download Gap Analysis (CSV)",
            data=_gap_table.to_csv(index=True).encode(),
            file_name="portfolio_gap_analysis.csv",
            mime="text/csv",
        )
