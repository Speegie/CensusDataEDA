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
from census_insurance.fetcher import fetch_state_overview, fetch_tracts

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

@st.cache_data(show_spinner="Fetching tract data from Census API…")
def load_live(state: str, county: str, year: int) -> pd.DataFrame:
    return fetch_tracts(state, county, year)


@st.cache_data(show_spinner="Fetching county overview from Census API…")
def load_state_overview(state: str, year: int) -> pd.DataFrame:
    return fetch_state_overview(state, year)


def load_csv(uploaded) -> pd.DataFrame:
    return pd.read_csv(uploaded)

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

if df is None:
    st.title("Census Insurance EDA Dashboard")
    st.info(
        "👈  Select a data source in the sidebar and click **Fetch Data** (or upload a CSV) to get started.\n\n"
        "**Default demo:** Cook County, IL — 1,300+ census tracts, 2024 ACS 5-year estimates."
    )
    st.stop()

source_label = st.session_state.get("source_label", "")
st.title(f"Census Insurance EDA — {source_label}")

# Export button
csv_bytes = df.to_csv(index=False).encode()
st.sidebar.download_button(
    "⬇ Download current data (CSV)",
    data=csv_bytes,
    file_name="census_insurance_tracts.csv",
    mime="text/csv",
)

tab_state, tab_general, tab_homeowners = st.tabs(
    ["🗺️ State Overview", "📊 General Overview", "🏠 Homeowners / P&C"]
)

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


# ===========================================================================
# TAB 2 — GENERAL OVERVIEW  (tract level)
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
# TAB 2 — HOMEOWNERS / P&C
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
