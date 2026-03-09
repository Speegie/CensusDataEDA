#!/usr/bin/env python3
"""Create quick EDA plots from the local ACS insurance starter CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="acs_tract_insurance_starter.csv",
        help="Path to CSV created by fetch_acs_option_a.py",
    )
    parser.add_argument(
        "--output-dir",
        default="plots",
        help="Directory to write plot PNG files",
    )
    parser.add_argument("--top-n", type=int, default=15, help="Top N tracts for bar charts")
    return parser.parse_args()


def clean_label(name: str) -> str:
    return name.split(";")[0].strip()


def main() -> None:
    args = parse_args()
    in_path = Path(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    df = pd.read_csv(in_path)

    required = {"GEOID", "NAME", "opportunity_score", "risk_score"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(f"Missing required columns for plotting: {missing}")

    sns.set_theme(style="whitegrid")

    # Distribution plot: opportunity vs risk
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.kdeplot(df["opportunity_score"].dropna(), fill=True, label="Opportunity", ax=ax)
    sns.kdeplot(df["risk_score"].dropna(), fill=True, label="Risk", ax=ax)
    ax.set_title("Distribution of Opportunity and Risk Scores")
    ax.set_xlabel("Score")
    ax.set_ylabel("Density")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "score_distributions.png", dpi=150)
    plt.close(fig)

    # Scatter plot: opportunity vs risk
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.scatterplot(
        data=df,
        x="opportunity_score",
        y="risk_score",
        alpha=0.6,
        s=40,
        edgecolor=None,
        ax=ax,
    )
    ax.set_title("Opportunity vs Risk by Tract")
    ax.set_xlabel("Opportunity Score")
    ax.set_ylabel("Risk Score")
    fig.tight_layout()
    fig.savefig(out_dir / "opportunity_vs_risk_scatter.png", dpi=150)
    plt.close(fig)

    # Top opportunity tracts
    top_opp = (
        df.sort_values("opportunity_score", ascending=False)
        .head(args.top_n)
        .copy()
    )
    top_opp["label"] = top_opp["NAME"].map(clean_label)

    fig, ax = plt.subplots(figsize=(11, 7))
    sns.barplot(data=top_opp, x="opportunity_score", y="label", palette="Blues_r", ax=ax)
    ax.set_title(f"Top {args.top_n} Tracts by Opportunity Score")
    ax.set_xlabel("Opportunity Score")
    ax.set_ylabel("Tract")
    fig.tight_layout()
    fig.savefig(out_dir / "top_opportunity_tracts.png", dpi=150)
    plt.close(fig)

    # Top risk tracts
    top_risk = (
        df.sort_values("risk_score", ascending=False)
        .head(args.top_n)
        .copy()
    )
    top_risk["label"] = top_risk["NAME"].map(clean_label)

    fig, ax = plt.subplots(figsize=(11, 7))
    sns.barplot(data=top_risk, x="risk_score", y="label", palette="Reds_r", ax=ax)
    ax.set_title(f"Top {args.top_n} Tracts by Risk Score")
    ax.set_xlabel("Risk Score")
    ax.set_ylabel("Tract")
    fig.tight_layout()
    fig.savefig(out_dir / "top_risk_tracts.png", dpi=150)
    plt.close(fig)

    print(f"Wrote plots to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
