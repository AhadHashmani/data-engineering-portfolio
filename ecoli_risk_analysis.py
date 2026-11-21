"""
E. coli Exposure Risk Analysis
==============================
Scores zip-code zones from synthetic environmental records.
Computes a composite risk score, flags high-exposure zones,
prints a report, and writes risk_scores.csv.

Author: Ahad Hashmani
"""

import logging
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import zscore

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)


# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
@dataclass
class AnalysisConfig:
    high_risk_threshold: float = 1.5    # z-score above which a zone is high-risk
    min_samples_per_zone: int = 10      # zones with fewer samples are excluded
    output_csv: str = "risk_scores.csv"
    chunk_size: int = 50_000


# ─────────────────────────────────────────────
# Synthetic Data Generator (mirrors real pipeline)
# ─────────────────────────────────────────────
def generate_synthetic_data(n: int = 200_000, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic E. coli exposure records that mirror the
    structure of the Austin environmental dataset.
    """
    rng = np.random.default_rng(seed)
    zip_codes = [f"787{str(i).zfill(2)}" for i in range(10, 55)]

    # Simulate spatial heterogeneity: some zip codes have higher baseline risk
    zip_risk_bias = {z: rng.uniform(0, 2) for z in zip_codes}

    records = []
    for _ in range(n):
        z = rng.choice(zip_codes)
        bias = zip_risk_bias[z]
        records.append({
            "zip_code": z,
            "sample_date": pd.Timestamp("2022-01-01") + pd.Timedelta(days=int(rng.integers(0, 730))),
            "ecoli_cfu_per_100ml": max(0, rng.lognormal(mean=bias, sigma=0.8)),
            "water_temp_c": rng.uniform(10, 32),
            "turbidity_ntu": rng.exponential(scale=3 + bias),
            "rainfall_mm_24h": rng.exponential(scale=5),
            "site_type": rng.choice(["creek", "lake", "stormwater", "groundwater"],
                                     p=[0.4, 0.3, 0.2, 0.1]),
        })

    return pd.DataFrame(records)


# ─────────────────────────────────────────────
# Feature Engineering
# ─────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived columns used in risk scoring."""
    df = df.copy()

    df["log_ecoli"] = np.log1p(df["ecoli_cfu_per_100ml"])
    df["log_turbidity"] = np.log1p(df["turbidity_ntu"])

    # EPA threshold exceedance flag (235 CFU/100mL for recreational water)
    df["epa_exceedance"] = (df["ecoli_cfu_per_100ml"] > 235).astype(int)

    # Rainfall interaction: high rain + high E. coli = elevated concern
    df["rain_ecoli_interaction"] = df["rainfall_mm_24h"] * df["log_ecoli"]

    df["month"] = df["sample_date"].dt.month
    df["season"] = df["month"].map({
        12: "winter", 1: "winter", 2: "winter",
        3: "spring", 4: "spring", 5: "spring",
        6: "summer", 7: "summer", 8: "summer",
        9: "fall",   10: "fall",  11: "fall",
    })

    return df


# ─────────────────────────────────────────────
# Risk Scoring
# ─────────────────────────────────────────────
def compute_zone_risk(df: pd.DataFrame, config: AnalysisConfig) -> pd.DataFrame:
    """
    Aggregate sample-level measurements to zip-code risk scores.

    Risk score is a weighted composite of:
      - Mean log E. coli concentration      (weight 0.45)
      - EPA exceedance rate                  (weight 0.30)
      - Mean turbidity                       (weight 0.15)
      - Rainfall-E. coli interaction mean   (weight 0.10)
    """
    agg = (
        df.groupby("zip_code")
        .agg(
            n_samples=("ecoli_cfu_per_100ml", "count"),
            mean_log_ecoli=("log_ecoli", "mean"),
            epa_exceedance_rate=("epa_exceedance", "mean"),
            mean_turbidity=("log_turbidity", "mean"),
            mean_rain_interaction=("rain_ecoli_interaction", "mean"),
            median_ecoli=("ecoli_cfu_per_100ml", "median"),
            p95_ecoli=("ecoli_cfu_per_100ml", lambda x: np.percentile(x, 95)),
        )
        .reset_index()
    )

    # Exclude zones with too few samples
    agg = agg[agg["n_samples"] >= config.min_samples_per_zone].copy()

    # Normalise each component to [0, 1]
    components = ["mean_log_ecoli", "epa_exceedance_rate", "mean_turbidity", "mean_rain_interaction"]
    weights = [0.45, 0.30, 0.15, 0.10]

    for col in components:
        col_min, col_max = agg[col].min(), agg[col].max()
        span = col_max - col_min
        agg[f"{col}_norm"] = (agg[col] - col_min) / span if span > 0 else 0.0

    agg["composite_risk_score"] = sum(
        w * agg[f"{c}_norm"] for w, c in zip(weights, components)
    )

    # Z-score to identify statistically high-risk zones
    agg["risk_zscore"] = zscore(agg["composite_risk_score"])
    agg["high_risk"] = agg["risk_zscore"] > config.high_risk_threshold

    return agg.sort_values("composite_risk_score", ascending=False)


# ─────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────
def print_report(risk_df: pd.DataFrame, config: AnalysisConfig) -> None:
    high = risk_df[risk_df["high_risk"]]
    total = len(risk_df)
    pct_high = len(high) / total * 100

    print("\n" + "=" * 60)
    print("  E. COLI EXPOSURE RISK REPORT — SYNTHETIC DATA")
    print("=" * 60)
    print(f"  Zones analysed          : {total}")
    print(f"  High-risk zones (z>{config.high_risk_threshold}) : {len(high)} ({pct_high:.1f}%)")
    print(f"  EPA exceedance (top zone): {risk_df.iloc[0]['epa_exceedance_rate']:.1%}")
    print()
    print("  Top 10 Highest-Risk Zip Codes:")
    print(f"  {'Zip':>8} | {'Risk Score':>10} | {'Z-score':>8} | {'EPA Exceed%':>11} | {'N':>6}")
    print(f"  {'-'*8}-+-{'-'*10}-+-{'-'*8}-+-{'-'*11}-+-{'-'*6}")
    for _, row in risk_df.head(10).iterrows():
        flag = " ***" if row["high_risk"] else ""
        print(
            f"  {row['zip_code']:>8} | {row['composite_risk_score']:>10.4f} | "
            f"{row['risk_zscore']:>8.2f} | {row['epa_exceedance_rate']:>10.1%} | "
            f"{int(row['n_samples']):>6}{flag}"
        )
    print("\n  *** = statistically high-risk zone")
    print("=" * 60)


# ─────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────
def main() -> None:
    config = AnalysisConfig()

    log.info("Generating synthetic dataset...")
    raw = generate_synthetic_data(n=200_000)
    log.info("Dataset: %d rows, %d columns", *raw.shape)

    log.info("Engineering features...")
    featured = engineer_features(raw)

    log.info("Computing zone-level risk scores...")
    risk_scores = compute_zone_risk(featured, config)

    print_report(risk_scores, config)

    risk_scores.to_csv(config.output_csv, index=False)
    log.info("Risk scores saved to %s", config.output_csv)


if __name__ == "__main__":
    main()
