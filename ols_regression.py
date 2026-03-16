"""
Econometric Analysis: Housing Price Determinants
=================================================
OLS regression with heteroskedasticity-robust standard errors,
multicollinearity diagnostics, and residual analysis.

Author: Ahad Hashmani
"""

import numpy as np
import pandas as pd
from scipy import stats
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


# ─────────────────────────────────────────────
# Synthetic Data
# ─────────────────────────────────────────────
def generate_housing_data(n: int = 2000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    sqft = rng.normal(1800, 400, n).clip(600, 5000)
    bedrooms = rng.choice([2, 3, 4, 5], n, p=[0.2, 0.45, 0.25, 0.10])
    age = rng.uniform(0, 50, n)
    dist_downtown = rng.exponential(10, n).clip(0.5, 40)
    school_rating = rng.uniform(3, 10, n)

    noise = rng.normal(0, 20_000, n)
    price = (
        80_000
        + 95 * sqft
        + 12_000 * bedrooms
        - 800 * age
        - 4_500 * dist_downtown
        + 8_000 * school_rating
        + noise
    ).clip(50_000)

    return pd.DataFrame({
        "price": price,
        "sqft": sqft,
        "bedrooms": bedrooms,
        "age_years": age,
        "dist_downtown_mi": dist_downtown,
        "school_rating": school_rating,
    })


# ─────────────────────────────────────────────
# OLS Implementation
# ─────────────────────────────────────────────
@dataclass
class OLSResult:
    coefficients: np.ndarray
    feature_names: List[str]
    r_squared: float
    adj_r_squared: float
    robust_se: np.ndarray
    t_stats: np.ndarray
    p_values: np.ndarray
    n: int
    k: int
    residuals: np.ndarray
    vif: Dict[str, float] = field(default_factory=dict)


def ols_fit(X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> OLSResult:
    """
    OLS with HC3 heteroskedasticity-robust standard errors.
    """
    n, k = X.shape

    # Coefficients: beta = (X'X)^-1 X'y
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    y_hat = X @ beta
    residuals = y - y_hat

    # R-squared
    ss_res = residuals @ residuals
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - k)

    # HC3 robust standard errors
    h = np.einsum("ij,jk,ik->i", X, XtX_inv, X)   # leverage scores
    e_sq = (residuals / (1 - h)) ** 2
    meat = (X * e_sq[:, None]).T @ X
    cov_robust = XtX_inv @ meat @ XtX_inv
    robust_se = np.sqrt(np.diag(cov_robust))

    t_stats = beta / robust_se
    p_values = 2 * stats.t.sf(np.abs(t_stats), df=n - k)

    return OLSResult(
        coefficients=beta,
        feature_names=feature_names,
        r_squared=r2,
        adj_r_squared=adj_r2,
        robust_se=robust_se,
        t_stats=t_stats,
        p_values=p_values,
        n=n,
        k=k,
        residuals=residuals,
    )


def compute_vif(X_df: pd.DataFrame) -> Dict[str, float]:
    """Variance Inflation Factors for multicollinearity diagnostics."""
    vif = {}
    cols = X_df.columns.tolist()
    for i, col in enumerate(cols):
        others = [c for c in cols if c != col]
        X_other = X_df[others].values
        X_other = np.column_stack([np.ones(len(X_other)), X_other])
        y_col = X_df[col].values
        beta = np.linalg.lstsq(X_other, y_col, rcond=None)[0]
        y_hat = X_other @ beta
        ss_res = ((y_col - y_hat) ** 2).sum()
        ss_tot = ((y_col - y_col.mean()) ** 2).sum()
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        vif[col] = 1 / (1 - r2) if r2 < 1 else np.inf
    return vif


def print_results(result: OLSResult) -> None:
    print("\n" + "=" * 65)
    print("  OLS REGRESSION RESULTS (HC3 Robust Standard Errors)")
    print("=" * 65)
    print(f"  Observations : {result.n:,}   |   Parameters: {result.k}")
    print(f"  R²           : {result.r_squared:.4f}")
    print(f"  Adj. R²      : {result.adj_r_squared:.4f}")
    print("─" * 65)
    print(f"  {'Variable':<22} {'Coef':>12} {'Robust SE':>12} {'t-stat':>8} {'p-value':>10} {'Sig':>4}")
    print("─" * 65)

    def sig(p):
        if p < 0.001: return "***"
        if p < 0.01:  return "** "
        if p < 0.05:  return "*  "
        if p < 0.10:  return ".  "
        return "   "

    for name, b, se, t, p in zip(
        result.feature_names, result.coefficients,
        result.robust_se, result.t_stats, result.p_values
    ):
        print(f"  {name:<22} {b:>12.2f} {se:>12.2f} {t:>8.3f} {p:>10.4f} {sig(p):>4}")

    print("─" * 65)
    print("  Significance: *** p<0.001  ** p<0.01  * p<0.05  . p<0.1")

    if result.vif:
        print("\n  VIF (Multicollinearity Diagnostics):")
        for col, v in result.vif.items():
            flag = "  *** HIGH" if v > 10 else ("  * moderate" if v > 5 else "")
            print(f"    {col:<22}: {v:.2f}{flag}")

    print("=" * 65)


def main() -> None:
    df = generate_housing_data(n=2000)

    features = ["sqft", "bedrooms", "age_years", "dist_downtown_mi", "school_rating"]
    X_df = df[features].copy()
    X = np.column_stack([np.ones(len(X_df)), X_df.values])
    y = df["price"].values

    result = ols_fit(X, y, ["intercept"] + features)
    result.vif = compute_vif(X_df)
    print_results(result)


if __name__ == "__main__":
    main()
