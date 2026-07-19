#!/usr/bin/env python3
"""
Month 5, Week 1: IvyGAP Clinical Cohort Ingestion & Mapping

Loads calibrated dual-KO results, extracts top synergistic targets,
generates a mock IvyGAP-like patient cohort, maps target expression,
and performs survival analysis (log-rank test, Cox PH).
"""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize

warnings.filterwarnings("ignore")


# --------------------------------------------------------------------------- #
# Data Loading
# --------------------------------------------------------------------------- #
def load_dual_ko_results(path: str = "output/dual_ko_ti.json") -> List[dict]:
    """Load dual KO therapeutic index results from JSON."""
    with open(path, "r") as f:
        return json.load(f)


def extract_top_genes(
    dual_ko_results: List[dict],
    top_n_pairs: int = 4,
) -> List[str]:
    """Extract unique gene symbols from top N dual-KO pairs by Bliss synergy."""
    sorted_pairs = sorted(
        dual_ko_results,
        key=lambda x: x.get("bliss_synergy", -np.inf),
        reverse=True,
    )
    top_genes = set()
    for pair in sorted_pairs[:top_n_pairs]:
        top_genes.add(pair["gene_a"])
        top_genes.add(pair["gene_b"])
    return sorted(top_genes)


# --------------------------------------------------------------------------- #
# Mock IvyGAP Cohort Generation
# --------------------------------------------------------------------------- #
def generate_mock_ivygap_cohort(
    n_patients: int = 120,
    target_genes: Optional[List[str]] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a robust mock IvyGAP-like clinical cohort.

    Features:
      - patient_id: unique identifier
      - survival_time_days: realistic GBM survival (Weibull)
      - vital_status: 0=censored, 1=deceased (event)
      - age_at_diagnosis: normal distribution ~60±12 years
      - sex: M/F balanced
      - grade: WHO grade (all IV for GBM)
      - <gene>_expr: normalized RNA-seq (log2 TPM+1) for target genes
    """
    rng = np.random.default_rng(seed)

    if target_genes is None:
        target_genes = ["S100A8", "S100A11", "ZNF106", "LST1"]

    # GBM survival: median ~15 months (450 days), Weibull shape ~1.2
    shape, scale = 1.2, 450.0
    survival_time = rng.weibull(shape, n_patients) * scale

    # Censoring: ~30% censored (alive at last follow-up)
    censor_time = rng.uniform(100, 800, n_patients)
    observed_time = np.minimum(survival_time, censor_time)
    vital_status = (survival_time <= censor_time).astype(int)

    # Clinical covariates
    age = np.clip(rng.normal(60, 12, n_patients).astype(int), 20, 85)
    sex = rng.choice(["M", "F"], n_patients, p=[0.55, 0.45])

    # Gene expression: correlated modules (S100 family co-expressed)
    n_genes = len(target_genes)
    # Base correlation structure
    corr = np.eye(n_genes)
    s100_idx = [i for i, g in enumerate(target_genes) if g.startswith("S100")]
    if len(s100_idx) >= 2:
        for i in s100_idx:
            for j in s100_idx:
                if i != j:
                    corr[i, j] = 0.6  # S100 co-expression

    # Add mild correlation with ZNF106/LST1
    znf_idx = [i for i, g in enumerate(target_genes) if g == "ZNF106"]
    lst_idx = [i for i, g in enumerate(target_genes) if g == "LST1"]
    if znf_idx and lst_idx:
        corr[znf_idx[0], lst_idx[0]] = corr[lst_idx[0], znf_idx[0]] = 0.3

    # Sample from multivariate normal (log2 TPM scale)
    mean_expr = rng.uniform(2, 8, n_genes)  # gene-specific baselines
    expr = rng.multivariate_normal(mean_expr, corr * 1.5, size=n_patients)

    # Build DataFrame
    df = pd.DataFrame({
        "patient_id": [f"PAT_{i:04d}" for i in range(n_patients)],
        "survival_time_days": np.round(observed_time, 1),
        "vital_status": vital_status,
        "age_at_diagnosis": age,
        "sex": sex,
        "who_grade": 4,
    })
    for j, gene in enumerate(target_genes):
        df[f"{gene}_expr"] = np.round(expr[:, j], 3)

    return df


# --------------------------------------------------------------------------- #
# Clinical Mapping
# --------------------------------------------------------------------------- #
def map_targets_to_patients(
    patient_df: pd.DataFrame,
    top_genes: List[str],
) -> pd.DataFrame:
    """
    Isolate expression profiles of top KO targets across the cohort.

    Returns a tidy DataFrame with one row per (patient, gene).
    """
    expr_cols = [f"{g}_expr" for g in top_genes if f"{g}_expr" in patient_df.columns]
    if not expr_cols:
        raise ValueError(f"No expression columns found for genes: {top_genes}")

    id_vars = ["patient_id", "survival_time_days", "vital_status",
               "age_at_diagnosis", "sex", "who_grade"]

    tidy = patient_df.melt(
        id_vars=id_vars,
        value_vars=expr_cols,
        var_name="gene",
        value_name="expression_log2tpm",
    )
    tidy["gene"] = tidy["gene"].str.replace("_expr", "", regex=False)
    return tidy


# --------------------------------------------------------------------------- #
# Survival Analysis (Manual Implementation)
# --------------------------------------------------------------------------- #
def logrank_test(
    time: np.ndarray,
    event: np.ndarray,
    group: np.ndarray,
) -> Tuple[float, float]:
    """
    Two-sample log-rank test (Mantel-Cox).

    Returns: (chi2_statistic, p_value)
    """
    # Unique event times
    unique_times = np.unique(time[event == 1])
    if len(unique_times) == 0:
        return 0.0, 1.0

    n_groups = len(np.unique(group))
    if n_groups != 2:
        raise ValueError("logrank_test requires exactly 2 groups")

    g1, g2 = np.unique(group)
    mask1 = group == g1
    mask2 = group == g2

    O1 = O2 = E1 = E2 = 0.0

    for t in unique_times:
        at_risk1 = np.sum((time >= t) & mask1)
        at_risk2 = np.sum((time >= t) & mask2)
        at_risk = at_risk1 + at_risk2

        events1 = np.sum((time == t) & (event == 1) & mask1)
        events2 = np.sum((time == t) & (event == 1) & mask2)
        events = events1 + events2

        if at_risk > 0:
            exp1 = events * at_risk1 / at_risk
            exp2 = events * at_risk2 / at_risk
            O1 += events1
            O2 += events2
            E1 += exp1
            E2 += exp2

    # Variance of O1 - E1
    V = 0.0
    for t in unique_times:
        at_risk1 = np.sum((time >= t) & mask1)
        at_risk2 = np.sum((time >= t) & mask2)
        at_risk = at_risk1 + at_risk2

        events = np.sum((time == t) & (event == 1))
        if at_risk > 1 and events > 0:
            v_t = (at_risk1 * at_risk2 * events * (at_risk - events)) / \
                  (at_risk**2 * (at_risk - 1))
            V += v_t

    if V == 0:
        return 0.0, 1.0

    chi2 = (O1 - E1)**2 / V
    p = 1 - stats.chi2.cdf(chi2, df=1)
    return chi2, p


def cox_ph_fit(
    time: np.ndarray,
    event: np.ndarray,
    X: np.ndarray,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> Dict:
    """
    Cox Proportional Hazards model via Breslow partial likelihood.

    Returns: dict with coefficients, HR, CI, p-values
    """
    n, p = X.shape
    if p == 0:
        return {"coefficients": np.array([]), "hr": np.array([]),
                "ci_lower": np.array([]), "ci_upper": np.array([]),
                "p_values": np.array([])}

    # Center covariates for numerical stability
    X_centered = X - X.mean(axis=0)

    # Unique event times
    event_times = np.unique(time[event == 1])

    # Newton-Raphson optimization of partial log-likelihood
    beta = np.zeros(p)

    for _ in range(max_iter):
        U = np.zeros(p)  # Score vector
        I = np.zeros((p, p))  # Information matrix

        for t in event_times:
            risk_mask = time >= t
            X_risk = X_centered[risk_mask]
            n_risk = X_risk.shape[0]
            if n_risk == 0:
                continue

            exp_xb = np.exp(X_risk @ beta)
            s0 = exp_xb.sum()
            # s1: weighted mean of covariates
            s1 = (X_risk.T * exp_xb).sum(axis=1) / s0
            # s2: weighted covariance
            s2 = np.zeros((p, p))
            for k in range(n_risk):
                diff = X_risk[k] - s1
                s2 += exp_xb[k] * np.outer(diff, diff)
            s2 /= s0

            # Events at this time
            events_mask = (time == t) & (event == 1)
            X_events = X_centered[events_mask]
            n_events = X_events.shape[0]
            if n_events == 0:
                continue

            U += X_events.sum(axis=0) - n_events * s1
            I += n_events * s2

        # Update
        try:
            delta = np.linalg.solve(I + 1e-8 * np.eye(p), U)
        except np.linalg.LinAlgError:
            delta = np.linalg.pinv(I + 1e-8 * np.eye(p)) @ U

        beta += delta
        if np.max(np.abs(delta)) < tol:
            break

    # Standard errors
    try:
        var = np.diag(np.linalg.inv(I + 1e-8 * np.eye(p)))
    except np.linalg.LinAlgError:
        var = np.diag(np.linalg.pinv(I + 1e-8 * np.eye(p)))

    se = np.sqrt(np.maximum(var, 0))
    hr = np.exp(beta)
    z = beta / se
    p_values = 2 * (1 - stats.norm.cdf(np.abs(z)))

    # 95% CI for HR
    ci_lower = np.exp(beta - 1.96 * se)
    ci_upper = np.exp(beta + 1.96 * se)

    return {
        "coefficients": beta,
        "hr": hr,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "p_values": p_values,
        "standard_errors": se,
    }


def stratify_by_expression(
    df: pd.DataFrame,
    gene: str,
    method: str = "median",
) -> np.ndarray:
    """Stratify patients into high/low expression groups."""
    expr_col = f"{gene}_expr"
    if expr_col not in df.columns:
        raise ValueError(f"Column {expr_col} not found")

    values = df[expr_col].values
    if method == "median":
        cutoff = np.median(values)
    elif method == "tertile":
        cutoff = np.percentile(values, 66.67)
    elif method == "quartile":
        cutoff = np.percentile(values, 75)
    else:
        raise ValueError(f"Unknown method: {method}")

    return (values > cutoff).astype(int)


def run_univariate_survival(
    patient_df: pd.DataFrame,
    genes: List[str],
) -> pd.DataFrame:
    """Run log-rank test and Cox PH for each gene (high vs low expression)."""
    results = []
    time = patient_df["survival_time_days"].values
    event = patient_df["vital_status"].values

    for gene in genes:
        if f"{gene}_expr" not in patient_df.columns:
            continue

        # Stratify by median expression
        group = stratify_by_expression(patient_df, gene, method="median")

        # Log-rank test
        chi2, p_lr = logrank_test(time, event, group)

        # Cox PH (continuous expression)
        X = patient_df[f"{gene}_expr"].values.reshape(-1, 1)
        cox = cox_ph_fit(time, event, X)

        results.append({
            "gene": gene,
            "logrank_chi2": chi2,
            "logrank_p": p_lr,
            "cox_hr": cox["hr"][0],
            "cox_ci_lower": cox["ci_lower"][0],
            "cox_ci_upper": cox["ci_upper"][0],
            "cox_p": cox["p_values"][0],
            "n_high": int(group.sum()),
            "n_low": int((1 - group).sum()),
        })

    return pd.DataFrame(results)


def run_multivariate_survival(
    patient_df: pd.DataFrame,
    genes: List[str],
    clinical_covariates: List[str] = None,
) -> Dict:
    """Run multivariate Cox PH with gene expressions + clinical covariates."""
    if clinical_covariates is None:
        clinical_covariates = ["age_at_diagnosis"]

    time = patient_df["survival_time_days"].values
    event = patient_df["vital_status"].values

    # Build design matrix
    cols = [f"{g}_expr" for g in genes if f"{g}_expr" in patient_df.columns]
    cols += clinical_covariates

    X = patient_df[cols].values.astype(float)
    # Handle missing
    if np.isnan(X).any():
        X = np.nan_to_num(X, nan=np.nanmean(X))

    cox = cox_ph_fit(time, event, X)

    return {
        "features": cols,
        "coefficients": dict(zip(cols, cox["coefficients"])),
        "hr": dict(zip(cols, cox["hr"])),
        "ci_lower": dict(zip(cols, cox["ci_lower"])),
        "ci_upper": dict(zip(cols, cox["ci_upper"])),
        "p_values": dict(zip(cols, cox["p_values"])),
    }


# --------------------------------------------------------------------------- #
# Main Pipeline
# --------------------------------------------------------------------------- #
def main():
    print("=" * 60)
    print("MONTH 5 WEEK 1: IVYGAP CLINICAL COHORT INGESTION & MAPPING")
    print("=" * 60)

    # 1. Load dual-KO results
    print("\n[LOAD] Reading dual_ko_ti.json...")
    dual_ko = load_dual_ko_results()
    print(f"  Loaded {len(dual_ko)} dual-KO pairs")

    # 2. Extract top synergistic targets
    top_genes = extract_top_genes(dual_ko, top_n_pairs=4)
    print(f"\n[TARGETS] Top genes from Bliss-ranked pairs: {top_genes}")

    # 3. Generate mock IvyGAP cohort
    print(f"\n[COHORT] Generating mock IvyGAP cohort (n=120)...")
    cohort_df = generate_mock_ivygap_cohort(n_patients=120, target_genes=top_genes)
    print(f"  Shape: {cohort_df.shape}")
    print(f"  Events: {cohort_df['vital_status'].sum()}/{len(cohort_df)}")
    print(f"  Median survival: {cohort_df['survival_time_days'].median():.1f} days")

    # 4. Map targets to patients
    print("\n[MAP] Creating tidy expression matrix...")
    mapped_df = map_targets_to_patients(cohort_df, top_genes)
    print(f"  Tidy shape: {mapped_df.shape}")

    # 5. Survival analysis
    print("\n[SURVIVAL] Univariate analysis (log-rank + Cox)...")
    univariate = run_univariate_survival(cohort_df, top_genes)
    print(univariate.to_string(index=False))

    print("\n[SURVIVAL] Multivariate Cox PH (genes + age)...")
    multivariate = run_multivariate_survival(cohort_df, top_genes)
    for feat in multivariate["features"]:
        print(f"  {feat}: HR={multivariate['hr'][feat]:.3f} "
              f"({multivariate['ci_lower'][feat]:.3f}-{multivariate['ci_upper'][feat]:.3f}), "
              f"p={multivariate['p_values'][feat]:.4f}")

    # 6. Save artifacts
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    cohort_path = output_dir / "clinical_mapped_cohort.csv"
    cohort_df.to_csv(cohort_path, index=False)
    print(f"\n[SAVE] Clinical cohort -> {cohort_path}")

    mapped_path = output_dir / "clinical_mapped_tidy.csv"
    mapped_df.to_csv(mapped_path, index=False)
    print(f"[SAVE] Tidy expression -> {mapped_path}")

    univariate_path = output_dir / "univariate_survival.csv"
    univariate.to_csv(univariate_path, index=False)
    print(f"[SAVE] Univariate survival -> {univariate_path}")

    multivar_path = output_dir / "multivariate_survival.json"
    with open(multivar_path, "w") as f:
        json.dump(multivariate, f, indent=2, default=str)
    print(f"[SAVE] Multivariate survival -> {multivar_path}")

    print("\n[SUCCESS] Month 5 Week 1 Complete: IvyGAP Clinical Ingestion")
    print(f"  - {cohort_path} ({cohort_df.shape[0]} patients x {cohort_df.shape[1]} features)")
    print(f"  - {mapped_path} ({mapped_df.shape[0]} rows)")
    print(f"  - {univariate_path} ({len(univariate)} genes)")
    print(f"  - {multivar_path} (multivariate Cox)")


if __name__ == "__main__":
    main()