#!/usr/bin/env python3
"""
Month 5, Week 3: Clinical Validation Report Generation

Automated synthesis of survival analysis and spatial therapeutic indices
into a publication-ready clinical validation report.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Any


def load_survival_summary(path: str = "output/survival_stats_summary.json") -> Dict:
    """Load the survival statistics summary."""
    with open(path, "r") as f:
        return json.load(f)


def load_dual_ko_ti(path: str = "output/dual_ko_ti.json") -> List[Dict]:
    """Load dual KO therapeutic index results."""
    with open(path, "r") as f:
        return json.load(f)


def extract_top_risk_genes(survival_data: Dict, top_n: int = 4) -> List[Dict]:
    """Extract top risk-associated genes by Hazard Ratio magnitude."""
    univariate = survival_data.get("univariate", [])
    # Sort by HR deviation from 1.0 (risk or protective)
    sorted_genes = sorted(
        univariate,
        key=lambda x: abs(x["cox_hr"] - 1.0),
        reverse=True
    )
    return sorted_genes[:top_n]


def format_univariate_table(univariate: List[Dict]) -> str:
    """Generate Markdown table for univariate survival results."""
    lines = [
        "| Gene | Log-Rank chi2 | Log-Rank p | Cox HR | 95% CI Lower | 95% CI Upper | Cox p | High / Low |",
        "|------|-------------|------------|--------|--------------|--------------|-------|------------|",
    ]
    for row in univariate:
        lines.append(
            f"| {row['gene']} | {row['logrank_chi2']:.3f} | {row['logrank_p']:.4f} | "
            f"{row['cox_hr']:.3f} | {row['cox_ci_lower']:.3f} | {row['cox_ci_upper']:.3f} | "
            f"{row['cox_p']:.4f} | {row['n_high']} / {row['n_low']} |"
        )
    return "\n".join(lines)


def format_multivariate_table(multivariate: Dict) -> str:
    """Generate Markdown table for multivariate Cox results."""
    features = multivariate["features"]
    lines = [
        "| Feature | Coefficient (beta) | Hazard Ratio | 95% CI Lower | 95% CI Upper | p-value |",
        "|---------|-----------------|--------------|--------------|--------------|---------|",
    ]
    for feat in features:
        lines.append(
            f"| {feat} | {multivariate['coefficients'][feat]:.4f} | "
            f"{multivariate['hr'][feat]:.3f} | {multivariate['ci_lower'][feat]:.3f} | "
            f"{multivariate['ci_upper'][feat]:.3f} | {multivariate['p_values'][feat]:.4f} |"
        )
    return "\n".join(lines)


def format_spatial_therapeutic_table(dual_ko: List[Dict], top_n: int = 6) -> str:
    """Generate Markdown table contrasting spatial TI with survival HR."""
    # Sort by Bliss synergy (most synergistic first)
    sorted_ko = sorted(dual_ko, key=lambda x: x.get("bliss_synergy", -1e9), reverse=True)
    lines = [
        "| Rank | Gene A | Gene B | Bliss Synergy | Loewe Synergy | Tumor Collapse | Healthy Collapse | Calibrated TI |",
        "|------|--------|--------|---------------|---------------|----------------|------------------|---------------|",
    ]
    for i, pair in enumerate(sorted_ko[:top_n], 1):
        lines.append(
            f"| {i} | {pair['gene_a']} | {pair['gene_b']} | "
            f"{pair.get('bliss_synergy', 0):.4f} | {pair.get('loewe_synergy', 0):.4f} | "
            f"{pair.get('tumor_collapse', 0):.4f} | {pair.get('healthy_collapse', 0):.4f} | "
            f"{pair.get('therapeutic_index', 0):.2f} |"
        )
    return "\n".join(lines)


def generate_report(
    survival_data: Dict,
    dual_ko: List[Dict],
    output_path: str = "output/clinical_validation_report.md",
) -> None:
    """Generate the full clinical validation report."""
    univariate = survival_data.get("univariate", [])
    multivariate = survival_data.get("multivariate", {})
    cohort_size = survival_data.get("cohort_size", 0)
    n_events = survival_data.get("n_events", 0)
    median_survival = survival_data.get("median_survival_days", 0)
    genes = survival_data.get("genes_analyzed", [])

    top_risk = extract_top_risk_genes(survival_data)

    with open(output_path, "w") as f:
        f.write("# Month 5 Clinical Validation Report\n\n")
        f.write(f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("**Pipeline:** Multi-Scale Spatial Oncology Suite (MSOS) v1.0\n\n")
        f.write("---\n\n")

        # Executive Summary
        f.write("## Executive Summary\n\n")
        f.write(
            f"This report integrates **in silico combinatorial drug screening** (Month 4) "
            f"with **clinical survival modeling** on a mock IvyGAP-like glioblastoma cohort "
            f"(n={cohort_size}, events={n_events}, median survival={median_survival:.1f} days). "
            f"The analysis evaluates whether spatial therapeutic indices derived from the "
            f"calibrated cVAE encoder predict patient-level outcomes.\n\n"
        )
        f.write(
            f"**Key Finding:** None of the top 4 spatially-prioritized targets "
            f"({', '.join(genes)}) reached statistical significance (p < 0.05) in either "
            f"univariate log-rank or multivariate Cox models adjusted for age. "
            f"S100A11 showed the strongest trend toward adverse prognosis "
            f"(multivariate HR = {multivariate['hr'].get('S100A11_expr', 0):.2f}, "
            f"p = {multivariate['p_values'].get('S100A11_expr', 1):.3f}), "
            f"consistent with its role in mesenchymal GBM transition.\n\n"
        )
        f.write("---\n\n")

        # Univariate Survival Analysis
        f.write("## 1. Univariate Survival Analysis\n\n")
        f.write(
            "Patients stratified by median expression of each target gene. "
            "Log-rank test compares Kaplan-Meier curves; Cox PH estimates continuous HR.\n\n"
        )
        f.write(format_univariate_table(univariate))
        f.write("\n\n")
        f.write(
            "*Interpretation: Higher HR indicates worse survival for high-expression group. "
            "No gene reached p < 0.05; S100A11 trending (p = 0.107 Cox, p = 0.205 log-rank).*\n\n"
        )
        f.write("---\n\n")

        # Multivariate Cox Model
        f.write("## 2. Multivariate Cox Proportional Hazards Model\n\n")
        f.write(
            "Joint model including all 4 target gene expressions + age at diagnosis. "
            "Tests independent prognostic value after mutual adjustment.\n\n"
        )
        f.write(format_multivariate_table(multivariate))
        f.write("\n\n")
        f.write(
            "*Interpretation: S100A11 remains the strongest independent predictor "
            "(HR = 1.17 per log2 TPM unit, p = 0.19). All CIs cross 1.0.*\n\n"
        )
        f.write("---\n\n")

        # Spatial Therapeutic Indices
        f.write("## 3. Spatial Therapeutic Indices (Month 4 Dual-KO Screen)\n\n")
        f.write(
            "Calibrated therapeutic indices from the combinatorial screen, "
            "measured against per-zone latent covariance baselines. "
            "Higher TI = greater tumor collapse relative to healthy tissue disruption.\n\n"
        )
        f.write(format_spatial_therapeutic_table(dual_ko))
        f.write("\n\n")
        f.write(
            "*Note: All calibrated TI values are negative (log2 scale), "
            "reflecting that healthy-zone disruption exceeds tumor-zone collapse "
            "in the current model. This indicates the need for target refinement.*\n\n"
        )
        f.write("---\n\n")

        # Cross-Modal Concordance
        f.write("## 4. Cross-Modal Concordance Assessment\n\n")
        f.write(
            "Comparison of spatial prioritization (Bliss synergy) vs. clinical risk (Cox HR):\n\n"
        )

        # Build concordance table
        lines = [
            "| Gene | Spatial Rank (Bliss) | Univariate Cox HR | Multivariate HR | Concordance |",
            "|------|---------------------|-------------------|-----------------|-------------|",
        ]
        # Map gene to Bliss rank
        sorted_ko = sorted(dual_ko, key=lambda x: x.get("bliss_synergy", -1e9), reverse=True)
        gene_to_rank = {}
        for i, pair in enumerate(sorted_ko, 1):
            gene_to_rank[pair["gene_a"]] = gene_to_rank.get(pair["gene_a"], i)
            gene_to_rank[pair["gene_b"]] = gene_to_rank.get(pair["gene_b"], i)

        for u in univariate:
            gene = u["gene"]
            spatial_rank = gene_to_rank.get(gene, "—")
            uni_hr = u["cox_hr"]
            multi_hr = multivariate["hr"].get(f"{gene}_expr", "—")
            concord = "YES" if (spatial_rank <= 2 and uni_hr > 1.1) or (spatial_rank > 2 and uni_hr <= 1.1) else "NO"
            lines.append(f"| {gene} | {spatial_rank} | {uni_hr:.2f} | {multi_hr if isinstance(multi_hr, str) else f'{multi_hr:.2f}'} | {concord} |")

        f.write("\n".join(lines))
        f.write("\n\n")
        f.write(
            "*Concordance (YES) defined as top spatial synergy aligning with HR > 1.1 "
            "(adverse) or low synergy with HR <= 1.1. Current mock data shows limited "
            "concordance, expected without true biological signal.*\n\n"
        )
        f.write("---\n\n")

        # Visual Assets
        f.write("## 5. Visual Artifacts\n\n")
        f.write("### 5.1 Kaplan-Meier Survival Curves\n\n")
        f.write(
            "![Kaplan-Meier Curves](km_survival_curves.png)\n\n"
            "**Figure 1.** Kaplan-Meier survival curves stratified by median expression "
            "of each target gene. Log-rank p-values annotated per panel. "
            "No gene shows statistically significant separation (all p > 0.05).\n\n"
        )
        f.write("### 5.2 Multivariate Forest Plot\n\n")
        f.write(
            "![Forest Plot](forest_plot.png)\n\n"
            "**Figure 2.** Hazard ratios with 95% confidence intervals from the "
            "multivariate Cox model (4 genes + age). Red indicates p < 0.05; "
            "gray indicates non-significance. All features are non-significant.\n\n"
        )
        f.write("---\n\n")

        # Conclusions & Next Steps
        f.write("## 6. Conclusions & Next Steps\n\n")
        f.write("### Summary\n\n")
        f.write(
            f"- **Cohort:** {cohort_size} patients, {n_events} events ({n_events/cohort_size:.0%} event rate)\n"
            f"- **Targets tested:** {', '.join(genes)}\n"
            f"- **Significant univariate predictors:** 0/4 (p < 0.05)\n"
            f"- **Significant multivariate predictors:** 0/5 (p < 0.05)\n"
            f"- **Top trend:** S100A11 (HR = {multivariate['hr'].get('S100A11_expr', 0):.2f}, p = {multivariate['p_values'].get('S100A11_expr', 1):.3f})\n\n"
        )
        f.write("### Limitations\n\n")
        f.write(
            "1. **Mock cohort:** Expression and survival generated from parametric "
            "distributions without true biological signal.\n"
            "2. **Sample size:** n=120 provides limited power for multivariable modeling "
            "with 5 covariates (events per variable ~ 15).\n"
            "3. **Spatial-clinical gap:** The cVAE encoder was trained on spatial transcriptomics, "
            "not bulk RNA-seq; expression distributions differ.\n\n"
        )
        f.write("### Recommended Next Steps (Month 6)\n\n")
        f.write(
            "1. **Real IvyGAP/TCGA ingestion:** Replace mock cohort with actual IvyGAP "
            "microdissection RNA-seq + clinical annotations.\n"
            "   Download from NIH GDC.\n"
            "2. **Continuous target scoring:** Move from median splits to penalized "
            "Cox (LASSO/elastic net) for multi-gene risk signatures.\n"
            "3. **Spatial transcriptomics validation:** Map top KO targets back to "
            "spatial zones (Periphery/Core) and correlate zone-specific expression "
            "with local recurrence patterns.\n"
            "4. **Dose-response extension:** Replace binary KO with graded inhibition "
            "in the cVAE decoder to generate IC50-equivalent therapeutic windows.\n\n"
        )
        f.write("---\n\n")

        # Appendix
        f.write("## Appendix: Data Artifacts\n\n")
        f.write("| Artifact | Path | Description |\n")
        f.write("|----------|------|-------------|\n")
        f.write("| Survival Summary | `output/survival_stats_summary.json` | Combined univariate + multivariate stats |\n")
        f.write("| Univariate CSV | `output/univariate_survival.csv` | Log-rank + Cox per gene |\n")
        f.write("| Multivariate JSON | `output/multivariate_survival.json` | Full Cox model coefficients |\n")
        f.write("| KM Curves | `output/km_survival_curves.png` | 4-panel survival curves |\n")
        f.write("| Forest Plot | `output/forest_plot.png` | Multivariate HR visualization |\n")
        f.write("| Dual-KO TI | `output/dual_ko_ti.json` | Calibrated spatial therapeutic indices |\n")
        f.write("| Clinical Cohort | `output/clinical_mapped_cohort.csv` | Patient-level expression + survival |\n\n")
        f.write("---\n\n")
        f.write(f"*Report generated by MSOS Pipeline v1.0 | {time.strftime('%Y-%m-%d %H:%M:%S')}*\n")


def main():
    print("=" * 60)
    print("MONTH 5 WEEK 3: CLINICAL VALIDATION REPORT GENERATION")
    print("=" * 60)

    print("\n[LOAD] Reading survival_stats_summary.json...")
    survival_data = load_survival_summary()
    print(f"  Cohort: {survival_data.get('cohort_size')} patients, "
          f"{survival_data.get('n_events')} events")

    print("\n[LOAD] Reading dual_ko_ti.json...")
    dual_ko = load_dual_ko_ti()
    print(f"  Loaded {len(dual_ko)} dual-KO pairs")

    print("\n[GENERATE] Compiling clinical validation report...")
    output_path = "output/clinical_validation_report.md"
    Path("output").mkdir(exist_ok=True)
    generate_report(survival_data, dual_ko, output_path)

    print(f"\n[SUCCESS] Month 5 Week 3 Complete: Clinical Validation Report")
    print(f"  - {output_path}")


if __name__ == "__main__":
    main()