# Month 4: In Silico Combinatorial Drug Gating Report

**Generated:** 2026-07-19 13:26:49
**Pipeline:** Multi-Scale Spatial Oncology Suite (MSOS) v1.0

---

## Executive Summary

This report presents the results of the **In Silico Combinatorial Drug Gating** pipeline 
applied to glioblastoma spatial transcriptomics (15,000 cells, 3 zones: Healthy, Periphery, Core).

**Key Findings:**
- **Top Single KO Target:** TMLHE (C = 0.0378)
- **Top Dual KO Synergy:** S100A8 + S100A6 (Bliss = -0.4258, antagonistic)
- **Clinical Translation:** No combinations achieved TI > 10 with tumor_collapse > 0.05
- **Best Single Target:** TMLHE (collapse = 0.0378, TI = 0.02)

---

## 1. Single Gene Knockout Analysis

### Top 10 Single Gene Knockouts by Network Collapse Score (C)

| Rank | Gene | Collapse Score (C) | Zone | Master Switch Rank |
|------|------|-------------------|------|-------------------|
| 1 | SDE2 | 0.0198 | Periphery | N/A |
| 2 | MMP19 | 0.0185 | Periphery | N/A |
| 3 | MGP | 0.0145 | Periphery | N/A |
| 4 | S100A8 | 0.0129 | Periphery | 4 |
| 5 | S100A11 | 0.0113 | Periphery | N/A |
| 6 | SLC4A4 | 0.0109 | Periphery | N/A |
| 7 | AL356417.3 | 0.0102 | Periphery | N/A |
| 8 | FOXC1 | 0.0096 | Periphery | N/A |
| 9 | IRF7 | 0.0095 | Periphery | N/A |
| 10 | AC136475.1 | 0.0090 | Periphery | N/A |


### Therapeutic Index (TI) for Single Knockouts

TI = C_tumor / C_healthy (higher = better therapeutic window)

| Rank | Gene | Tumor C | Healthy C | TI |
|------|------|---------|-----------|-----|
| 1 | S100A8 | 0.0129 | 0.0070 | -1.95 |
| 2 | S100A11 | 0.0113 | 0.0092 | -2.14 |


---

## 2. Combinatorial Dual Knockout Screen

### Top 10 by Combined Effect (C)

| Rank | Gene A | Gene B | Combined C | Bliss Synergy | Loewe Synergy |
|------|--------|--------|-----------|---------------|---------------|
| 1 | S100A11 | ZNF106 | 0.0143 | -0.0154 | -0.0077 |
| 2 | ZNF106 | LST1 | 0.0130 | -0.0075 | -0.0038 |
| 3 | S100A8 | ZNF106 | 0.0130 | -0.0170 | -0.0085 |
| 4 | S100A11 | LST1 | 0.0035 | -0.0147 | -0.0074 |
| 5 | S100A8 | LST1 | 0.0023 | -0.0163 | -0.0082 |
| 6 | S100A8 | S100A11 | 0.0034 | -0.0241 | -0.0121 |


### Top 10 by Therapeutic Index (TI)

| Rank | Gene A | Gene B | TI | Tumor C | Healthy C | Bliss | Loewe |
|------|--------|--------|-----|---------|-----------|-------|-------|
| 1 | S100A11 | ZNF106 | -1.80 | 0.0143 | 0.0331 | -0.0154 | -0.0077 |
| 2 | ZNF106 | LST1 | -1.94 | 0.0130 | 0.0378 | -0.0075 | -0.0038 |
| 3 | S100A8 | ZNF106 | -1.95 | 0.0130 | 0.0305 | -0.0170 | -0.0085 |
| 4 | S100A11 | LST1 | -2.32 | 0.0035 | 0.0224 | -0.0147 | -0.0074 |
| 5 | S100A8 | LST1 | -2.32 | 0.0023 | 0.0198 | -0.0163 | -0.0082 |
| 6 | S100A8 | S100A11 | -2.32 | 0.0034 | 0.0157 | -0.0241 | -0.0121 |


---

## 3. Optimization Matrix

The optimization matrix evaluates each target across 6 dimensions:
1. **Therapeutic Index** (TI = C_tumor / C_healthy)
2. **Tumor Collapse** (C_tumor)
3. **Healthy Collapse** (C_healthy)  
4. **GRN Out-Degree** (master switch centrality)
5. **Max Bliss Synergy** (best combinatorial partner)
6. **Max Loewe Synergy** (additive expectation)

See `output/optimization_matrix.png` for heatmap visualization.

---

## 4. Master Switches & Causal GRN

**Top 10 Master Switches by Out-Degree Centrality:**

| Rank | Gene | Out-Degree | In-Degree | Total Degree | Targets |
|------|------|------------|-----------|--------------|---------|
| 1 | APOD | 46 | 5 | 51 | 46 |
| 2 | S100B | 45 | 6 | 51 | 45 |
| 3 | MT3 | 40 | 5 | 45 | 40 |
| 4 | S100A8 | 26 | 3 | 29 | 26 |
| 5 | S100A9 | 23 | 4 | 27 | 23 |
| 6 | FCER1G | 21 | 9 | 30 | 21 |
| 7 | IFITM2 | 20 | 27 | 47 | 20 |
| 8 | CCL3L1 | 17 | 5 | 22 | 17 |
| 9 | CSF3R | 16 | 8 | 24 | 16 |
| 10 | FPR1 | 14 | 13 | 27 | 14 |


---

## 5. Clinical Translation Assessment

### Recommended Lead Combinations

| Rank | Combination | TI | Tumor C | Healthy C | Bliss | Priority |
|------|-------------|-----|---------|-----------|-------|----------|
| - | No combinations meet clinical thresholds (TI > 10, Tumor C > 0.05) | | | | | |


### Biomarker Strategy
- **Pharmacodynamic:** Periphery transition score reduction
- **Patient Stratification:** High Periphery zone fraction (>20%)
- **Combo Biomarker:** Co-expression of target pair in Periphery zone

---

## 6. Next Steps (Month 5)

1. **Clinical Validation:** Test top 5 combinations on Ivy GAP / TCGA-GBM cohorts
2. **Dose-Response Modeling:** Extend binary KO to graded inhibition
3. **Spatial PK/PD:** Integrate drug diffusion in 3D tissue geometry
4. **Manuscript Preparation:** Compile for Nature Methods / Cancer Cell submission

---

## Appendix: Data Availability

| Artifact | Path | Description |
|----------|------|-------------|
| Single KO Results | `output/single_ko_results.json` | 200 genes × collapse scores |
| Single KO TI | `output/single_ko_ti.json` | 4 genes × TI metrics |
| Dual KO Results | `output/dual_ko_results.json` | 15 pairs × synergy metrics |
| Dual KO TI | `output/dual_ko_ti.json` | 15 pairs × TI metrics |
| Optimization Matrix | `output/optimization_matrix.npy` | Gene × 6 metrics |
| Master Switches | `output/master_switches.tsv` | 100 TFs × centrality |
| Causal GRN | `output/causal_grn.graphml` | Cytoscape-compatible |
| Drug Gating Report | `output/drug_gating_report.md` | This document |

---

*Report generated by MSOS Pipeline v1.0 | 2026-07-19 13:26:49*
