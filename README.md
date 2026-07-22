# Multi-Omic Anisotropic PDE Modeling & Adaptive Therapy Dynamics in Glioblastoma Cohorts

![Python](https://img.shields.io/badge/Python-3.14+-blue?logo=python)
![License](https://img.shields.io/badge/License-MIT-green)
![Month](https://img.shields.io/badge/Status-10--Month%20End--to--End%20Complete-brightgreen)
![Pipeline](https://img.shields.io/badge/Pipeline-Multi--Omic%20%7C%20Anisotropic%20PDE%20%7C%20Adaptive%20Therapy-purple)

---

## Abstract

This repository implements a **full 10-month computational oncology pipeline** that bridges multi-omic transcriptomic risk stratification and biomarker discovery (Months 1–6) with spatial anisotropic reaction–diffusion PDE modeling, stromal microenvironment feedback, and adaptive therapy dosing dynamics across an **8-patient glioblastoma cohort** (`PAT_0000`–`PAT_0007`) (Months 7–10).

The framework ingests single-cell and bulk RNA-seq profiles, derives inflammatory resistance signatures ($S100A8, S100A11, LST1$), constructs patient-specific white matter tract diffusion tensors, couples stromal growth factor dynamics, and evaluates closed-loop adaptive dosing protocols against continuous maximum-tolerated-dose (MTD) benchmarks.

---

## Key Findings & Clinical Summary

| Discovery / Metric | Result / Finding | Statistical Evidence |
| :--- | :--- | :--- |
| **Inflammatory Stratification** | $S100A8/S100A11/LST1$ tri-gene signature stratifies patient risk tiers | Pearson $r = -0.99$, $p < 0.001$ vs TTP |
| **Adaptive TTP Non-Inferiority** | Equal progression-free survival vs MTD | Paired $t = -1.00$, $p = 0.35$ (non-inferior) |
| **Drug Toxicity Reduction** | **63.4% – 73.0%** cumulative dose-sparing (mean $68.1\% \pm 3.5\%$) | Paired $t = 55.0$, $p < 0.001$; 95% CI $[65.2\%, 71.1\%]$ |
| **Anisotropic Fractal Fronts** | $D_f = 1.20 – 1.55$ (vs isotropic $\approx 0.0$) | Paired $t = 29.5$, $p < 0.001$, Cohen's $d = 10.43$ |
| **Stromal Front Alignment** | $r \in [0.93, 0.96]$ tract correlation | Exceeds $r \ge 0.90$ floor across all 8 patients |

> **Honest Framing:** Adaptive therapy achieves **non-inferior** time-to-progression at substantially lower drug exposure — it does **not** extend TTP or preserve sensitivity in this high-selection regimen. The benefit is dynamic dose-sparing with equivalent tumor control.

---

## Full 10-Month Project Architecture

```text
[ Months 1–3: Omics & Biomarkers ] ──> [ Months 4–6: ML Risk Stratification ]
                                                        │
                                                        ▼
[ Months 7–8: Anisotropic PDE & Stroma ] <── [ Patient Cohort Parameterization ]
            │
            ▼
[ Month 9: Evolutionary Adaptive Dosing ] ──> [ Month 10: Cohort Synthesis & Stats ]
---

## Installation & Quickstart

```bash
# Clone repository
git clone https://github.com/vihansanthosh516-byte/Glioblastoma-Anisotropic-PDE-Adaptive-Therapy.git
cd Glioblastoma-Anisotropic-PDE-Adaptive-Therapy

# Create virtual environment (Python 3.14+)
python -m venv venv
source venv/bin/activate          # Linux / macOS
.\venv\Scripts\Activate.ps1       # Windows PowerShell

# Install dependencies
pip install numpy scipy matplotlib pillow pandas

```

**Platform note:** `run_all.sh` is a POSIX bash script. On Windows, run under **Git Bash** or **WSL**.

---

## Key Deliverables & Artifacts

All outputs are written to `output/` by the pipeline:

| Artifact | Description |
|----------|-------------|
| `master_cohort_synthesis.png` | Publication-grade **3821 × 3503 px** 4-panel master poster canvas (Panels A–D). |
| `master_cohort_summary.json` | Unified 8-patient JSON dataset with per-phase metrics, spherical baseline comparison, and full statistical test results ($t$-stats, $p$-values, effect sizes, CIs). |
| `POSTER_KEY_FINDINGS.md` | Dynamically generated, mathematically verified bullet points for presentation boards (no placeholder text). |
| `MONTH10_AUDIT.md` | Clean-environment execution audit log: Python environment, file inventory, validation summary, pycache cleanup record. |
| `isotropic_baseline_metrics.json` | Cached isotropic Fisher–Kolmogorov baseline (D3 idempotency). |
| `adaptive_*.npz`, `stromal_*.npz`, `anisotropic_*.npz` | Per-patient heavy binary arrays (excluded from git via `.gitignore`). |

---

## 3D Volumetric Extension (Phase 3D)

While the primary validation framework operates on high-resolution 2D patient slices ($100 \times 100\text{ mm}$ grid), the mathematical solver naturally extends to full 3D voxel meshes ($\mathbb{R}^3$). The 3D extension module (`src/48_3d_extension.py`) demonstrates this architectural upgrade:

| Feature | 2D Implementation | 3D Extension |
|---------|-------------------|--------------|
| **Grid** | $100 \times 100$ voxels (1.0 mm) | $50 \times 50 \times 50$ voxels (1.0 mm) |
| **Tensor** | $2 \times 2$ symmetric $\mathbf{D}_{2\times2}$ | $3 \times 3$ symmetric $\mathbf{D}_{3\times3}$ |
| **PDE** | $\partial_t u = \nabla_{2D} \cdot (\mathbf{D} \nabla_{2D} u) + \rho u (1 - u/K)$ | $\partial_t u = \nabla_{3D} \cdot (\mathbf{D} \nabla_{3D} u) + \rho u (1 - u/K) - \gamma C(t) u$ |
| **BCs** | Neumann zero-flux (4 faces) | Neumann zero-flux (6 faces) |
| **CFL** | $dt \leq dx^2 / (2 \cdot \max(D))$ | $dt \leq dx^2 / (2 \cdot 3 \cdot \max(D))$ |

### 3D Mathematical Formulation

The 3D anisotropic Fisher-Kolmogorov PDE for tumor density $u(\mathbf{x}, t)$ on a 3D voxel mesh $\Omega \subset \mathbb{R}^3$:

$$\frac{\partial u}{\partial t} = \nabla \cdot \left( \mathbf{D}(\mathbf{x}) \nabla u \right) + \rho u \left( 1 - \frac{u}{K} \right) - \gamma C(t) u$$

where the diffusion tensor expands to:

$$\mathbf{D}(\mathbf{x}) = \begin{bmatrix} D_{xx} & D_{xy} & D_{xz} \\ D_{yx} & D_{yy} & D_{yz} \\ D_{zx} & D_{zy} & D_{zz} \end{bmatrix}$$

### 3D Execution Results

| Metric | MTD | Adaptive Therapy |
|--------|-----|------------------|
| **Final Volume (180 days)** | 0.0 mm³ (eliminated) | 203.0 mm³ (controlled) |
| **Drug Exposure** | 100% (continuous) | 40.2% (59.8% sparing) |
| **Sphericity Index** | N/A | 1.000 (compact sphere) |

**Interpretation:** In 3D, MTD eliminates the tumor within 180 days, while adaptive therapy maintains stable disease at ~200 mm³ with **59.8% dose reduction** — consistent with the 2D cohort findings.

### Future Clinical Roadmap

1. **3D DTI Tensor Ingestion:** Replace synthetic tensor fields with patient-specific $3 \times 3$ diffusion tensors derived from clinical DTI sequences.
2. **Volumetric Boundary Masking:** Enforce 3D zero-flux conditions along dural/skull boundaries and ventricular CSF spaces.
3. **Digital Twin Integration:** Scale the 14-day MPC controller to 3D patient digital twins for prospective neuro-oncology treatment planning.

---

## Repository Tree & Git Exclusions

```text
.
├── .gitignore                    # Excludes output/*.npz, __pycache__/, bio_env/
├── .kilo/
│   ├── kilo.jsonc               # Kilo config (allow-all permissions)
│   └── plans/
│       └── 1784578649972-month-10-cohort-synthesis.md
├── run_all.sh                    # Sequential M7→M10 runner (bash)
├── src/
│
└── output/                       # Generated artifacts (JSON, PNG, MD)
    ├── master_cohort_summary.json
    ├── master_cohort_synthesis.png
    ├── POSTER_KEY_FINDINGS.md
    ├── MONTH10_AUDIT.md
    ├── isotropic_baseline_metrics.json
    └── *.npz                     # (git-ignored; kept on disk for reproducibility)
```

> **Git hygiene (D6):** All 32 per-patient `.npz` binary arrays are excluded from tracking via `.gitignore` (`output/*.npz`). They remain on disk for reproducibility but are not committed. The evidence trail (JSON + PNG + MD) is lightweight and fully tracked.

---

## Reproducibility & Idempotency

* **D5 Idempotency:** Re-running `45_validation_synthesis.py` produces byte-identical statistics (verified SHA-256 hash match across runs). The spherical baseline is cached to `output/isotropic_baseline_metrics.json` and reused unless `--force` is passed.
* **Validation:** All three phase metric JSONs pass schema verification and range checks (8/8 patients PASS per phase).
* **Audit:** Full environment capture (`MONTH10_AUDIT.md`) includes Python version, library versions, file sizes, pixel dimensions, validation table, and `__pycache__` cleanup count.

---

## Citation

If you use this pipeline in your work, please cite:

```bibtex
@software{gbm_pde_cohort_2026,
  title = {Multi-Omic Anisotropic PDE Modeling & Adaptive Therapy Dynamics in Glioblastoma Cohorts},
  author = {Vihan et al.},
  year = {2026},
  note = {10-month computational pipeline: Months 1–6 multi-omic → Months 7–10 spatial PDE + adaptive therapy}
}
```

---

## License

MIT License — see `LICENSE` for details.