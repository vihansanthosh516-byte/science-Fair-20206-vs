"""Phase 3: MPC Optimal Control & Dual-Drug Protocol (scaffold).

Implements Model Predictive Control (MPC) with reduced ODE for dual-drug
adaptive therapy optimization. Uses scipy.optimize.minimize with a 14-day
rolling horizon.

Baseline weights: w_tumor=1.0, w_drug=0.1
Pareto sweep: w_drug in {0.01, 0.05, 0.1, 0.2, 0.5}

Usage:
    python src/47_optimal_control.py

Output:
    output/optimal_control_policy.npz
    output/dual_drug_comparison.json
"""

import json
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

# --------------------------------------------------------------------------- #
# Phase 1 physical constants (mm, days, ug/mL)
# --------------------------------------------------------------------------- #
RHO_S = 0.02       # sensitive proliferation rate (/day)
RHO_R = 0.015      # resistant proliferation rate (/day)
K = 1.0            # carrying capacity (normalized)
K_EL = np.log(2) / 0.075  # TMZ elimination rate (~9.24 /day)
C_PEAK = 10.0      # peak TMZ concentration (ug/mL)
EC50 = 5.0         # TMZ half-max kill (ug/mL)
HILL_COEFF = 2.0
E_MAX = 0.8

# MPC parameters
MPC_HORIZON_DAYS = 14
PARETO_W_DRUG_GRID = [0.01, 0.05, 0.1, 0.2, 0.5]
W_TUMOR = 1.0
W_DRUG_BASELINE = 0.1


def main() -> None:
    print("=" * 70)
    print("Phase 3: MPC Optimal Control & Dual-Drug Protocol")
    print("=" * 70)
    print(f"MPC horizon: {MPC_HORIZON_DAYS} days")
    print(f"Baseline weights: w_tumor={W_TUMOR}, w_drug={W_DRUG_BASELINE}")
    print(f"Pareto sweep: w_drug in {PARETO_W_DRUG_GRID}")

    # Placeholder: run reduced ODE with MPC optimization
    # Full implementation will:
    #   1. Solve reduced ODE: dM_s/dt, dM_r/dt, dC/dt
    #   2. For each day, solve finite-horizon optimal control (H=14)
    #   3. Apply first control action, recede horizon
    #   4. Evaluate 3 arms: MTD, Adaptive (single), Adaptive (dual)
    #   5. Compute TTP, AUC, resistant fraction for each arm
    results = {
        "status": "scaffold",
        "notes": "Replace with full MPC optimization loop",
        "params": {
            "MPC_HORIZON_DAYS": MPC_HORIZON_DAYS,
            "W_TUMOR": W_TUMOR,
            "W_DRUG_BASELINE": W_DRUG_BASELINE,
            "PARETO_W_DRUG_GRID": PARETO_W_DRUG_GRID,
        },
    }

    out_path = OUTPUT_DIR / "dual_drug_comparison.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved -> {out_path}")
    print("\nPhase 3 scaffold complete.")


if __name__ == "__main__":
    main()