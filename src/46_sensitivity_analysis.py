"""Phase 2b: Global Sobol Sensitivity Analysis (scaffold).

Performs Sobol sensitivity analysis on key PDE parameters using SALib.
Parameter space: ±20% around Phase 1 baseline values.
N=500 base samples → ~3500 model evaluations.

Usage:
    python src/46_sensitivity_analysis.py

Output:
    output/sobol_sensitivity_results.json
    output/sobol_tornado_plot.png
"""

import json
import warnings
from pathlib import Path

import numpy as np
from SALib.analyze import sobol
from SALib.sample import saltelli

warnings.filterwarnings("ignore")

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

# --------------------------------------------------------------------------- #
# Phase 1 baseline parameters (physical units: mm, days, ug/mL)
# --------------------------------------------------------------------------- #
problem = {
    "num_vars": 5,
    "names": ["rho_s", "aniso_ratio", "mu_r", "EC50", "D_white"],
    "bounds": [
        [0.016, 0.024],   # rho_s: proliferation rate (/day)
        [8.0, 12.0],      # aniso_ratio: D_parallel / D_perp
        [8e-6, 1.2e-5],   # mu_r: mutation rate (per division)
        [4.0, 6.0],       # EC50: TMZ half-max kill (ug/mL)
        [0.0104, 0.0156], # D_white: white matter diffusivity (mm^2/day)
    ],
}


def run_simulation_and_get_ttp(params: np.ndarray) -> float:
    """Run a reduced PDE simulation and return time-to-progression (TTP).

    Placeholder: returns a synthetic TTP value for scaffold testing.
    Replace with actual ODE/PDE integration for full Sobol analysis.
    """
    rho_s, aniso_ratio, mu_r, ec50, d_white = params
    ttp = 2000.0 - 500.0 * rho_s / 0.02 + 200.0 * (ec50 - 5.0) / 5.0
    ttp += np.random.default_rng(seed=None).normal(0, 50)
    return max(ttp, 100.0)


def main() -> None:
    print("=" * 70)
    print("Phase 2b: Global Sobol Sensitivity Analysis")
    print("=" * 70)

    N = 500
    print(f"Generating {N} Saltelli samples ({N * (2 * problem['num_vars'] + 2)} evaluations)...")
    param_values = saltelli.sample(problem, N)
    print(f"  -> {len(param_values)} parameter combinations")

    print("Running simulations...")
    Y = np.array([run_simulation_and_get_ttp(p) for p in param_values])

    print("Computing Sobol indices...")
    Si = sobol.analyze(problem, Y, print_to_console=True)

    results = {
        "S1": {name: float(Si["S1"][i]) for i, name in enumerate(problem["names"])},
        "ST": {name: float(Si["ST"][i]) for i, name in enumerate(problem["names"])},
        "S1_conf": {name: float(Si["S1_conf"][i]) for i, name in enumerate(problem["names"])},
        "ST_conf": {name: float(Si["ST_conf"][i]) for i, name in enumerate(problem["names"])},
    }

    out_path = OUTPUT_DIR / "sobol_sensitivity_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved -> {out_path}")

    print("\nPhase 2b scaffold complete.")


if __name__ == "__main__":
    main()