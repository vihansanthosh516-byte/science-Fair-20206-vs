"""Phase 2b: Global Sobol Sensitivity Analysis.

Reduced-ODE evaluator + SALib Sobol indices + publication-ready tornado plot.
N=500 base samples → ~3500 model evaluations.
"""
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from SALib.analyze import sobol as sobol_analyze
from SALib.sample import sobol as sobol_sample

warnings.filterwarnings("ignore")

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

# Physical constants (Phase 1, mm / days / ug/mL)
DT = 0.1
SIM_DAYS = 360
N_STEPS = int(SIM_DAYS / DT)

RHO_R_BASE = 0.015
K = 1.0
K_EL = np.log(2) / 0.075   # ~9.24 /day
C_PEAK = 10.0
HILL_COEFF = 2.0
E_MAX = 0.35              # reduced: drug suppresses but does not overwhelm growth

# TTP threshold: total cell mass fraction
TTP_FRACTION = 0.4

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


def reduced_ode(params: np.ndarray) -> float:
    """Reduced spatially-averaged ODE returning TTP (days).

    All 5 Sobol parameters enter the dynamics:
        rho_s       -> sensitive proliferation rate (/day, direct)
        aniso_ratio -> amplifies effective growth (anisotropic invasion)
        mu_r        -> resistant clone emergence rate
        EC50        -> TMZ drug sensitivity (ug/mL)
        D_white     -> white matter diffusivity (mm^2/day) -> effective growth

    eff_rho_s = rho_s * (1 + k_diff * D_white) * (1 + k_aniso * (aniso_ratio - 1))

    ODE system:
        dM_s/dt = eff_rho_s * M_s * (1 - M_s - M_r) - gamma(C) * M_s
        dM_r/dt = rho_r * M_r * (1 - M_s - M_r) + mu * eff_rho_s * M_s
        dC/dt   = -k_el * C  (bolus, reset on dose days)

    TTP = first time total mass >= TTP_FRACTION of carrying capacity.
    """
    rho_s, aniso_ratio, mu_r, ec50, d_white = params
    rho_r = RHO_R_BASE
    mu = mu_r

    # Effective sensitive growth: diffusion + anisotropy amplify invasion
    k_diff = 15.0       # D_white coupling (mm^2/day -> day^-1)
    k_aniso = 0.2       # aniso_ratio coupling
    eff_rho_s = rho_s * (1.0 + k_diff * d_white) * (1.0 + k_aniso * (aniso_ratio - 1.0))

    M_s = 0.05
    M_r = 1e-4
    C = 0.0

    days_on = 5
    cycle_days = 28

    for step in range(N_STEPS):
        t = step * DT
        day_in_cycle = int(t) % cycle_days

        if day_in_cycle < days_on:
            C = C_PEAK
        else:
            C *= np.exp(-K_EL * DT)

        kill = E_MAX * (C ** HILL_COEFF) / (ec50 ** HILL_COEFF + C ** HILL_COEFF + 1e-12)

        total = M_s + M_r
        dMs = (eff_rho_s * M_s * (1.0 - total) - kill * M_s) * DT
        # Resistant clone: emerges from sensitive mutation, grows with own rate.
        # mu scaled up so mutation rate range produces meaningful resistant pop.
        dMr = (rho_r * M_r * (1.0 - total) + mu * 5e4 * eff_rho_s * M_s) * DT

        M_s = max(M_s + dMs, 0.0)
        M_r = max(M_r + dMr, 0.0)

        if M_s + M_r >= TTP_FRACTION:
            return t + DT

    return float(SIM_DAYS)


def run_simulation_and_get_ttp(params: np.ndarray) -> float:
    return reduced_ode(params)


def plot_tornado(Si: dict, path: Path) -> None:
    """Publication-ready tornado plot: horizontal bars of ST sorted descending.

    Si values are numpy arrays indexed by parameter position; we map them
    to names via `problem['names']`.
    """
    names = list(problem["names"])
    st_vals = np.asarray(Si["ST"])
    s1_vals = np.asarray(Si["S1"])
    st_conf = np.asarray(Si["ST_conf"])

    order = np.argsort(st_vals)[::-1]
    names_sorted = [names[i] for i in order]
    st_sorted = st_vals[order]
    s1_sorted = s1_vals[order]
    conf_sorted = st_conf[order]

    y = np.arange(len(names_sorted))
    fig, ax = plt.subplots(figsize=(8, 4.5))

    ax.barh(y, st_sorted, xerr=conf_sorted, color="steelblue", alpha=0.85,
            label="Total-effect ST", capsize=3, height=0.5)
    ax.scatter(s1_sorted, y, color="crimson", s=50, zorder=5,
               label="First-order S1")

    ax.set_yticks(y)
    ax.set_yticklabels(names_sorted, fontsize=10)
    ax.set_xlabel("Sobol index", fontsize=11)
    ax.set_title("Sobol Sensitivity: TTP variance decomposition", fontsize=12,
                 fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    ax.set_xlim(-0.05, 1.05)
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"[Tornado] Saved -> {path}")


def main() -> None:
    print("=" * 70)
    print("Phase 2b: Global Sobol Sensitivity Analysis")
    print("=" * 70)
    np.random.seed(42)

    N = 500
    n_evals = N * (2 * problem["num_vars"] + 2)
    print(f"Generating {N} Saltelli samples ({n_evals} evaluations)...")
    param_values = sobol_sample.sample(problem, N)
    print(f"  -> {len(param_values)} parameter combinations")

    print("Running reduced-ODE simulations...")
    Y = np.array([run_simulation_and_get_ttp(p) for p in param_values])
    print(f"  TTP range: [{Y.min():.1f}, {Y.max():.1f}] days, "
          f"mean={Y.mean():.1f} +/- {Y.std():.1f}")

    print("Computing Sobol indices...")
    Si = sobol_analyze.analyze(problem, Y, print_to_console=False)

    results = {
        "S1": {name: float(Si["S1"][i]) for i, name in enumerate(problem["names"])},
        "ST": {name: float(Si["ST"][i]) for i, name in enumerate(problem["names"])},
        "S1_conf": {name: float(Si["S1_conf"][i]) for i, name in enumerate(problem["names"])},
        "ST_conf": {name: float(Si["ST_conf"][i]) for i, name in enumerate(problem["names"])},
        "n_samples": N,
        "ttp_mean": float(Y.mean()),
        "ttp_std": float(Y.std()),
    }

    out_path = OUTPUT_DIR / "sobol_sensitivity_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved -> {out_path}")

    plot_tornado(Si, OUTPUT_DIR / "sobol_tornado_plot.png")

    print("\nPhase 2b complete.")


if __name__ == "__main__":
    main()