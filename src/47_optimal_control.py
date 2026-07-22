"""Phase 3: MPC Optimal Control & Dual-Drug Protocol.

Full implementation:
- Reduced ODE: dM_s/dt, dM_r/dt, dC/dt, dC2/dt
- MPC: scipy.optimize.minimize over 14-day horizon, receding
- Dual-drug: stromal inhibitor during TMZ holidays
- 3-arm benchmark: MTD, Single-agent adaptive, Dual-agent adaptive
- All 8 patients (different initial conditions)
"""
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

warnings.filterwarnings("ignore")

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

# --------------------------------------------------------------------------- #
# Phase 1 physical constants
# --------------------------------------------------------------------------- #
RHO_S = 0.02
RHO_R = 0.015
K = 1.0
K_EL = np.log(2) / 0.075    # ~9.24 /day, TMZ
K_EL2 = np.log(2) / 0.15    # stromal inhibitor (~2x longer half-life)
C_PEAK = 10.0               # ug/mL TMZ
C2_PEAK = 5.0               # ug/mL stromal inhibitor
EC50 = 5.0                  # TMZ EC50 ug/mL
EC50_2 = 3.0                # stromal inhibitor EC50 ug/mL
HILL_COEFF = 2.0
E_MAX = 0.35
GAMMA_R = 0.4              # secondary drug kill rate on resistant cells
MUTATION_RATE = 1e-5
INFUSION_RATE = C_PEAK

# MPC
MPC_HORIZON_DAYS = 14
DT = 1.0                     # 1-day steps for MPC (ODE integrated daily)
SIM_DAYS = 360
N_SIM_STEPS = int(SIM_DAYS / DT)
W_TUMOR = 1.0
W_DRUG_BASELINE = 0.1


# --------------------------------------------------------------------------- #
# Reduced ODE (1-day step)
# --------------------------------------------------------------------------- #
def step_reduced_ode(state: np.ndarray, a: float, a2: float,
                      dt: float = DT) -> np.ndarray:
    """Single-day reduced ODE step.

    state = [M_s, M_r, C, C2]
    a in [0,1]: TMZ dose rate (fraction of C_PEAK bolus)
    a2 in [0,1]: stromal inhibitor dose rate

    dM_s/dt = rho_s * M_s * (1 - M_s - M_r) - kill_tmz * M_s
    dM_r/dt = rho_r * M_r * (1 - M_s - M_r) + mu * rho_s * M_s - kill_stromal * M_r
    dC/dt   = -k_el * C + a * infusion_rate
    dC2/dt  = -k_el2 * C2 + a2 * C2_peak
    """
    M_s, M_r, C, C2 = state
    total = M_s + M_r

    kill_tmz = E_MAX * (C ** HILL_COEFF) / (EC50 ** HILL_COEFF + C ** HILL_COEFF + 1e-12)
    # Secondary drug only kills resistant during TMZ holidays
    kill_stromal = 0.0 if a >= 0.01 else GAMMA_R * (C2 ** HILL_COEFF) / (
        EC50_2 ** HILL_COEFF + C2 ** HILL_COEFF + 1e-12)

    dMs = (RHO_S * M_s * (1.0 - total) - kill_tmz * M_s) * dt
    dMr = (RHO_R * M_r * (1.0 - total) + MUTATION_RATE * RHO_S * M_s * 1e4
           - kill_stromal * M_r) * dt
    dC = (-K_EL * C + a * INFUSION_RATE) * dt
    dC2 = (-K_EL2 * C2 + a2 * C2_PEAK) * dt

    new = np.array(state) + np.array([dMs, dMr, dC, dC2])
    return np.maximum(new, 0.0)


# --------------------------------------------------------------------------- #
# MPC cost function: simulate 14-day horizon with control sequence
# --------------------------------------------------------------------------- #
def mpc_cost(control_seq: np.ndarray, state0: np.ndarray,
              w_tumor: float, w_drug: float) -> float:
    """Cost = sum over horizon of [w_tumor * (M_s+M_r) + w_drug * a(t)^2]."""
    state = state0.copy()
    cost = 0.0
    for i in range(len(control_seq)):
        a = float(np.clip(control_seq[i], 0.0, 1.0))
        state = step_reduced_ode(state, a, 0.0)
        cost += w_tumor * (state[0] + state[1]) + w_drug * a * a
    return float(cost)


def solve_mpc_horizon(state0: np.ndarray, w_tumor: float, w_drug: float
                      ) -> np.ndarray:
    """Solve 14-day MPC horizon, return optimal daily controls [0,1]^14."""
    res = minimize(
        mpc_cost,
        x0=np.full(MPC_HORIZON_DAYS, 0.5),
        args=(state0, w_tumor, w_drug),
        method="L-BFGS-B",
        bounds=[(0.0, 1.0)] * MPC_HORIZON_DAYS,
        options={"maxiter": 200, "ftol": 1e-6},
    )
    return np.clip(res.x, 0.0, 1.0)


# --------------------------------------------------------------------------- #
# Arm 1: MTD (continuous TMZ 5-on/23-off)
# --------------------------------------------------------------------------- #
def run_mtd(state0: np.ndarray) -> dict:
    state = state0.copy()
    n = N_SIM_STEPS
    Ms, Mr, C, a_hist = (np.zeros(n) for _ in range(4))
    for step in range(n):
        t = step * DT
        day_in_cycle = int(t) % 28
        a = 1.0 if day_in_cycle < 5 else 0.0
        state = step_reduced_ode(state, a, 0.0)
        Ms[step], Mr[step], C[step], a_hist[step] = state[0], state[1], state[2], a
    return {"M_s": Ms, "M_r": Mr, "C": C, "a": a_hist, "a2": np.zeros(n)}


# --------------------------------------------------------------------------- #
# Arm 2: Single-agent MPC adaptive (TMZ only)
# --------------------------------------------------------------------------- #
def run_single_adaptive(state0: np.ndarray, w_tumor: float = W_TUMOR,
                         w_drug: float = W_DRUG_BASELINE) -> dict:
    state = state0.copy()
    n = N_SIM_STEPS
    Ms, Mr, C, a_hist = (np.zeros(n) for _ in range(4))
    daily_controls = None
    ctrl_idx = 0
    for step in range(n):
        if step % MPC_HORIZON_DAYS == 0:
            daily_controls = solve_mpc_horizon(state, w_tumor, w_drug)
            ctrl_idx = 0
        a = float(daily_controls[ctrl_idx])
        state = step_reduced_ode(state, a, 0.0)
        Ms[step], Mr[step], C[step], a_hist[step] = state[0], state[1], state[2], a
        ctrl_idx += 1
    return {"M_s": Ms, "M_r": Mr, "C": C, "a": a_hist, "a2": np.zeros(n)}


# --------------------------------------------------------------------------- #
# Arm 3: Dual-agent MPC adaptive (TMZ + stromal inhibitor on holidays)
# --------------------------------------------------------------------------- #
def run_dual_adaptive(state0: np.ndarray, w_tumor: float = W_TUMOR,
                       w_drug: float = W_DRUG_BASELINE) -> dict:
    state = state0.copy()
    n = N_SIM_STEPS
    Ms, Mr, C, a_hist, a2_hist = (np.zeros(n) for _ in range(5))
    daily_controls = None
    ctrl_idx = 0
    for step in range(n):
        if step % MPC_HORIZON_DAYS == 0:
            daily_controls = solve_mpc_horizon(state, w_tumor, w_drug)
            ctrl_idx = 0
        a = float(daily_controls[ctrl_idx])
        a2 = 1.0 if a < 0.01 else 0.0
        state = step_reduced_ode(state, a, a2)
        Ms[step], Mr[step], C[step] = state[0], state[1], state[2]
        a_hist[step], a2_hist[step] = a, a2
        ctrl_idx += 1
    return {"M_s": Ms, "M_r": Mr, "C": C, "a": a_hist, "a2": a2_hist}


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def compute_ttp(Ms: np.ndarray, Mr: np.ndarray, threshold: float = 0.4) -> float:
    total = Ms + Mr
    for step in range(len(total)):
        if total[step] >= threshold * K:
            return float(step * DT)
    return float(SIM_DAYS)


def compute_auc(C: np.ndarray) -> float:
    return float(np.trapezoid(C, dx=DT))


def resistant_fraction(Ms: np.ndarray, Mr: np.ndarray) -> float:
    return float(Mr[-1] / (Ms[-1] + Mr[-1] + 1e-12))


# --------------------------------------------------------------------------- #
# Patient initial conditions (8 patients, fixed seed for reproducibility)
# --------------------------------------------------------------------------- #
def get_patient_state(pid: int) -> np.ndarray:
    rng = np.random.default_rng(42 + pid)
    M_s0 = 0.15 + 0.10 * rng.random()
    M_r0 = 1e-4 * (1 + rng.random())
    return np.array([M_s0, M_r0, 0.0, 0.0])


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    print("=" * 70)
    print("Phase 3: MPC Optimal Control & Dual-Drug Protocol")
    print("=" * 70)
    print(f"MPC horizon: {MPC_HORIZON_DAYS} days | "
          f"baseline w_tumor={W_TUMOR}, w_drug={W_DRUG_BASELINE}")
    print(f"Trial horizon: {SIM_DAYS} days\n")

    patient_ids = [f"PAT_{i:04d}" for i in range(8)]
    all_results = []

    for pid_str in patient_ids:
        pid_num = int(pid_str.split("_")[1])
        print(f"--- {pid_str} ---")
        state0 = get_patient_state(pid_num)

        print("  Arm 1: MTD (continuous TMZ 5/23)...")
        r_mtd = run_mtd(state0)

        print("  Arm 2: Single-agent MPC adaptive...")
        r_sa = run_single_adaptive(state0)

        print("  Arm 3: Dual-agent MPC adaptive...")
        r_da = run_dual_adaptive(state0)

        ttp_m = compute_ttp(r_mtd["M_s"], r_mtd["M_r"])
        ttp_s = compute_ttp(r_sa["M_s"], r_sa["M_r"])
        ttp_d = compute_ttp(r_da["M_s"], r_da["M_r"])
        auc_m = compute_auc(r_mtd["C"])
        auc_s = compute_auc(r_sa["C"])
        auc_d = compute_auc(r_da["C"])
        rf_m = resistant_fraction(r_mtd["M_s"], r_mtd["M_r"])
        rf_s = resistant_fraction(r_sa["M_s"], r_sa["M_r"])
        rf_d = resistant_fraction(r_da["M_s"], r_da["M_r"])

        all_results.append({
            "patient_id": pid_str,
            "mtd": {"ttp_days": ttp_m, "auc": auc_m, "resistant_fraction": rf_m},
            "single_adaptive": {"ttp_days": ttp_s, "auc": auc_s, "resistant_fraction": rf_s},
            "dual_adaptive": {"ttp_days": ttp_d, "auc": auc_d, "resistant_fraction": rf_d},
        })
        print(f"    MTD    : TTP={ttp_m:6.1f}d  AUC={auc_m:8.1f}  Rfrac={rf_m:.3f}")
        print(f"    Single : TTP={ttp_s:6.1f}d  AUC={auc_s:8.1f}  Rfrac={rf_s:.3f}")
        print(f"    Dual   : TTP={ttp_d:6.1f}d  AUC={auc_d:8.1f}  Rfrac={rf_d:.3f}\n")

    # Save JSON
    out = {
        "parameters": {
            "MPC_HORIZON_DAYS": MPC_HORIZON_DAYS,
            "W_TUMOR": W_TUMOR,
            "W_DRUG_BASELINE": W_DRUG_BASELINE,
            "SIM_DAYS": SIM_DAYS,
            "DT": DT,
        },
        "patients": all_results,
    }
    json_path = OUTPUT_DIR / "dual_drug_comparison.json"
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Results saved -> {json_path}")

    print("\n" + "=" * 70)
    print("3-Arm Benchmark Summary")
    print("=" * 70)
    hdr = (f"{'Patient':<10} {'MTD TTP':>8} {'SA TTP':>8} {'DA TTP':>8} "
           f"{'MTD AUC':>10} {'SA AUC':>10} {'DA AUC':>10} "
           f"{'MTD Rf':>7} {'SA Rf':>7} {'DA Rf':>7}")
    print(hdr)
    print("-" * len(hdr))
    for e in all_results:
        m, s, d = e["mtd"], e["single_adaptive"], e["dual_adaptive"]
        print(f"{e['patient_id']:<10} {m['ttp_days']:>8.1f} {s['ttp_days']:>8.1f} "
              f"{d['ttp_days']:>8.1f} {m['auc']:>10.1f} {s['auc']:>10.1f} "
              f"{d['auc']:>10.1f} {m['resistant_fraction']:>7.3f} "
              f"{s['resistant_fraction']:>7.3f} {d['resistant_fraction']:>7.3f}")

    print("\nPhase 3 complete.")


if __name__ == "__main__":
    main()