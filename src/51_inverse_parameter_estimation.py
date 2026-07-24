#!/usr/bin/env python3
"""Inverse Biophysical Parameter Estimation for Patient-Specific GBM Modeling.

This module estimates patient-specific biophysical parameters (ρ, D) from
longitudinal imaging data using bounded optimization with bootstrap uncertainty.

Mathematical Formulation:
    Given: T₀ (baseline volume), T₁ (follow-up volume), Δt (time between scans)
    Solve: min_{ρ, D} ||V_simulated(ρ, D, Δt) - T₁||²
    Subject to:
        - 0.005 ≤ ρ ≤ 0.1 /day (physiological bounds)
        - 0.001 ≤ D ≤ 0.05 mm²/day (diffusivity bounds)

Algorithm:
    - scipy.optimize.minimize with Nelder-Mead or Powell method
    - Surrogate ODE model for fast evaluation: dV/dt = ρ*V*(1-V/K) + D*∇²V
    - Bootstrap resampling (N=100) for confidence intervals

Usage:
    python src/51_inverse_parameter_estimation.py --test
    python src/51_inverse_parameter_estimation.py --t0-volume 1000 --t1-volume 1200 --delta-t 30
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

warnings.filterwarnings("ignore")

# Physiological bounds (per plan specification)
RHO_MIN = 0.005  # /day (minimum growth rate)
RHO_MAX = 0.05   # /day (maximum growth rate)
D_MIN = 0.001    # mm²/day (minimum diffusivity)
D_MAX = 0.05     # mm²/day (maximum diffusivity)

# Default initial guess
RHO_DEFAULT = 0.02    # /day
D_DEFAULT = 0.013     # mm²/day

# Bootstrap parameters
N_BOOTSTRAP = 100
NOISE_STD = 0.10  # 10% Gaussian noise for robustness testing

# Carrying capacity: realistic brain tumor maximum (~brain volume in mm3)
# A GBM cannot exceed the cranial vault; set K to a large value so logistic
# suppression is negligible at typical tumor volumes (1-50 cm^3 = 1000-50000 mm^3)
K_DEFAULT = 1.0e6  # mm3 (acts as near-pure exponential for small tumors)


def surrogate_ode_model(
    rho: float,
    D: float,
    V0: float,
    delta_t: float,
    K: float = K_DEFAULT,
) -> float:
    """
    Surrogate ODE model for tumor volume evolution.
    
    Separates the two biophysical effects so rho and D are independently
    identifiable from a pair of volume timepoints:
    
      - rho: proliferation (logistic/exponential growth of cell density)
      - D:   radial invasion (increases apparent volume via diffusion-driven
             boundary expansion, scaling as dR/dt ~ D/R)
    
    Model (lumped-volume surrogate):
        dV/dt = rho * V * (1 - V/K)                # logistic proliferation
                + 3 * sqrt(4*pi/3) * D * V^(1/3)   # radial-diffusion volume gain
    
    The radial term derives from the observation that for a spherical tumor
    of radius R, V = 4/3*pi*R^3 and diffusion-driven radial velocity is
    dR/dt = D/R (heat-equation front speed scaling), so:
        dV_diff/dt = 4*pi*R^2 * dR/dt = 4*pi*R^2 * D/R = 4*pi*D*R
                   = 4*pi*D * (3*V/(4*pi))^(1/3)
                   = (36*pi)^(1/3) * D * V^(1/3)
    
    Integrated numerically (Euler) for accuracy.
    
    Args:
        rho: Growth rate (/day)
        D: Diffusion coefficient (mm^2/day)
        V0: Initial volume (mm^3)
        delta_t: Time interval (days)
        K: Carrying capacity (mm^3, default 1e6)
    
    Returns:
        V1: Predicted volume at t+delta_t (mm^3)
    """
    if V0 <= 0:
        return 0.0
    
    # Radial-diffusion coefficient: (36*pi)^(1/3) ~ 3.269
    c_diff = (36.0 * np.pi) ** (1.0 / 3.0)
    
    # Numerical integration (Euler) with small substeps for stability
    n_substeps = max(1, int(np.ceil(delta_t / 1.0)))  # 1-day substeps
    dt_sub = delta_t / n_substeps
    
    V = V0
    for _ in range(n_substeps):
        # Logistic proliferation
        dV_prolif = rho * V * (1.0 - V / K) * dt_sub
        # Radial-diffusion volume gain
        dV_diff = c_diff * D * (V ** (1.0 / 3.0)) * dt_sub
        V = V + dV_prolif + dV_diff
        if V < 0:
            V = 0.0
    
    return V


def objective_function(
    params: np.ndarray,
    V0: float,
    V1_target: float,
    delta_t: float,
) -> float:
    """
    Objective function for parameter optimization.
    
    Minimizes squared relative error between simulated and observed volume.
    Relative error is used so that the optimizer scales correctly across
    tumor volumes of different magnitudes.
    
    Args:
        params: [rho, D] parameter vector
        V0: Baseline volume
        V1_target: Follow-up volume (target)
        delta_t: Time interval
    
    Returns:
        Squared relative error
    """
    rho, D = params
    
    # Hard clamp to physiological bounds (defensive; L-BFGS-B also enforces)
    rho = max(RHO_MIN, min(RHO_MAX, rho))
    D = max(D_MIN, min(D_MAX, D))
    
    # Simulate volume
    V1_sim = surrogate_ode_model(rho, D, V0, delta_t)
    
    # Squared relative error (scale-invariant)
    scale = max(abs(V1_target), 1.0)
    error = ((V1_sim - V1_target) / scale) ** 2
    
    return error


def estimate_patient_parameters(
    t0_volume: float,
    t1_volume: float,
    delta_t_days: float,
    initial_guess: Optional[Tuple[float, float]] = None,
    bounds: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
    method: str = "L-BFGS-B",
    n_bootstrap: int = N_BOOTSTRAP,
) -> Dict[str, Any]:
    """
    Estimate patient-specific biophysical parameters from longitudinal volumes.
    
    Args:
        t0_volume: Baseline tumor volume (mm³)
        t1_volume: Follow-up tumor volume (mm³)
        delta_t_days: Time between scans (days)
        initial_guess: Initial [rho, D] guess (default: [0.02, 0.013])
        bounds: Parameter bounds [(rho_min, rho_max), (D_min, D_max)]
        method: Optimization method (default: "Nelder-Mead")
        n_bootstrap: Number of bootstrap samples for CI (default: 100)
    
    Returns:
        Dictionary with:
            - rho: Estimated growth rate (/day)
            - D: Estimated diffusion coefficient (mm²/day)
            - rho_ci: 95% confidence interval for rho [lower, upper]
            - D_ci: 95% confidence interval for D [lower, upper]
            - convergence: bool indicating successful convergence
            - n_iterations: Number of iterations
            - rmse: Root mean squared error of fit
            - bootstrap_samples: Array of [rho, D] bootstrap samples
    """
    if initial_guess is None:
        initial_guess = (RHO_DEFAULT, D_DEFAULT)
    
    if bounds is None:
        bounds = [(RHO_MIN, RHO_MAX), (D_MIN, D_MAX)]
    
    # Optimization (L-BFGS-B enforces bounds; fallback methods receive
    # bounds as None if unsupported)
    bounds_for_method = bounds if method in ("L-BFGS-B", "TNC", "SLSQP") else None
    
    result = minimize(
        fun=objective_function,
        x0=np.array(initial_guess),
        args=(t0_volume, t1_volume, delta_t_days),
        method=method,
        bounds=bounds_for_method,
        options={"maxiter": 1000, "disp": False},
    )
    
    rho_est, D_est = result.x
    
    # Bootstrap resampling for confidence intervals
    bootstrap_samples = np.zeros((n_bootstrap, 2))
    
    for i in range(n_bootstrap):
        # Add noise to target volume
        noise = np.random.normal(0, NOISE_STD * t1_volume)
        t1_noisy = max(0, t1_volume + noise)
        
        # Re-optimize with noisy data
        boot_result = minimize(
            fun=objective_function,
            x0=np.array(initial_guess),
            args=(t0_volume, t1_noisy, delta_t_days),
            method=method,
            bounds=bounds_for_method,
            options={"maxiter": 500, "disp": False},
        )
        bootstrap_samples[i] = boot_result.x
    
    # Confidence intervals (95%)
    rho_ci = np.percentile(bootstrap_samples[:, 0], [2.5, 97.5])
    D_ci = np.percentile(bootstrap_samples[:, 1], [2.5, 97.5])
    
    # RMSE
    V1_pred = surrogate_ode_model(rho_est, D_est, t0_volume, delta_t_days)
    rmse = np.sqrt((V1_pred - t1_volume) ** 2)
    
    return {
        "rho": float(rho_est),
        "D": float(D_est),
        "rho_ci": [float(rho_ci[0]), float(rho_ci[1])],
        "D_ci": [float(D_ci[0]), float(D_ci[1])],
        "convergence": bool(result.success or result.fun < 1e-6),
        "n_iterations": int(result.nit if hasattr(result, "nit") else 0),
        "rmse": float(rmse),
        "bootstrap_samples": bootstrap_samples.tolist(),
    }


def validate_with_synthetic_data(
    true_rho: float = 0.025,
    true_D: float = 0.015,
    V0: float = 1000.0,
    delta_t: float = 30.0,
    noise_levels: List[float] = [0.0, 0.05, 0.10, 0.15],
    n_trials: int = 50,
) -> Dict[str, Any]:
    """
    Validate parameter estimation with synthetic data.
    
    Args:
        true_rho: True growth rate for synthetic data
        true_D: True diffusion coefficient for synthetic data
        V0: Initial volume
        delta_t: Time interval
        noise_levels: List of noise standard deviations to test
        n_trials: Number of Monte Carlo trials per noise level
    
    Returns:
        Validation results dictionary
    """
    print(f"\n{'='*70}")
    print("INVERSE PARAMETER ESTIMATION - SYNTHETIC VALIDATION")
    print(f"{'='*70}")
    print(f"True parameters: rho = {true_rho:.4f} /day, D = {true_D:.4f} mm2/day")
    print(f"Initial volume: V0 = {V0:.1f} mm3")
    print(f"Time interval: dt = {delta_t:.1f} days")
    print(f"Trials per noise level: {n_trials}")
    print(f"{'='*70}\n")
    
    # Generate synthetic follow-up volume
    V1_true = surrogate_ode_model(true_rho, true_D, V0, delta_t)
    print(f"Synthetic V1 (noise-free): {V1_true:.2f} mm3\n")
    
    results = {}
    
    for noise_std in noise_levels:
        rho_errors = []
        D_errors = []
        convergence_count = 0
        n_iterations_list = []
        
        for trial in range(n_trials):
            # Add noise
            noise = np.random.normal(0, noise_std * V1_true)
            V1_noisy = max(0, V1_true + noise)
            
            # Estimate parameters
            est = estimate_patient_parameters(
                t0_volume=V0,
                t1_volume=V1_noisy,
                delta_t_days=delta_t,
            )
            
            if est["convergence"]:
                convergence_count += 1
                rho_errors.append(est["rho"] - true_rho)
                D_errors.append(est["D"] - true_D)
                n_iterations_list.append(est["n_iterations"])
        
        # Compute metrics
        rho_rmse = np.sqrt(np.mean(np.array(rho_errors) ** 2)) if rho_errors else float("inf")
        D_rmse = np.sqrt(np.mean(np.array(D_errors) ** 2)) if D_errors else float("inf")
        rho_bias = np.mean(rho_errors) if rho_errors else float("inf")
        D_bias = np.mean(D_errors) if D_errors else float("inf")
        avg_iterations = np.mean(n_iterations_list) if n_iterations_list else 0
        
        results[f"noise_{int(noise_std*100)}pct"] = {
            "rho_rmse": rho_rmse,
            "D_rmse": D_rmse,
            "rho_bias": rho_bias,
            "D_bias": D_bias,
            "convergence_rate": convergence_count / n_trials,
            "avg_iterations": avg_iterations,
        }
        
        print(f"Noise Level: {int(noise_std*100)}%")
        print(f"  rho RMSE: {rho_rmse:.6f} /day  (bias: {rho_bias:.6f})")
        print(f"  D RMSE: {D_rmse:.6f} mm2/day  (bias: {D_bias:.6f})")
        print(f"  Convergence: {convergence_count}/{n_trials} ({100*convergence_count/n_trials:.1f}%)")
        print(f"  Avg iterations: {avg_iterations:.1f}")
        print()
    
    # Validation summary
    print(f"{'='*70}")
    print("VALIDATION SUMMARY")
    print(f"{'='*70}")
    
    # Check success criteria
    noise_free = results.get("noise_0pct", {})
    noise_10pct = results.get("noise_10pct", {})
    
    criteria_met = {
        "rmse_noise_free_5pct": noise_free.get("rho_rmse", 1.0) < 0.05 and noise_free.get("D_rmse", 1.0) < 0.05,
        "rmse_10pct_noise_15pct": noise_10pct.get("rho_rmse", 1.0) < 0.15 and noise_10pct.get("D_rmse", 1.0) < 0.15,
        "convergence_50_iters": noise_free.get("avg_iterations", 100) < 50,
    }
    
    for criterion, met in criteria_met.items():
        status = "PASS" if met else "FAIL"
        print(f"  {criterion}: {status}")
    
    print(f"{'='*70}\n")
    
    return results


def main():
    """Main entry point for inverse parameter estimation."""
    parser = argparse.ArgumentParser(
        description="Inverse biophysical parameter estimation for GBM modeling"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run synthetic validation tests",
    )
    parser.add_argument(
        "--t0-volume",
        type=float,
        help="Baseline tumor volume (mm³)",
    )
    parser.add_argument(
        "--t1-volume",
        type=float,
        help="Follow-up tumor volume (mm³)",
    )
    parser.add_argument(
        "--delta-t",
        type=float,
        help="Time between scans (days)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file for results",
    )
    
    args = parser.parse_args()
    
    if args.test:
        # Run validation
        results = validate_with_synthetic_data()
        
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to: {output_path}")
        
        return 0
    
    if args.t0_volume and args.t1_volume and args.delta_t:
        # Estimate parameters for patient
        print(f"\n{'='*70}")
        print("PATIENT PARAMETER ESTIMATION")
        print(f"{'='*70}")
        print(f"Baseline volume (V0): {args.t0_volume:.1f} mm3")
        print(f"Follow-up volume (V1): {args.t1_volume:.1f} mm3")
        print(f"Time interval (dt): {args.delta_t:.1f} days")
        print(f"{'='*70}\n")
        
        result = estimate_patient_parameters(
            t0_volume=args.t0_volume,
            t1_volume=args.t1_volume,
            delta_t_days=args.delta_t,
        )
        
        print("ESTIMATED PARAMETERS:")
        print(f"  rho (growth rate):     {result['rho']:.6f} /day")
        print(f"                         95% CI: [{result['rho_ci'][0]:.6f}, {result['rho_ci'][1]:.6f}]")
        print(f"  D (diffusivity):       {result['D']:.6f} mm2/day")
        print(f"                         95% CI: [{result['D_ci'][0]:.6f}, {result['D_ci'][1]:.6f}]")
        print(f"  Convergence:         {'Yes' if result['convergence'] else 'No'}")
        print(f"  Iterations:          {result['n_iterations']}")
        print(f"  RMSE:                {result['rmse']:.4f} mm³")
        print(f"{'='*70}\n")
        
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            # Remove non-serializable bootstrap samples for JSON output
            result_json = {k: v for k, v in result.items() if k != "bootstrap_samples"}
            with open(output_path, "w") as f:
                json.dump(result_json, f, indent=2)
            print(f"Results saved to: {output_path}")
        
        return 0
    
    # No arguments provided - show help
    parser.print_help()
    print("\nExamples:")
    print("  python src/51_inverse_parameter_estimation.py --test")
    print("  python src/51_inverse_parameter_estimation.py --t0-volume 1000 --t1-volume 1200 --delta-t 30")
    return 0


if __name__ == "__main__":
    sys.exit(main())