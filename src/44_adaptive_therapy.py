#!/usr/bin/env python3
"""
Month 9: Advanced Clonal Optimization - Adaptive Therapy Engine
================================================================
Implements dual-clone competitive dynamics with adaptive dosing control.

PDE System:
    du_s/dt = div(D grad u_s) + rho_s * u_s * (1 - u_s - u_r) - drug_kill(C) * u_s + mutation
    du_r/dt = div(D grad u_r) + rho_r * u_r * (1 - u_s - u_r) + mutation

    drug_kill(C) = E_max * C^H / (EC50^H + C^H)  [Hill equation]

Phases:
    Week 1: Dual-clone competitive state space & mutation engine
    Week 2: Pharmacodynamic clearance & MTD simulation (competitive release)
    Week 3: Closed-loop adaptive dosing engine (drug holidays)
    Week 4: Evolutionary timeline analytics & visualization

Deliverables:
    output/adaptive_initial_clones.png
    output/adaptive_therapy_data.npz
    output/adaptive_therapy_comparison.png
    output/adaptive_geometry_metrics.json
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.ndimage import gaussian_filter

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------- #
# Global constants (Phase 1: physical units, mm/days)
#   dx=1.0 mm, dt=0.1 day, D_white=0.013 mm^2/day, D_gray=0.0013,
#   rho_s=0.02 /day, rho_r=0.015 /day (resistance fitness cost).
# --------------------------------------------------------------------------- #
GRID_SIZE = 100
DX_MM = 1.0
DX = DX_MM
TARGET_GENES = ["LST1", "S100A11", "S100A8", "ZNF106"]

ZONE_REGIONS = {
    "Cellular Tumor": (0, 33),
    "Infiltrating Tumor": (33, 66),
    "Leading Edge": (66, 100),
}

# Base anisotropic parameters (from Month 7 / Phase 1 physical units)
D_WHITE = 0.013
D_GRAY = 0.0013
D_BASE = D_GRAY
D_PARALLEL_DEFAULT = D_WHITE
D_PERPENDICULAR_DEFAULT = 0.0013

# Clone-specific proliferation rates (Phase 1, /day)
RHO_SENSITIVE = 0.02       # ~35 day doubling
RHO_RESISTANT = 0.015      # fitness cost of resistance

# Mutation rate (per division)
MUTATION_RATE = 1e-5

# Carrying capacity
K = 1.0

# ----- Phase 1: TMZ pharmacokinetics (1-compartment IV bolus) ----- #
# TMZ half-life: 1.8 hours = 0.075 days (literature)
TMZ_HALF_LIFE_DAYS = 0.075
K_EL = float(np.log(2) / TMZ_HALF_LIFE_DAYS)   # ~9.24 day^-1
# Peak plasma concentration after standard 150-200 mg/m^2 dose (µg/mL)
C_PEAK = 10.0
# TMZ Hill-equation PD (Phase 1): EC50 in µg/mL (literature 2-10)
TMZ_EC50_UG_ML = 5.0
# Drug pharmacodynamics (Hill equation) — kept for legacy compatibility
# NOTE: EC50 below is in abstract units; the Phase-1 path uses TMZ_EC50_UG_ML.
E_MAX = 0.8
EC50 = TMZ_EC50_UG_ML
HILL_COEFF = 2.0

# Drug spatial diffusion - kept small; PK overrides bulk concentration
# (kept for legacy drug_field pathway / mass-conservation sanity plots)
D_DRUG = 0.13
DRUG_DECAY = K_EL

# Standard TMZ dosing schedule: 5 days on, 23 days off (28-day cycle)
TMZ_DOSE_DAYS_ON = 5
TMZ_CYCLE_DAYS = 28

# Dosing
MTD_DOSE = C_PEAK          # µg/mL peak per dose (Phase 1)
ADAPTIVE_DT = 0.1          # days (Phase 1)

# MTD protocol: continuous dosing (abstract steps retained for legacy code;
# under Phase 1 PK, "MTD" still means daily TMZ, C_RESET to C_PEAK each dose day)
MTD_STEPS = 5000

# Adaptive protocol
ADAPTIVE_STEPS = 5000
THRESHOLD_OFF = 0.5  # 50% of baseline -> drug holiday
THRESHOLD_ON = 1.0   # 100% of baseline -> resume drug

# Cohort patients
COHORT_PATIENTS = [f"PAT_{i:04d}" for i in range(8)]

# Save intervals
SAVE_INTERVAL = 250


# =========================================================================== #
# Phase 1: TMZ PK helper functions
# =========================================================================== #
def tmz_concentration(t_since_dose_days: float,
                      c_peak: float = C_PEAK,
                      k_el: float = K_EL) -> float:
    """1-compartment IV bolus concentration at time t (days) after a dose.

    C(t) = C_peak * exp(-k_el * t)
    Returns drug concentration in µg/mL.
    """
    return float(c_peak * np.exp(-k_el * float(t_since_dose_days)))


def tmz_dose_active(step_idx: int,
                    dt_days: float = ADAPTIVE_DT,
                    days_on: int = TMZ_DOSE_DAYS_ON,
                    cycle_days: int = TMZ_CYCLE_DAYS) -> bool:
    """Standard TMZ schedule: days 1-5 ON, 6-28 OFF (28-day cycle).

    Returns True if the simulation step `step_idx` (zero-based, with
    cumulative time = step_idx * dt_days) falls on a dosing day.
    """
    t_days = step_idx * dt_days
    day_in_cycle = int(t_days) % cycle_days
    return day_in_cycle < days_on


def compute_tmz_schedule_C(step_idx: int,
                           dt_days: float = ADAPTIVE_DT,
                           c_peak: float = C_PEAK,
                           k_el: float = K_EL,
                           days_on: int = TMZ_DOSE_DAYS_ON,
                           cycle_days: int = TMZ_CYCLE_DAYS) -> float:
    """Concentration at simulation step assuming daily bolus doses during ON
    days (each ON day resets plasma back to C_peak at the start of that day,
    then decays). Returns µg/mL at the *end* of the current step (i.e. dt_days
    after the most recent dosing event within this day).

    For an ON day, residual decay from any earlier dose the same day is small
    because k_el * dt_days << 1 (TMZ clears within hours); we approximate by
    assuming each ON step starts at C_peak.
    """
    t_days = step_idx * dt_days
    day_in_cycle = int(t_days) % cycle_days
    if day_in_cycle < days_on:
        # On a dosing day: peak reached, then decays over dt_days
        return tmz_concentration(dt_days, c_peak=c_peak, k_el=k_el)
    # Off day: time since last ON-day end
    # day_in_cycle in [days_on, cycle_days)
    days_since_last_dose = day_in_cycle - (days_on - 1) + (t_days - int(t_days))
    return tmz_concentration(days_since_last_dose, c_peak=c_peak, k_el=k_el)


def hill_kill_physical(C_ug_ml: float,
                       E_max: float = E_MAX,
                       EC50: float = TMZ_EC50_UG_ML,
                       H: float = HILL_COEFF) -> float:
    """Hill equation drug efficacy with physical concentration (µg/mL).

    kill = E_max * C^H / (EC50^H + C^H)
    Returns fractional kill rate per day (in [0, E_max]).
    Accepts scalars or numpy arrays.
    """
    C = np.asarray(C_ug_ml, dtype=float)
    return E_max * (C ** H) / (EC50 ** H + C ** H + 1e-12)


# =========================================================================== #
# PHASE 1: Tensor Field Builder (reused from Month 7)
# =========================================================================== #
class TensorFieldBuilder:
    """Constructs 2x2 symmetric anisotropic diffusion tensor field."""

    def __init__(
        self,
        grid_size: int = GRID_SIZE,
        d_parallel: float = D_PARALLEL_DEFAULT,
        d_perpendicular: float = D_PERPENDICULAR_DEFAULT,
        d_base: float = D_BASE,
        tract_angle_deg: float = 45.0,
        tract_width: int = 15,
        tract_curvature: float = 0.15,
    ) -> None:
        self.N = grid_size
        self.d_parallel = float(d_parallel)
        self.d_perpendicular = float(d_perpendicular)
        self.d_base = float(d_base)
        self.tract_angle = np.deg2rad(tract_angle_deg)
        self.tract_width = int(tract_width)
        self.tract_curvature = float(tract_curvature)

        self.tract_mask: np.ndarray | None = None
        self.theta_field: np.ndarray | None = None
        self.D_xx: np.ndarray | None = None
        self.D_xy: np.ndarray | None = None
        self.D_yx: np.ndarray | None = None
        self.D_yy: np.ndarray | None = None
        self.lambda_1: np.ndarray | None = None
        self.lambda_2: np.ndarray | None = None

    def build_tract_mask(self) -> np.ndarray:
        N = self.N
        yy, xx = np.mgrid[0:N, 0:N].astype(float)
        t = xx / max(N - 1, 1)
        center_line = 0.5 * (N - 1) + (xx - 0.5 * (N - 1)) * np.tan(self.tract_angle)
        center_line += self.tract_curvature * (N - 1) * np.sin(2 * np.pi * t)
        dist_to_tract = np.abs(yy - center_line)
        sigma = self.tract_width / 2.355
        soft = np.exp(-(dist_to_tract ** 2) / (2 * sigma ** 2))
        hard_mask = soft > 0.5
        self.tract_mask = hard_mask.astype(bool)
        return self.tract_mask

    def build_orientation_field(self) -> np.ndarray:
        if self.tract_mask is None:
            self.build_tract_mask()
        N = self.N
        yy, xx = np.mgrid[0:N, 0:N].astype(float)
        t = xx / max(N - 1, 1)
        tangent_slope = np.tan(self.tract_angle) + (
            2 * np.pi * self.tract_curvature * np.cos(2 * np.pi * t)
        )
        theta = np.arctan(tangent_slope)
        theta_field = np.where(self.tract_mask, theta, 0.0)
        theta_field = gaussian_filter(theta_field, sigma=2.0) * self.tract_mask + (
            theta_field * (~self.tract_mask)
        )
        self.theta_field = theta_field
        return theta_field

    def build_tensor_field(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self.tract_mask is None:
            self.build_tract_mask()
        if self.theta_field is None:
            self.build_orientation_field()

        theta = self.theta_field
        c = np.cos(theta)
        s = np.sin(theta)

        lam1_field = np.where(self.tract_mask, self.d_parallel, self.d_base)
        lam2_field = np.where(self.tract_mask, self.d_perpendicular, self.d_base)

        D_xx = lam1_field * c * c + lam2_field * s * s
        D_yy = lam1_field * s * s + lam2_field * c * c
        D_xy = (lam1_field - lam2_field) * s * c

        self.D_xx = D_xx
        self.D_yy = D_yy
        self.D_xy = D_xy
        self.D_yx = D_xy.copy()
        self.lambda_1 = lam1_field
        self.lambda_2 = lam2_field
        return D_xx, D_xy, self.D_yx, D_yy

    def validate_tensor(self) -> Dict[str, float]:
        assert self.D_xx is not None
        sym_err = float(np.max(np.abs(self.D_xy - self.D_yx)))
        trace = self.D_xx + self.D_yy
        det = self.D_xx * self.D_yy - self.D_xy * self.D_yx
        min_trace = float(trace.min())
        min_det = float(det.min())
        min_eig = float(np.minimum(self.lambda_1, self.lambda_2).min())
        metrics = {
            "symmetry_max_error": sym_err,
            "min_trace": min_trace,
            "min_determinant": min_det,
            "min_eigenvalue": min_eig,
            "symmetry_pass": bool(sym_err < 1e-12),
            "positive_definite_pass": bool(min_det > 0 and min_trace > 0),
        }
        print(f"[Tensor] Symmetry: {sym_err:.2e}, Min eig: {min_eig:.2e}")
        return metrics

    def save_npz(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            D_xx=self.D_xx, D_xy=self.D_xy, D_yx=self.D_yx, D_yy=self.D_yy,
            theta_field=self.theta_field, tract_mask=self.tract_mask,
            lambda_1=self.lambda_1, lambda_2=self.lambda_2,
            anisotropy_ratio=np.where(self.lambda_2 > 0,
                self.lambda_1 / np.maximum(self.lambda_2, 1e-12), 1.0),
            grid_size=np.array([self.N, self.N]),
        )
        print(f"[Tensor] Saved -> {path}")

    def plot_validation(self, path: Path) -> None:
        assert self.D_xx is not None
        path.parent.mkdir(parents=True, exist_ok=True)
        step = 5
        yy, xx = np.mgrid[0:self.N, 0:self.N]
        xs = xx[::step, ::step]
        ys = yy[::step, ::step]
        theta_sub = self.theta_field[::step, ::step]
        lam1_sub = self.lambda_1[::step, ::step]
        lam2_sub = self.lambda_2[::step, ::step]
        aniso_sub = np.where(lam2_sub > 0, lam1_sub / np.maximum(lam2_sub, 1e-9), 1.0)
        aniso_norm = aniso_sub / max(aniso_sub.max(), 1e-9)
        U = np.cos(theta_sub) * aniso_norm
        V = np.sin(theta_sub) * aniso_norm

        fig, axes = plt.subplots(2, 2, figsize=(13, 11))
        axes[0, 0].imshow(self.tract_mask, origin="lower", cmap="gray", alpha=0.6, extent=[0, self.N, 0, self.N])
        axes[0, 0].quiver(xs, ys, U, -V, color="red", scale=15, width=0.004, headwidth=3, headlength=4)
        axes[0, 0].set_title("White Matter Tract + Eigenvectors")
        for ax, comp, title in [(axes[0, 1], self.D_xx, r"$D_{xx}$"),
                                 (axes[1, 0], self.D_yy, r"$D_{yy}$"),
                                 (axes[1, 1], self.D_xy, r"$D_{xy}=D_{yx}$")]:
            im = ax.imshow(comp, origin="lower", cmap="viridis" if "xy" not in title.lower() else "RdBu_r", extent=[0, self.N, 0, self.N])
            ax.set_title(title)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.suptitle("Anisotropic Tensor Field", fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"[Tensor] Plot saved -> {path}")


# =========================================================================== #
# Anisotropic FK Solver (baseline for comparison)
# =========================================================================== #
class AnisotropicFKSolver:
    """Baseline anisotropic Fisher-Kolmogorov solver (single population)."""

    def __init__(
        self,
        D_xx: np.ndarray, D_xy: np.ndarray, D_yy: np.ndarray,
        rho: np.ndarray | float = 0.0,
        dt: float = ADAPTIVE_DT,
        dx: float = DX,
        carrying_capacity: float = K,
    ) -> None:
        assert D_xx.shape == D_xy.shape == D_yy.shape
        self.H, self.W = D_xx.shape
        self.D_xx = D_xx.astype(float)
        self.D_xy = D_xy.astype(float)
        self.D_yy = D_yy.astype(float)
        self.rho = rho if np.isscalar(rho) else rho.astype(float)
        self.dt = float(dt)
        self.dx = float(dx)
        self.K = float(carrying_capacity)

        per_pixel_eig = 0.5 * (self.D_xx + self.D_yy) + np.sqrt(
            np.maximum(0.0, 0.25 * (self.D_xx - self.D_yy) ** 2 + self.D_xy ** 2)
        )
        max_eig = float(per_pixel_eig.max())
        self.cfl_limit = (self.dx ** 2) / (2.0 * max(max_eig, 1e-12))
        if self.dt > self.cfl_limit:
            self.dt = 0.9 * self.cfl_limit
        print(f"[BaselineFK] dt={self.dt:.4f}, CFL={self.cfl_limit:.4f}")

    def anisotropic_divergence(self, u: np.ndarray) -> np.ndarray:
        dx = self.dx
        Dxx, Dxy, Dyy = self.D_xx, self.D_xy, self.D_yy
        H, W = self.H, self.W

        u_p = np.pad(u, 1, mode="constant", constant_values=0)
        Dxx_p = np.pad(Dxx, 1, mode="edge")
        Dxy_p = np.pad(Dxy, 1, mode="edge")
        Dyy_p = np.pad(Dyy, 1, mode="edge")

        u_x_vface = (u_p[1:-1, 1:] - u_p[1:-1, :-1]) / dx
        u_y_cc = (u_p[2:, 1:-1] - u_p[:-2, 1:-1]) / (2.0 * dx)
        u_y_cc_wpad = np.pad(u_y_cc, ((0, 0), (1, 1)), mode="edge")
        u_y_vface = 0.5 * (u_y_cc_wpad[:, :-1] + u_y_cc_wpad[:, 1:])

        Dxx_vface = 0.5 * (Dxx_p[1:-1, :-1] + Dxx_p[1:-1, 1:])
        Dxy_vface = 0.5 * (Dxy_p[1:-1, :-1] + Dxy_p[1:-1, 1:])

        Fx_face = Dxx_vface * u_x_vface + Dxy_vface * u_y_vface
        Fx_E, Fx_W = Fx_face[:, 1:], Fx_face[:, :-1]

        u_y_hface = (u_p[1:, 1:-1] - u_p[:-1, 1:-1]) / dx
        u_x_cc = (u_p[1:-1, 2:] - u_p[1:-1, :-2]) / (2.0 * dx)
        u_x_cc_hpad = np.pad(u_x_cc, ((1, 1), (0, 0)), mode="edge")
        u_x_hface = 0.5 * (u_x_cc_hpad[:-1, :] + u_x_cc_hpad[1:, :])

        Dyy_hface = 0.5 * (Dyy_p[:-1, 1:-1] + Dyy_p[1:, 1:-1])
        Dyx_hface = 0.5 * (Dxy_p[:-1, 1:-1] + Dxy_p[1:, 1:-1])

        Fy_face = Dyy_hface * u_y_hface + Dyx_hface * u_x_hface
        Fy_N, Fy_S = Fy_face[1:, :], Fy_face[:-1, :]

        return (Fx_E - Fx_W + Fy_N - Fy_S) / (dx ** 2)

    def reaction_term(self, u: np.ndarray) -> np.ndarray:
        if np.isscalar(self.rho):
            return float(self.rho) * u * (1.0 - u / self.K)
        return self.rho * u * (1.0 - u / self.K)

    def step(self, u: np.ndarray, clamp: bool = True) -> np.ndarray:
        div_term = self.anisotropic_divergence(u)
        react_term = self.reaction_term(u)
        u_new = u + self.dt * (div_term + react_term)
        if clamp:
            u_new = np.clip(u_new, 0.0, self.K)
        return u_new

    @staticmethod
    def initial_gaussian_seed(grid_shape: Tuple[int, int], center: Tuple[int, int] | None = None,
                               sigma: float = 3.0, amplitude: float = 0.8) -> np.ndarray:
        H, W = grid_shape
        if center is None:
            cy, cx = H // 2, W // 2
        else:
            cy, cx = center
        yy, xx = np.mgrid[0:H, 0:W].astype(float)
        r2 = (yy - cy) ** 2 + (xx - cx) ** 2
        return amplitude * np.exp(-r2 / (2.0 * sigma ** 2))


# =========================================================================== #
# Dual-Clone Adaptive Therapy Solver
# =========================================================================== #
class AdaptiveTherapySolver:
    """
    Coupled dual-clone solver with Lotka-Volterra competition, mutation,
    and pharmacodynamic drug killing.
    """

    def __init__(
        self,
        D_xx: np.ndarray, D_xy: np.ndarray, D_yy: np.ndarray,
        rho_s: float = RHO_SENSITIVE,
        rho_r: float = RHO_RESISTANT,
        mutation_rate: float = MUTATION_RATE,
        E_max: float = E_MAX,
        EC50: float = EC50,
        hill_coeff: float = HILL_COEFF,
        D_drug: float = D_DRUG,
        drug_decay: float = DRUG_DECAY,
        dt: float = ADAPTIVE_DT,
        dx: float = DX,
        carrying_capacity: float = K,
        use_physical_PK: bool = True,
    ) -> None:
        assert D_xx.shape == D_xy.shape == D_yy.shape
        self.H, self.W = D_xx.shape
        self.D_xx = D_xx.astype(float)
        self.D_xy = D_xy.astype(float)
        self.D_yy = D_yy.astype(float)

        # Support spatially-varying proliferation rates
        self.rho_s = rho_s if np.isscalar(rho_s) else rho_s.astype(float)
        self.rho_r = rho_r if np.isscalar(rho_r) else rho_r.astype(float)
        self.mutation_rate = float(mutation_rate)

        self.E_max = float(E_max)
        self.EC50 = float(EC50)
        self.hill_coeff = float(hill_coeff)

        self.D_drug = float(D_drug)
        self.drug_decay = float(drug_decay)

        self.dt = float(dt)
        self.dx = float(dx)
        self.K = float(carrying_capacity)

        # Phase 1: physical 1-compartment TMZ PK (µg/mL, days)
        # When enabled, `dose` in coupled_step is the bulk concentration
        # C(t) [µg/mL] computed by the protocol runner from the dosing
        # schedule (see compute_tmz_schedule_C / tmz_concentration).
        self.use_physical_PK = bool(use_physical_PK)
        self.k_el = K_EL
        self.tmz_EC50 = TMZ_EC50_UG_ML

        # CFL for tumor diffusion
        per_pixel_eig = 0.5 * (self.D_xx + self.D_yy) + np.sqrt(
            np.maximum(0.0, 0.25 * (self.D_xx - self.D_yy) ** 2 + self.D_xy ** 2)
        )
        max_eig = float(per_pixel_eig.max())
        self.cfl_tumor = (self.dx ** 2) / (2.0 * max(max_eig, 1e-12))
        if self.dt > self.cfl_tumor:
            self.dt = 0.9 * self.cfl_tumor

        # CFL for drug diffusion (isotropic)
        self.cfl_drug = (self.dx ** 2) / (4.0 * self.D_drug)
        self.dt_drug = min(self.dt, 0.9 * self.cfl_drug)
        self.drug_substeps = max(1, int(np.ceil(self.dt / self.dt_drug)))
        self.dt_drug_eff = self.dt / self.drug_substeps

        print(f"[AdaptiveSolver] dt={self.dt:.4f}, drug_substeps={self.drug_substeps}, "
              f"dt_drug={self.dt_drug_eff:.4f}")

        # Drug concentration field
        self.C = np.zeros((self.H, self.W), dtype=float)

    # ------------------------------------------------------------------ #
    def anisotropic_divergence(self, u: np.ndarray) -> np.ndarray:
        dx = self.dx
        Dxx, Dxy, Dyy = self.D_xx, self.D_xy, self.D_yy

        u_p = np.pad(u, 1, mode="constant", constant_values=0)
        Dxx_p = np.pad(Dxx, 1, mode="edge")
        Dxy_p = np.pad(Dxy, 1, mode="edge")
        Dyy_p = np.pad(Dyy, 1, mode="edge")

        u_x_vface = (u_p[1:-1, 1:] - u_p[1:-1, :-1]) / dx
        u_y_cc = (u_p[2:, 1:-1] - u_p[:-2, 1:-1]) / (2.0 * dx)
        u_y_cc_wpad = np.pad(u_y_cc, ((0, 0), (1, 1)), mode="edge")
        u_y_vface = 0.5 * (u_y_cc_wpad[:, :-1] + u_y_cc_wpad[:, 1:])

        Dxx_vface = 0.5 * (Dxx_p[1:-1, :-1] + Dxx_p[1:-1, 1:])
        Dxy_vface = 0.5 * (Dxy_p[1:-1, :-1] + Dxy_p[1:-1, 1:])

        Fx_face = Dxx_vface * u_x_vface + Dxy_vface * u_y_vface
        Fx_E, Fx_W = Fx_face[:, 1:], Fx_face[:, :-1]

        u_y_hface = (u_p[1:, 1:-1] - u_p[:-1, 1:-1]) / dx
        u_x_cc = (u_p[1:-1, 2:] - u_p[1:-1, :-2]) / (2.0 * dx)
        u_x_cc_hpad = np.pad(u_x_cc, ((1, 1), (0, 0)), mode="edge")
        u_x_hface = 0.5 * (u_x_cc_hpad[:-1, :] + u_x_cc_hpad[1:, :])

        Dyy_hface = 0.5 * (Dyy_p[:-1, 1:-1] + Dyy_p[1:, 1:-1])
        Dyx_hface = 0.5 * (Dxy_p[:-1, 1:-1] + Dxy_p[1:, 1:-1])

        Fy_face = Dyy_hface * u_y_hface + Dyx_hface * u_x_hface
        Fy_N, Fy_S = Fy_face[1:, :], Fy_face[:-1, :]

        return (Fx_E - Fx_W + Fy_N - Fy_S) / (dx ** 2)

    # ------------------------------------------------------------------ #
    def drug_laplacian(self, C: np.ndarray) -> np.ndarray:
        """5-point isotropic Laplacian for drug diffusion."""
        C_p = np.pad(C, 1, mode="constant", constant_values=0)
        return (C_p[2:, 1:-1] + C_p[:-2, 1:-1] +
                C_p[1:-1, 2:] + C_p[1:-1, :-2] -
                4.0 * C_p[1:-1, 1:-1]) / (self.dx ** 2)

    # ------------------------------------------------------------------ #
    def drug_step(self, C: np.ndarray, dose: float, dt_drug: float) -> np.ndarray:
        """Single drug diffusion-decay step with source at boundary."""
        laplacian = self.drug_laplacian(C)
        # Drug enters from boundaries (simplified: uniform source when dosing)
        source = dose if dose > 0 else 0.0
        decay = self.drug_decay * C
        C_new = C + dt_drug * (self.D_drug * laplacian + source - decay)
        return np.maximum(C_new, 0.0)

    # ------------------------------------------------------------------ #
    def hill_kill(self, C: np.ndarray) -> np.ndarray:
        """Hill equation drug efficacy on sensitive cells."""
        return self.E_max * (C ** self.hill_coeff) / (self.EC50 ** self.hill_coeff + C ** self.hill_coeff + 1e-12)

    # ------------------------------------------------------------------ #
    def coupled_step(
        self,
        u_s: np.ndarray,
        u_r: np.ndarray,
        C: np.ndarray,
        dose: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Full coupled step:
        Phase 1 (use_physical_PK=True):
          - `dose` is the bulk plasma concentration C(t) in µg/mL
          - Drug field is set uniform = dose (spatial diffusion skipped)
          - kill_rate = hill_kill_physical(dose)
        Legacy (use_physical_PK=False):
          - Spatial drug diffusion-decay with boundary source
        """
        if self.use_physical_PK:
            # Physical PK: bulk concentration directly drives kill rate
            C_curr = np.full((self.H, self.W), float(dose), dtype=float)
            # Use Hill with physical EC50 (µg/mL)
            kill_rate = hill_kill_physical(
                C_curr, E_max=self.E_max, EC50=self.tmz_EC50, H=self.hill_coeff
            )
        else:
            # Legacy spatial drug diffusion
            C_curr = C.copy()
            for _ in range(self.drug_substeps):
                C_curr = self.drug_step(C_curr, dose, self.dt_drug_eff)

            # Drug kill rate at current concentration (legacy EC50 in abstract units)
            kill_rate = self.hill_kill(C_curr)  # (H, W)

        # Total local density for competition
        u_total = u_s + u_r
        competition = 1.0 - u_total / self.K
        competition = np.clip(competition, 0.0, 1.0)

        # Diffusion terms
        div_s = self.anisotropic_divergence(u_s)
        div_r = self.anisotropic_divergence(u_r)

        # Reaction + competition + drug kill + mutation
        # Sensitive: growth - drug kill
        growth_s = self.rho_s * u_s * competition
        death_s = kill_rate * u_s
        # Mutation: fraction of dividing sensitive cells become resistant
        mutation = self.mutation_rate * np.maximum(growth_s, 0)

        # Resistant: growth (no drug kill) + gain from mutation
        growth_r = self.rho_r * u_r * competition

        u_s_new = u_s + self.dt * (div_s + growth_s - death_s - mutation)
        u_r_new = u_r + self.dt * (div_r + growth_r + mutation)

        # Clamp
        u_s_new = np.clip(u_s_new, 0.0, self.K)
        u_r_new = np.clip(u_r_new, 0.0, self.K)
        # Ensure total doesn't exceed K (redistribute excess proportionally)
        excess = u_s_new + u_r_new - self.K
        if excess.max() > 1e-6:
            mask = excess > 0
            scale = self.K / (u_s_new[mask] + u_r_new[mask] + 1e-12)
            u_s_new[mask] *= scale
            u_r_new[mask] *= scale

        return u_s_new, u_r_new, C_curr

    # ------------------------------------------------------------------ #
    @staticmethod
    def initial_seeds(grid_shape: Tuple[int, int],
                       center_s: Tuple[int, int],
                       center_r: Tuple[int, int],
                       sigma: float = 3.0,
                       amp_s: float = 0.6,
                       amp_r: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
        """Dual Gaussian seeds for sensitive and resistant clones."""
        H, W = grid_shape
        yy, xx = np.mgrid[0:H, 0:W].astype(float)

        cy_s, cx_s = center_s
        r2_s = (yy - cy_s) ** 2 + (xx - cx_s) ** 2
        u_s = amp_s * np.exp(-r2_s / (2.0 * sigma ** 2))

        cy_r, cx_r = center_r
        r2_r = (yy - cy_r) ** 2 + (xx - cx_r) ** 2
        u_r = amp_r * np.exp(-r2_r / (2.0 * sigma ** 2))

        return u_s, u_r

    # ------------------------------------------------------------------ #
    def patient_seed_centers(self, patient_id: str) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """Deterministic but patient-specific seed positions."""
        # Both clones seeded at (40, 40) inside the diagonal white matter tract corridor
        return (40, 40), (40, 40)


# =========================================================================== #
# Patient Parameter Mapper (multi-omic calibration)
# =========================================================================== #
class PatientParameterMapper:
    """Maps patient expression to clone-specific parameters."""

    INFLAMMATORY_GENES = ["S100A8", "S100A11", "LST1"]

    def __init__(
        self,
        cohort_npz: Path = Path("output/spatial_recurrence_profiles.npz"),
        zone_csv_root: Path = Path("output"),
    ) -> None:
        self.cohort_npz = cohort_npz
        self.zone_csv_root = zone_csv_root
        self.data: Dict = {}
        self.zone_expr_dfs: Dict[str, pd.DataFrame] = {}

    def load(self) -> Dict:
        d = np.load(self.cohort_npz, allow_pickle=True)
        self.data = {k: d[k] for k in d.files}
        suffix_map = {
            "Leading Edge": "real_cohort_le.csv",
            "Cellular Tumor": "real_cohort_ct.csv",
            "Infiltrating Tumor": "real_cohort_it.csv",
        }
        for zone, fname in suffix_map.items():
            fp = self.zone_csv_root / fname
            df = pd.read_csv(fp)
            self.zone_expr_dfs[zone] = df
        print(f"[Mapper] Loaded cohort: {list(self.data['patient_ids'])}")
        return self.data

    def get_patient_inflammation(self, patient_id: str) -> Dict[str, float]:
        scores = {}
        for zone, df in self.zone_expr_dfs.items():
            sub = df[df["patient_id"] == patient_id]
            vals = []
            for g in self.INFLAMMATORY_GENES:
                row = sub[sub["gene"] == g]
                if len(row) > 0:
                    vals.append(float(row["expression_log2tpm"].values[0]))
            if not vals:
                scores[zone] = 1.0
                continue
            raw = np.mean(vals)
            score = 1.0 + np.tanh((raw - 5.0) / 1.5)
            scores[zone] = float(score)
        return scores

    def build_patient_parameters(self, patient_id: str) -> Dict:
        if not self.data:
            self.load()
        pids = list(self.data["patient_ids"])
        if patient_id not in pids:
            raise KeyError(f"Patient {patient_id} not found")
        pidx = pids.index(patient_id)

        zone_scores = self.get_patient_inflammation(patient_id)
        N = GRID_SIZE
        scale_s = np.ones((N, N))
        scale_r = np.ones((N, N))

        for zone, (y0, y1) in ZONE_REGIONS.items():
            sc = zone_scores[zone]
            # Higher inflammation -> higher sensitive proliferation, slightly lower resistant fitness cost
            scale_s[y0:y1, :] *= sc
            scale_r[y0:y1, :] *= (1.0 + 0.2 * (sc - 1.0))  # mild boost

        scale_s = gaussian_filter(scale_s, sigma=2.0)
        scale_r = gaussian_filter(scale_r, sigma=2.0)

        rho_s_field = RHO_SENSITIVE * scale_s
        rho_r_field = RHO_RESISTANT * scale_r

        return {
            "rho_s_field": rho_s_field,
            "rho_r_field": rho_r_field,
            "zone_scores": zone_scores,
            "mean_inflammation": float(np.mean(list(zone_scores.values()))),
        }


# =========================================================================== #
# MTD Simulation (Phase 1: TMZ PK schedule)
# =========================================================================== #
def run_mtd_protocol(
    solver: AdaptiveTherapySolver,
    patient_id: str,
    u_s0: np.ndarray,
    u_r0: np.ndarray,
    n_steps: int = MTD_STEPS,
    save_interval: int = SAVE_INTERVAL,
) -> Dict:
    """MTD with standard TMZ 5-days-on / 23-days-off schedule.

    At each step, C(t) is computed from the dosing schedule via
    compute_tmz_schedule_C(), which implements 1-compartment PK.
    """
    u_s = u_s0.copy()
    u_r = u_r0.copy()
    C = np.zeros((solver.H, solver.W), dtype=float)

    n_saves = n_steps // save_interval + 1
    u_s_hist = np.zeros((n_saves, solver.H, solver.W), dtype=np.float32)
    u_r_hist = np.zeros((n_saves, solver.H, solver.W), dtype=np.float32)
    C_hist = np.zeros((n_saves, solver.H, solver.W), dtype=np.float32)
    mass_s_hist = np.zeros(n_steps + 1, dtype=np.float64)
    mass_r_hist = np.zeros(n_steps + 1, dtype=np.float64)
    dose_hist = np.zeros(n_steps + 1, dtype=np.float64)

    u_s_hist[0] = u_s.astype(np.float32)
    u_r_hist[0] = u_r.astype(np.float32)
    C_hist[0] = C.astype(np.float32)
    mass_s_hist[0] = float(u_s.sum())
    mass_r_hist[0] = float(u_r.sum())
    dose_hist[0] = compute_tmz_schedule_C(0)

    save_idx = 1
    for step in range(1, n_steps + 1):
        # TMZ concentration at this step from PK schedule
        dose = compute_tmz_schedule_C(step)
        u_s, u_r, C = solver.coupled_step(u_s, u_r, C, dose)
        mass_s_hist[step] = float(u_s.sum())
        mass_r_hist[step] = float(u_r.sum())
        dose_hist[step] = dose

        if step % save_interval == 0 and save_idx < n_saves:
            u_s_hist[save_idx] = u_s.astype(np.float32)
            u_r_hist[save_idx] = u_r.astype(np.float32)
            C_hist[save_idx] = C.astype(np.float32)
            save_idx += 1

    print(f"[MTD] {patient_id}: final u_s={mass_s_hist[-1]:.1f}, u_r={mass_r_hist[-1]:.1f}")
    return {
        "patient_id": patient_id,
        "protocol": "MTD",
        "u_s_history": u_s_hist,
        "u_r_history": u_r_hist,
        "C_history": C_hist,
        "mass_s": mass_s_hist,
        "mass_r": mass_r_hist,
        "dose_history": dose_hist,
        "final_u_s": u_s,
        "final_u_r": u_r,
        "final_C": C,
    }


# =========================================================================== #
# Adaptive Therapy Simulation (Phase 1: TMZ PK with drug-holiday gating)
# =========================================================================== #
def run_adaptive_protocol(
    solver: AdaptiveTherapySolver,
    patient_id: str,
    u_s0: np.ndarray,
    u_r0: np.ndarray,
    n_steps: int = ADAPTIVE_STEPS,
    save_interval: int = SAVE_INTERVAL,
    threshold_off: float = THRESHOLD_OFF,
    threshold_on: float = THRESHOLD_ON,
) -> Dict:
    """Closed-loop adaptive dosing with TMZ PK schedule gated by drug_on state.

    - When drug_on=True: TMZ follows 5-on/23-off schedule (C(t) from schedule)
    - When drug_on=False: C(t) decays from last dose (schedule OFF days)
    """
    u_s = u_s0.copy()
    u_r = u_r0.copy()
    C = np.zeros((solver.H, solver.W), dtype=float)

    # Baseline total mass
    baseline_mass = float(u_s0.sum() + u_r0.sum())
    drug_on = True

    n_saves = n_steps // save_interval + 1
    u_s_hist = np.zeros((n_saves, solver.H, solver.W), dtype=np.float32)
    u_r_hist = np.zeros((n_saves, solver.H, solver.W), dtype=np.float32)
    C_hist = np.zeros((n_saves, solver.H, solver.W), dtype=np.float32)
    mass_s_hist = np.zeros(n_steps + 1, dtype=np.float64)
    mass_r_hist = np.zeros(n_steps + 1, dtype=np.float64)
    dose_hist = np.zeros(n_steps + 1, dtype=np.float64)
    drug_state_hist = np.zeros(n_steps + 1, dtype=bool)

    u_s_hist[0] = u_s.astype(np.float32)
    u_r_hist[0] = u_r.astype(np.float32)
    C_hist[0] = C.astype(np.float32)
    mass_s_hist[0] = float(u_s.sum())
    mass_r_hist[0] = float(u_r.sum())
    dose_hist[0] = compute_tmz_schedule_C(0) if drug_on else 0.0
    drug_state_hist[0] = drug_on

    save_idx = 1
    for step in range(1, n_steps + 1):
        current_mass = float(u_s.sum() + u_r.sum())

        # Adaptive control logic (same thresholds as before)
        if drug_on and current_mass < threshold_off * baseline_mass:
            drug_on = False
        elif not drug_on and current_mass > threshold_on * baseline_mass:
            drug_on = True

        # PK: compute C(t) from schedule, but only if drug_on
        if drug_on:
            dose = compute_tmz_schedule_C(step)
        else:
            dose = 0.0  # No dosing during holiday

        u_s, u_r, C = solver.coupled_step(u_s, u_r, C, dose)

        mass_s_hist[step] = float(u_s.sum())
        mass_r_hist[step] = float(u_r.sum())
        dose_hist[step] = dose
        drug_state_hist[step] = drug_on

        if step % save_interval == 0 and save_idx < n_saves:
            u_s_hist[save_idx] = u_s.astype(np.float32)
            u_r_hist[save_idx] = u_r.astype(np.float32)
            C_hist[save_idx] = C.astype(np.float32)
            save_idx += 1

    print(f"[Adaptive] {patient_id}: final u_s={mass_s_hist[-1]:.1f}, u_r={mass_r_hist[-1]:.1f}, "
          f"drug_on_fraction={drug_state_hist.mean():.2f}")
    return {
        "patient_id": patient_id,
        "protocol": "Adaptive",
        "u_s_history": u_s_hist,
        "u_r_history": u_r_hist,
        "C_history": C_hist,
        "mass_s": mass_s_hist,
        "mass_r": mass_r_hist,
        "dose_history": dose_hist,
        "drug_state": drug_state_hist,
        "final_u_s": u_s,
        "final_u_r": u_r,
        "final_C": C,
        "baseline_mass": baseline_mass,
    }


# =========================================================================== #
# Cohort Simulator
# =========================================================================== #
class CohortSimulator:
    """Runs both MTD and Adaptive protocols for the cohort."""

    def __init__(
        self,
        mapper: PatientParameterMapper,
        builder: TensorFieldBuilder,
        mtd_steps: int = MTD_STEPS,
        adaptive_steps: int = ADAPTIVE_STEPS,
        save_interval: int = SAVE_INTERVAL,
    ) -> None:
        self.mapper = mapper
        self.builder = builder
        self.mtd_steps = mtd_steps
        self.adaptive_steps = adaptive_steps
        self.save_interval = save_interval

    @staticmethod
    def patient_seed_centers(patient_id: str, N: int = GRID_SIZE) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        seed = sum(ord(c) for c in patient_id)
        rng = np.random.default_rng(seed)
        cy_s = 40
        cx_s = 40
        cy_r = 40
        cx_r = 40
        return (cy_s, cx_s), (cy_r, cx_r)

    def run_patient(self, patient_id: str) -> Dict:
        params = self.mapper.build_patient_parameters(patient_id)

        # Create solver with patient-specific rho fields
        solver = AdaptiveTherapySolver(
            D_xx=self.builder.D_xx, D_xy=self.builder.D_xy, D_yy=self.builder.D_yy,
            rho_s=params["rho_s_field"], rho_r=params["rho_r_field"],
        )

        # Initial seeds
        center_s, center_r = self.patient_seed_centers(patient_id)
        u_s0, u_r0 = AdaptiveTherapySolver.initial_seeds(
            (GRID_SIZE, GRID_SIZE), center_s, center_r
        )

        # Run MTD
        mtd_result = run_mtd_protocol(solver, patient_id, u_s0, u_r0,
                                       n_steps=self.mtd_steps, save_interval=self.save_interval)

        # Run Adaptive (reset solver state)
        solver2 = AdaptiveTherapySolver(
            D_xx=self.builder.D_xx, D_xy=self.builder.D_xy, D_yy=self.builder.D_yy,
            rho_s=params["rho_s_field"], rho_r=params["rho_r_field"],
        )
        adaptive_result = run_adaptive_protocol(solver2, patient_id, u_s0, u_r0,
                                                 n_steps=self.adaptive_steps, save_interval=self.save_interval)

        return {
            "patient_id": patient_id,
            "params": params,
            "seed_centers": {"sensitive": center_s, "resistant": center_r},
            "mtd": mtd_result,
            "adaptive": adaptive_result,
        }

    def run_cohort(self, output_dir: Path = Path("output")) -> List[Dict]:
        if not self.mapper.data:
            self.mapper.load()
        output_dir.mkdir(parents=True, exist_ok=True)

        results = []
        for pid in COHORT_PATIENTS:
            if pid not in self.mapper.data["patient_ids"]:
                print(f"[Cohort] Skipping {pid} (not in cohort data)")
                continue
            print(f"\n[Cohort] Processing {pid}...")
            res = self.run_patient(pid)

            # Save individual patient data
            np.savez_compressed(
                output_dir / f"adaptive_{pid}.npz",
                patient_id=pid,
                mtd_u_s=res["mtd"]["u_s_history"],
                mtd_u_r=res["mtd"]["u_r_history"],
                mtd_C=res["mtd"]["C_history"],
                mtd_mass_s=res["mtd"]["mass_s"],
                mtd_mass_r=res["mtd"]["mass_r"],
                mtd_dose=res["mtd"]["dose_history"],
                adaptive_u_s=res["adaptive"]["u_s_history"],
                adaptive_u_r=res["adaptive"]["u_r_history"],
                adaptive_C=res["adaptive"]["C_history"],
                adaptive_mass_s=res["adaptive"]["mass_s"],
                adaptive_mass_r=res["adaptive"]["mass_r"],
                adaptive_dose=res["adaptive"]["dose_history"],
                adaptive_drug_state=res["adaptive"]["drug_state"],
                seed_sensitive=np.array(res["seed_centers"]["sensitive"]),
                seed_resistant=np.array(res["seed_centers"]["resistant"]),
                rho_s_field=res["params"]["rho_s_field"],
                rho_r_field=res["params"]["rho_r_field"],
            )
            results.append(res)

        # Save combined cohort data
        all_mtd_mass_s = np.stack([r["mtd"]["mass_s"] for r in results])
        all_mtd_mass_r = np.stack([r["mtd"]["mass_r"] for r in results])
        all_adaptive_mass_s = np.stack([r["adaptive"]["mass_s"] for r in results])
        all_adaptive_mass_r = np.stack([r["adaptive"]["mass_r"] for r in results])

        np.savez_compressed(
            output_dir / "adaptive_therapy_data.npz",
            patient_ids=np.array([r["patient_id"] for r in results]),
            mtd_mass_s=all_mtd_mass_s,
            mtd_mass_r=all_mtd_mass_r,
            adaptive_mass_s=all_adaptive_mass_s,
            adaptive_mass_r=all_adaptive_mass_r,
        )
        print(f"\n[Cohort] Saved combined data -> {output_dir / 'adaptive_therapy_data.npz'}")
        return results


# =========================================================================== #
# Visualization & Metrics
# =========================================================================== #
class AdaptiveVisualizer:
    """Generates comparative visualizations and metrics."""

    def __init__(self, builder: TensorFieldBuilder) -> None:
        self.builder = builder

    def plot_initial_clones(self, results: List[Dict], output_path: Path) -> None:
        """8-panel showing initial sensitive vs resistant seeds."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        n = len(results)
        n_cols = 4
        n_rows = (n + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
        axes = axes.flatten() if n > 1 else [axes]

        for idx, res in enumerate(results):
            if idx >= len(axes):
                break
            ax = axes[idx]
            mtd = res["mtd"]
            # Show initial state from MTD (same initial conditions)
            u_s0 = mtd["u_s_history"][0]
            u_r0 = mtd["u_r_history"][0]

            im = ax.imshow(u_s0, origin="lower", cmap="Blues", alpha=0.7, vmin=0, vmax=1)
            ax.imshow(u_r0, origin="lower", cmap="Reds", alpha=0.7, vmin=0, vmax=1)
            ax.contour(self.builder.tract_mask, levels=[0.5], colors="cyan", linewidths=0.8, alpha=0.5)
            cy_s, cx_s = res["seed_centers"]["sensitive"]
            cy_r, cx_r = res["seed_centers"]["resistant"]
            ax.plot(cx_s, cy_s, '+', color='blue', ms=12, mew=2, label='Sensitive')
            ax.plot(cx_r, cy_r, 'x', color='red', ms=12, mew=2, label='Resistant')
            ax.set_title(f"{res['patient_id']}\nInfl={res['params']['mean_inflammation']:.2f}", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])

        for idx in range(n, len(axes)):
            axes[idx].set_visible(False)
        axes[0].legend(fontsize=8)
        plt.suptitle("Initial Clonal Seeding: Sensitive (blue) vs Resistant (red)", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(output_path, dpi=220, bbox_inches="tight")
        plt.close()
        print(f"[Viz] Initial clones saved -> {output_path}")

    def plot_comparison_canvas(self, results: List[Dict], output_path: Path) -> None:
        """
        Enhanced adaptive therapy vs MTD comparison dashboard.
        Top: Spatial timeline (Start, Mid, End) for MTD and Adaptive
        Bottom: Synchronized line charts with drug holidays and clonal ratios
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        n = len(results)
        n_cols = 4
        
        # We'll create a figure with:
        # - Top section: MTD spatial timeline (Start, Mid, End) 
        # - Middle section: Adaptive spatial timeline (Start, Mid, End)
        # - Bottom section: 2 rows x n patients (Volume timeline, Clonal ratio timeline)
        
        fig = plt.figure(figsize=(20, 16))
        
        # GridSpec for flexible layout: 4 rows x n_cols
        gs = fig.add_gridspec(4, n_cols, hspace=0.35, wspace=0.25)
        
        timepoints = [0, MTD_STEPS // 2, MTD_STEPS - 1]
        timepoint_labels = ['Start (t=0)', f'Mid (t={MTD_STEPS//2})', f'End (t={MTD_STEPS-1})']
        
        for idx, res in enumerate(results):
            col = idx % n_cols
            mtd = res["mtd"]
            adapt = res["adaptive"]
            pid = res["patient_id"]
            
            u_s_mtd = mtd["u_s_history"]
            u_r_mtd = mtd["u_r_history"]
            u_s_adapt = adapt["u_s_history"]
            u_r_adapt = adapt["u_r_history"]
            
            # Map timepoints to saved indices
            save_interval = SAVE_INTERVAL
            tp_indices = [0, len(u_s_mtd) // 2, len(u_s_mtd) - 1]
            
            # ========== ROW 0: MTD Spatial Timeline (Midpoint) ==========
            tp_idx = 1  # midpoint
            tp_saved = tp_indices[tp_idx]
            ax = fig.add_subplot(gs[0, col])
            u_s = u_s_mtd[tp_saved]
            u_r = u_r_mtd[tp_saved]
            
            # Sensitive = Green, Resistant = Red
            im_s = ax.imshow(u_s, origin="lower", cmap="Greens", alpha=0.8, vmin=0, vmax=1)
            im_r = ax.imshow(u_r, origin="lower", cmap="Reds", alpha=0.8, vmin=0, vmax=1)
            ax.contour(self.builder.tract_mask, levels=[0.5], colors="cyan", linewidths=0.8, alpha=0.7)
            
            total = u_s.sum() + u_r.sum()
            res_frac = u_r.sum() / max(total, 1e-6)
            ax.set_title(f"{pid} MTD - {timepoint_labels[tp_idx]}\nS:{u_s.sum():.0f} R:{u_r.sum():.0f} (R%:{res_frac:.0%})", 
                        fontsize=9, fontweight='bold')
            ax.set_xticks([])
            ax.set_yticks([])
            
            # ========== ROW 1: Adaptive Spatial Timeline (Midpoint) ==========
            tp_saved = tp_indices[tp_idx]
            ax = fig.add_subplot(gs[1, col])
            u_s = u_s_adapt[tp_saved]
            u_r = u_r_adapt[tp_saved]
            
            im_s = ax.imshow(u_s, origin="lower", cmap="Greens", alpha=0.8, vmin=0, vmax=1)
            im_r = ax.imshow(u_r, origin="lower", cmap="Reds", alpha=0.8, vmin=0, vmax=1)
            ax.contour(self.builder.tract_mask, levels=[0.5], colors="cyan", linewidths=0.8, alpha=0.7)
            
            total = u_s.sum() + u_r.sum()
            res_frac = u_r.sum() / max(total, 1e-6)
            ax.set_title(f"{pid} Adaptive - {timepoint_labels[tp_idx]}\nS:{u_s.sum():.0f} R:{u_r.sum():.0f} (R%:{res_frac:.0%})", 
                        fontsize=9, fontweight='bold')
            ax.set_xticks([])
            ax.set_yticks([])
        
        # Add colorbar for spatial plots
        cax = fig.add_axes([0.92, 0.55, 0.015, 0.35])
        fig.colorbar(im_s, cax=cax, label='Density')
        
        # ========== ROW 2: Normalized Tumor Volume Timeline ==========
        for idx, res in enumerate(results):
            col = idx % n_cols
            ax = fig.add_subplot(gs[2, col])
            mtd = res["mtd"]
            adapt = res["adaptive"]
            pid = res["patient_id"]
            
            t_mtd = np.arange(len(mtd["mass_s"]))
            t_adapt = np.arange(len(adapt["mass_s"]))
            
            mtd_total = mtd["mass_s"] + mtd["mass_r"]
            adapt_total = adapt["mass_s"] + adapt["mass_r"]
            baseline = adapt["baseline_mass"]
            
            # Normalize to baseline
            mtd_norm = mtd_total / baseline
            adapt_norm = adapt_total / baseline
            
            ax.plot(t_mtd, mtd_norm, 'r-', linewidth=2, label='MTD', alpha=0.8)
            ax.plot(t_adapt, adapt_norm, 'g-', linewidth=2, label='Adaptive', alpha=0.8)
            
            # Drug holiday markers for adaptive
            drug_on = adapt["drug_state"]
            # Find holiday transitions
            drug_changes = np.diff(drug_on.astype(int))
            holiday_starts = np.where(drug_changes < 0)[0]  # ON -> OFF
            holiday_ends = np.where(drug_changes > 0)[0]    # OFF -> ON
            
            for hs in holiday_starts:
                ax.axvline(hs, color='green', linestyle='--', alpha=0.4, linewidth=1.5)
            for he in holiday_ends:
                ax.axvline(he, color='red', linestyle='--', alpha=0.4, linewidth=1.5)
            
            # Shade drug-on periods
            drug_on_mask = drug_on.astype(bool)
            ax.fill_between(t_adapt, 0, max(mtd_norm.max(), adapt_norm.max()) * 1.1,
                           where=drug_on_mask, alpha=0.1, color='green', label='Drug ON')
            
            # Threshold lines
            ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5, label='Drug OFF threshold')
            ax.axhline(1.0, color='black', linestyle=':', alpha=0.5, label='Drug ON threshold')
            ax.axhline(1.5, color='red', linestyle=':', alpha=0.5, label='Progression')
            
            ax.set_title(f"{pid} - Normalized Volume", fontsize=10, fontweight='bold')
            ax.set_xlabel("Time Step")
            ax.set_ylabel("Volume / Baseline")
            ax.set_ylim(0, max(mtd_norm.max(), adapt_norm.max()) * 1.2)
            ax.grid(alpha=0.3)
            if idx == 0:
                ax.legend(fontsize=7, loc='upper left', ncol=2)
        
        # ========== ROW 3: Clonal Composition Ratio (S/R) Timeline ==========
        for idx, res in enumerate(results):
            col = idx % n_cols
            ax = fig.add_subplot(gs[3, col])
            mtd = res["mtd"]
            adapt = res["adaptive"]
            pid = res["patient_id"]
            
            t_mtd = np.arange(len(mtd["mass_s"]))
            t_adapt = np.arange(len(adapt["mass_s"]))
            
            # Resistant fraction over time
            mtd_total = mtd["mass_s"] + mtd["mass_r"]
            adapt_total = adapt["mass_s"] + adapt["mass_r"]
            
            mtd_r_frac = np.where(mtd_total > 0, mtd["mass_r"] / mtd_total, 0)
            adapt_r_frac = np.where(adapt_total > 0, adapt["mass_r"] / adapt_total, 0)
            mtd_s_frac = np.where(mtd_total > 0, mtd["mass_s"] / mtd_total, 0)
            adapt_s_frac = np.where(adapt_total > 0, adapt["mass_s"] / adapt_total, 0)
            
            # Stacked area chart for MTD
            ax.fill_between(t_mtd, 0, mtd_s_frac, color='green', alpha=0.5, label='Sensitive (MTD)')
            ax.fill_between(t_mtd, mtd_s_frac, mtd_s_frac + mtd_r_frac, color='red', alpha=0.5, label='Resistant (MTD)')
            
            # Lines for Adaptive
            ax.plot(t_adapt, adapt_s_frac, 'g--', linewidth=2, alpha=0.8, label='Sensitive (Adapt)')
            ax.plot(t_adapt, adapt_r_frac, 'r--', linewidth=2, alpha=0.8, label='Resistant (Adapt)')
            
            # Drug holiday markers
            drug_on = adapt["drug_state"]
            drug_changes = np.diff(drug_on.astype(int))
            holiday_starts = np.where(drug_changes < 0)[0]
            for hs in holiday_starts:
                ax.axvline(hs, color='green', linestyle='--', alpha=0.4, linewidth=1.5)
            
            ax.set_title(f"{pid} - Clonal Composition (R/S)", fontsize=10, fontweight='bold')
            ax.set_xlabel("Time Step")
            ax.set_ylabel("Fraction")
            ax.set_ylim(0, 1.05)
            ax.grid(alpha=0.3)
            if idx == 0:
                ax.legend(fontsize=7, loc='upper left', ncol=2)
        
        # Main title
        fig.suptitle(
            "Adaptive Therapy vs Continuous MTD: Evolutionary Dynamics Dashboard\n"
            "Sensitive=Green, Resistant=Red | Dashed vertical lines = Drug Holiday transitions",
            fontsize=15, fontweight="bold", y=0.98
        )
        
        plt.savefig(output_path, dpi=220, bbox_inches="tight")
        plt.close()
        print(f"[Viz] Enhanced comparison canvas saved -> {output_path}")

    def plot_temporal_dynamics(self, results: List[Dict], output_path: Path) -> None:
        """
        Legacy method - now integrated into plot_comparison_canvas.
        Kept for backward compatibility.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        n = len(results)
        n_cols = 4
        n_rows = (n + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
        axes = axes.flatten() if n > 1 else [axes]

        for idx, res in enumerate(results):
            ax = axes[idx]
            mtd = res["mtd"]
            adapt = res["adaptive"]

            t_mtd = np.arange(len(mtd["mass_s"]))
            t_adapt = np.arange(len(adapt["mass_s"]))

            ax.plot(t_mtd, mtd["mass_s"], 'b-', alpha=0.7, label='MTD Sensitive')
            ax.plot(t_mtd, mtd["mass_r"], 'r-', alpha=0.7, label='MTD Resistant')
            ax.plot(t_adapt, adapt["mass_s"], 'b--', alpha=0.7, label='Adapt Sensitive')
            ax.plot(t_adapt, adapt["mass_r"], 'r--', alpha=0.7, label='Adapt Resistant')

            drug_on = adapt["drug_state"]
            ax.fill_between(t_adapt, 0, max(mtd["mass_s"].max(), mtd["mass_r"].max()) * 1.1,
                           where=drug_on, alpha=0.1, color='green', label='Drug ON')

            ax.set_title(f"{res['patient_id']}", fontsize=9)
            ax.set_xlabel("Time step")
            ax.set_ylabel("Mass")
            ax.grid(alpha=0.3)
            if idx == 0:
                ax.legend(fontsize=6, loc='upper right')

        for idx in range(n, len(axes)):
            axes[idx].set_visible(False)

        plt.suptitle("Temporal Dynamics: MTD (solid) vs Adaptive (dashed)", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"[Viz] Temporal dynamics saved -> {output_path}")

    def compute_ttp(self, mass_total: np.ndarray, baseline: float, threshold: float = 1.5) -> int:
        """Time to progression: when total mass exceeds threshold * baseline."""
        # Progression: mass > 1.5x baseline (or never)
        exceeded = np.where(mass_total > threshold * baseline)[0]
        if len(exceeded) > 0:
            return int(exceeded[0])
        return len(mass_total)  # censored

    def compute_metrics(self, results: List[Dict]) -> List[Dict]:
        """Compute quantitative metrics for each patient."""
        metrics_list = []
        for res in results:
            mtd = res["mtd"]
            adapt = res["adaptive"]
            pid = res["patient_id"]

            mtd_total = mtd["mass_s"] + mtd["mass_r"]
            adapt_total = adapt["mass_s"] + adapt["mass_r"]
            baseline = adapt["baseline_mass"]

            # TTP
            ttp_mtd = self.compute_ttp(mtd_total, baseline)
            ttp_adapt = self.compute_ttp(adapt_total, baseline)

            # Volume reduction at end
            vr_mtd = mtd_total[-1] / baseline
            vr_adapt = adapt_total[-1] / baseline

            # Resistant fraction at end
            rf_mtd = mtd["mass_r"][-1] / max(mtd_total[-1], 1e-6)
            rf_adapt = adapt["mass_r"][-1] / max(adapt_total[-1], 1e-6)

            # Drug exposure (AUC)
            drug_auc_mtd = mtd["dose_history"].sum()
            drug_auc_adapt = adapt["dose_history"].sum()

            # Number of drug holidays
            drug_state = adapt["drug_state"]
            holidays = np.sum(np.diff(drug_state.astype(int)) < 0)  # ON->OFF transitions

            # Area under curve for total mass
            auc_mtd = float(np.trapezoid(mtd_total))
            auc_adapt = float(np.trapezoid(adapt_total))

            metrics = {
                "patient_id": pid,
                "ttp_mtd": ttp_mtd,
                "ttp_adaptive": ttp_adapt,
                "ttp_ratio": ttp_adapt / max(ttp_mtd, 1),
                "volume_ratio_mtd": vr_mtd,
                "volume_ratio_adaptive": vr_adapt,
                "resistant_fraction_mtd": rf_mtd,
                "resistant_fraction_adaptive": rf_adapt,
                "drug_auc_mtd": drug_auc_mtd,
                "drug_auc_adaptive": drug_auc_adapt,
                "drug_reduction": 1.0 - drug_auc_adapt / max(drug_auc_mtd, 1e-6),
                "num_holidays": int(holidays),
                "mean_mass_mtd": float(mtd_total.mean()),
                "mean_mass_adaptive": float(adapt_total.mean()),
                "auc_total_mtd": float(auc_mtd),
                "auc_total_adaptive": float(auc_adapt),
                "zone_scores": res["params"]["zone_scores"],
                "mean_inflammation": res["params"]["mean_inflammation"],
            }
            metrics_list.append(metrics)
            print(f"[Metrics] {pid}: TTP_MTD={ttp_mtd}, TTP_Adapt={ttp_adapt}, "
                  f"VR_MTD={vr_mtd:.2f}, VR_Adapt={vr_adapt:.2f}, "
                  f"Holidays={holidays}, DrugRed={metrics['drug_reduction']:.2f}")

        return metrics_list

    def save_metrics_json(self, metrics: List[Dict], output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        def convert(obj):
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert(v) for v in obj]
            return obj
        with open(output_path, "w") as f:
            json.dump(convert(metrics), f, indent=2)
        print(f"[Metrics] Saved -> {output_path}")

    def plot_metrics_summary(self, metrics: List[Dict], output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pids = [m["patient_id"] for m in metrics]

        fig, axes = plt.subplots(2, 3, figsize=(15, 9))

        # TTP comparison
        ax = axes[0, 0]
        ttp_mtd = [m["ttp_mtd"] for m in metrics]
        ttp_adapt = [m["ttp_adaptive"] for m in metrics]
        x = np.arange(len(pids))
        w = 0.35
        ax.bar(x - w/2, ttp_mtd, w, label='MTD', color='red', alpha=0.7)
        ax.bar(x + w/2, ttp_adapt, w, label='Adaptive', color='green', alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(pids, rotation=45)
        ax.set_ylabel("Time to Progression")
        ax.set_title("TTP: MTD vs Adaptive")
        ax.legend()
        ax.grid(alpha=0.3, axis='y')

        # Volume ratio
        ax = axes[0, 1]
        vr_mtd = [m["volume_ratio_mtd"] for m in metrics]
        vr_adapt = [m["volume_ratio_adaptive"] for m in metrics]
        ax.bar(x - w/2, vr_mtd, w, label='MTD', color='red', alpha=0.7)
        ax.bar(x + w/2, vr_adapt, w, label='Adaptive', color='green', alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(pids, rotation=45)
        ax.set_ylabel("Final Volume / Baseline")
        ax.set_title("Tumor Volume Ratio")
        ax.legend()
        ax.grid(alpha=0.3, axis='y')

        # Resistant fraction
        ax = axes[0, 2]
        rf_mtd = [m["resistant_fraction_mtd"] for m in metrics]
        rf_adapt = [m["resistant_fraction_adaptive"] for m in metrics]
        ax.bar(x - w/2, rf_mtd, w, label='MTD', color='red', alpha=0.7)
        ax.bar(x + w/2, rf_adapt, w, label='Adaptive', color='green', alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(pids, rotation=45)
        ax.set_ylabel("Resistant Fraction")
        ax.set_title("Resistant Clone Dominance")
        ax.legend()
        ax.grid(alpha=0.3, axis='y')

        # Drug exposure
        ax = axes[1, 0]
        auc_mtd = [m["drug_auc_mtd"] for m in metrics]
        auc_adapt = [m["drug_auc_adaptive"] for m in metrics]
        ax.bar(x - w/2, auc_mtd, w, label='MTD', color='red', alpha=0.7)
        ax.bar(x + w/2, auc_adapt, w, label='Adaptive', color='green', alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(pids, rotation=45)
        ax.set_ylabel("Drug AUC")
        ax.set_title("Cumulative Drug Exposure")
        ax.legend()
        ax.grid(alpha=0.3, axis='y')

        # Holidays
        ax = axes[1, 1]
        holidays = [m["num_holidays"] for m in metrics]
        ax.bar(pids, holidays, color='blue', alpha=0.7)
        ax.set_ylabel("Number of Drug Holidays")
        ax.set_title("Adaptive Therapy Cycles")
        ax.tick_params(axis='x', rotation=45)
        ax.grid(alpha=0.3, axis='y')

        # Inflammation vs TTP benefit
        ax = axes[1, 2]
        infl = [m["mean_inflammation"] for m in metrics]
        ttp_ratio = [m["ttp_ratio"] for m in metrics]
        ax.scatter(infl, ttp_ratio, c='purple', s=80)
        for i, pid in enumerate(pids):
            ax.annotate(pid, (infl[i], ttp_ratio[i]), fontsize=7, xytext=(4, 4), textcoords="offset points")
        ax.axhline(1.0, color='red', ls='--', alpha=0.5, label='No benefit')
        ax.set_xlabel("Mean Inflammation Score")
        ax.set_ylabel("TTP Ratio (Adaptive/MTD)")
        ax.set_title("Inflammation Predicts Adaptive Benefit")
        ax.legend()
        ax.grid(alpha=0.3)

        plt.suptitle("Adaptive Therapy Metrics Summary (8-Patient Cohort)", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"[Viz] Metrics summary saved -> {output_path}")


# =========================================================================== #
# MAIN PIPELINE
# =========================================================================== #
def main():
    print("=" * 70)
    print("MONTH 9: ADVANCED CLONAL OPTIMIZATION - ADAPTIVE THERAPY ENGINE")
    print("=" * 70)

    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    # ---------------------------------------------------------
    # WEEK 1: Tensor field + dual-clone initialization
    # ---------------------------------------------------------
    print("\n" + "#" * 70)
    print("# WEEK 1: Dual-Clone Competitive State Space")
    print("#" * 70)

    builder = TensorFieldBuilder()
    builder.build_tract_mask()
    builder.build_orientation_field()
    builder.build_tensor_field()
    builder.validate_tensor()
    builder.save_npz(output_dir / "adaptive_tensor_profiles.npz")
    builder.plot_validation(output_dir / "adaptive_tensor_validation.png")

    # Quick initial clone visualization
    mapper = PatientParameterMapper()
    mapper.load()
    # Dummy results for initial clone plot - use (40, 40) for both clones inside diagonal tract
    dummy_results = []
    for pid in COHORT_PATIENTS:
        if pid not in mapper.data["patient_ids"]:
            continue
        params = mapper.build_patient_parameters(pid)
        u_s0, u_r0 = AdaptiveTherapySolver.initial_seeds((GRID_SIZE, GRID_SIZE), (40, 40), (40, 40))
        dummy_results.append({
            "patient_id": pid,
            "mtd": {"u_s_history": u_s0[None], "u_r_history": u_r0[None]},
            "params": params,
            "seed_centers": {"sensitive": (40, 40), "resistant": (40, 40)},
        })

    visualizer = AdaptiveVisualizer(builder)
    visualizer.plot_initial_clones(dummy_results, output_dir / "adaptive_initial_clones.png")

    # ---------------------------------------------------------
    # WEEK 2 & 3: Cohort simulation (MTD + Adaptive)
    # ---------------------------------------------------------
    print("\n" + "#" * 70)
    print("# WEEKS 2-3: MTD & Adaptive Therapy Simulation")
    print("#" * 70)

    cohort_sim = CohortSimulator(mapper, builder)
    results = cohort_sim.run_cohort(output_dir)

    # ---------------------------------------------------------
    # WEEK 4: Visualization & Metrics
    # ---------------------------------------------------------
    print("\n" + "#" * 70)
    print("# WEEK 4: Evolutionary Timeline Analytics & Visualization")
    print("#" * 70)

    visualizer.plot_comparison_canvas(results, output_dir / "adaptive_therapy_comparison.png")
    visualizer.plot_temporal_dynamics(results, output_dir / "adaptive_therapy_dynamics.png")

    metrics = visualizer.compute_metrics(results)
    visualizer.save_metrics_json(metrics, output_dir / "adaptive_geometry_metrics.json")
    visualizer.plot_metrics_summary(metrics, output_dir / "adaptive_therapy_metrics_summary.png")

    # ---------------------------------------------------------
    # SUMMARY
    # ---------------------------------------------------------
    print("\n" + "=" * 70)
    print("MONTH 9 COMPLETE: All deliverables generated in output/")
    print("=" * 70)
    print("Deliverables:")
    print("  output/adaptive_tensor_profiles.npz")
    print("  output/adaptive_tensor_validation.png")
    print("  output/adaptive_initial_clones.png")
    print("  output/adaptive_therapy_data.npz (combined cohort)")
    print("  output/adaptive_PAT_XXXX.npz (8 individual)")
    print("  output/adaptive_therapy_comparison.png")
    print("  output/adaptive_therapy_dynamics.png")
    print("  output/adaptive_geometry_metrics.json")
    print("  output/adaptive_therapy_metrics_summary.png")
    print("\nKey findings:")
    for m in metrics:
        print(f"  {m['patient_id']}: TTP_MTD={m['ttp_mtd']}, TTP_Adapt={m['ttp_adaptive']}, "
              f"VR_ratio={m['volume_ratio_adaptive']/max(m['volume_ratio_mtd'],1e-6):.2f}, "
              f"Holidays={m['num_holidays']}, DrugRed={m['drug_reduction']:.0%}")

    print("\n[SUCCESS] Month 9 complete.")


if __name__ == "__main__":
    main()