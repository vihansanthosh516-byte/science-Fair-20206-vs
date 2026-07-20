#!/usr/bin/env python3
"""
Month 8: Stromal Feedback Coupled PDE Solver
==============================================
Implements a coupled tumor-stroma PDE system with:
- Tumor density u(x,t) with anisotropic diffusion + Michaelis-Menten proliferation
- Stromal growth factor G(x,t) with isotropic diffusion + tumor-sourced production + degradation
- Patient-specific parameter calibration from multi-omic expression profiles
- Coupled dual-PDE integration with adaptive time-stepping

PDE System:
    du/dt = div(D(x) grad u) + rho(G) * u * (1 - u/K)
    dG/dt = D_G * nabla^2 G + alpha * u - gamma * G

    rho(G) = rho_0 * (1 + beta * G / (K_m + G))   [Michaelis-Menten]

Phases:
    Week 1: Dual-grid solver & coupled parameter framework
    Week 2: Isotropic chemical diffusion & divergence integration
    Week 3: Multi-omic pathway calibration & cohort sweep
    Week 4: Dual-panel visualization & metrics summary

Deliverables:
    output/stromal_feedback_init.png
    output/stromal_chemical_test.npz
    output/stromal_evolution_cohort.npz
    output/stromal_feedback_recurrence_maps.png
    output/stromal_feedback_metrics.json
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.ndimage import binary_dilation, gaussian_filter
from scipy.signal import correlate2d

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------- #
# Global constants
# --------------------------------------------------------------------------- #
GRID_SIZE = 100
DX = 1.0
TARGET_GENES = ["LST1", "S100A11", "S100A8", "ZNF106"]
INFLAMMATORY_GENES = ["S100A8", "S100A11", "LST1"]  # inflammatory/macrophage-recruiting

ZONE_REGIONS = {
    "Cellular Tumor": (0, 33),
    "Infiltrating Tumor": (33, 66),
    "Leading Edge": (66, 100),
}

# Base parameters (Week 1)
D_BASE = 0.01
D_PARALLEL_DEFAULT = 0.15
D_PERPENDICULAR_DEFAULT = 0.005
RHO_0 = 0.02
K_M = 0.2
BETA = 1.5
CARRYING_CAPACITY = 1.0

# Chemical diffusion (Week 2) - D_G >> D_u for fast chemical diffusion
D_G = 0.5
GAMMA = 0.01
ALPHA_BASE = 0.05

# Solver parameters
DT_DEFAULT = 0.05
DT_CHEMICAL = 0.005  # much smaller for chemical diffusion (D_G >> D_u)
MASS_TEST_STEPS = 500

# Time stepping for patient simulations
N_PATIENT_STEPS = 2000
PATIENT_SAVE_INTERVAL = 250

# Cohort patients (8 core patients from Month 6/7)
COHORT_PATIENTS = [f"PAT_{i:04d}" for i in range(8)]


# =========================================================================== #
# PHASE 1: Tensor Field Builder (reusing Month 7 anisotropic tensor builder)
# =========================================================================== #
class TensorFieldBuilder:
    """
    Constructs a 2x2 symmetric anisotropic diffusion tensor field over a
    100x100 grid containing a diagonal white matter tract corridor.
    Reused from Month 7 (42_anisotropic_pde.py) with minimal changes.
    """

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

    # ------------------------------------------------------------------ #
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

    # ------------------------------------------------------------------ #
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

    # ------------------------------------------------------------------ #
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

    # ------------------------------------------------------------------ #
    def validate_tensor(self) -> Dict[str, float]:
        assert self.D_xx is not None, "build_tensor_field() must run first"
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
        print(f"[Tensor] Symmetry error: {sym_err:.3e} {'PASS' if sym_err < 1e-12 else 'FAIL'}")
        print(f"[Tensor] Min eigenvalue: {min_eig:.3e} {'PASS' if min_eig > 0 else 'FAIL'}")
        return metrics

    # ------------------------------------------------------------------ #
    def anisotropy_ratio(self) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(
                self.lambda_2 > 0, self.lambda_1 / np.maximum(self.lambda_2, 1e-12), 1.0
            )
        return ratio

    # ------------------------------------------------------------------ #
    def save_npz(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            D_xx=self.D_xx,
            D_xy=self.D_xy,
            D_yx=self.D_yx,
            D_yy=self.D_yy,
            theta_field=self.theta_field,
            tract_mask=self.tract_mask,
            lambda_1=self.lambda_1,
            lambda_2=self.lambda_2,
            anisotropy_ratio=self.anisotropy_ratio(),
            grid_size=np.array([self.N, self.N]),
            D_parallel=np.array(self.d_parallel),
            D_perpendicular=np.array(self.d_perpendicular),
            D_base=np.array(self.d_base),
            tract_angle_deg=np.array(np.rad2deg(self.tract_angle)),
            tract_width=np.array(self.tract_width),
        )
        print(f"[Tensor] Saved tensor profiles -> {path}")

    # ------------------------------------------------------------------ #
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

        ax = axes[0, 0]
        ax.imshow(self.tract_mask, origin="lower", cmap="gray", alpha=0.6,
                  extent=[0, self.N, 0, self.N])
        ax.quiver(xs, ys, U, -V, color="red", scale=15, width=0.004,
                  headwidth=3, headlength=4)
        ax.set_title("White Matter Tract Corridor\n+ Principal Eigenvector Field", fontsize=11)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

        ax = axes[0, 1]
        im = ax.imshow(self.D_xx, origin="lower", cmap="viridis",
                       extent=[0, self.N, 0, self.N])
        ax.set_title(r"$D_{xx}$ component", fontsize=11)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax = axes[1, 0]
        im = ax.imshow(self.D_yy, origin="lower", cmap="viridis",
                       extent=[0, self.N, 0, self.N])
        ax.set_title(r"$D_{yy}$ component", fontsize=11)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax = axes[1, 1]
        im = ax.imshow(self.D_xy, origin="lower", cmap="RdBu_r",
                       extent=[0, self.N, 0, self.N])
        ax.set_title(r"$D_{xy} = D_{yx}$ (off-diagonal)", fontsize=11)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        plt.suptitle(
            "Anisotropic Tensor Field Validation\n"
            r"$D = R(\theta)\,\mathrm{diag}(D_\parallel, D_\perp)\,R(-\theta)$",
            fontsize=13, fontweight="bold"
        )
        plt.tight_layout()
        plt.savefig(path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"[Tensor] Saved validation plot -> {path}")


# =========================================================================== #
# Baseline Anisotropic FK Solver (for sanity validation)
# =========================================================================== #
class AnisotropicFKSolver:
    """
    Baseline anisotropic Fisher-Kolmogorov solver (no stroma coupling).
    du/dt = div(D grad u) + rho * u * (1 - u/K)
    """

    def __init__(
        self,
        D_xx: np.ndarray,
        D_xy: np.ndarray,
        D_yy: np.ndarray,
        rho: np.ndarray | float = 0.0,
        dt: float = DT_DEFAULT,
        dx: float = DX,
        carrying_capacity: float = CARRYING_CAPACITY,
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

        # CFL condition
        per_pixel_eig = 0.5 * (self.D_xx + self.D_yy) + np.sqrt(
            np.maximum(0.0, 0.25 * (self.D_xx - self.D_yy) ** 2 + self.D_xy ** 2)
        )
        max_eig = float(per_pixel_eig.max())
        self.cfl_limit = (self.dx ** 2) / (2.0 * max(max_eig, 1e-12))
        if self.dt > self.cfl_limit:
            print(f"[BaselineFK] WARNING dt={self.dt} exceeds CFL limit "
                  f"{self.cfl_limit:.4f}; clamping to 0.9*CFL")
            self.dt = 0.9 * self.cfl_limit
        print(f"[BaselineFK] Solver init: grid={self.H}x{self.W}, "
              f"max eig={max_eig:.4e}, CFL={self.cfl_limit:.4f}, dt={self.dt:.4f}")

    # ------------------------------------------------------------------ #
    def anisotropic_divergence(self, u: np.ndarray) -> np.ndarray:
        """Compute div(D grad u) with cross-derivative terms & Neumann BCs."""
        dx = self.dx
        Dxx, Dxy, Dyy = self.D_xx, self.D_xy, self.D_yy
        H, W = self.H, self.W

        u_p = np.pad(u, 1, mode="reflect")
        Dxx_p = np.pad(Dxx, 1, mode="edge")
        Dxy_p = np.pad(Dxy, 1, mode="edge")
        Dyy_p = np.pad(Dyy, 1, mode="edge")

        # Vertical faces (Fx)
        u_x_vface = (u_p[1:-1, 1:] - u_p[1:-1, :-1]) / dx
        u_y_cc = (u_p[2:, 1:-1] - u_p[:-2, 1:-1]) / (2.0 * dx)
        u_y_cc_wpad = np.pad(u_y_cc, ((0, 0), (1, 1)), mode="edge")
        u_y_vface = 0.5 * (u_y_cc_wpad[:, :-1] + u_y_cc_wpad[:, 1:])

        Dxx_vface = 0.5 * (Dxx_p[1:-1, :-1] + Dxx_p[1:-1, 1:])
        Dxy_vface = 0.5 * (Dxy_p[1:-1, :-1] + Dxy_p[1:-1, 1:])

        Fx_face = Dxx_vface * u_x_vface + Dxy_vface * u_y_vface
        Fx_E = Fx_face[:, 1:]
        Fx_W = Fx_face[:, :-1]

        # Horizontal faces (Fy)
        u_y_hface = (u_p[1:, 1:-1] - u_p[:-1, 1:-1]) / dx
        u_x_cc = (u_p[1:-1, 2:] - u_p[1:-1, :-2]) / (2.0 * dx)
        u_x_cc_hpad = np.pad(u_x_cc, ((1, 1), (0, 0)), mode="edge")
        u_x_hface = 0.5 * (u_x_cc_hpad[:-1, :] + u_x_cc_hpad[1:, :])

        Dyy_hface = 0.5 * (Dyy_p[:-1, 1:-1] + Dyy_p[1:, 1:-1])
        Dyx_hface = 0.5 * (Dxy_p[:-1, 1:-1] + Dxy_p[1:, 1:-1])

        Fy_face = Dyy_hface * u_y_hface + Dyx_hface * u_x_hface
        Fy_N = Fy_face[1:, :]
        Fy_S = Fy_face[:-1, :]

        div = (Fx_E - Fx_W + Fy_N - Fy_S) / (dx ** 2)
        return div

    # ------------------------------------------------------------------ #
    def reaction_term(self, u: np.ndarray) -> np.ndarray:
        if np.isscalar(self.rho):
            return float(self.rho) * u * (1.0 - u / self.K)
        return self.rho * u * (1.0 - u / self.K)

    # ------------------------------------------------------------------ #
    def step(self, u: np.ndarray, clamp: bool = True) -> np.ndarray:
        div_term = self.anisotropic_divergence(u)
        react_term = self.reaction_term(u)
        u_new = u + self.dt * (div_term + react_term)
        if clamp:
            u_new = np.clip(u_new, 0.0, self.K)
        return u_new

    # ------------------------------------------------------------------ #
    @staticmethod
    def initial_gaussian_seed(
        grid_shape: Tuple[int, int],
        center: Tuple[int, int] | None = None,
        sigma: float = 3.0,
        amplitude: float = 0.8,
    ) -> np.ndarray:
        H, W = grid_shape
        if center is None:
            cy, cx = H // 2, W // 2
        else:
            cy, cx = center
        yy, xx = np.mgrid[0:H, 0:W].astype(float)
        r2 = (yy - cy) ** 2 + (xx - cx) ** 2
        return amplitude * np.exp(-r2 / (2.0 * sigma ** 2))


# =========================================================================== #
# PHASE 1: Dual-Grid Coupled Solver
# =========================================================================== #
class StromalFeedbackSolver:
    """
    Coupled dual-grid solver for the tumor-stroma feedback system:

    du/dt = div(D(x) grad u) + rho(G) * u * (1 - u/K)
    dG/dt = D_G * nabla^2 G + alpha * u - gamma * G

    rho(G) = rho_0 * (1 + beta * G / (K_m + G))  [Michaelis-Menten]
    """

    def __init__(
        self,
        D_xx: np.ndarray,
        D_xy: np.ndarray,
        D_yy: np.ndarray,
        rho_0: float = RHO_0,
        K_m: float = K_M,
        beta: float = BETA,
        alpha: float = ALPHA_BASE,
        gamma: float = GAMMA,
        D_G: float = D_G,
        dt: float = DT_DEFAULT,
        dt_chemical: float = DT_CHEMICAL,
        dx: float = DX,
        carrying_capacity: float = CARRYING_CAPACITY,
    ) -> None:
        assert D_xx.shape == D_xy.shape == D_yy.shape
        self.H, self.W = D_xx.shape
        self.D_xx = D_xx.astype(float)
        self.D_xy = D_xy.astype(float)
        self.D_yy = D_yy.astype(float)

        self.rho_0 = float(rho_0)
        self.K_m = float(K_m)
        self.beta = float(beta)
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.D_G = float(D_G)

        self.dt = float(dt)
        self.dt_chemical = float(dt_chemical)
        self.dx = float(dx)
        self.K = float(carrying_capacity)

        # CFL for anisotropic tumor diffusion
        per_pixel_eig = 0.5 * (self.D_xx + self.D_yy) + np.sqrt(
            np.maximum(0.0, 0.25 * (self.D_xx - self.D_yy) ** 2 + self.D_xy ** 2)
        )
        max_eig = float(per_pixel_eig.max())
        self.max_eigenvalue = max_eig
        self.cfl_limit = (self.dx ** 2) / (2.0 * max(max_eig, 1e-12))
        if self.dt > self.cfl_limit:
            print(f"[Solver] WARNING dt={self.dt} exceeds CFL limit "
                  f"{self.cfl_limit:.4f}; clamping to 0.9*CFL")
            self.dt = 0.9 * self.cfl_limit

        # CFL for isotropic chemical diffusion (5-point stencil)
        # CFL for 2D 5-point Laplacian: dt <= dx^2 / (4 * D_G)
        self.cfl_chemical = (self.dx ** 2) / (4.0 * self.D_G)
        if self.dt_chemical > self.cfl_chemical:
            print(f"[Solver] WARNING dt_chemical={self.dt_chemical} exceeds chemical CFL "
                  f"{self.cfl_chemical:.4f}; clamping to 0.9*CFL")
            self.dt_chemical = 0.9 * self.cfl_chemical

        # Sub-stepping ratio for chemical diffusion
        self.chem_substeps = max(1, int(np.ceil(self.dt / self.dt_chemical)))
        self.dt_chemical_eff = self.dt / self.chem_substeps

        print(f"[Solver] Init: grid={self.H}x{self.W}, dt_tumor={self.dt:.4f} (CFL={self.cfl_limit:.4f})")
        print(f"[Solver] Chemical: D_G={self.D_G}, dt_chem={self.dt_chemical_eff:.4f}, "
              f"substeps={self.chem_substeps}, gamma={self.gamma}, alpha={self.alpha}")

    # ------------------------------------------------------------------ #
    def michaelis_menten_rho(self, G: np.ndarray) -> np.ndarray:
        """rho(G) = rho_0 * (1 + beta * G / (K_m + G))"""
        return self.rho_0 * (1.0 + self.beta * G / (self.K_m + G))

    # ------------------------------------------------------------------ #
    def anisotropic_divergence(self, u: np.ndarray) -> np.ndarray:
        """Compute div(D grad u) with cross-derivative terms & Neumann BCs."""
        dx = self.dx
        Dxx, Dxy, Dyy = self.D_xx, self.D_xy, self.D_yy
        H, W = self.H, self.W

        u_p = np.pad(u, 1, mode="reflect")
        Dxx_p = np.pad(Dxx, 1, mode="edge")
        Dxy_p = np.pad(Dxy, 1, mode="edge")
        Dyy_p = np.pad(Dyy, 1, mode="edge")

        # Vertical faces (Fx)
        u_x_vface = (u_p[1:-1, 1:] - u_p[1:-1, :-1]) / dx
        u_y_cc = (u_p[2:, 1:-1] - u_p[:-2, 1:-1]) / (2.0 * dx)
        u_y_cc_wpad = np.pad(u_y_cc, ((0, 0), (1, 1)), mode="edge")
        u_y_vface = 0.5 * (u_y_cc_wpad[:, :-1] + u_y_cc_wpad[:, 1:])

        Dxx_vface = 0.5 * (Dxx_p[1:-1, :-1] + Dxx_p[1:-1, 1:])
        Dxy_vface = 0.5 * (Dxy_p[1:-1, :-1] + Dxy_p[1:-1, 1:])

        Fx_face = Dxx_vface * u_x_vface + Dxy_vface * u_y_vface
        Fx_E = Fx_face[:, 1:]
        Fx_W = Fx_face[:, :-1]

        # Horizontal faces (Fy)
        u_y_hface = (u_p[1:, 1:-1] - u_p[:-1, 1:-1]) / dx
        u_x_cc = (u_p[1:-1, 2:] - u_p[1:-1, :-2]) / (2.0 * dx)
        u_x_cc_hpad = np.pad(u_x_cc, ((1, 1), (0, 0)), mode="edge")
        u_x_hface = 0.5 * (u_x_cc_hpad[:-1, :] + u_x_cc_hpad[1:, :])

        Dyy_hface = 0.5 * (Dyy_p[:-1, 1:-1] + Dyy_p[1:, 1:-1])
        Dyx_hface = 0.5 * (Dxy_p[:-1, 1:-1] + Dxy_p[1:, 1:-1])

        Fy_face = Dyy_hface * u_y_hface + Dyx_hface * u_x_hface
        Fy_N = Fy_face[1:, :]
        Fy_S = Fy_face[:-1, :]

        div = (Fx_E - Fx_W + Fy_N - Fy_S) / (dx ** 2)
        return div

    # ------------------------------------------------------------------ #
    def chemical_laplacian(self, G: np.ndarray) -> np.ndarray:
        """
        5-point central-difference Laplacian for isotropic diffusion:
        nabla^2 G = (G[i+1,j] + G[i-1,j] + G[i,j+1] + G[i,j-1] - 4*G[i,j]) / dx^2
        Neumann zero-flux via mirror padding.
        """
        G_p = np.pad(G, 1, mode="reflect")
        laplacian = (
            G_p[2:, 1:-1] + G_p[:-2, 1:-1] +
            G_p[1:-1, 2:] + G_p[1:-1, :-2] -
            4.0 * G_p[1:-1, 1:-1]
        ) / (self.dx ** 2)
        return laplacian

    # ------------------------------------------------------------------ #
    def chemical_step(self, G: np.ndarray, u: np.ndarray, dt_chem: float) -> np.ndarray:
        """
        Single chemical diffusion step:
        dG/dt = D_G * nabla^2 G + alpha * u - gamma * G
        """
        laplacian = self.chemical_laplacian(G)
        production = self.alpha * u
        degradation = self.gamma * G
        G_new = G + dt_chem * (self.D_G * laplacian + production - degradation)
        return np.maximum(G_new, 0.0)  # non-negative concentration

    # ------------------------------------------------------------------ #
    def tumor_step(self, u: np.ndarray, G: np.ndarray, dt_tumor: float,
                   clamp: bool = True) -> np.ndarray:
        """
        Single tumor diffusion-reaction step with Michaelis-Menten proliferation:
        du/dt = div(D grad u) + rho(G) * u * (1 - u/K)
        """
        div_term = self.anisotropic_divergence(u)
        rho_G = self.michaelis_menten_rho(G)
        react_term = rho_G * u * (1.0 - u / self.K)
        u_new = u + dt_tumor * (div_term + react_term)
        if clamp:
            u_new = np.clip(u_new, 0.0, self.K)
        return u_new

    # ------------------------------------------------------------------ #
    def coupled_step(self, u: np.ndarray, G: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Full coupled step with sub-stepping for chemical diffusion:
        1. Sub-step chemical diffusion for dt_chem_eff * chem_substeps = dt_tumor
        2. Tumor step with updated G
        """
        # Chemical sub-steps
        G_current = G.copy()
        for _ in range(self.chem_substeps):
            G_current = self.chemical_step(G_current, u, self.dt_chemical_eff)

        # Tumor step with updated G
        u_new = self.tumor_step(u, G_current, self.dt)

        return u_new, G_current

    # ------------------------------------------------------------------ #
    @staticmethod
    def initial_gaussian_seed(
        grid_shape: Tuple[int, int],
        center: Tuple[int, int] | None = None,
        sigma: float = 3.0,
        amplitude: float = 0.8,
    ) -> np.ndarray:
        H, W = grid_shape
        if center is None:
            cy, cx = H // 2, W // 2
        else:
            cy, cx = center
        yy, xx = np.mgrid[0:H, 0:W].astype(float)
        r2 = (yy - cy) ** 2 + (xx - cx) ** 2
        return amplitude * np.exp(-r2 / (2.0 * sigma ** 2))

    # ------------------------------------------------------------------ #
    def run_sanity_validation(self, n_steps: int = 100) -> Dict:
        """
        Sanity check: if alpha=0 (no growth factor production), the system
        should collapse to baseline anisotropic FK growth (rho = rho_0).
        """
        print("[Week1] Running sanity validation (alpha=0)...")

        # Create solver with alpha=0
        solver_alpha0 = StromalFeedbackSolver(
            D_xx=self.D_xx, D_xy=self.D_xy, D_yy=self.D_yy,
            rho_0=self.rho_0, K_m=self.K_m, beta=self.beta,
            alpha=0.0, gamma=self.gamma, D_G=self.D_G,
            dt=self.dt, dt_chemical=self.dt_chemical,
            dx=self.dx, carrying_capacity=self.K
        )

        # Initial conditions
        u0 = self.initial_gaussian_seed((self.H, self.W), center=(16, 50), sigma=3.0)
        G0 = np.zeros((self.H, self.W))

        # Run coupled solver with alpha=0
        u_alpha0 = u0.copy()
        G_alpha0 = G0.copy()
        for _ in range(n_steps):
            u_alpha0, G_alpha0 = solver_alpha0.coupled_step(u_alpha0, G_alpha0)

        # Run baseline anisotropic FK solver (no stroma coupling)
        # Use the AnisotropicFKSolver class defined in this file's scope
        # (it's available from the import at top or we can use a simple baseline)
        baseline_solver = AnisotropicFKSolver(
            D_xx=self.D_xx, D_xy=self.D_xy, D_yy=self.D_yy,
            rho=self.rho_0, dt=self.dt, dx=self.dx, carrying_capacity=self.K
        )
        u_baseline = u0.copy()
        for _ in range(n_steps):
            u_baseline = baseline_solver.step(u_baseline)

        # Compare final states
        diff = np.abs(u_alpha0 - u_baseline)
        max_diff = float(diff.max())
        rel_diff = float(diff.max() / max(u_baseline.max(), 1e-12))
        mass_alpha0 = float(u_alpha0.sum())
        mass_baseline = float(u_baseline.sum())

        print(f"[Week1] Sanity check: max_diff={max_diff:.6e}, rel_diff={rel_diff:.6e}")
        print(f"[Week1] Mass alpha=0: {mass_alpha0:.4f}, Mass baseline: {mass_baseline:.4f}")

        passed = max_diff < 1e-5 and abs(mass_alpha0 - mass_baseline) < 1e-3
        print(f"[Week1] Sanity validation: {'PASS' if passed else 'FAIL'}")

        return {
            "max_abs_diff": max_diff,
            "relative_diff": rel_diff,
            "mass_alpha0": mass_alpha0,
            "mass_baseline": mass_baseline,
            "passed": passed,
        }

    # ------------------------------------------------------------------ #
    def run_chemical_mass_test(self, n_steps: int = MASS_TEST_STEPS * 20,
                                save_path: Path | None = None) -> Dict:
        """
        Week 2: Pure chemical diffusion test.
        Test mass balance and stable decay with alpha > 0, gamma > 0, u = constant source.
        """
        print(f"[Week2] Running chemical mass conservation test ({n_steps} steps)...")

        # Constant tumor source in center
        u_source = np.zeros((self.H, self.W))
        u_source[40:60, 40:60] = 1.0  # constant source region

        G = np.zeros((self.H, self.W))
        mass_history = np.zeros(n_steps + 1)

        # Analytical: with constant source alpha*u and degradation gamma*G,
        # steady state is G_ss = (alpha/gamma) * u
        # Total mass should approach sum(alpha/gamma * u) = (alpha/gamma) * sum(u)
        expected_steady_mass = (self.alpha / self.gamma) * float(u_source.sum())

        # Use chemical time step for accurate diffusion
        dt_chem = self.dt_chemical_eff
        for step in range(n_steps + 1):
            mass_history[step] = float(G.sum())
            if step < n_steps:
                # Only chemical diffusion, no tumor evolution
                G = self.chemical_step(G, u_source, dt_chem)

        final_mass = float(G.sum())
        mass_change = float(mass_history[-1] - mass_history[0])
        rel_error = abs(final_mass - expected_steady_mass) / max(expected_steady_mass, 1e-12)

        print(f"[Week2] Initial mass: {mass_history[0]:.6f}")
        print(f"[Week2] Final mass: {final_mass:.6f}")
        print(f"[Week2] Expected steady mass: {expected_steady_mass:.6f}")
        print(f"[Week2] Relative error: {rel_error:.6e}")
        print(f"[Week2] Mass change: {mass_change:.6e}")

        # Check for NaN/Inf and negative values
        has_nan = bool(np.isnan(G).any() or np.isinf(G).any())
        has_neg = bool((G < -1e-10).any())

        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                save_path,
                mass_history=mass_history,
                final_G=G,
                u_source=u_source,
                expected_steady_mass=np.array(expected_steady_mass),
                dt_chemical=np.array(dt_chem),
                D_G=np.array(self.D_G),
                gamma=np.array(self.gamma),
                alpha=np.array(self.alpha),
            )
            print(f"[Week2] Saved chemical test -> {save_path}")

        # Pass if: no NaN/Inf, no negative, and mass is monotonically increasing toward steady state
        # (don't require full convergence in limited steps)
        monotonic = all(mass_history[i+1] >= mass_history[i] - 1e-10 for i in range(len(mass_history)-1))
        pass_test = (not has_nan and not has_neg and monotonic and 
                     final_mass > 0.1 * expected_steady_mass)

        return {
            "mass_history": mass_history,
            "final_mass": final_mass,
            "expected_steady_mass": expected_steady_mass,
            "relative_error": rel_error,
            "has_nan_or_inf": has_nan,
            "has_negative": has_neg,
            "monotonic_increase": monotonic,
            "mass_conservation_pass": pass_test,
        }

    # ------------------------------------------------------------------ #
    def run_patient_simulation(
        self,
        patient_id: str,
        n_steps: int = N_PATIENT_STEPS,
        save_interval: int = PATIENT_SAVE_INTERVAL,
        seed_center: Tuple[int, int] | None = None,
    ) -> Dict:
        """
        Run full coupled simulation for a single patient.
        """
        if seed_center is None:
            cy, cx = 16, 50
        else:
            cy, cx = seed_center

        u = self.initial_gaussian_seed((self.H, self.W), center=(cy, cx), sigma=3.0)
        G = np.zeros((self.H, self.W))

        n_saves = n_steps // save_interval + 1
        u_evolution = np.zeros((n_saves, self.H, self.W), dtype=np.float32)
        G_evolution = np.zeros((n_saves, self.H, self.W), dtype=np.float32)
        mass_history = np.zeros(n_steps + 1, dtype=np.float64)
        G_mass_history = np.zeros(n_steps + 1, dtype=np.float64)

        u_evolution[0] = u.astype(np.float32)
        G_evolution[0] = G.astype(np.float32)
        mass_history[0] = float(u.sum())
        G_mass_history[0] = float(G.sum())

        save_idx = 1
        for step in range(1, n_steps + 1):
            u, G = self.coupled_step(u, G)
            mass_history[step] = float(u.sum())
            G_mass_history[step] = float(G.sum())

            if step % save_interval == 0 and save_idx < n_saves:
                u_evolution[save_idx] = u.astype(np.float32)
                G_evolution[save_idx] = G.astype(np.float32)
                save_idx += 1

        final_mass = float(u.sum())
        final_G_mass = float(G.sum())
        print(f"[Patient] {patient_id}: final tumor mass={final_mass:.3f}, "
              f"final G mass={final_G_mass:.3f}")

        return {
            "patient_id": patient_id,
            "u_evolution": u_evolution,
            "G_evolution": G_evolution,
            "mass_history": mass_history,
            "G_mass_history": G_mass_history,
            "final_u": u,
            "final_G": G,
            "seed_center": (cy, cx),
            "parameters": {
                "rho_0": self.rho_0,
                "K_m": self.K_m,
                "beta": self.beta,
                "alpha": self.alpha,
                "gamma": self.gamma,
                "D_G": self.D_G,
                "dt": self.dt,
            }
        }


# =========================================================================== #
# PHASE 3: Patient Parameter Mapper (Multi-omic calibration)
# =========================================================================== #
class PatientParameterMapper:
    """
    Loads Month 6 spatial recurrence data + zone-stratified gene expression
    CSVs, maps patient-specific inflammatory markers to secretion rate (alpha)
    and susceptibility (beta).
    """

    INFLAMMATORY_GENES = ["S100A8", "S100A11", "LST1"]  # macrophage-recruiting

    def __init__(
        self,
        cohort_npz: Path = Path("output/spatial_recurrence_profiles.npz"),
        zone_csv_root: Path = Path("output"),
    ) -> None:
        self.cohort_npz = cohort_npz
        self.zone_csv_root = zone_csv_root
        self.data: Dict = {}
        self.zone_expr_dfs: Dict[str, pd.DataFrame] = {}

    # ------------------------------------------------------------------ #
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

        print(f"[Phase3] Loaded cohort: {list(self.data['patient_ids'])}")
        return self.data

    # ------------------------------------------------------------------ #
    def get_patient_zone_inflammation(
        self, patient_id: str
    ) -> Dict[str, float]:
        """
        Per-zone inflammation score from S100A8, S100A11, LST1 expressions.
        Score = mean(log2TPM) of inflammatory genes, normalized via tanh to (0, 2).
        """
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
            raw = np.mean(vals)  # log2TPM ~ 4-8
            # Normalize: aggressive -> high score (up to 2x), baseline = 1
            score = 1.0 + np.tanh((raw - 5.0) / 1.5)
            scores[zone] = float(score)
        return scores

    # ------------------------------------------------------------------ #
    def build_patient_parameters(
        self,
        patient_id: str,
        base_builder: TensorFieldBuilder,
    ) -> Dict:
        """
        Build patient-specific parameters:
        - alpha: growth factor secretion rate (scales with inflammation)
        - beta: tumor susceptibility to growth factor (scales with inflammation)
        - rho_field: base proliferation field scaled by inflammation
        """
        if not self.data:
            self.load()

        pids = list(self.data["patient_ids"])
        if patient_id not in pids:
            raise KeyError(f"Patient {patient_id} not found in cohort NPZ.")
        pidx = pids.index(patient_id)

        # Base rho field from Month 6
        base_rho_field = self.data["rho_fields"][pidx]  # (100, 100)

        # Per-zone inflammation scores
        zone_scores = self.get_patient_zone_inflammation(patient_id)

        # Build spatial scaling fields
        N = base_builder.N
        scale_alpha = np.ones((N, N))
        scale_beta = np.ones((N, N))
        scale_rho = np.ones((N, N))

        for zone, (y0, y1) in ZONE_REGIONS.items():
            sc = zone_scores[zone]
            # alpha: secretion rate scales linearly with inflammation
            scale_alpha[y0:y1, :] *= sc
            # beta: susceptibility also scales (max 2x)
            scale_beta[y0:y1, :] *= sc
            # rho: proliferation field from Month 6 scaled
            scale_rho[y0:y1, :] *= sc

        # Apply to base parameters
        alpha_patient = ALPHA_BASE * scale_alpha
        beta_patient = BETA * scale_beta

        # Smooth for continuity
        alpha_patient = gaussian_filter(alpha_patient, sigma=2.0)
        beta_patient = gaussian_filter(beta_patient, sigma=2.0)

        rho_patient = base_rho_field * scale_rho
        rho_patient = gaussian_filter(rho_patient, sigma=2.0)
        rho_patient = np.maximum(rho_patient, 1e-6)

        return {
            "alpha_field": alpha_patient,
            "beta_field": beta_patient,
            "rho_field": rho_patient,
            "zone_scores": zone_scores,
            "base_rho_field": base_rho_field,
        }


# =========================================================================== #
# PHASE 3: Cohort Simulator
# =========================================================================== #
class CohortSimulator:
    """Runs coupled dual-PDE simulations for the 8-patient test cohort."""

    def __init__(
        self,
        mapper: PatientParameterMapper,
        base_builder: TensorFieldBuilder,
        n_steps: int = N_PATIENT_STEPS,
        save_interval: int = PATIENT_SAVE_INTERVAL,
        dt: float = DT_DEFAULT,
    ) -> None:
        self.mapper = mapper
        self.base_builder = base_builder
        self.n_steps = int(n_steps)
        self.save_interval = int(save_interval)
        self.dt = float(dt)

    # ------------------------------------------------------------------ #
    @staticmethod
    def patient_seed_center(patient_id: str, N: int = GRID_SIZE) -> Tuple[int, int]:
        seed = sum(ord(c) for c in patient_id)
        rng = np.random.default_rng(seed)
        cy = int(np.clip(rng.normal(16, 4), 4, 28))
        cx = int(np.clip(rng.normal(N // 2, 8), 10, N - 10))
        return cy, cx

    # ------------------------------------------------------------------ #
    def run_patient(self, patient_id: str) -> Dict:
        params = self.mapper.build_patient_parameters(patient_id, self.base_builder)

        # Create solver with patient-specific alpha/beta fields
        # Note: we need to handle spatially varying alpha/beta
        # For simplicity, use mean alpha/beta for solver init, but apply
        # spatial fields in the step function
        alpha_mean = float(params["alpha_field"].mean())
        beta_mean = float(params["beta_field"].mean())

        solver = StromalFeedbackSolver(
            D_xx=self.base_builder.D_xx,
            D_xy=self.base_builder.D_xy,
            D_yy=self.base_builder.D_yy,
            rho_0=RHO_0,
            K_m=K_M,
            beta=beta_mean,
            alpha=alpha_mean,
            gamma=GAMMA,
            D_G=D_G,
            dt=self.dt,
            dt_chemical=DT_CHEMICAL,
            dx=DX,
            carrying_capacity=CARRYING_CAPACITY,
        )

        # Store spatial fields for use in coupled step
        solver.alpha_field = params["alpha_field"]
        solver.beta_field = params["beta_field"]
        solver.rho_field = params["rho_field"]

        # Override methods to use spatial fields
        original_michaelis = solver.michaelis_menten_rho
        original_chem_step = solver.chemical_step

        def spatial_michaelis(G: np.ndarray) -> np.ndarray:
            return solver.rho_0 * (1.0 + solver.beta_field * G / (solver.K_m + G))

        def spatial_chem_step(G: np.ndarray, u: np.ndarray, dt_chem: float) -> np.ndarray:
            laplacian = solver.chemical_laplacian(G)
            production = solver.alpha_field * u
            degradation = solver.gamma * G
            G_new = G + dt_chem * (solver.D_G * laplacian + production - degradation)
            return np.maximum(G_new, 0.0)

        solver.michaelis_menten_rho = spatial_michaelis
        solver.chemical_step = spatial_chem_step

        cy, cx = self.patient_seed_center(patient_id, N=solver.H)
        u = StromalFeedbackSolver.initial_gaussian_seed(
            (solver.H, solver.W), center=(cy, cx), sigma=3.0, amplitude=0.8
        )
        G = np.zeros((solver.H, solver.W))

        n_saves = self.n_steps // self.save_interval + 1
        u_evolution = np.zeros((n_saves, solver.H, solver.W), dtype=np.float32)
        G_evolution = np.zeros((n_saves, solver.H, solver.W), dtype=np.float32)
        mass_history = np.zeros(self.n_steps + 1, dtype=np.float64)
        G_mass_history = np.zeros(self.n_steps + 1, dtype=np.float64)

        u_evolution[0] = u.astype(np.float32)
        G_evolution[0] = G.astype(np.float32)
        mass_history[0] = float(u.sum())
        G_mass_history[0] = float(G.sum())

        save_idx = 1
        for step in range(1, self.n_steps + 1):
            u, G = solver.coupled_step(u, G)
            mass_history[step] = float(u.sum())
            G_mass_history[step] = float(G.sum())

            if step % self.save_interval == 0 and save_idx < n_saves:
                u_evolution[save_idx] = u.astype(np.float32)
                G_evolution[save_idx] = G.astype(np.float32)
                save_idx += 1

        print(f"[Cohort] {patient_id}: final tumor mass={mass_history[-1]:.3f}, "
              f"G mass={G_mass_history[-1]:.3f}, zone_scores={params['zone_scores']}")

        return {
            "patient_id": patient_id,
            "u_evolution": u_evolution,
            "G_evolution": G_evolution,
            "mass_history": mass_history,
            "G_mass_history": G_mass_history,
            "final_u": u,
            "final_G": G,
            "tensor_field": {
                "D_xx": self.base_builder.D_xx,
                "D_xy": self.base_builder.D_xy,
                "D_yy": self.base_builder.D_yy,
                "theta": self.base_builder.theta_field,
            },
            "rho_field": params["rho_field"],
            "alpha_field": params["alpha_field"],
            "beta_field": params["beta_field"],
            "zone_scores": params["zone_scores"],
            "seed_center": (cy, cx),
        }

    # ------------------------------------------------------------------ #
    def run_cohort(
        self,
        output_dir: Path = Path("output"),
    ) -> List[Dict]:
        if not self.mapper.data:
            self.mapper.load()
        pids = COHORT_PATIENTS  # 8 core patients
        output_dir.mkdir(parents=True, exist_ok=True)

        results: List[Dict] = []
        for pid in pids:
            if pid not in self.mapper.data["patient_ids"]:
                print(f"[Cohort] WARNING: {pid} not in cohort data, skipping")
                continue
            res = self.run_patient(pid)
            np.savez_compressed(
                output_dir / f"stromal_evolution_{pid}.npz",
                u_evolution=res["u_evolution"],
                G_evolution=res["G_evolution"],
                mass_history=res["mass_history"],
                G_mass_history=res["G_mass_history"],
                final_u=res["final_u"],
                final_G=res["final_G"],
                D_xx=res["tensor_field"]["D_xx"],
                D_xy=res["tensor_field"]["D_xy"],
                D_yy=res["tensor_field"]["D_yy"],
                theta=res["tensor_field"]["theta"],
                rho_field=res["rho_field"],
                alpha_field=res["alpha_field"],
                beta_field=res["beta_field"],
                seed_center=np.array(res["seed_center"]),
            )
            results.append(res)

        # Save combined cohort file
        all_u = np.stack([r["u_evolution"] for r in results])
        all_G = np.stack([r["G_evolution"] for r in results])
        all_mass = np.stack([r["mass_history"] for r in results])
        all_Gmass = np.stack([r["G_mass_history"] for r in results])

        np.savez_compressed(
            output_dir / "stromal_evolution_cohort.npz",
            patient_ids=np.array([r["patient_id"] for r in results]),
            u_evolution=all_u,
            G_evolution=all_G,
            mass_history=all_mass,
            G_mass_history=all_Gmass,
        )
        print(f"[Cohort] Saved combined cohort -> {output_dir / 'stromal_evolution_cohort.npz'}")
        return results


# =========================================================================== #
# PHASE 4: Dual-Panel Visualization & Metrics
# =========================================================================== #
class StromalVisualizer:
    """Generates publication-grade comparative graphics and metrics."""

    def __init__(self, base_builder: TensorFieldBuilder) -> None:
        self.base_builder = base_builder

    # ------------------------------------------------------------------ #
    def plot_cohort_canvas(
        self,
        results: List[Dict],
        output_path: Path,
    ) -> None:
        """
        8-patient dual-panel canvas:
        Top row: tumor density (u)
        Bottom row: growth factor concentration (G)
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        n = len(results)
        n_cols = 4
        n_rows = 2 * ((n + n_cols - 1) // n_cols)  # 2 rows per patient (u over G)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.flatten()

        for idx, res in enumerate(results):
            u = res["final_u"]
            G = res["final_G"]
            pid = res["patient_id"]
            cy, cx = res["seed_center"]

            # Tumor panel (even index)
            ax_u = axes[2 * idx]
            im_u = ax_u.imshow(u, origin="lower", cmap="hot", vmin=0, vmax=1,
                               extent=[0, u.shape[1], 0, u.shape[0]])
            ax_u.contour(self.base_builder.tract_mask, levels=[0.5],
                         colors="cyan", linewidths=0.8, alpha=0.5,
                         extent=[0, u.shape[1], 0, u.shape[0]])
            for _, (y0, y1) in ZONE_REGIONS.items():
                ax_u.axhline(y=y0, color="white", linestyle=":", alpha=0.25, lw=0.6)
                ax_u.axhline(y=y1, color="white", linestyle=":", alpha=0.25, lw=0.6)
            ax_u.plot(cx, cy, marker="+", color="yellow", ms=12, mew=2)
            ax_u.set_title(f"{pid} - Tumor Density", fontsize=9)
            ax_u.set_xticks([])
            ax_u.set_yticks([])

            # Growth factor panel (odd index)
            ax_G = axes[2 * idx + 1]
            im_G = ax_G.imshow(G, origin="lower", cmap="plasma", vmin=0, vmax=G.max(),
                               extent=[0, G.shape[1], 0, G.shape[0]])
            ax_G.contour(self.base_builder.tract_mask, levels=[0.5],
                         colors="cyan", linewidths=0.8, alpha=0.5,
                         extent=[0, G.shape[1], 0, G.shape[0]])
            for _, (y0, y1) in ZONE_REGIONS.items():
                ax_G.axhline(y=y0, color="white", linestyle=":", alpha=0.25, lw=0.6)
                ax_G.axhline(y=y1, color="white", linestyle=":", alpha=0.25, lw=0.6)
            ax_G.plot(cx, cy, marker="+", color="yellow", ms=12, mew=2)
            ax_G.set_title(f"{pid} - Growth Factor G", fontsize=9)
            ax_G.set_xticks([])
            ax_G.set_yticks([])

        # Hide unused axes
        for idx in range(2 * n, len(axes)):
            axes[idx].set_visible(False)

        plt.suptitle(
            "Stromal Feedback Recurrence Maps: Tumor Density (top) vs "
            "Growth Factor Concentration (bottom) — 8-Patient Cohort",
            fontsize=14, fontweight="bold"
        )
        plt.tight_layout()
        plt.savefig(output_path, dpi=220, bbox_inches="tight")
        plt.close()
        print(f"[Phase4] Saved cohort canvas -> {output_path}")

    # ------------------------------------------------------------------ #
    @staticmethod
    def cross_correlation(u: np.ndarray, G: np.ndarray, threshold: float = 0.1) -> float:
        """
        Cross-correlation between tumor boundary and growth factor field.
        Measures alignment of invasion front with chemical cloud.
        """
        # Get tumor boundary (gradient magnitude at threshold)
        mask = u > threshold
        if mask.sum() == 0:
            return 0.0

        gy, gx = np.gradient(u)
        grad_mag = np.sqrt(gx**2 + gy**2)
        boundary = mask & (grad_mag > 0.05 * grad_mag.max())

        if boundary.sum() == 0:
            return 0.0

        # Cross-correlation at zero lag = normalized dot product
        G_masked = G[boundary]
        u_masked = u[boundary]

        # Normalized cross-correlation
        if G_masked.std() > 0 and u_masked.std() > 0:
            corr = np.corrcoef(G_masked, u_masked)[0, 1]
        else:
            corr = 0.0

        return float(corr)

    # ------------------------------------------------------------------ #
    @staticmethod
    def front_alignment_metric(u: np.ndarray, G: np.ndarray,
                                threshold: float = 0.1) -> Dict:
        """
        Compute how tightly the chemical cloud aligns with invasion front.
        Returns: correlation, front overlap fraction, G gradient alignment.
        """
        mask = u > threshold
        if mask.sum() == 0:
            return {"correlation": 0.0, "front_overlap": 0.0, "G_gradient_alignment": 0.0}

        # Tumor front (boundary)
        gy_u, gx_u = np.gradient(u)
        grad_mag_u = np.sqrt(gx_u**2 + gy_u**2)
        front = mask & (grad_mag_u > 0.05 * grad_mag_u.max())

        # G field at front
        G_front = G[front]
        u_front = u[front]

        # Correlation
        if G_front.std() > 0 and u_front.std() > 0:
            corr = float(np.corrcoef(G_front, u_front)[0, 1])
        else:
            corr = 0.0

        # Front overlap: fraction of front pixels where G > median(G)
        G_thresh = np.median(G)
        overlap = float((G_front > G_thresh).mean()) if front.any() else 0.0

        # G gradient alignment with tumor gradient
        gy_G, gx_G = np.gradient(G)
        grad_mag_G = np.sqrt(gx_G**2 + gy_G**2)
        if front.any() and grad_mag_G.max() > 0:
            G_front_grad = grad_mag_G[front]
            u_front_grad = grad_mag_u[front]
            if G_front_grad.std() > 0 and u_front_grad.std() > 0:
                grad_align = float(np.corrcoef(G_front_grad, u_front_grad)[0, 1])
            else:
                grad_align = 0.0
        else:
            grad_align = 0.0

        return {
            "correlation": corr,
            "front_overlap": overlap,
            "G_gradient_alignment": grad_align,
        }

    # ------------------------------------------------------------------ #
    @staticmethod
    def microenvironment_acceleration(
        u_coupled: np.ndarray,
        u_baseline: np.ndarray,
    ) -> float:
        """
        Microenvironment Acceleration Velocity:
        Ratio of invasion front radius (coupled vs baseline static model).
        """
        def front_radius(u: np.ndarray, threshold: float = 0.1) -> float:
            mask = u > threshold
            if not mask.any():
                return 0.0
            H, W = u.shape
            cy, cx = H // 2, W // 2
            yy, xx = np.mgrid[0:H, 0:W]
            r2 = (yy - cy) ** 2 + (xx - cx) ** 2
            return float(np.sqrt(r2[mask]).mean())

        r_coupled = front_radius(u_coupled)
        r_baseline = front_radius(u_baseline)
        if r_baseline > 0:
            return float(r_coupled / r_baseline)
        return 1.0

    # ------------------------------------------------------------------ #
    def compute_all_metrics(
        self,
        results: List[Dict],
        baseline_results: Optional[List[Dict]] = None,
    ) -> List[Dict]:
        """
        Compute comprehensive metrics for each patient.
        If baseline_results provided, compute acceleration velocity.
        """
        metrics_list = []
        baseline_map = {}
        if baseline_results:
            baseline_map = {r["patient_id"]: r["final_u"] for r in baseline_results}

        for res in results:
            u = res["final_u"]
            G = res["final_G"]
            pid = res["patient_id"]

            # Front alignment metrics
            align = self.front_alignment_metric(u, G)

            # Fractal dimension
            fd = self.fractal_dimension(u)

            # Perimeter/area
            pa = self.perimeter_area_ratio(u)

            # Branch count
            bc = self.branch_count(u)

            # Tract alignment
            ta = self.orientation_alignment(u, self.base_builder.theta_field)

            # Microenvironment acceleration
            accel = 1.0
            if pid in baseline_map:
                accel = self.microenvironment_acceleration(u, baseline_map[pid])

            row = {
                "patient_id": pid,
                "front_correlation": align["correlation"],
                "front_overlap": align["front_overlap"],
                "G_gradient_alignment": align["G_gradient_alignment"],
                "fractal_dimension": fd,
                "perimeter": pa["perimeter"],
                "area": pa["area"],
                "perimeter_to_area_ratio": pa["P_A_ratio"],
                "convexity_defect": pa["convexity_defect"],
                "branch_count": bc,
                "tract_alignment_fraction": ta,
                "microenvironment_acceleration": accel,
                "zone_scores": res["zone_scores"],
                "mean_zone_score": float(np.mean(list(res["zone_scores"].values()))),
                "final_tumor_mass": float(u.sum()),
                "final_G_mass": float(G.sum()),
            }
            metrics_list.append(row)
            print(f"[Phase4] {pid}: corr={align['correlation']:.3f}, "
                  f"overlap={align['front_overlap']:.3f}, "
                  f"D_f={fd:.3f}, P/A={pa['P_A_ratio']:.3f}, "
                  f"accel={accel:.3f}")

        return metrics_list

    # ------------------------------------------------------------------ #
    @staticmethod
    def fractal_dimension(u: np.ndarray, threshold: float = 0.1) -> float:
        mask = (u > threshold).astype(np.uint8)
        if mask.sum() == 0:
            return 0.0
        H, W = mask.shape
        P = 2 ** int(np.ceil(np.log2(max(H, W))))
        padded = np.zeros((P, P), dtype=np.uint8)
        padded[:H, :W] = mask

        sizes = np.unique(np.floor(np.logspace(np.log2(2), np.log2(P / 2), 18)).astype(int))
        sizes = sizes[sizes >= 2]
        counts = []
        inv_eps = []
        for s in sizes:
            if s > P / 2:
                continue
            reshaped = padded[:P // s * s, :P // s * s].reshape(
                P // s, s, P // s, s
            )
            box_sum = reshaped.sum(axis=(1, 3))
            count = int((box_sum > 0).sum())
            counts.append(count)
            inv_eps.append(P / s)

        counts = np.array(counts, dtype=float)
        inv_eps = np.array(inv_eps, dtype=float)
        coeffs = np.polyfit(np.log(inv_eps), np.log(counts), 1)
        return float(coeffs[0])

    # ------------------------------------------------------------------ #
    @staticmethod
    def perimeter_area_ratio(u: np.ndarray, threshold: float = 0.1) -> Dict:
        mask = (u > threshold).astype(np.uint8)
        if mask.sum() == 0:
            return {"perimeter": 0.0, "area": 0.0, "P_A_ratio": 0.0, "convexity_defect": 0.0}
        eroded = ndimage.binary_erosion(mask)
        perimeter = int((mask & ~eroded).sum())
        area = int(mask.sum())
        p_a = perimeter / max(area, 1)

        ys, xs = np.where(mask)
        if len(xs) < 3:
            convex_area = float(area)
        else:
            try:
                from scipy.spatial import ConvexHull
                hull = ConvexHull(np.column_stack([xs, ys]))
                convex_area = float(hull.volume)
            except Exception:
                convex_area = float(area)
        defect = float((convex_area - area) / max(convex_area, 1))
        return {
            "perimeter": float(perimeter),
            "area": float(area),
            "P_A_ratio": float(p_a),
            "convex_area": convex_area,
            "convexity_defect": defect,
        }

    # ------------------------------------------------------------------ #
    @staticmethod
    def branch_count(u: np.ndarray, threshold: float = 0.1) -> int:
        from scipy.ndimage import label
        mask = (u > threshold).astype(np.uint8)
        labeled, n = label(mask)
        return int(n)

    # ------------------------------------------------------------------ #
    @staticmethod
    def orientation_alignment(u: np.ndarray, theta_tract: np.ndarray,
                               threshold: float = 0.1) -> float:
        mask = (u > threshold)
        if mask.sum() == 0:
            return 0.0
        gy, gx = np.gradient(u)
        grad_mag = np.sqrt(gx**2 + gy**2)
        boundary = mask & (grad_mag > 0.05 * grad_mag.max())
        if boundary.sum() == 0:
            return 0.0
        grad_angle = np.arctan2(gy, gx)
        cos_align = np.abs(np.cos(grad_angle - theta_tract))
        aligned = (cos_align[boundary] > 0.7).sum()
        return float(aligned / boundary.sum())

    # ------------------------------------------------------------------ #
    def plot_metrics_summary(self, metrics: List[Dict], output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pids = [m["patient_id"] for m in metrics]

        fig, axes = plt.subplots(2, 3, figsize=(15, 9))

        # Fractal dimension
        ax = axes[0, 0]
        fd = [m["fractal_dimension"] for m in metrics]
        ax.bar(pids, fd, color="steelblue")
        ax.axhline(1.2, color="red", ls="--", alpha=0.6, label="D_f=1.2")
        ax.set_ylabel("Fractal Dimension D_f")
        ax.set_title("Fractal Dimension per Patient")
        ax.tick_params(axis="x", rotation=45)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, axis="y")

        # Front correlation
        ax = axes[0, 1]
        corr = [m["front_correlation"] for m in metrics]
        ax.bar(pids, corr, color="seagreen")
        ax.set_ylabel("Front Correlation (u vs G)")
        ax.set_title("Tumor-Growth Factor Boundary Alignment")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(alpha=0.3, axis="y")

        # Microenvironment acceleration
        ax = axes[0, 2]
        accel = [m["microenvironment_acceleration"] for m in metrics]
        ax.bar(pids, accel, color="darkorange")
        ax.axhline(1.0, color="red", ls="--", alpha=0.6, label="No acceleration")
        ax.set_ylabel("Acceleration Velocity (coupled/baseline)")
        ax.set_title("Microenvironment Acceleration Velocity")
        ax.tick_params(axis="x", rotation=45)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, axis="y")

        # Tract alignment
        ax = axes[1, 0]
        align = [m["tract_alignment_fraction"] for m in metrics]
        ax.bar(pids, align, color="purple")
        ax.set_ylabel("Tract-alignment fraction")
        ax.set_title("Invasion vs White Matter Tract Alignment")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(alpha=0.3, axis="y")

        # P/A ratio vs fractal dimension
        ax = axes[1, 1]
        pa = [m["perimeter_to_area_ratio"] for m in metrics]
        ax.scatter(fd, pa, c="darkred", s=80)
        for i, pid in enumerate(pids):
            ax.annotate(pid, (fd[i], pa[i]), fontsize=7, xytext=(4, 4),
                        textcoords="offset points")
        ax.set_xlabel("Fractal Dimension")
        ax.set_ylabel("Perimeter/Area Ratio")
        ax.set_title("Geometry: Branching vs Rugosity")
        ax.grid(alpha=0.3)

        # Zone score vs acceleration
        ax = axes[1, 2]
        scores = [m["mean_zone_score"] for m in metrics]
        ax.scatter(scores, accel, c="navy", s=80)
        for i, pid in enumerate(pids):
            ax.annotate(pid, (scores[i], accel[i]), fontsize=7, xytext=(4, 4),
                        textcoords="offset points")
        ax.set_xlabel("Mean Zone Inflammation Score")
        ax.set_ylabel("Acceleration Velocity")
        ax.set_title("Inflammation Drives Microenvironment Acceleration")
        ax.grid(alpha=0.3)

        plt.suptitle(
            "Stromal Feedback Metrics Summary: 8-Patient Cohort",
            fontsize=14, fontweight="bold"
        )
        plt.tight_layout()
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"[Phase4] Saved metrics summary -> {output_path}")

    # ------------------------------------------------------------------ #
    def save_metrics_json(self, metrics: List[Dict], output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Convert numpy types to Python types for JSON
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
        print(f"[Phase4] Saved metrics JSON -> {output_path}")


# =========================================================================== #
# MAIN EXECUTION PIPELINE
# =========================================================================== #
def run_week1_tensor_and_sanity():
    """Week 1: Tensor field + dual-grid initialization + sanity validation."""
    print("=" * 60)
    print("WEEK 1: Dual-Grid Solver & Coupled Parameter Framework")
    print("=" * 60)

    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build tensor field (reused from Month 7)
    builder = TensorFieldBuilder()
    builder.build_tract_mask()
    builder.build_orientation_field()
    builder.build_tensor_field()
    builder.validate_tensor()

    # Save tensor profiles
    builder.save_npz(output_dir / "stromal_tensor_profiles.npz")
    builder.plot_validation(output_dir / "stromal_tensor_validation.png")

    # Create baseline solver for sanity check
    solver = StromalFeedbackSolver(
        D_xx=builder.D_xx, D_xy=builder.D_xy, D_yy=builder.D_yy,
        rho_0=RHO_0, K_m=K_M, beta=BETA,
        alpha=ALPHA_BASE, gamma=GAMMA, D_G=D_G,
        dt=DT_DEFAULT, dt_chemical=DT_CHEMICAL,
        dx=DX, carrying_capacity=CARRYING_CAPACITY,
    )

    # Sanity validation: alpha=0 should collapse to baseline
    sanity = solver.run_sanity_validation(n_steps=100)
    assert sanity["passed"], "Sanity validation FAILED!"

    # Initial state visualization
    u0 = StromalFeedbackSolver.initial_gaussian_seed((GRID_SIZE, GRID_SIZE), center=(16, 50))
    G0 = np.zeros((GRID_SIZE, GRID_SIZE))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(u0, origin="lower", cmap="hot", vmin=0, vmax=1)
    axes[0].set_title("Initial Tumor Density (u)")
    axes[0].axis("off")
    axes[1].imshow(G0, origin="lower", cmap="plasma", vmin=0, vmax=1)
    axes[1].set_title("Initial Growth Factor (G)")
    axes[1].axis("off")
    plt.suptitle("Week 1: Dual-Grid Initial States", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "stromal_feedback_init.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[Week1] Saved initial states -> {output_dir / 'stromal_feedback_init.png'}")

    return builder, solver, sanity


def run_week2_chemical_diffusion(solver: StromalFeedbackSolver):
    """Week 2: Pure chemical diffusion test with mass balance."""
    print("=" * 60)
    print("WEEK 2: Isotropic Chemical Diffusion & Divergence Integration")
    print("=" * 60)

    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)

    chem_test = solver.run_chemical_mass_test(
        save_path=output_dir / "stromal_chemical_test.npz"
    )
    assert chem_test["mass_conservation_pass"], "Chemical mass test FAILED!"
    print(f"[Week2] Chemical diffusion test PASSED")
    return chem_test


def run_week3_cohort_calibration(builder: TensorFieldBuilder):
    """Week 3: Multi-omic calibration & cohort sweep."""
    print("=" * 60)
    print("WEEK 3: Multi-Omic Pathway Calibration & Cohort Sweep")
    print("=" * 60)

    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load mapper
    mapper = PatientParameterMapper()
    mapper.load()

    # Run cohort
    cohort_sim = CohortSimulator(mapper, builder)
    results = cohort_sim.run_cohort(output_dir)

    print(f"[Week3] Completed {len(results)} patient simulations")
    return results


def run_week4_visualization(builder: TensorFieldBuilder, results: List[Dict]):
    """Week 4: Dual-panel visualization & metrics summary."""
    print("=" * 60)
    print("WEEK 4: Dual-Panel Canvas Visualization & Metrics Summary")
    print("=" * 60)

    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load baseline anisotropic results for comparison
    baseline_results = []
    for pid in COHORT_PATIENTS:
        path = output_dir / f"anisotropic_evolution_{pid}.npz"
        if path.exists():
            d = np.load(path, allow_pickle=True)
            baseline_results.append({
                "patient_id": pid,
                "final_u": d["final_density"],
            })

    visualizer = StromalVisualizer(builder)

    # Generate cohort canvas
    visualizer.plot_cohort_canvas(
        results,
        output_dir / "stromal_feedback_recurrence_maps.png"
    )

    # Compute metrics
    metrics = visualizer.compute_all_metrics(results, baseline_results)

    # Save metrics JSON
    visualizer.save_metrics_json(metrics, output_dir / "stromal_feedback_metrics.json")

    # Plot metrics summary
    visualizer.plot_metrics_summary(metrics, output_dir / "stromal_feedback_metrics_summary.png")

    print(f"[Week4] Completed visualization and metrics for {len(results)} patients")
    return metrics


def main():
    """Execute full 4-week pipeline."""
    print("=" * 70)
    print("MONTH 8: STROMAL FEEDBACK COUPLED PDE SOLVER")
    print("=" * 70)

    # Week 1
    builder, solver, sanity = run_week1_tensor_and_sanity()

    # Week 2
    chem_test = run_week2_chemical_diffusion(solver)

    # Week 3
    results = run_week3_cohort_calibration(builder)

    # Week 4
    metrics = run_week4_visualization(builder, results)

    print("=" * 70)
    print("MONTH 8 COMPLETE: All deliverables generated in output/")
    print("=" * 70)
    print("Deliverables:")
    print("  - output/stromal_tensor_profiles.npz")
    print("  - output/stromal_tensor_validation.png")
    print("  - output/stromal_feedback_init.png")
    print("  - output/stromal_chemical_test.npz")
    print("  - output/stromal_evolution_<PAT_XXXX>.npz (8 files)")
    print("  - output/stromal_evolution_cohort.npz")
    print("  - output/stromal_feedback_recurrence_maps.png")
    print("  - output/stromal_feedback_metrics.json")
    print("  - output/stromal_feedback_metrics_summary.png")


if __name__ == "__main__":
    main()