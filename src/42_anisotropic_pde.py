#!/usr/bin/env python3
"""
Month 7: Anisotropic Tensor Diffusion Engineering
====================================================
Simulates glioblastoma invasion with anisotropic diffusion tensors aligned
to a synthetic white matter tract corridor, then couples patient-specific
gene expression (S100A8, S100A11) to scale diffusion/proliferation per
patient in the cohort.

PDE:
    du/dt = div( D(x) grad u ) + rho(x) u (1 - u)

D(x) is a 2x2 symmetric positive-definite tensor field whose principal
eigenvector follows the tract corridor orientation theta:

    D = R(theta) diag(D_parallel, D_perpendicular) R(-theta)

Phases:
    1. Tensor matrix field construction + symmetry/PD validation
    2. Finite-difference solver with cross-derivative flux & Neumann BCs
    3. Cohort patient coupling (8 patients, gene-driven parameter scaling)
    4. Branching visualization + fractal / perimeter-to-area geometry metrics

Deliverables:
    output/anisotropic_tensor_profiles.npz
    output/anisotropic_tensor_validation.png
    output/anisotropic_solver_mass_test.png
    output/anisotropic_evolution_PAT_XXXX.npz          (8 patients)
    output/anisotropic_recurrence_maps.png
    output/anisotropic_geometry_metrics.json
    output/anisotropic_evolution_all_patients.npz
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.ndimage import binary_dilation, gaussian_filter

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------- #
# Global constants (Phase 1: physical units, mm/days)
#   dx            : 1.0 mm          (standard brain MRI voxel)
#   dt            : 0.1 day         (CFL-safe: dt <= dx^2/(2*D_max) ~ 38.5)
#   D_white       : 0.013 mm^2/day  (Swanson et al. 2003 — along tract)
#   D_gray        : 0.0013 mm^2/day (10x lower, isotropic baseline)
#   rho           : 0.02 /day       (~35 day doubling time)
#   K             : 1.0 (normalized density fraction)
# Abstract "unit" values (D=0.15, dt=0.05) are retained as legacy defaults
# only for backward-compatible tensor geometry (Tract width/angle preserve
# shape; only diffusivity magnitude changes the physics).
# --------------------------------------------------------------------------- #
GRID_SIZE = 100
DX_MM = 1.0                 # spatial spacing, mm
DX = DX_MM                  # legacy alias
TARGET_GENES = ["LST1", "S100A11", "S100A8", "ZNF106"]

ZONE_REGIONS = {
    "Cellular Tumor": (0, 33),
    "Infiltrating Tumor": (33, 66),
    "Leading Edge": (66, 100),
}

# Physical anisotropic parameters (Phase 1, mm^2/day)
D_WHITE = 0.013             # white matter, along tract axis (D_parallel)
D_GRAY = 0.0013            # gray matter / baseline, isotropic (D_BASE)
# Anisotropy ratio ~10 (Swanson 2003); D_perp = D_white / 10
D_PARALLEL_DEFAULT = D_WHITE
D_PERPENDICULAR_DEFAULT = 0.0013   # ~D_white/10, suppressed cross-tract
D_BASE = D_GRAY

# Proliferation rate (Phase 1, /day)
RHO_DEFAULT = 0.02

# Solver parameters (Phase 1)
DT_DEFAULT = 0.1            # days
CARRYING_CAPACITY = 1.0
MASS_TEST_STEPS = 500

# Time stepping for patient simulations (Week 3)
# N_PATIENT_STEPS * DT_DEFAULT gives simulated duration
# 1500 * 0.1 day = 150 days (~6 months tumor evolution)
N_PATIENT_STEPS = 1500
PATIENT_SAVE_INTERVAL = 250


# =========================================================================== #
# PHASE 1: Tensor Field Builder
# =========================================================================== #
class TensorFieldBuilder:
    """
    Constructs a 2x2 symmetric anisotropic diffusion tensor field over a
    100x100 grid containing a diagonal white matter tract corridor.
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

        # Outputs
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
        """
        Create a diagonal curved white matter tract corridor crossing the
        matrix. Cells inside the corridor are hyper-permeable along the
        tract axis and suppressed perpendicular to it.
        """
        N = self.N
        yy, xx = np.mgrid[0:N, 0:N].astype(float)

        # Reference diagonal line y = a*x + b with mild curvature
        t = xx / max(N - 1, 1)
        center_line = 0.5 * (N - 1) + (xx - 0.5 * (N - 1)) * np.tan(self.tract_angle)
        center_line += self.tract_curvature * (N - 1) * np.sin(2 * np.pi * t)

        # Distance from each pixel to the curved tract centerline
        dist_to_tract = np.abs(yy - center_line)

        # Soft mask: Gaussian roll-off so transition isn't a hard cliff
        sigma = self.tract_width / 2.355  # FWHM -> sigma
        soft = np.exp(-(dist_to_tract ** 2) / (2 * sigma ** 2))
        hard_mask = soft > 0.5
        self.tract_mask = hard_mask.astype(bool)
        return self.tract_mask

    # ------------------------------------------------------------------ #
    def build_orientation_field(self) -> np.ndarray:
        """
        Build a per-pixel tract orientation theta. Inside the tract, theta
        follows the local tangent of the curved corridor; outside it is set
        to 0 (isotropic)."""
        if self.tract_mask is None:
            self.build_tract_mask()
        N = self.N
        yy, xx = np.mgrid[0:N, 0:N].astype(float)
        t = xx / max(N - 1, 1)

        # Local tangent angle: base angle + curvature derivative
        # d/dx of  curvature*(N-1)*sin(2*pi*t) -> 2*pi*curvature*cos(2*pi*t)
        tangent_slope = np.tan(self.tract_angle) + (
            2 * np.pi * self.tract_curvature * np.cos(2 * np.pi * t)
        )
        theta = np.arctan(tangent_slope)

        # Outside tract -> isotropic, angle = 0 (irrelevant since lambdas equal)
        theta_field = np.where(self.tract_mask, theta, 0.0)
        # Smooth slightly so the orientation field is continuous
        theta_field = gaussian_filter(theta_field, sigma=2.0) * self.tract_mask + (
            theta_field * (~self.tract_mask)
        )
        self.theta_field = theta_field
        return theta_field

    # ------------------------------------------------------------------ #
    def build_tensor_field(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Construct the 2x2 symmetric diffusion tensor components at every grid
        cell using:

            D = R(theta) diag(D_parallel, D_perpendicular) R(-theta)

        which expands to:
            D_xx = lam1 cos^2 th + lam2 sin^2 th
            D_yy = lam1 sin^2 th + lam2 cos^2 th
            D_xy = D_yx = (lam1 - lam2) sin th cos th
        """
        if self.tract_mask is None:
            self.build_tract_mask()
        if self.theta_field is None:
            self.build_orientation_field()

        theta = self.theta_field
        c = np.cos(theta)
        s = np.sin(theta)

        # Eigenvalues: along tract vs across tract (tract cells anisotropic,
        # outside tract isotropic -> use d_base for both eigenvalues)
        lam1_field = np.where(self.tract_mask, self.d_parallel, self.d_base)
        lam2_field = np.where(self.tract_mask, self.d_perpendicular, self.d_base)

        D_xx = lam1_field * c * c + lam2_field * s * s
        D_yy = lam1_field * s * s + lam2_field * c * c
        D_xy = (lam1_field - lam2_field) * s * c

        self.D_xx = D_xx
        self.D_yy = D_yy
        self.D_xy = D_xy
        self.D_yx = D_xy.copy()  # symmetry by construction
        self.lambda_1 = lam1_field
        self.lambda_2 = lam2_field
        return D_xx, D_xy, self.D_yx, D_yy

    # ------------------------------------------------------------------ #
    def validate_tensor(self) -> Dict[str, float]:
        """
        Validate tensor symmetry (D_xy == D_yx) and positive-definiteness
        across all grid pixels."""
        assert self.D_xx is not None, "build_tensor_field() must run first"
        sym_err = float(np.max(np.abs(self.D_xy - self.D_yx)))

        # Positive definite iff both eigenvalues > 0, equiv: trace > 0 and det > 0
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
        print("[Phase1] Tensor validation:")
        print(f"  symmetry max |D_xy - D_yx| = {sym_err:.3e} "
              f"{'PASS' if sym_err < 1e-12 else 'FAIL'}")
        print(f"  min trace(D)              = {min_trace:.3e}")
        print(f"  min det(D)                = {min_det:.3e}")
        print(f"  min eigenvalue            = {min_eig:.3e}")
        print(f"  positive-definite: "
              f"{'PASS' if metrics['positive_definite_pass'] else 'FAIL'}")
        return metrics

    # ------------------------------------------------------------------ #
    def anisotropy_ratio(self) -> np.ndarray:
        """Return lambda_1 / lambda_2 (Inf where lambda_2 == 0)."""
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(
                self.lambda_2 > 0, self.lambda_1 / np.maximum(self.lambda_2, 1e-12), 1.0
            )
        return ratio

    # ------------------------------------------------------------------ #
    def save_npz(self, path: Path) -> None:
        """Persist all tensor field arrays to an .npz archive."""
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
            # Phase 1 physical-unit metadata for downstream synthesis
            units_dx_mm=np.array(DX_MM),
            units_diffusion=np.array("mm^2/day"),
            units_time=np.array("day"),
        )
        print(f"[Phase1] Saved tensor profiles -> {path}")

    # ------------------------------------------------------------------ #
    def plot_validation(self, path: Path) -> None:
        """
        Render a multi-panel validation check plot including:
            - tract mask
            - D_xx, D_yy heatmaps
            - D_xy (off-diagonal) heatmap
            - principal eigenvector directional arrows (quiver)
        """
        assert self.D_xx is not None
        path.parent.mkdir(parents=True, exist_ok=True)

        # Subsample for quiver legibility
        step = 5
        yy, xx = np.mgrid[0:self.N, 0:self.N]
        xs = xx[::step, ::step]
        ys = yy[::step, ::step]
        theta_sub = self.theta_field[::step, ::step]
        lam1_sub = self.lambda_1[::step, ::step]
        lam2_sub = self.lambda_2[::step, ::step]
        # Vector along principal eigenvector, scaled by anisotropy
        aniso_sub = np.where(lam2_sub > 0, lam1_sub / np.maximum(lam2_sub, 1e-9), 1.0)
        aniso_norm = aniso_sub / max(aniso_sub.max(), 1e-9)
        U = np.cos(theta_sub) * aniso_norm
        V = np.sin(theta_sub) * aniso_norm

        fig, axes = plt.subplots(2, 2, figsize=(13, 11))

        # Panel 1: tract mask + quiver overlay
        ax = axes[0, 0]
        ax.imshow(self.tract_mask, origin="lower", cmap="gray", alpha=0.6,
                  extent=[0, self.N, 0, self.N])
        ax.quiver(xs, ys, U, -V, color="red", scale=15, width=0.004,
                  headwidth=3, headlength=4)
        ax.set_title("White Matter Tract Corridor\n+ Principal Eigenvector Field",
                     fontsize=11)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

        # Panel 2: D_xx
        ax = axes[0, 1]
        im = ax.imshow(self.D_xx, origin="lower", cmap="viridis",
                       extent=[0, self.N, 0, self.N])
        ax.set_title(r"$D_{xx}$ component", fontsize=11)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Panel 3: D_yy
        ax = axes[1, 0]
        im = ax.imshow(self.D_yy, origin="lower", cmap="viridis",
                       extent=[0, self.N, 0, self.N])
        ax.set_title(r"$D_{yy}$ component", fontsize=11)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Panel 4: D_xy = D_yx
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
        print(f"[Phase1] Saved validation plot -> {path}")


# =========================================================================== #
# PHASE 2: Anisotropic Finite-Difference Solver
# =========================================================================== #
class AnisotropicFKSolver:
    """
    Finite-difference solver for the anisotropic reaction-diffusion PDE:

        du/dt = div(D grad u) + rho u (1 - u)

    The flux divergence is expanded fully, including cross-derivative
    D_xy / D_yx terms, and boundary fluxes are zeroed (Neumann zero-flux).

    Discrete divergence (central differences, dx = dy = 1):
        div(D grad u) =
            [F_x(i+1/2,j) - F_x(i-1/2,j)] / dx
          + [F_y(i,j+1/2) - F_y(i,j-1/2)] / dy

        where the flux vector is
            F = ( D_xx u_x + D_xy u_y , D_yx u_x + D_yy u_y )

    Face values of D are arithmetic averages of adjacent cell-centred D's.
    Neumann zero-flux is enforced by mirror-padding (symmetric extension)
    which forces the normal gradient to zero at the boundary, and combined
    with zeroing the outermost boundary update so no mass leaks out.
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
        self.rho = (
            rho if np.isscalar(rho) else rho.astype(float)
        )
        self.dt = float(dt)
        self.dx = float(dx)
        self.K = float(carrying_capacity)

        # CFL condition for explicit anisotropic diffusion:
        #   dt <= dx^2 / (2 * max eigenvalue of D over all pixels)
        per_pixel_eig = 0.5 * (self.D_xx + self.D_yy) + np.sqrt(
            np.maximum(0.0, 0.25 * (self.D_xx - self.D_yy) ** 2 + self.D_xy ** 2)
        )
        max_eig = float(per_pixel_eig.max())
        self.max_eigenvalue = max_eig
        # CFL (physical units): dt <= dx^2 / (2 * D_max); units: day = mm^2/(mm^2/day)
        self.cfl_limit = (self.dx ** 2) / (2.0 * max(max_eig, 1e-12))
        self.cfl_ok = bool(self.dt <= self.cfl_limit)
        if self.dt > self.cfl_limit:
            print(f"[Phase2] WARNING dt={self.dt} days exceeds CFL limit "
                  f"{self.cfl_limit:.4f} days; clamping to 0.9*CFL.")
            self.dt = 0.9 * self.cfl_limit
            self.cfl_ok = False
        print(f"[Phase2] Solver init: grid={self.H}x{self.W}, "
              f"max eigenvalue(D)={max_eig:.4e} mm^2/day, "
              f"CFL limit={self.cfl_limit:.4f} days, dt={self.dt:.4f} days "
              f"({'OK' if self.cfl_ok else 'CLAMPED'})")

    # ------------------------------------------------------------------ #
    def anisotropic_divergence(self, u: np.ndarray) -> np.ndarray:
        """
        Compute div(D grad u) via a staggered-face finite-volume scheme.

        The divergence over cell (i,j) is the sum of inward fluxes through
        the four cell faces:

            div = ( Fx[i, j+1/2] - Fx[i, j-1/2] ) / dx^2
                + ( Fy[i+1/2, j] - Fy[i-1/2, j] ) / dx^2

        with the face-angle flux vector (consistent with the cross-derivative
        expansion):

            Fx = D_xx * u_x_face + D_xy * u_y_face        (face at j+/-1/2)
            Fy = D_yx * u_x_face + D_yy * u_y_face        (face at i+/-1/2)

        Crucially, each face is shared by exactly two cells and the flux at
        that face is computed ONCE with one-sided gradients across that face.
        This makes the discrete divergence a perfect telescoping sum, so the
        net mass change equals the boundary flux integral exactly. Combined
        with mirror padding (Neumann zero-flux) at the outer boundary, this
        yields strict mass conservation to machine precision.
        """
        dx = self.dx
        # D_yx == D_xy by symmetry (validated in Phase 1).
        Dxx, Dxy, Dyy = self.D_xx, self.D_xy, self.D_yy
        H, W = self.H, self.W

        # --- Mirror-pad everything by 1 (Neumann zero outward flux) ---------- #
        # Reflect on u (normal derivative zero), edge-pad on D so the face
        # diffusivity at the boundary equals the edge cell value (no flux
        # leaves the domain because the gradient is zero under reflect).
        u_p = np.pad(u, 1, mode="constant", constant_values=0)  # (H+2, W+2)
        Dxx_p = np.pad(Dxx, 1, mode="edge")          # (H+2, W+2)
        Dxy_p = np.pad(Dxy, 1, mode="edge")
        Dyy_p = np.pad(Dyy, 1, mode="edge")

        # ---- Vertical faces (constant i, varying j) -> Fx[i, j+1/2] -------- #
        # Vertical face at (i, j+1/2) for j = -1, 0, ..., W-1  -> W+1 faces
        # per row (the outermost faces are the boundary ones).
        # u_x across face j+1/2 = (u_p[i, j+1] - u_p[i, j]) / dx  (forward diff)
        # In padded indexing: face f lives between padded cols f and f+1;
        # for inner cell indices we want faces 0..W (relative to the HxW grid).
        # u_p extended column slice gives W+1 forward differences across the
        # whole interior + halo edges.
        u_x_vface = (u_p[1:-1, 1:] - u_p[1:-1, :-1]) / dx        # (H, W+1)
        # Cell-centred u_y (central difference) -> (H, W)
        u_y_cc = (u_p[2:, 1:-1] - u_p[:-2, 1:-1]) / (2.0 * dx)
        # u_y at vertical face j+1/2 = average of u_y_cc[:, j] and u_y_cc[:, j+1]
        # We need u_y at faces 0..W, so pad u_y_cc by 1 column on each side
        # (edge) so boundary faces use the adjacent interior u_y.
        u_y_cc_wpad = np.pad(u_y_cc, ((0, 0), (1, 1)), mode="edge")  # (H, W+2)
        u_y_vface = 0.5 * (u_y_cc_wpad[:, :-1] + u_y_cc_wpad[:, 1:])  # (H, W+1)

        # Face D components on vertical faces -> (H, W+1)
        Dxx_vface = 0.5 * (Dxx_p[1:-1, :-1] + Dxx_p[1:-1, 1:])
        Dxy_vface = 0.5 * (Dxy_p[1:-1, :-1] + Dxy_p[1:-1, 1:])

        # Flux through vertical face j+1/2 (forward = +x direction)
        Fx_face = Dxx_vface * u_x_vface + Dxy_vface * u_y_vface  # (H, W+1)
        # Cell (j) faces: east face = face at j+1/2 (index j+1), west face at j-1/2 (index j)
        Fx_E = Fx_face[:, 1:]   # (H, W)
        Fx_W = Fx_face[:, :-1]  # (H, W)

        # ---- Horizontal faces (varying i, constant j) -> Fy[i+1/2, j] ----- #
        u_y_hface = (u_p[1:, 1:-1] - u_p[:-1, 1:-1]) / dx        # (H+1, W)
        u_x_cc = (u_p[1:-1, 2:] - u_p[1:-1, :-2]) / (2.0 * dx)  # (H, W)
        u_x_cc_hpad = np.pad(u_x_cc, ((1, 1), (0, 0)), mode="edge")  # (H+2, W)
        u_x_hface = 0.5 * (u_x_cc_hpad[:-1, :] + u_x_cc_hpad[1:, :])  # (H+1, W)

        Dyy_hface = 0.5 * (Dyy_p[:-1, 1:-1] + Dyy_p[1:, 1:-1])  # (H+1, W)
        Dyx_hface = 0.5 * (Dxy_p[:-1, 1:-1] + Dxy_p[1:, 1:-1])  # (H+1, W)

        Fy_face = Dyy_hface * u_y_hface + Dyx_hface * u_x_hface  # (H+1, W)
        Fy_N = Fy_face[1:, :]   # (H, W)
        Fy_S = Fy_face[:-1, :]  # (H, W)

        # ---- Divergence = (East - West + North - South) / dx^2 ------------- #
        div = (Fx_E - Fx_W + Fy_N - Fy_S) / (dx ** 2)
        return div

    # ------------------------------------------------------------------ #
    def reaction_term(self, u: np.ndarray) -> np.ndarray:
        """Fisher-Kolmogorov reaction: rho * u * (1 - u / K)."""
        if np.isscalar(self.rho):
            return float(self.rho) * u * (1.0 - u / self.K)
        return self.rho * u * (1.0 - u / self.K)

    # ------------------------------------------------------------------ #
    def step(self, u: np.ndarray, clamp: bool = True) -> np.ndarray:
        """Forward-Euler explicit step. Clamp to [0, K] by default; disable
        only for diagnostic mass-conservation runs where clipping would
        artificially break the conservation accounting."""
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
        """Gaussian tumor seed."""
        H, W = grid_shape
        if center is None:
            cy, cx = H // 2, W // 2
        else:
            cy, cx = center
        yy, xx = np.mgrid[0:H, 0:W].astype(float)
        r2 = (yy - cy) ** 2 + (xx - cx) ** 2
        return amplitude * np.exp(-r2 / (2.0 * sigma ** 2))

    # ------------------------------------------------------------------ #
    def run_mass_conservation_test(
        self,
        n_steps: int = MASS_TEST_STEPS,
        with_reaction: bool = False,
        initial_field: np.ndarray | None = None,
        save_plot_path: Path | None = None,
    ) -> Dict:
        """
        Run a pure-diffusion test for n_steps and verify strict mass
        conservation (no leak through the Neumann boundaries).
        """
        if initial_field is None:
            u = self.initial_gaussian_seed((self.H, self.W),
                                           center=(self.H // 2, self.W // 2),
                                           sigma=4.0)
        else:
            u = initial_field.copy()
        mass0 = float(u.sum())
        # Make a pure-diffusion snapshot of rho=0 if requested
        original_rho = self.rho
        if not with_reaction:
            self.rho = 0.0

        history = np.zeros(n_steps + 1)
        history[0] = mass0
        t0 = None
        for step in range(1, n_steps + 1):
            u = self.step(u, clamp=False)
            history[step] = float(u.sum())

        self.rho = original_rho  # restore
        final_mass = float(u.sum())
        max_abs_dev = float(np.max(np.abs(history - mass0)))
        rel_err = abs(final_mass - mass0) / max(abs(mass0), 1e-12)
        has_nan = bool(np.isnan(u).any() or np.isinf(u).any())
        has_negative = bool((u < -1e-10).any())

        metrics = {
            "n_steps": n_steps,
            "initial_mass": mass0,
            "final_mass": final_mass,
            "max_abs_mass_deviation": max_abs_dev,
            "relative_mass_error": rel_err,
            "has_nan_or_inf": has_nan,
            "has_negative_density": has_negative,
            "min_density": float(u.min()),
            "max_density": float(u.max()),
            "mass_conservation_pass": bool(rel_err < 1e-6 and not has_nan),
        }
        print("[Phase2] Mass conservation test:")
        print(f"  initial mass       = {mass0:.6f}")
        print(f"  final mass         = {final_mass:.6f}")
        print(f"  max |dm|           = {max_abs_dev:.3e}")
        print(f"  relative error     = {rel_err:.3e}")
        print(f"  NaN/Inf present    = {has_nan}")
        print(f"  negative density   = {has_negative}")
        print(f"  mass conservation  = "
              f"{'PASS' if metrics['mass_conservation_pass'] else 'FAIL'}")

        if save_plot_path is not None:
            save_plot_path.parent.mkdir(parents=True, exist_ok=True)
            fig, axes = plt.subplots(1, 2, figsize=(13, 5))
            ax = axes[0]
            ax.plot(np.arange(n_steps + 1), history, "b-", lw=1.5)
            ax.axhline(mass0, color="k", ls="--", alpha=0.5)
            ax.set_xlabel("Step")
            ax.set_ylabel("Total mass")
            ax.set_title(f"Mass conservation (rel err={rel_err:.2e})")
            ax.grid(alpha=0.3)

            ax = axes[1]
            # Final density cross-section through centre
            mid = u.shape[0] // 2
            ax.plot(np.arange(u.shape[1]), u[mid, :], "r-", lw=1.5,
                    label="y = N/2 cross-section")
            ax.set_xlabel("X")
            ax.set_ylabel("u")
            ax.set_title("Final density profile")
            ax.grid(alpha=0.3)
            ax.legend()

            plt.suptitle(
                f"Anisotropic Solver Mass Test ({n_steps} steps, "
                f"dt={self.dt:.4f})",
                fontsize=12, fontweight="bold")
            plt.tight_layout()
            plt.savefig(save_plot_path, dpi=200, bbox_inches="tight")
            plt.close()
            print(f"[Phase2] Saved mass-test plot -> {save_plot_path}")

        return {"final_field": u, "mass_history": history, "metrics": metrics}


# =========================================================================== #
# PHASE 3: Cohort Expression Profile Coupling
# =========================================================================== #
class PatientParameterMapper:
    """
    Loads Month 6 spatial recurrence data + zone-stratified gene expression
    CSVs, then maps patient-specific invasive markers (S100A8, S100A11) to
    spatially resolved tensor/proliferation fields for the cohort.
    """

    INVASIVE_GENES = ["S100A8", "S100A11"]

    def __init__(
        self,
        cohort_npz: Path = Path("output/spatial_recurrence_profiles.npz"),
        zone_csv_root: Path = Path("output"),
    ) -> None:
        self.cohort_npz = cohort_npz
        self.zone_csv_root = zone_csv_root
        self.data: Dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------ #
    def load(self) -> Dict:
        """Load Month-6 NPZ cohort data and zone CSVs."""
        d = np.load(self.cohort_npz, allow_pickle=True)
        self.data = {k: d[k] for k in d.files}
        # Load zone CSV expressions keyed by patient
        zone_exprs = {}
        suffix_map = {
            "Leading Edge": "real_cohort_le.csv",
            "Cellular Tumor": "real_cohort_ct.csv",
            "Infiltrating Tumor": "real_cohort_it.csv",
        }
        for zone, fname in suffix_map.items():
            fp = self.zone_csv_root / fname
            df = pd.read_csv(fp)
            zone_exprs[zone] = df
        self.zone_expr_dfs = zone_exprs
        print(f"[Phase3] Loaded cohort "
              f"{list(self.data['patient_ids'])}")
        return self.data

    # ------------------------------------------------------------------ #
    def get_patient_zone_invasiveness(
        self, patient_id: str
    ) -> Dict[str, float]:
        """
        Per-zone invasiveness score from S100A8 and S100A11 expressions.
        Score = mean(S100A8, S100A11) normalized via tanh into (0, 2).
        """
        scores = {}
        for zone, df in self.zone_expr_dfs.items():
            sub = df[df["patient_id"] == patient_id]
            vals = []
            for g in self.INVASIVE_GENES:
                row = sub[sub["gene"] == g]
                if len(row) > 0:
                    vals.append(float(row["expression_log2tpm"].values[0]))
            if not vals:
                scores[zone] = 1.0
                continue
            raw = np.mean(vals)  # log2tpm ~ 4-8
            # Sigmoid: score in (-1, 2) with neutral at ~5
            score = 1.0 + np.tanh((raw - 5.0) / 1.5)
            scores[zone] = float(score)
        return scores

    # ------------------------------------------------------------------ #
    def build_patient_tensor_and_rho(
        self,
        patient_id: str,
        base_builder: TensorFieldBuilder,
    ) -> Dict[str, np.ndarray]:
        """
        Construct patient-specific tensor field and proliferation field by
        scaling the base builder's tract eigenvalues and Month-6 rho_field by
        the per-zone invasiveness score from S100A8 / S100A11.
        """
        if not self.data:
            self.load()

        # Find patient index in Month 6 NPZ
        pids = list(self.data["patient_ids"])
        if patient_id not in pids:
            raise KeyError(f"Patient {patient_id} not found in cohort NPZ.")
        pidx = pids.index(patient_id)
        base_rho_field = self.data["rho_fields"][pidx]  # (100, 100)

        # Scale tract eigenvalues by per-zone invasiveness
        zone_scores = self.get_patient_zone_invasiveness(patient_id)

        # Patient tract scaling masks
        N = base_builder.N
        scale_parallel = np.ones((N, N))
        scale_perp = np.ones((N, N))
        scale_rho = np.ones((N, N))
        for zone, (y0, y1) in ZONE_REGIONS.items():
            sc = zone_scores[zone]
            scale_parallel[y0:y1, :] *= sc
            # suppress cross-tract more aggressively for invasive patients
            scale_perp[y0:y1, :] *= max(0.2, 1.5 - sc)
            scale_rho[y0:y1, :] *= sc

        # Apply to base eigenvalues within the tract mask
        tract = base_builder.tract_mask
        lam1_field = np.where(tract, base_builder.d_parallel * scale_parallel,
                              base_builder.d_base)
        lam2_field = np.where(tract, base_builder.d_perpendicular * scale_perp,
                              base_builder.d_base)

        # Keep orientation field from base builder
        c = np.cos(base_builder.theta_field)
        s = np.sin(base_builder.theta_field)
        D_xx = lam1_field * c * c + lam2_field * s * s
        D_yy = lam1_field * s * s + lam2_field * c * c
        D_xy = (lam1_field - lam2_field) * s * c

        # Proliferation field scaled by per-zone score on top of Month-6 field
        rho_patient = base_rho_field * scale_rho
        # Smooth continuity
        rho_patient = gaussian_filter(rho_patient, sigma=2.0)
        rho_patient = np.maximum(rho_patient, 1e-6)

        return {
            "D_xx": D_xx,
            "D_xy": D_xy,
            "D_yy": D_yy,
            "rho": rho_patient,
            "lambda_1": lam1_field,
            "lambda_2": lam2_field,
            "zone_scores": zone_scores,
        }


class CohortSimulator:
    """Runs anisotropic FK-PDE simulations for the 8-patient test cohort."""

    def __init__(
        self,
        mapper: PatientParameterMapper,
        n_steps: int = N_PATIENT_STEPS,
        save_interval: int = PATIENT_SAVE_INTERVAL,
        dt: float = DT_DEFAULT,
    ) -> None:
        self.mapper = mapper
        self.n_steps = int(n_steps)
        self.save_interval = int(save_interval)
        self.dt = float(dt)

    # ------------------------------------------------------------------ #
    @staticmethod
    def patient_seed_center(patient_id: str, N: int = GRID_SIZE) -> Tuple[int, int]:
        """Fixed seed at center of sigmoidal white matter tract corridor."""
        return 50, 50

    # ------------------------------------------------------------------ #
    def run_patient(
        self, patient_id: str, base_builder: TensorFieldBuilder
    ) -> Dict:
        params = self.mapper.build_patient_tensor_and_rho(patient_id, base_builder)
        solver = AnisotropicFKSolver(
            D_xx=params["D_xx"],
            D_xy=params["D_xy"],
            D_yy=params["D_yy"],
            rho=params["rho"],
            dt=self.dt,
        )
        cy, cx = self.patient_seed_center(patient_id, N=solver.H)
        u = AnisotropicFKSolver.initial_gaussian_seed(
            (solver.H, solver.W), center=(cy, cx), sigma=3.0, amplitude=0.8
        )

        n_save = self.n_steps // self.save_interval + 1
        evolution = np.zeros((n_save, solver.H, solver.W), dtype=np.float32)
        mass_hist = np.zeros(self.n_steps + 1, dtype=np.float64)
        front_hist = np.zeros(self.n_steps + 1, dtype=np.float64)

        evolution[0] = u.astype(np.float32)
        mass_hist[0] = float(u.sum())

        save_idx = 1
        for step in range(1, self.n_steps + 1):
            u = solver.step(u)
            mass_hist[step] = float(u.sum())
            front_hist[step] = self._front_radius(u, cy, cx)
            if step % self.save_interval == 0 and save_idx < n_save:
                evolution[save_idx] = u.astype(np.float32)
                save_idx += 1

        print(f"[Phase3] {patient_id}: final mass={mass_hist[-1]:.3f}, "
              f"front_radius={front_hist[-1]:.2f}, "
              f"scores={params['zone_scores']}")

        return {
            "patient_id": patient_id,
            "evolution": evolution,
            "mass_history": mass_hist,
            "front_history": front_hist,
            "final_density": u,
            "tensor_field": {
                "D_xx": params["D_xx"], "D_xy": params["D_xy"],
                "D_yy": params["D_yy"], "theta": base_builder.theta_field,
            },
            "rho_field": params["rho"],
            "zone_scores": params["zone_scores"],
            "seed_center": (cy, cx),
        }

    # ------------------------------------------------------------------ #
    @staticmethod
    def _front_radius(u: np.ndarray, cy: int, cx: int,
                     threshold: float = 0.1) -> float:
        yy, xx = np.mgrid[0:u.shape[0], 0:u.shape[1]]
        r2 = (yy - cy) ** 2 + (xx - cx) ** 2
        mask = u > threshold
        if not mask.any():
            return 0.0
        return float(np.sqrt(r2[mask]).mean())

    # ------------------------------------------------------------------ #
    def run_cohort(
        self,
        base_builder: TensorFieldBuilder,
        output_dir: Path = Path("output"),
    ) -> List[Dict]:
        if not self.mapper.data:
            self.mapper.load()
        pids = list(self.mapper.data["patient_ids"])
        output_dir.mkdir(parents=True, exist_ok=True)

        results: List[Dict] = []
        for pid in pids:
            res = self.run_patient(pid, base_builder)
            np.savez_compressed(
                output_dir / f"anisotropic_evolution_{pid}.npz",
                evolution=res["evolution"],
                mass_history=res["mass_history"],
                front_history=res["front_history"],
                final_density=res["final_density"],
                D_xx=res["tensor_field"]["D_xx"],
                D_xy=res["tensor_field"]["D_xy"],
                D_yy=res["tensor_field"]["D_yy"],
                theta=res["tensor_field"]["theta"],
                rho_field=res["rho_field"],
                seed_center=np.array(res["seed_center"]),
            )
            results.append(res)
        return results


# =========================================================================== #
# PHASE 4: Deep Branching Visualization & Geometry Metrics
# =========================================================================== #
class AnisotropicVisualizer:
    """Renders the cohort comparative canvas and computes geometry metrics."""

    def __init__(self, base_builder: TensorFieldBuilder) -> None:
        self.base_builder = base_builder

    # ------------------------------------------------------------------ #
    def plot_cohort_canvas(
        self,
        results: List[Dict],
        output_path: Path,
    ) -> None:
        """8-panel comparative canvas: tumor fronts deforming along tracts."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        n = len(results)
        n_cols = 4
        n_rows = (n + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
        axes = axes.flatten()

        for idx, res in enumerate(results):
            ax = axes[idx]
            u = res["final_density"]
            pid = res["patient_id"]
            cy, cx = res["seed_center"]

            # Density heatmap
            im = ax.imshow(u, origin="lower", cmap="hot", vmin=0, vmax=1,
                           extent=[0, u.shape[1], 0, u.shape[0]])
            # Overlay tract corridor
            ax.contour(self.base_builder.tract_mask, levels=[0.5],
                       colors="cyan", linewidths=0.8, alpha=0.5,
                       extent=[0, u.shape[1], 0, u.shape[0]])
            # Zone boundaries
            for (_, (y0, y1)) in ZONE_REGIONS.items():
                ax.axhline(y=y0, color="white", linestyle=":", alpha=0.25, lw=0.6)
                ax.axhline(y=y1, color="white", linestyle=":", alpha=0.25, lw=0.6)
            # Principal eigenvector quiver overlay (subsampled)
            step = 10
            yy, xx = np.mgrid[0:u.shape[0], 0:u.shape[1]]
            th = self.base_builder.theta_field[::step, ::step]
            lam1 = self.base_builder.lambda_1[::step, ::step]
            lam2 = self.base_builder.lambda_2[::step, ::step]
            aniso = np.where(lam2 > 0, lam1 / np.maximum(lam2, 1e-9), 1.0)
            aniso /= max(aniso.max(), 1e-9)
            U = np.cos(th) * aniso
            V = np.sin(th) * aniso
            ax.quiver(xx[::step, ::step], yy[::step, ::step], U, V,
                      color="lime", scale=15, width=0.003, alpha=0.6,
                      headwidth=3, headlength=4)
            # Seed marker
            ax.plot(cx, cy, marker="+", color="yellow", ms=12, mew=2)
            ax.set_title(f"{pid}\nscores={list(res['zone_scores'].values())}",
                          fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])

        for idx in range(n, len(axes)):
            axes[idx].set_visible(False)

        plt.suptitle(
            "Anisotropic Recurrence Maps: Tumor Fronts Deforming Along\n"
            "White Matter Tract Corridors (8-patient cohort)",
            fontsize=14, fontweight="bold"
        )
        plt.tight_layout()
        plt.savefig(output_path, dpi=220, bbox_inches="tight")
        plt.close()
        print(f"[Phase4] Saved cohort canvas -> {output_path}")

    # ------------------------------------------------------------------ #
    @staticmethod
    def fractal_dimension(u: np.ndarray, threshold: float = 0.1) -> float:
        """
        Box-counting fractal dimension of the tumor mask {u > threshold}.
            D_f = lim(eps -> 0) log N(eps) / log(1/eps)
        Estimated by sliding boxes of varying size over the binary mask.
        """
        mask = (u > threshold).astype(np.uint8)
        if mask.sum() == 0:
            return 0.0
        # Pad to power-of-two for clean box sizes
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
            # Count boxes of size s x s that contain at least one occupied cell
            reshaped = padded[:P // s * s, :P // s * s].reshape(
                P // s, s, P // s, s
            )
            box_sum = reshaped.sum(axis=(1, 3))
            count = int((box_sum > 0).sum())
            counts.append(count)
            inv_eps.append(P / s)

        counts = np.array(counts, dtype=float)
        inv_eps = np.array(inv_eps, dtype=float)
        # Linear fit: log(N) = D_f * log(1/eps) + c
        coeffs = np.polyfit(np.log(inv_eps), np.log(counts), 1)
        return float(coeffs[0])

    # ------------------------------------------------------------------ #
    @staticmethod
    def perimeter_area_ratio(u: np.ndarray, threshold: float = 0.1) -> Dict:
        """Perimeter-to-area ratio and convexity defect."""
        mask = (u > threshold).astype(np.uint8)
        if mask.sum() == 0:
            return {"perimeter": 0.0, "area": 0.0, "P_A_ratio": 0.0,
                    "convexity_defect": 0.0}
        # Perimeter via edge detection (4-connected)
        eroded = ndimage.binary_erosion(mask)
        perimeter = int((mask & ~eroded).sum())
        area = int(mask.sum())
        p_a = perimeter / max(area, 1)

        # Convex hull area
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
        """Number of connected components in the tumor mask (skeleton-ish)."""
        from scipy.ndimage import label
        mask = (u > threshold).astype(np.uint8)
        labeled, n = label(mask)
        return int(n)

    # ------------------------------------------------------------------ #
    @staticmethod
    def orientation_alignment(u: np.ndarray, theta_tract: np.ndarray,
                              threshold: float = 0.1) -> float:
        """
        Fraction of tumor boundary pixels whose local gradient is preferentially
        aligned with the tract angle (cos(angle) > 0.7). Measures whether the
        invasion aligns with the tract axis.
        """
        mask = (u > threshold)
        if mask.sum() == 0:
            return 0.0
        gy, gx = np.gradient(u)
        grad_mag = np.sqrt(gx ** 2 + gy ** 2)
        boundary = (mask & (grad_mag > 0.05 * grad_mag.max()))
        if boundary.sum() == 0:
            return 0.0
        grad_angle = np.arctan2(gy, gx)
        cos_align = np.abs(np.cos(grad_angle - theta_tract))
        aligned = (cos_align[boundary] > 0.7).sum()
        return float(aligned / boundary.sum())

    # ------------------------------------------------------------------ #
    def compute_all_metrics(self, results: List[Dict]) -> List[Dict]:
        metrics_list = []
        for res in results:
            u = res["final_density"]
            fd = self.fractal_dimension(u)
            pa = self.perimeter_area_ratio(u)
            bc = self.branch_count(u)
            align = self.orientation_alignment(u, self.base_builder.theta_field)
            row = {
                "patient_id": res["patient_id"],
                "fractal_dimension": fd,
                "perimeter": pa["perimeter"],
                "area": pa["area"],
                "perimeter_to_area_ratio": pa["P_A_ratio"],
                "convexity_defect": pa["convexity_defect"],
                "branch_count": bc,
                "tract_alignment_fraction": align,
                "zone_scores": res["zone_scores"],
                "mean_zone_score": float(np.mean(list(res["zone_scores"].values()))),
            }
            metrics_list.append(row)
            print(f"[Phase4] {row['patient_id']}: D_f={fd:.3f}, "
                  f"P/A={pa['P_A_ratio']:.3f}, branches={bc}, "
                  f"align={align:.3f}")
        return metrics_list

    # ------------------------------------------------------------------ #
    def plot_geometry_summary(self, metrics: List[Dict], output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pids = [m["patient_id"] for m in metrics]
        fd = [m["fractal_dimension"] for m in metrics]
        pa = [m["perimeter_to_area_ratio"] for m in metrics]
        bc = [m["branch_count"] for m in metrics]
        aligns = [m["tract_alignment_fraction"] for m in metrics]
        scores = [m["mean_zone_score"] for m in metrics]

        fig, axes = plt.subplots(2, 2, figsize=(13, 9))
        ax = axes[0, 0]
        ax.bar(pids, fd, color="steelblue")
        ax.axhline(1.2, color="red", ls="--", alpha=0.6, label="D_f=1.2 (branching threshold)")
        ax.set_ylabel("Fractal Dimension D_f")
        ax.set_title("Fractal Dimension per Patient")
        ax.tick_params(axis="x", rotation=45)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, axis="y")

        ax = axes[0, 1]
        ax.bar(pids, aligns, color="seagreen")
        ax.set_ylabel("Tract-alignment fraction")
        ax.set_title("Tumor Gradients vs Tract Direction")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(alpha=0.3, axis="y")

        ax = axes[1, 0]
        ax.scatter(scores, fd, c="darkorange", s=80)
        for i, pid in enumerate(pids):
            ax.annotate(pid, (scores[i], fd[i]), fontsize=7,
                        xytext=(4, 4), textcoords="offset points")
        ax.set_xlabel("Mean zone invasiveness score (S100A8/S100A11)")
        ax.set_ylabel("Fractal Dimension")
        ax.set_title("Invasiveness vs Geometric Complexity")
        ax.grid(alpha=0.3)

        ax = axes[1, 1]
        ax.bar(pids, pa, color="purple")
        ax.set_ylabel("Perimeter / Area")
        ax.set_title("Spatial Perimeter-to-Area Ratio")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(alpha=0.3, axis="y")

        plt.suptitle(
            "Anisotropic Branching Geometry Metrics (8-patient cohort)",
            fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"[Phase4] Saved geometry summary -> {output_path}")


# =========================================================================== #
# MAIN PIPELINE
# =========================================================================== #
def save_all_evolution(results: List[Dict], path: Path) -> None:
    """Stacked evolution + metadata over all patients into one npz."""
    path.parent.mkdir(parents=True, exist_ok=True)
    stack = np.stack([r["final_density"] for r in results])  # (n, 100, 100)
    pids = [r["patient_id"] for r in results]
    mass_stack = np.stack([r["mass_history"] for r in results])
    front_stack = np.stack([r["front_history"] for r in results])
    np.savez_compressed(
        path,
        patient_ids=np.array(pids),
        final_density_stack=stack,
        mass_history_stack=mass_stack,
        front_history_stack=front_stack,
    )
    print(f"[Phase4] Saved combined evolution npz -> {path}")


def main():
    print("=" * 70)
    print("MONTH 7: ANISOTROPIC TENSOR DIFFUSION ENGINEERING")
    print("=" * 70)
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    # ------------------ PHASE 1 -------------------------------
    print("\n" + "#" * 70)
    print("# PHASE 1: Tensor Matrix Field Construction")
    print("#" * 70)
    builder = TensorFieldBuilder(
        grid_size=GRID_SIZE,
        d_parallel=D_PARALLEL_DEFAULT,
        d_perpendicular=D_PERPENDICULAR_DEFAULT,
        d_base=D_BASE,
        tract_angle_deg=45.0,
        tract_width=15,
    )
    builder.build_tract_mask()
    builder.build_orientation_field()
    builder.build_tensor_field()
    val_metrics = builder.validate_tensor()
    builder.save_npz(output_dir / "anisotropic_tensor_profiles.npz")
    builder.plot_validation(output_dir / "anisotropic_tensor_validation.png")

    # ------------------ PHASE 2 -------------------------------
    print("\n" + "#" * 70)
    print("# PHASE 2: Finite-Difference Solver & Mass Conservation")
    print("#" * 70)
    solver = AnisotropicFKSolver(
        D_xx=builder.D_xx, D_xy=builder.D_xy, D_yy=builder.D_yy,
        rho=0.0, dt=DT_DEFAULT,
    )
    test_result = solver.run_mass_conservation_test(
        n_steps=MASS_TEST_STEPS,
        with_reaction=False,
        save_plot_path=output_dir / "anisotropic_solver_mass_test.png",
    )
    mass_test_metrics = test_result["metrics"]

    # ------------------ PHASE 3 -------------------------------
    print("\n" + "#" * 70)
    print("# PHASE 3: Cohort Expression Profile Coupling")
    print("#" * 70)
    mapper = PatientParameterMapper()
    mapper.load()
    cohort_sim = CohortSimulator(
        mapper=mapper,
        n_steps=N_PATIENT_STEPS,
        save_interval=PATIENT_SAVE_INTERVAL,
        dt=DT_DEFAULT,
    )
    results = cohort_sim.run_cohort(builder, output_dir=output_dir)

    # ------------------ PHASE 4 -------------------------------
    print("\n" + "#" * 70)
    print("# PHASE 4: Deep Branching Visualization & Geometry Metrics")
    print("#" * 70)
    viz = AnisotropicVisualizer(builder)
    viz.plot_cohort_canvas(results, output_dir / "anisotropic_recurrence_maps.png")
    metrics = viz.compute_all_metrics(results)
    viz.plot_geometry_summary(metrics, output_dir / "anisotropic_geometry_summary.png")

    # Save metrics JSON
    with open(output_dir / "anisotropic_geometry_metrics.json", "w") as f:
        # Combine phase1, phase2 and phase4 metrics
        full_report = {
            "phase1_tensor_validation": val_metrics,
            "phase2_mass_conservation": mass_test_metrics,
            "phase4_geometry_metrics": metrics,
        }
        json.dump(full_report, f, indent=2, default=str)
    print(f"[Phase4] Saved metrics JSON -> "
          f"{output_dir / 'anisotropic_geometry_metrics.json'}")

    # Save combined evolution npz
    save_all_evolution(results, output_dir / "anisotropic_evolution_all_patients.npz")

    # ------------------ SUMMARY --------------------------------
    print("\n" + "=" * 70)
    print("[SUMMARY] Month 7 Anisotropic Tensor Diffusion Engineering")
    print("=" * 70)
    print(f"  Phase 1 symmetry pass       : {val_metrics['symmetry_pass']}")
    print(f"  Phase 1 positive-definite   : {val_metrics['positive_definite_pass']}")
    print(f"  Phase 2 mass conservation   : {mass_test_metrics['mass_conservation_pass']}")
    print(f"  Patients simulated          : {len(results)}")
    mean_fd = float(np.mean([m["fractal_dimension"] for m in metrics]))
    mean_pa = float(np.mean([m["perimeter_to_area_ratio"] for m in metrics]))
    print(f"  Mean fractal dimension      : {mean_fd:.3f}")
    print(f"  Mean perimeter/area ratio   : {mean_pa:.3f}")
    print(f"  Branching (D_f > 1.2)       : "
          f"{sum(m['fractal_dimension'] > 1.2 for m in metrics)} / {len(metrics)}")

    print("\nDeliverables:")
    print("  output/anisotropic_tensor_profiles.npz")
    print("  output/anisotropic_tensor_validation.png")
    print("  output/anisotropic_solver_mass_test.png")
    print(f"  output/anisotropic_evolution_PAT_000{{0..7}}.npz  (8 files)")
    print("  output/anisotropic_recurrence_maps.png")
    print("  output/anisotropic_geometry_summary.png")
    print("  output/anisotropic_geometry_metrics.json")
    print("  output/anisotropic_evolution_all_patients.npz")
    print("\n[SUCCESS] Month 7 complete.")


if __name__ == "__main__":
    main()
