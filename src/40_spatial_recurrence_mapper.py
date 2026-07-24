#!/usr/bin/env python3
"""
Month 6, Week 3: Spatial Recurrence & Microenvironment Mapping

Maps zone-stratified gene expression to Fisher-Kolmogorov PDE parameters,
simulates tumor invasion front on 2D brain geometry, and generates
spatial recurrence risk heatmaps.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

warnings.filterwarnings("ignore")


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
TARGET_GENES = ["LST1", "S100A11", "S100A8", "ZNF106"]
SPATIAL_ZONES = ["Leading Edge", "Cellular Tumor", "Infiltrating Tumor"]
GRID_SIZE = 100
DT = 0.1  # Time step
N_STEPS = 1200  # Number of PDE steps (increased from 200 for full invasion)
BASE_DIFFUSION = 0.05  # Base D (mm^2/day)
BASE_PROLIFERATION = 0.03  # Base ρ (1/day)
CARRYING_CAPACITY = 1.0
SEED = 42

# Zone-to-grid region mapping (y-slices on 100x100 grid)
ZONE_REGIONS = {
    "Cellular Tumor": (0, 33),      # Core: rows 0-32
    "Infiltrating Tumor": (33, 66), # Middle: rows 33-65
    "Leading Edge": (66, 100),      # Margin: rows 66-99
}

# Gene weights for invasion score (derived from literature/Month 4 TI)
GENE_WEIGHTS = {
    "S100A8": 1.5,    # Strong invasion driver at leading edge
    "S100A11": 1.2,   # Inflammation/invasion
    "LST1": 1.0,      # Immune/vascular
    "ZNF106": -0.5,   # Transcriptional repressor (protective)
}


# --------------------------------------------------------------------------- #
# Data Loading
# --------------------------------------------------------------------------- #
def load_zone_data(zone: str) -> pd.DataFrame:
    """Load zone-stratified cohort from Week 1."""
    suffix_map = {
        "Leading Edge": "le",
        "Cellular Tumor": "ct",
        "Infiltrating Tumor": "it",
    }
    suffix = suffix_map[zone]
    path = Path(f"output/real_cohort_{suffix}.csv")
    df = pd.read_csv(path)
    print(f"  Loaded {zone}: {df.shape[0]} samples, {df['patient_id'].nunique()} patients")
    return df


def get_patient_expression(zone_df: pd.DataFrame, patient_id: str) -> Dict[str, float]:
    """Extract target gene expressions for a patient in a zone."""
    patient_data = zone_df[zone_df["patient_id"] == patient_id]
    expr = {}
    for gene in TARGET_GENES:
        gene_data = patient_data[patient_data["gene"] == gene]
        if len(gene_data) > 0:
            expr[gene] = gene_data["expression_log2tpm"].values[0]
        else:
            expr[gene] = 0.0
    return expr


# --------------------------------------------------------------------------- #
# PDE Parameter Coupling
# --------------------------------------------------------------------------- #
def compute_invasion_score(expression: Dict[str, float]) -> float:
    """
    Compute Zone Invasion Score (Z_inv) from gene expression.
    Z_inv = sum(w_i * Expression_i)
    """
    score = 0.0
    for gene, weight in GENE_WEIGHTS.items():
        score += weight * expression.get(gene, 0.0)
    return score


def map_expression_to_pde_params(
    zone_expressions: Dict[str, Dict[str, float]],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Map patient-specific zone expressions to spatial D and ρ fields.

    Returns:
        D_field: (GRID_SIZE, GRID_SIZE) diffusion field
        rho_field: (GRID_SIZE, GRID_SIZE) proliferation field
    """
    D_field = np.full((GRID_SIZE, GRID_SIZE), BASE_DIFFUSION, dtype=float)
    rho_field = np.full((GRID_SIZE, GRID_SIZE), BASE_PROLIFERATION, dtype=float)

    # Compute invasion score per zone
    zone_scores = {}
    for zone, expr in zone_expressions.items():
        zone_scores[zone] = compute_invasion_score(expr)

    # Map to grid regions
    for zone, (y_start, y_end) in ZONE_REGIONS.items():
        score = zone_scores.get(zone, 0.0)

        # Normalize score to reasonable scaling factor
        # Typical expression range ~4-8, weights ~1-1.5, so score ~10-30
        # Scale to 0.5x - 3x base parameters
        scale_factor = 1.0 + np.tanh(score / 15.0)  # Range ~0.5 to 2.0

        # Apply to region
        D_field[y_start:y_end, :] *= scale_factor
        rho_field[y_start:y_end, :] *= scale_factor

    # Add spatial smoothing (tissue continuity)
    D_field = gaussian_filter(D_field, sigma=3.0)
    rho_field = gaussian_filter(rho_field, sigma=3.0)

    # Ensure positive
    D_field = np.maximum(D_field, 1e-6)
    rho_field = np.maximum(rho_field, 1e-6)

    return D_field, rho_field


# --------------------------------------------------------------------------- #
# Fisher-Kolmogorov PDE Solver (Finite Difference)
# --------------------------------------------------------------------------- #
def fk_pde_step(
    u: np.ndarray,
    D_field: np.ndarray,
    rho_field: np.ndarray,
    dt: float = DT,
    dx: float = 1.0,
) -> np.ndarray:
    """
    One forward step of Fisher-Kolmogorov:
    ∂u/∂t = ∇·(D∇u) + ρ u (1 - u/K)

    Using finite differences with variable D.
    """
    # Laplacian with variable diffusion: ∇·(D∇u)
    # Central differences
    u_pad = np.pad(u, 1, mode='edge')

    # D at cell faces (average of adjacent cells)
    D_pad = np.pad(D_field, 1, mode='edge')
    D_x_plus = 0.5 * (D_pad[1:-1, 2:] + D_pad[1:-1, 1:-1])
    D_x_minus = 0.5 * (D_pad[1:-1, 1:-1] + D_pad[1:-1, :-2])
    D_y_plus = 0.5 * (D_pad[2:, 1:-1] + D_pad[1:-1, 1:-1])
    D_y_minus = 0.5 * (D_pad[1:-1, 1:-1] + D_pad[:-2, 1:-1])

    # Flux differences
    flux_x = D_x_plus * (u_pad[1:-1, 2:] - u_pad[1:-1, 1:-1]) - \
             D_x_minus * (u_pad[1:-1, 1:-1] - u_pad[1:-1, :-2])
    flux_y = D_y_plus * (u_pad[2:, 1:-1] - u_pad[1:-1, 1:-1]) - \
             D_y_minus * (u_pad[1:-1, 1:-1] - u_pad[:-2, 1:-1])

    laplacian = (flux_x + flux_y) / (dx * dx)

    # Reaction term: ρ u (1 - u/K)
    reaction = rho_field * u * (1.0 - u / CARRYING_CAPACITY)

    # Forward Euler
    u_new = u + dt * (laplacian + reaction)

    # Clamp to [0, K]
    u_new = np.clip(u_new, 0.0, CARRYING_CAPACITY)

    return u_new


def simulate_fk_pde(
    D_field: np.ndarray,
    rho_field: np.ndarray,
    initial_condition: np.ndarray = None,
    n_steps: int = N_STEPS,
    dt: float = DT,
    patient_id: str = None,
) -> np.ndarray:
    """
    Run FK-PDE simulation to steady state or n_steps.
    Returns final tumor density field.
    """
    if initial_condition is None:
        # Start with small tumor at patient-specific core center
        u = np.zeros((GRID_SIZE, GRID_SIZE))
        
        # Deterministic but patient-specific seed location
        if patient_id is not None:
            # Hash patient_id to get reproducible but varied seed
            pid_hash = hash(patient_id) % 10000
            rng = np.random.default_rng(pid_hash)
            # Core zone is rows 0-32, so center around row 16
            center_y = int(rng.normal(16, 4))
            center_x = int(rng.normal(GRID_SIZE // 2, 8))
            # Clamp to core region
            center_y = np.clip(center_y, 5, 27)
            center_x = np.clip(center_x, 10, 90)
            radius = int(rng.integers(2, 5))
        else:
            center_y, center_x = GRID_SIZE // 4, GRID_SIZE // 2
            radius = 3
        
        y_min, y_max = max(0, center_y - radius), min(GRID_SIZE, center_y + radius)
        x_min, x_max = max(0, center_x - radius), min(GRID_SIZE, center_x + radius)
        u[y_min:y_max, x_min:x_max] = 0.5
    else:
        u = initial_condition.copy()

    for step in range(n_steps):
        u = fk_pde_step(u, D_field, rho_field, dt)

    return u


def compute_recurrence_risk(
    final_density: np.ndarray,
    threshold: float = 0.1,
) -> np.ndarray:
    """
    Compute recurrence risk map from final tumor density.
    Risk = probability of tumor presence > threshold at each location.
    """
    # Risk increases sigmoidally with density
    risk = 1.0 / (1.0 + np.exp(-20.0 * (final_density - threshold)))
    return risk


# --------------------------------------------------------------------------- #
# Main Pipeline per Patient
# --------------------------------------------------------------------------- #
def process_patient(
    patient_id: str,
    zone_dfs: Dict[str, pd.DataFrame],
) -> Dict:
    """Process one patient: get expressions, map to PDE, simulate, return risk map."""
    # Get expressions per zone
    zone_expressions = {}
    for zone in SPATIAL_ZONES:
        zone_expressions[zone] = get_patient_expression(zone_dfs[zone], patient_id)

    # Map to PDE parameters
    D_field, rho_field = map_expression_to_pde_params(zone_expressions)

    # Simulate with patient-specific tumor seed
    final_density = simulate_fk_pde(D_field, rho_field, patient_id=patient_id)

    # Compute risk
    risk_map = compute_recurrence_risk(final_density)

    return {
        "patient_id": patient_id,
        "D_field": D_field,
        "rho_field": rho_field,
        "final_density": final_density,
        "risk_map": risk_map,
        "zone_expressions": zone_expressions,
    }


# --------------------------------------------------------------------------- #
# Visualization
# --------------------------------------------------------------------------- #
def plot_spatial_recurrence(
    patient_results: List[Dict],
    output_path: Path,
) -> None:
    """Generate multi-panel spatial recurrence heatmap."""
    n_patients = len(patient_results)
    n_cols = min(4, n_patients)
    n_rows = (n_patients + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    if n_patients == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)

    axes = axes.flatten()

    for idx, result in enumerate(patient_results):
        ax = axes[idx]
        risk = result["risk_map"]
        patient_id = result["patient_id"]

        im = ax.imshow(risk, cmap='hot_r', origin='lower', vmin=0, vmax=1,
                       extent=[0, GRID_SIZE, 0, GRID_SIZE])

        # Add zone boundaries
        for zone, (y_start, y_end) in ZONE_REGIONS.items():
            ax.axhline(y=y_start, color='cyan', linestyle='--', alpha=0.5, linewidth=0.8)
            ax.axhline(y=y_end, color='cyan', linestyle='--', alpha=0.5, linewidth=0.8)
        ax.text(5, 5, 'Core', color='cyan', fontsize=8, alpha=0.8)
        ax.text(5, 38, 'Infiltrating', color='cyan', fontsize=8, alpha=0.8)
        ax.text(5, 72, 'Leading Edge', color='cyan', fontsize=8, alpha=0.8)

        ax.set_title(f'{patient_id}', fontsize=10)
        ax.set_xlabel('X (mm)', fontsize=8)
        ax.set_ylabel('Y (mm)', fontsize=8)

    # Hide unused
    for idx in range(n_patients, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle('Spatial Recurrence Risk Maps\n(Fisher-Kolmogorov PDE, Gene-Driven D/ρ Fields)',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [PLOT] Saved recurrence heatmap: {output_path}")


def plot_pde_parameters(
    patient_results: List[Dict],
    output_path: Path,
) -> None:
    """Plot D and ρ fields for first few patients."""
    n_show = min(3, len(patient_results))
    fig, axes = plt.subplots(n_show, 2, figsize=(8, 4 * n_show))
    if n_show == 1:
        axes = axes.reshape(1, -1)

    for idx in range(n_show):
        result = patient_results[idx]
        D_field = result["D_field"]
        rho_field = result["rho_field"]
        patient_id = result["patient_id"]

        im0 = axes[idx, 0].imshow(D_field, cmap='viridis', origin='lower')
        axes[idx, 0].set_title(f'{patient_id}: Diffusion (D)', fontsize=10)
        plt.colorbar(im0, ax=axes[idx, 0], fraction=0.046, pad=0.04)

        im1 = axes[idx, 1].imshow(rho_field, cmap='plasma', origin='lower')
        axes[idx, 1].set_title(f'{patient_id}: Proliferation (ρ)', fontsize=10)
        plt.colorbar(im1, ax=axes[idx, 1], fraction=0.046, pad=0.04)

    plt.suptitle('PDE Parameter Fields (Gene-Driven)', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [PLOT] Saved PDE parameter fields: {output_path}")


def plot_risk_profile_summary(
    patient_results: List[Dict],
    output_path: Path,
) -> None:
    """Plot aggregate risk profile across zones."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Mean risk by zone
    zone_names = list(ZONE_REGIONS.keys())
    mean_risk_by_zone = {zone: [] for zone in zone_names}

    for result in patient_results:
        risk = result["risk_map"]
        for zone, (y_start, y_end) in ZONE_REGIONS.items():
            zone_risk = risk[y_start:y_end, :].mean()
            mean_risk_by_zone[zone].append(zone_risk)

    # Box plot
    ax = axes[0]
    data = [mean_risk_by_zone[zone] for zone in zone_names]
    bp = ax.boxplot(data, tick_labels=zone_names, patch_artist=True)
    colors = ['#2ecc71', '#f39c12', '#e74c3c']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel('Mean Recurrence Risk', fontsize=11)
    ax.set_title('Risk Distribution by Spatial Zone', fontsize=12)
    ax.grid(True, axis='y', alpha=0.3)

    # Radial profile (distance from core center)
    ax = axes[1]
    center_y = ZONE_REGIONS["Cellular Tumor"][1]  # ~33
    radial_risks = []

    for result in patient_results:
        risk = result["risk_map"]
        radial = []
        for y in range(GRID_SIZE):
            dist = abs(y - center_y)
            radial.append(risk[y, :].mean())
        radial_risks.append(radial)

    radial_risks = np.array(radial_risks)
    mean_radial = radial_risks.mean(axis=0)
    std_radial = radial_risks.std(axis=0)

    y_vals = np.arange(GRID_SIZE)
    ax.plot(mean_radial, y_vals, 'r-', linewidth=2, label='Mean Risk')
    ax.fill_betweenx(y_vals, mean_radial - std_radial, mean_radial + std_radial,
                     alpha=0.3, color='red', label='±1 STD')
    ax.axhline(y=ZONE_REGIONS["Cellular Tumor"][1], color='green', linestyle='--', alpha=0.5, label='Core/Infiltrating')
    ax.axhline(y=ZONE_REGIONS["Infiltrating Tumor"][1], color='orange', linestyle='--', alpha=0.5, label='Infiltrating/Leading')
    ax.set_xlabel('Recurrence Risk', fontsize=11)
    ax.set_ylabel('Distance from Core (grid units)', fontsize=11)
    ax.set_title('Radial Risk Profile', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()

    plt.suptitle('Spatial Recurrence Risk Analysis', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [PLOT] Saved risk profile summary: {output_path}")


# --------------------------------------------------------------------------- #
# Data Export
# --------------------------------------------------------------------------- #
def export_spatial_profiles(
    patient_results: List[Dict],
    output_path: Path,
) -> None:
    """Export grid coordinates and recurrence matrices to NPY."""
    # Stack all patient risk maps
    risk_stack = np.stack([r["risk_map"] for r in patient_results])  # (n_pat, 100, 100)
    density_stack = np.stack([r["final_density"] for r in patient_results])
    D_stack = np.stack([r["D_field"] for r in patient_results])
    rho_stack = np.stack([r["rho_field"] for r in patient_results])

    patient_ids = [r["patient_id"] for r in patient_results]

    # Save as compressed NPZ
    np.savez_compressed(
        output_path,
        patient_ids=np.array(patient_ids),
        risk_maps=risk_stack,
        density_maps=density_stack,
        D_fields=D_stack,
        rho_fields=rho_stack,
        grid_size=np.array([GRID_SIZE, GRID_SIZE]),
        zone_regions=np.array([list(v) for v in ZONE_REGIONS.values()]),
        target_genes=np.array(TARGET_GENES),
        gene_weights=np.array([GENE_WEIGHTS[g] for g in TARGET_GENES]),
    )
    print(f"  [EXPORT] Saved spatial profiles: {output_path}")


def export_patient_summary(
    patient_results: List[Dict],
    output_path: Path,
) -> None:
    """Export per-patient summary metrics as JSON."""
    summary = []
    for result in patient_results:
        risk = result["risk_map"]
        zone_risks = {}
        for zone, (y_start, y_end) in ZONE_REGIONS.items():
            zone_risks[zone] = float(risk[y_start:y_end, :].mean())

        summary.append({
            "patient_id": result["patient_id"],
            "zone_recurrence_risk": zone_risks,
            "total_risk_mass": float(risk.sum()),
            "max_risk": float(risk.max()),
            "invasion_scores": {
                zone: compute_invasion_score(expr)
                for zone, expr in result["zone_expressions"].items()
            },
        })

    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  [EXPORT] Saved patient summary: {output_path}")


# --------------------------------------------------------------------------- #
# Main Pipeline
# --------------------------------------------------------------------------- #
def main():
    print("=" * 60)
    print("MONTH 6 WEEK 3: SPATIAL RECURRENCE & MICROENVIRONMENT MAPPING")
    print("=" * 60)

    # 1. Load zone-stratified data
    print("\n[LOAD] Reading zone-stratified cohorts...")
    zone_dfs = {}
    for zone in SPATIAL_ZONES:
        zone_dfs[zone] = load_zone_data(zone)

    # Get common patient IDs
    patient_ids = set(zone_dfs[SPATIAL_ZONES[0]]["patient_id"].unique())
    for zone in SPATIAL_ZONES[1:]:
        patient_ids &= set(zone_dfs[zone]["patient_id"].unique())
    patient_ids = sorted(list(patient_ids))
    print(f"  Common patients across all zones: {len(patient_ids)}")

    # Process first 8 patients for visualization (or all if fewer)
    n_vis = min(8, len(patient_ids))
    vis_patient_ids = patient_ids[:n_vis]
    print(f"\n[PROCESS] Simulating PDE for {n_vis} patients...")

    patient_results = []
    for pid in vis_patient_ids:
        print(f"  Processing {pid}...")
        result = process_patient(pid, zone_dfs)
        patient_results.append(result)

    # 2. Visualizations
    print("\n[PLOT] Generating spatial recurrence artifacts...")
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    plot_spatial_recurrence(patient_results, output_dir / "spatial_recurrence_risk.png")
    plot_pde_parameters(patient_results, output_dir / "spatial_pde_parameters.png")
    plot_risk_profile_summary(patient_results, output_dir / "spatial_risk_profile.png")

    # 3. Data export
    print("\n[EXPORT] Saving spatial profile data...")
    export_spatial_profiles(patient_results, output_dir / "spatial_recurrence_profiles.npz")
    export_patient_summary(patient_results, output_dir / "spatial_recurrence_summary.json")

    # 4. Aggregate stats
    print("\n[STATS] Aggregate recurrence risk by zone:")
    for zone in SPATIAL_ZONES:
        risks = []
        for result in patient_results:
            risk = result["risk_map"]
            y_start, y_end = ZONE_REGIONS[zone]
            risks.append(risk[y_start:y_end, :].mean())
        print(f"  {zone}: mean={np.mean(risks):.4f}, std={np.std(risks):.4f}")

    print("\n" + "=" * 60)
    print("[SUCCESS] Month 6 Week 3 Complete: Spatial Recurrence Mapping")
    print("=" * 60)
    print(f"  - output/spatial_recurrence_risk.png")
    print(f"  - output/spatial_pde_parameters.png")
    print(f"  - output/spatial_risk_profile.png")
    print(f"  - output/spatial_recurrence_profiles.npz")
    print(f"  - output/spatial_recurrence_summary.json")


if __name__ == "__main__":
    main()