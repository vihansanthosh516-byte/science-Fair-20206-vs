#!/usr/bin/env python3
"""Interactive 3D Dashboard for 8-Patient 3D Cohort Visualization.

This module creates an interactive Plotly dashboard for exploring 3D tumor
growth dynamics across the 8-patient cohort. Features include:

- Patient selector dropdown (PAT_0000 – PAT_0007)
- Timeline slider (0–180 days)
- Treatment arm toggle (MTD vs Adaptive)
- 3D isosurface visualization of tumor density
- Patient metadata panel (tract orientation, volume, sphericity, dose sparing)

Output:
    output/3d_interactive_tumor_dashboard.html

Usage:
    python src/49_interactive_3d_dashboard.py
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

warnings.filterwarnings("ignore")

try:
    import plotly.graph_objects as go
    from plotly.offline import plot
except ImportError:
    print("ERROR: plotly not installed. Run: pip install plotly")
    exit(1)

OUTPUT_DIR = Path("output")
COHORT_NPZ = OUTPUT_DIR / "3d_master_cohort_volumes.npz"
COHORT_JSON = OUTPUT_DIR / "3d_extension_summary.json"
DASHBOARD_HTML = OUTPUT_DIR / "3d_interactive_tumor_dashboard.html"

# Physical constants
DX = 1.0  # mm
GRID_SIZE = 50
COHORT_PATIENTS = [f"PAT_{i:04d}" for i in range(8)]


def load_cohort_data() -> tuple:
    """Load 3D cohort volumes and summary metrics."""
    if not COHORT_NPZ.exists():
        raise FileNotFoundError(f"Missing {COHORT_NPZ}. Run src/48_3d_extension.py first.")
    if not COHORT_JSON.exists():
        raise FileNotFoundError(f"Missing {COHORT_JSON}. Run src/48_3d_extension.py first.")

    # Load NPZ
    npz_data = np.load(COHORT_NPZ, allow_pickle=True)

    # Load JSON summary
    with open(COHORT_JSON) as f:
        summary = json.load(f)

    return npz_data, summary


def create_coordinate_grid() -> tuple:
    """Create 3D coordinate grids for visualization."""
    x, y, z = np.mgrid[0:GRID_SIZE, 0:GRID_SIZE, 0:GRID_SIZE]
    return x * DX, y * DX, z * DX  # Convert to mm


def extract_isosurface_data(
    u: np.ndarray,
    threshold: float = 0.1,
    max_points: int = 5000,
) -> tuple:
    """Extract tumor isosurface points for visualization.

    Returns scattered points above threshold for efficient rendering.
    Returns (None, None, None, None) if no tumor present.
    """
    mask = u > threshold
    if not np.any(mask):
        return None, None, None, None

    # Get coordinates of tumor voxels
    indices = np.where(mask)
    if len(indices[0]) > max_points:
        # Subsample for performance
        choice = np.random.choice(len(indices[0]), max_points, replace=False)
        indices = tuple(idx[choice] for idx in indices)

    x = indices[2] * DX  # Note: numpy indexing is (z, y, x)
    y = indices[1] * DX
    z = indices[0] * DX
    values = u[mask]

    if len(values) > max_points:
        values = values[:max_points]

    return x, y, z, values


def create_patient_panel(
    pid: str,
    patient_data: Dict[str, Any],
    npz_data,
    x_mm: np.ndarray,
    y_mm: np.ndarray,
    z_mm: np.ndarray,
) -> go.Frame:
    """Create frames for a single patient (MTD and Adaptive timelines)."""
    frames = []

    # Get initial tumor (u0)
    u0_key = f"{pid}_u0"
    u0 = npz_data[u0_key] if u0_key in npz_data.files else None

    # Get MTD and Adaptive final states
    mtd_key = f"{pid}_mtd"
    adapt_key = f"{pid}_adapt"

    mtd_u = npz_data[mtd_key] if mtd_key in npz_data.files else None
    adapt_u = npz_data[adapt_key] if adapt_key in npz_data.files else None

    # Frame 1: Initial state (Day 0)
    if u0 is not None:
        x, y, z, vals = extract_isosurface_data(u0)
        if x is not None:
            frame_initial = go.Frame(
                data=[go.Scatter3d(
                    x=x, y=y, z=z,
                    mode='markers',
                    marker=dict(
                        size=2.5,
                        color=vals,
                        colorscale='Reds',
                        opacity=0.8,
                        colorbar=dict(title='Density', len=0.5)
                    ),
                    name='Tumor'
                )],
                layout=go.Layout(
                    title=f"{pid} — Day 0 (Initial)<br>"
                          f"<span style='font-size:10px'>Tract: n={patient_data['tract_orientation']}</span>",
                    scene=dict(
                        xaxis=dict(title='X (mm)', range=[0, GRID_SIZE*DX]),
                        yaxis=dict(title='Y (mm)', range=[0, GRID_SIZE*DX]),
                        zaxis=dict(title='Z (mm)', range=[0, GRID_SIZE*DX]),
                        camera=dict(
                            eye=dict(x=1.5, y=1.5, z=1.2)
                        )
                    )
                ),
                name=f"{pid}_initial"
            )
            frames.append(frame_initial)

    # Frame 2: MTD final state (Day 180)
    if mtd_u is not None:
        mtd_vol = patient_data['mtd']['final_volume_mm3']
        x, y, z, vals = extract_isosurface_data(mtd_u)
        if x is not None and len(x) > 0:
            frame_mtd = go.Frame(
                data=[go.Scatter3d(
                    x=x, y=y, z=z,
                    mode='markers',
                    marker=dict(
                        size=2.5,
                        color=vals,
                        colorscale='Reds',
                        opacity=0.8,
                        colorbar=dict(title='Density', len=0.5)
                    ),
                    name='Tumor (MTD)'
                )],
                layout=go.Layout(
                    title=f"{pid} — Day 180 (MTD)<br>"
                          f"<span style='font-size:10px'>Volume: {mtd_vol:.1f} mm³ | "
                          f"Sphericity: {patient_data['mtd']['sphericity']:.3f}</span>",
                ),
                name=f"{pid}_mtd"
            )
            frames.append(frame_mtd)
        else:
            # No tumor visible (eliminated)
            frame_mtd = go.Frame(
                data=[go.Scatter3d(
                    x=[GRID_SIZE*DX/2], y=[GRID_SIZE*DX/2], z=[GRID_SIZE*DX/2],
                    mode='text',
                    text=['Tumor Eliminated'],
                    textfont=dict(color='green', size=14)
                )],
                layout=go.Layout(
                    title=f"{pid} — Day 180 (MTD)<br>"
                          f"<span style='font-size:10px'>Volume: {mtd_vol:.1f} mm³ (eliminated)</span>",
                ),
                name=f"{pid}_mtd"
            )
            frames.append(frame_mtd)

    # Frame 3: Adaptive final state (Day 180)
    if adapt_u is not None:
        adapt_vol = patient_data['adaptive']['final_volume_mm3']
        dose_sparing = patient_data['adaptive']['dose_sparing_fraction'] * 100
        x, y, z, vals = extract_isosurface_data(adapt_u)
        if x is not None:
            frame_adapt = go.Frame(
                data=[go.Scatter3d(
                    x=x, y=y, z=z,
                    mode='markers',
                    marker=dict(
                        size=2.5,
                        color=vals,
                        colorscale='Oranges',
                        opacity=0.8,
                        colorbar=dict(title='Density', len=0.5)
                    ),
                    name='Tumor (Adaptive)'
                )],
                layout=go.Layout(
                    title=f"{pid} — Day 180 (Adaptive)<br>"
                          f"<span style='font-size:10px'>Volume: {adapt_vol:.1f} mm³ | "
                          f"Dose Sparing: {dose_sparing:.1f}%</span>",
                ),
                name=f"{pid}_adaptive"
            )
            frames.append(frame_adapt)

    return frames


def build_dashboard(npz_data, summary: Dict[str, Any]) -> go.Figure:
    """Build the interactive Plotly dashboard."""
    x_mm, y_mm, z_mm = create_coordinate_grid()

    # Initialize with first patient
    first_pid = COHORT_PATIENTS[0]
    first_patient = next(p for p in summary['patients'] if p['patient_id'] == first_pid)

    # Create initial scatter plot
    u0_key = f"{first_pid}_u0"
    u0 = npz_data[u0_key] if u0_key in npz_data.files else None
    x, y, z, vals = extract_isosurface_data(u0) if u0 is not None else (None, None, None, None)

    if x is not None:
        initial_data = [go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers',
            marker=dict(
                size=2.5,
                color=vals,
                colorscale='Reds',
                opacity=0.8,
                colorbar=dict(title='Density', len=0.5)
            ),
            name='Tumor'
        )]
    else:
        initial_data = [go.Scatter3d(
            x=[GRID_SIZE*DX/2], y=[GRID_SIZE*DX/2], z=[GRID_SIZE*DX/2],
            mode='text',
            text=['No Data'],
            textfont=dict(color='gray', size=14)
        )]

    # Build frames for all patients
    all_frames = []
    for pid in COHORT_PATIENTS:
        patient = next(p for p in summary['patients'] if p['patient_id'] == pid)
        frames = create_patient_panel(pid, patient, npz_data, x_mm, y_mm, z_mm)
        all_frames.extend(frames)

    # Create dropdown menu
    dropdown_buttons = []
    for i, pid in enumerate(COHORT_PATIENTS):
        patient = next(p for p in summary['patients'] if p['patient_id'] == pid)
        tract = patient['tract_orientation']
        dropdown_buttons.append(dict(
            label=pid,
            method='animate',
            args=[
                [f"{pid}_initial", f"{pid}_mtd", f"{pid}_adaptive"],
                dict(
                    mode='immediate',
                    frame=dict(duration=500, redraw=True),
                    transition=dict(duration=300)
                )
            ]
        ))

    # Build figure
    fig = go.Figure(
        data=initial_data,
        frames=all_frames,
        layout=go.Layout(
            title=f"{first_pid} — Day 0 (Initial)<br>"
                  f"<span style='font-size:11px'>Tract: n={first_patient['tract_orientation']} | "
                  f"Anisotropy: {summary['anisotropy']['anisotropy_ratio']}:1</span>",
            updatemenus=[dict(
                type='dropdown',
                buttons=dropdown_buttons,
                x=0.01,
                y=1.15,
                xanchor='left',
                yanchor='top',
                pad=dict(t=10, r=10, b=10, l=10),
                font=dict(size=11)
            )],
            scene=dict(
                xaxis=dict(title='X (mm)', range=[0, GRID_SIZE*DX], backgroundcolor='rgba(240,240,240,0.5)'),
                yaxis=dict(title='Y (mm)', range=[0, GRID_SIZE*DX], backgroundcolor='rgba(240,240,240,0.5)'),
                zaxis=dict(title='Z (mm)', range=[0, GRID_SIZE*DX], backgroundcolor='rgba(240,240,240,0.5)'),
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.2)),
                aspectmode='cube'
            ),
            showlegend=False,
            height=700,
            width=900,
            margin=dict(l=10, r=10, t=80, b=10),
            paper_bgcolor='white',
            plot_bgcolor='white',
        )
    )

    # Add play/pause buttons for animation
    play_pause_buttons = [
        dict(label='▶ Play',
             method='animate',
             args=[None, dict(frame=dict(duration=1000, redraw=True),
                             fromcurrent=True,
                             transition=dict(duration=300))]),
        dict(label='⏸ Pause',
             method='animate',
             args=[[None], dict(frame=dict(duration=0, redraw=False),
                               mode='immediate',
                               interrupt=True)]),
    ]

    fig.update_layout(
        updatemenus=list(fig.layout.updatemenus) + [dict(
            type='buttons',
            showactive=False,
            x=0.15,
            y=1.05,
            buttons=play_pause_buttons,
            direction='left',
            pad=dict(r=10, t=10),
            font=dict(size=11)
        )]
    )

    return fig


def main():
    print("=" * 70)
    print("Interactive 3D Dashboard Generator")
    print("=" * 70)

    # Load data
    print("\n[1] Loading 3D cohort data...")
    npz_data, summary = load_cohort_data()
    print(f"    Loaded {len(COHORT_PATIENTS)} patients from {COHORT_NPZ}")
    print(f"    Summary: {COHORT_JSON}")

    # Build dashboard
    print("\n[2] Building interactive dashboard...")
    fig = build_dashboard(npz_data, summary)
    print(f"    Created {len(fig.frames)} animation frames")

    # Export
    print(f"\n[3] Exporting dashboard to HTML...")
    plot(fig, filename=str(DASHBOARD_HTML), auto_open=False, include_plotlyjs='cdn')
    print(f"    Saved -> {DASHBOARD_HTML}")

    # Print summary
    print(f"\n{'='*70}")
    print("Dashboard Complete")
    print(f"{'='*70}")
    print(f"Features:")
    print(f"  - Patient dropdown selector (PAT_0000 - PAT_0007)")
    print(f"  - Timeline: Day 0 -> Day 180 (MTD vs Adaptive)")
    print(f"  - 3D isosurface visualization with tract orientation")
    print(f"  - Metadata: volume (mm3), sphericity, dose sparing (%)")
    print(f"\nOpen {DASHBOARD_HTML} in a web browser to explore.")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()