# Month 7: Anisotropic Tensor Diffusion Engineering - Implementation Plan

## Overview
4-week implementation of anisotropic tensor diffusion for glioblastoma invasion modeling on a 100×100 grid with synthetic white matter tract corridors and patient-specific gene-expression-driven parameter scaling.

---

## Week 1: Mathematical Foundations & Tensor Matrix Fields
**File**: `src/42_anisotropic_pde.py` (Phase 1)

### Objectives
1. Create synthetic brain anatomy mask with diagonal white matter tract corridor
2. Construct 2×2 diffusion tensor arrays (D_xx, D_xy, D_yx, D_yy) at each grid point
3. Validate tensor symmetry (D_xy = D_yx) and positive-definiteness
4. Save tensor fields to `output/anisotropic_tensor_profiles.npz`
5. Render validation plot with directional vector arrows

### Implementation Details

#### 1.1 Synthetic Brain Anatomy & Tract Mask
- Grid: 100×100 (matching Month 6 grid_size)
- Create diagonal white matter tract corridor (e.g., from top-left to bottom-right, width ~15-20 pixels)
- Tract orientation angle θ = π/4 (45° diagonal) or configurable
- Mask array: `tract_mask[100, 100]` boolean

#### 1.2 Tensor Construction
For each grid cell (i,j):
- If inside tract: 
  - Eigenvalues: λ₁ = D_parallel (high, e.g., 0.15), λ₂ = D_perpendicular (low, e.g., 0.005)
  - Eigenvectors: v₁ = [cos(θ), sin(θ)], v₂ = [-sin(θ), cos(θ)]
  - Tensor: D = R(θ) diag(λ₁, λ₂) R(-θ) where R(θ) is rotation matrix
  - Components: 
    - D_xx = λ₁ cos²θ + λ₂ sin²θ
    - D_xy = D_yx = (λ₁ - λ₂) sinθ cosθ
    - D_yy = λ₁ sin²θ + λ₂ cos²θ
- If outside tract: isotropic D = D_base * I (e.g., D_base = 0.01)

#### 1.3 Validation
- Verify D_xy == D_yx everywhere (symmetry)
- Verify positive-definiteness: det(D) > 0 and trace(D) > 0 everywhere
- Visualize with quiver plot of principal eigenvector (v₁) scaled by λ₁/λ₂ anisotropy ratio

#### 1.4 Deliverables
- `output/anisotropic_tensor_profiles.npz` with arrays:
  - `D_xx`, `D_xy`, `D_yx`, `D_yy` (100, 100) tensor components
  - `tract_mask` (100, 100) boolean
  - `theta_field` (100, 100) orientation angle
  - `lambda_1`, `lambda_2` (100, 100) eigenvalues
  - `anisotropy_ratio` (100, 100) λ₁/λ₂
  - `grid_size`, `D_parallel`, `D_perpendicular`, `D_base`
- `output/anisotropic_tensor_validation.png` quiver plot

---

## Week 2: Advanced Finite-Difference Numerical Solver
**File**: `src/42_anisotropic_pde.py` (Phase 2 - extending same file)

### Objectives
1. Implement full anisotropic divergence: ∇·(D∇u) with cross-derivative terms
2. Implement Neumann zero-flux boundary conditions
3. Establish CFL stability limits for explicit time stepping
4. Run 500-step test with mass conservation verification

### Implementation Details

#### 2.1 Anisotropic Divergence Stencil
∇·(D∇u) = ∂/∂x(D_xx ∂u/∂x + D_xy ∂u/∂y) + ∂/∂y(D_yx ∂u/∂x + D_yy ∂u/∂y)

Expanded finite difference (central differences, dx=dy=1):
```
∂/∂x(D_xx u_x + D_xy u_y) ≈ 
  [D_xx(i+½,j) * (u(i+1,j)-u(i,j)) - D_xx(i-½,j) * (u(i,j)-u(i-1,j))]
+ [D_xy(i+½,j) * (u(i+1,j+1)-u(i+1,j-1))/2 - D_xy(i-½,j) * (u(i-1,j+1)-u(i-1,j-1))/2]

∂/∂y(D_yx u_x + D_yy u_y) ≈
  [D_yy(i,j+½) * (u(i,j+1)-u(i,j)) - D_yy(i,j-½) * (u(i,j)-u(i,j-1))]
+ [D_yx(i,j+½) * (u(i+1,j+1)-u(i-1,j+1))/2 - D_yx(i,j-½) * (u(i+1,j-1)-u(i-1,j-1))/2]
```

Face-centered D values via arithmetic averaging of adjacent cell centers.

#### 2.2 Neumann Zero-Flux Boundary Conditions
At boundaries, set flux = 0:
- Left boundary (i=0): D_xx u_x + D_xy u_y = 0
- Right boundary (i=W-1): D_xx u_x + D_xy u_y = 0
- Bottom (j=0): D_yx u_x + D_yy u_y = 0
- Top (j=H-1): D_yx u_x + D_yy u_y = 0

Implement via ghost cells or modified stencils at boundaries.

#### 2.3 CFL Stability Condition
For anisotropic diffusion: dt ≤ min(dx², dy²) / (2 * max_eigenvalue(D)) 
With dx=dy=1 and max λ₁ = D_parallel: dt ≤ 1/(2*D_parallel)
For D_parallel=0.15: dt ≤ 3.33 → use dt=0.1 for safety

Use explicit Euler: uⁿ⁺¹ = uⁿ + dt * (∇·(D∇u) + ρ u(1-u))

#### 2.4 Mass Conservation Test
- Initialize: Gaussian tumor seed at center (mass = sum(u))
- Run 500 steps with ρ=0 (pure diffusion)
- Verify: |mass(t) - mass(0)| / mass(0) < 1e-6
- Verify: no NaN, no negative values, no boundary leakage

#### 2.5 Deliverables
- Extended `src/42_anisotropic_pde.py` with `AnisotropicFKSolver` class
- Test script output: mass conservation metrics, max/min values, NaN check
- Validation plot: initial vs final mass, density profile cross-section

---

## Week 3: Cohort Expression Profile Coupling
**File**: `src/42_anisotropic_pde.py` (Phase 3) + optional driver script

### Objectives
1. Load Month 6 patient data from `output/spatial_recurrence_profiles.npz`
2. Extract patient-specific gene weights (S100A8, S100A11) for parameter scaling
3. Scale D_parallel, D_perpendicular, ρ per patient per zone
4. Run parallel simulations for 8 patients (PAT_0000-PAT_0007)
5. Save evolution data matrices for all patients

### Implementation Details

#### 3.1 Patient Parameter Scaling
From Month 6 data:
- `target_genes`: ['LST1', 'S100A11', 'S100A8', 'ZNF106']
- `gene_weights`: [1.0, 1.2, 1.5, -0.5]
- `D_fields`: (8, 100, 100) base isotropic diffusion per patient
- `rho_fields`: (8, 100, 100) base proliferation per patient
- `zone_regions`: Core (0:33), Infiltrating (33:66), Leading Edge (66:100)

For each patient and zone:
- Compute zone-specific invasion score using S100A8, S100A11 expressions
- Scale D_parallel = D_base_parallel * (1 + α * invasion_score_zone)
- Scale D_perpendicular = D_base_perp * (1 - β * invasion_score_zone)  (suppress cross-tract)
- Scale ρ = ρ_base * (1 + γ * invasion_score_zone)

#### 3.2 Tensor Field Per Patient Per Zone
- Tract orientation θ varies by zone (e.g., Core: 45°, Infiltrating: 30°, Leading Edge: 60°)
- Or use patient-specific tract orientation from DTI if available (simulated)

#### 3.3 Parallel Simulation
- Run 8 simulations with patient-specific tensor fields
- Time steps: sufficient for visible invasion (e.g., 2000 steps with dt=0.1)
- Save full evolution: u(t, x, y) for t in [0, 500, 1000, 1500, 2000]

#### 3.4 Deliverables
- `output/anisotropic_evolution_PAT_XXXX.npz` per patient (or combined):
  - `density_evolution` (n_steps, 100, 100)
  - `tensor_fields` (100, 100, 4) D_xx, D_xy, D_yy, theta
  - `mass_history` (n_steps,)
  - `front_position_history` (n_steps,)

---

## Week 4: Deep Branching Visualization & Data Export
**File**: `src/42_anisotropic_pde.py` (Phase 4) + visualization script

### Objectives
1. Render publication-ready 8-panel comparative heatmap
2. Verify anisotropic branching morphology (not circular)
3. Compute fractal dimension and perimeter-to-area ratios
4. Final code check-in and validation loop execution

### Implementation Details

#### 4.1 Multi-Panel Visualization
- 8 patients × 1 row (or 4×2 grid)
- Each panel: final tumor density heatmap (hot colormap)
- Overlay: white matter tract corridor (white dashed line)
- Overlay: principal eigenvector quiver field (subsampled)
- Zone boundaries (dashed cyan lines)
- Title: PAT_XXXX + key metrics

#### 4.2 Geometry Metrics
For each patient final density field (threshold at 0.1):
- **Fractal Dimension** (box-counting): D_f = lim(ε→0) log(N(ε))/log(1/ε)
- **Perimeter-to-Area Ratio**: P/A (higher = more complex/invasive)
- **Convexity Defect**: Area(convex_hull) - Area(tumor) / Area(convex_hull)
- **Branch Count**: Number of connected components in skeleton
- **Orientation Anisotropy**: Alignment of tumor boundary with tract direction

#### 4.3 Statistical Comparison
- Compare anisotropic vs isotropic (Month 6) metrics
- Box plots of fractal dimension across 8 patients
- Correlation: S100A8 expression vs fractal dimension

#### 4.4 Deliverables
- `output/anisotropic_recurrence_maps.png` (publication-ready 8-panel)
- `output/anisotropic_geometry_metrics.json` (per-patient metrics)
- `output/anisotropic_evolution_all_patients.npz` (combined evolution data)
- Final validation summary printed to console

---

## Technical Architecture

### File Structure
```
src/
  42_anisotropic_pde.py          # Main implementation (all 4 phases)
  
output/
  anisotropic_tensor_profiles.npz      # Week 1
  anisotropic_tensor_validation.png    # Week 1
  anisotropic_evolution_PAT_XXXX.npz   # Week 3 (per patient)
  anisotropic_recurrence_maps.png      # Week 4
  anisotropic_geometry_metrics.json    # Week 4
  anisotropic_evolution_all_patients.npz  # Week 4
```

### Key Classes in `src/42_anisotropic_pde.py`
```python
class TensorFieldBuilder:      # Week 1 - builds 2x2 tensor fields
class AnisotropicFKSolver:     # Week 2 - finite difference solver
class PatientParameterMapper:  # Week 3 - maps gene expression to tensor params
class CohortSimulator:         # Week 3 - runs parallel patient sims
class AnisotropicVisualizer:   # Week 4 - plotting and metrics
```

### Dependencies
- numpy, scipy (ndimage, fftpack), matplotlib
- torch (optional, for GPU acceleration if needed)
- Existing: `output/spatial_recurrence_profiles.npz`, `output/real_cohort_*.csv`

---

## Validation Checkpoints

### Week 1
- [ ] `anisotropic_tensor_profiles.npz` saves all 6 tensor arrays + metadata
- [ ] Validation plot shows diagonal tract with aligned vectors
- [ ] Symmetry check: max|D_xy - D_yx| < 1e-10
- [ ] Positive-definiteness: min(det(D)) > 0, min(trace(D)) > 0

### Week 2
- [ ] 500-step pure diffusion test: mass conservation error < 1e-6
- [ ] No NaN values, no negative densities
- [ ] Zero flux at boundaries verified
- [ ] CFL condition respected (dt ≤ 0.1)

### Week 3
- [ ] 8 patient simulations complete without errors
- [ ] Each patient has unique tensor field driven by gene expression
- [ ] Evolution data saved for all time points

### Week 4
- [ ] 8-panel figure renders without errors
- [ ] Fractal dimension > 1.2 (proving branching, not circular)
- [ ] Perimeter/area ratio significantly > isotropic baseline
- [ ] Metrics JSON exports correctly

---

## Questions for Clarification

1. **Tract geometry**: Should the white matter tract be a straight diagonal (45°), curved (spline), or multiple crossing tracts? (Default: single diagonal 45° corridor, width 15px)

2. **Base diffusion values**: D_parallel=0.15, D_perpendicular=0.005, D_base=0.01? (Ratio 30:1 for strong anisotropy)

3. **Time stepping**: Use explicit Euler (simple, CFL-limited) or implicit/Crank-Nicolson (stable, more complex)? (Default: explicit with dt=0.05 for safety)

4. **Patient-specific tract orientation**: Use same tract for all patients (standardized) or vary by patient? (Default: same tract geometry, only D/ρ scaling varies by patient)

5. **GPU acceleration**: Use torch/CUDA for Week 2-3 simulations? (Default: numpy/scipy CPU; torch optional)

6. **Output grid size**: Keep 100×100 (Month 6) or increase to 200×200 for finer detail? (Default: 100×100 for consistency)

Please confirm or adjust these defaults before implementation begins.