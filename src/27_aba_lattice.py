#!/usr/bin/env python3
"""
Month 3, Week 1: Stochastic Agent-Based Lattice Simulator — CALIBRATED
2D/3D Cellular Automaton for tumor invasion with rules derived from empirical graph metrics.

CALIBRATION TARGETS:
- 512x512 lattice at 5 µm/pixel
- Transition rates in [0.5, 0.8] to match PDE velocity
- Core_necrose = 0.005, Core_proliferate = 0.2
- Healthy tissue homeostasis to prevent depletion
- Target: 10-50 µm/hr front speed, 10-40% necrotic fraction
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch


class CellularAutomaton:
    """
    2D Lattice-based tumor invasion simulator.
    
    Cell states:
    - 0: Empty
    - 1: Healthy
    - 2: Periphery (transition zone)
    - 3: Core (malignant)
    - 4: Necrotic
    """
    
    def __init__(
        self,
        grid_size: Tuple[int, int] = (512, 512),
        initial_healthy_frac: float = 0.75,
        initial_core_frac: float = 0.03,
        device: torch.device = None,
    ):
        self.grid_size = grid_size
        self.H, self.W = grid_size
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize grid on GPU
        self.grid = torch.zeros(grid_size, dtype=torch.int8, device=self.device)
        self.morphogen = torch.zeros(grid_size, dtype=torch.float32, device=self.device)
        self.velocity_field = torch.zeros(grid_size, dtype=torch.float32, device=self.device)
        
        # Load velocity gradient magnitude from Month 1
        self._load_velocity_field()
        
        # Initialize tissue
        self._initialize_tissue(initial_healthy_frac, initial_core_frac)
        
        # CALIBRATED Transition rules (P0 patch)
        # 512x512 at 5µm/pixel, transition rates [0.5, 0.8]
        self.rules = {
            'healthy_proliferate': 0.60,
            'healthy_to_periphery_base': 0.70,      # P0: increased
            'healthy_velocity_sensitivity': 0.60,    # P0: stronger velocity coupling
            'periphery_proliferate': 0.75,
            'periphery_to_core_base': 0.65,         # P0: increased
            'periphery_velocity_sensitivity': 0.55,  # P0: stronger velocity coupling
            'periphery_secrete_morphogen': 0.80,
            'core_proliferate': 0.20,                # P0: reduced to 0.2
            'core_necrose': 0.005,                   # P0: reduced to 0.005
            'diffusion_rate': 0.30,
            # Healthy homeostasis
            'healthy_replenish_rate': 0.02,          # Homeostatic replenishment
        }
        
        # Neighborhood offsets
        self.neighbors = self._get_neighborhood()
        
        print(f"[CA] Initialized {self.H}x{self.W} lattice on {self.device}")
        print(f"[CA] Velocity field loaded: range=[{self.velocity_field.min():.4f}, {self.velocity_field.max():.4f}]")
        print(f"[CA] Cell counts: {self.count_cells()}")
    
    def _load_velocity_field(self) -> None:
        """Load phenotypic velocity magnitude from Month 1 and map to grid."""
        try:
            vel_mag = np.load("output/phenotypic_velocity_magnitude.npy")  # (15000,)
            pca_coords = np.load("output/phenotypic_velocity_pca2d.npy")   # (15000, 2)
            
            # Normalize coords to grid
            coords = pca_coords.copy()
            coords -= coords.min(axis=0)
            coords /= (coords.max(axis=0) + 1e-8)
            coords = (coords * np.array([self.H - 1, self.W - 1])).astype(int)
            coords = np.clip(coords, 0, [self.H - 1, self.W - 1])
            
            # Bin velocity onto grid (mean per bin)
            vel_grid = np.zeros((self.H, self.W), dtype=np.float32)
            count_grid = np.zeros((self.H, self.W), dtype=np.int32)
            
            for i in range(len(vel_mag)):
                h, w = coords[i]
                vel_grid[h, w] += vel_mag[i]
                count_grid[h, w] += 1
            
            mask = count_grid > 0
            vel_grid[mask] /= count_grid[mask]
            
            # Smooth with Gaussian filter
            from scipy.ndimage import gaussian_filter
            vel_grid = gaussian_filter(vel_grid, sigma=3.0)
            
            self.velocity_field = torch.from_numpy(vel_grid).to(self.device, dtype=torch.float32)
            
            # Normalize to [0, 1] for probability scaling
            v_min, v_max = self.velocity_field.min(), self.velocity_field.max()
            if v_max > v_min:
                self.velocity_field = (self.velocity_field - v_min) / (v_max - v_min)
            
            print(f"[VEL] Loaded and normalized velocity field: range=[{v_min:.4f}, {v_max:.4f}]")
            
        except Exception as e:
            print(f"[VEL] Warning: Could not load velocity field: {e}")
            self.velocity_field = torch.zeros(self.grid_size, dtype=torch.float32, device=self.device)
    
    def _get_neighborhood(self) -> List[Tuple[int, int]]:
        """Get Moore neighborhood offsets (8 neighbors)."""
        return [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    
    def _initialize_tissue(self, healthy_frac: float, core_frac: float) -> None:
        """Initialize grid with healthy tissue and core tumor seed."""
        n_cells = self.H * self.W
        n_healthy = int(n_cells * healthy_frac)
        n_core = int(n_cells * core_frac)
        
        # Random placement
        flat = torch.zeros(n_cells, dtype=torch.int8, device=self.device)
        flat[:n_healthy] = 1  # Healthy
        flat[n_healthy:n_healthy + n_core] = 3  # Core
        
        # Shuffle
        perm = torch.randperm(n_cells, device=self.device)
        flat = flat[perm]
        
        self.grid = flat.view(self.H, self.W)
        
        # Initialize morphogen at core locations
        core_mask = (self.grid == 3)
        self.morphogen[core_mask] = 1.0
    
    def count_cells(self) -> Dict[str, int]:
        """Count cells of each type."""
        counts = {}
        for state, name in [(0, 'empty'), (1, 'healthy'), (2, 'periphery'), (3, 'core'), (4, 'necrotic')]:
            counts[name] = int((self.grid == state).sum().item())
        return counts
    
    def count_neighbors(self, state: int) -> torch.Tensor:
        """Count neighbors of given state for each cell."""
        padded = torch.nn.functional.pad(self.grid.float(), (1, 1, 1, 1), mode='constant', value=0)
        count = torch.zeros_like(self.grid, dtype=torch.float32)
        
        for dh, dw in self.neighbors:
            count += (padded[1+dh:self.H+1+dh, 1+dw:self.W+1+dw] == state).float()
        
        return count
    
    def step(self) -> Dict[str, int]:
        """Execute one simulation step (no sub-stepping - base step)."""
        return self._execute_ca_step()
    
    def _execute_ca_step(self) -> Dict[str, int]:
        """Execute a single CA step (core logic)."""
        new_grid = self.grid.clone()
        changes = {'proliferation': 0, 'transition': 0, 'necrosis': 0, 'secretion': 0, 'replenish': 0}
        
        # 1. Morphogen diffusion (Fisher-Kolmogorov style)
        self._diffuse_morphogen()
        
        # 2. Cell state transitions with velocity-coupled probabilities
        
        # Healthy cell rules
        healthy_mask = (self.grid == 1)
        if healthy_mask.any():
            # Neighbor counts
            periphery_neighbors = self.count_neighbors(2)
            core_neighbors = self.count_neighbors(3)
            total_tumor_neighbors = periphery_neighbors + core_neighbors
            empty_neighbors = self.count_neighbors(0)
            
            # Velocity field at healthy cell locations
            vel_at_healthy = self.velocity_field * healthy_mask.float()
            
            # Healthy -> Periphery: base + velocity-coupled
            morphogen_threshold = 0.25
            transition_prob = (
                self.rules['healthy_to_periphery_base'] + 
                self.rules['healthy_velocity_sensitivity'] * vel_at_healthy
            ) * (self.morphogen > morphogen_threshold).float()
            
            transition_mask = (
                healthy_mask & 
                (torch.rand_like(self.grid.float()) < transition_prob) & 
                (total_tumor_neighbors > 0)
            )
            new_grid[transition_mask] = 2
            changes['transition'] += int(transition_mask.sum().item())
            
            # Healthy proliferation
            prolif_mask = healthy_mask & (torch.rand_like(self.grid.float()) < self.rules['healthy_proliferate']) & (empty_neighbors > 0)
            if prolif_mask.any():
                self._proliferate_cells(prolif_mask, 1, new_grid)
                changes['proliferation'] += int(prolif_mask.sum().item())
            
            # Healthy homeostasis: replenish healthy in empty spaces near healthy tissue
            replenish_mask = (self.grid == 0) & (empty_neighbors > 0) & (torch.rand_like(self.grid.float()) < self.rules['healthy_replenish_rate'])
            if replenish_mask.any():
                new_grid[replenish_mask] = 1
                changes['replenish'] += int(replenish_mask.sum().item())
        
        # Periphery cell rules
        periphery_mask = (self.grid == 2)
        if periphery_mask.any():
            empty_neighbors = self.count_neighbors(0)
            healthy_neighbors = self.count_neighbors(1)
            vel_at_periphery = self.velocity_field * periphery_mask.float()
            
            # Periphery -> Core: base + velocity-coupled
            core_prob = self.rules['periphery_to_core_base'] + self.rules['periphery_velocity_sensitivity'] * vel_at_periphery
            core_transition = periphery_mask & (torch.rand_like(self.grid.float()) < core_prob)
            new_grid[core_transition] = 3
            changes['transition'] += int(core_transition.sum().item())
            
            # Periphery proliferation
            prolif_mask = periphery_mask & (torch.rand_like(self.grid.float()) < self.rules['periphery_proliferate']) & (empty_neighbors > 0)
            if prolif_mask.any():
                self._proliferate_cells(prolif_mask, 2, new_grid)
                changes['proliferation'] += int(prolif_mask.sum().item())
            
            # Morphogen secretion
            secret_mask = periphery_mask & (torch.rand_like(self.grid.float()) < self.rules['periphery_secrete_morphogen'])
            self.morphogen[secret_mask] = torch.clamp(self.morphogen[secret_mask] + 0.5, max=1.0)
            changes['secretion'] += int(secret_mask.sum().item())
        
        # Core cell rules
        core_mask = (self.grid == 3)
        if core_mask.any():
            empty_neighbors = self.count_neighbors(0)
            
            # Core proliferation
            prolif_mask = core_mask & (torch.rand_like(self.grid.float()) < self.rules['core_proliferate']) & (empty_neighbors > 0)
            if prolif_mask.any():
                self._proliferate_cells(prolif_mask, 3, new_grid)
                changes['proliferation'] += int(prolif_mask.sum().item())
            
            # Necrosis (P0: core_necrose = 0.005)
            necro_mask = core_mask & (torch.rand_like(self.grid.float()) < self.rules['core_necrose'])
            new_grid[necro_mask] = 4
            changes['necrosis'] += int(necro_mask.sum().item())
        
        self.grid = new_grid
        return changes
    
    def execute_substeps(self, n_substeps: int = 10) -> Dict[str, int]:
        """Execute multiple CA sub-steps (for multi-rate time-stepping with PDE)."""
        total_changes = {'proliferation': 0, 'transition': 0, 'necrosis': 0, 'secretion': 0, 'replenish': 0}
        for _ in range(n_substeps):
            changes = self._execute_ca_step()
            for k, v in changes.items():
                total_changes[k] += v
        return total_changes
    
    def _proliferate_cells(self, source_mask: torch.Tensor, cell_type: int, target_grid: torch.Tensor) -> None:
        """Proliferate cells into random empty neighbors."""
        source_coords = torch.nonzero(source_mask, as_tuple=False)
        for coord in source_coords:
            h, w = coord[0].item(), coord[1].item()
            # Find empty neighbors
            empty_neighbors = []
            for dh, dw in self.neighbors:
                nh, nw = h + dh, w + dw
                if 0 <= nh < self.H and 0 <= nw < self.W and target_grid[nh, nw] == 0:
                    empty_neighbors.append((nh, nw))
            if empty_neighbors:
                nh, nw = empty_neighbors[np.random.randint(len(empty_neighbors))]
                target_grid[nh, nw] = cell_type
    
    def _diffuse_morphogen(self) -> None:
        """Diffuse morphogen field (Fisher-Kolmogorov style)."""
        padded = torch.nn.functional.pad(self.morphogen, (1, 1, 1, 1), mode='constant', value=0)
        laplacian = (
            padded[:-2, 1:-1] + padded[2:, 1:-1] + padded[1:-1, :-2] + padded[1:-1, 2:] - 4 * self.morphogen
        )
        self.morphogen += self.rules['diffusion_rate'] * laplacian
        self.morphogen = torch.clamp(self.morphogen, 0, 1)
    
    def get_invasion_front(self) -> Tuple[torch.Tensor, float]:
        """Detect invasion front and compute velocity."""
        tumor_mask = (self.grid >= 2).float()
        healthy_mask = (self.grid == 1).float()
        
        padded_tumor = torch.nn.functional.pad(tumor_mask, (1, 1, 1, 1), mode='constant', value=0)
        dx = padded_tumor[1:-1, 2:] - padded_tumor[1:-1, :-2]
        dy = padded_tumor[2:, 1:-1] - padded_tumor[:-2, 1:-1]
        gradient_mag = torch.sqrt(dx**2 + dy**2)
        
        front = healthy_mask * (gradient_mag > 0)
        front_velocity = front.sum().item() / max(healthy_mask.sum().item(), 1)
        
        return front, front_velocity
    
    def to_numpy(self) -> Tuple[np.ndarray, np.ndarray]:
        """Convert grid and morphogen to numpy arrays."""
        return self.grid.cpu().numpy(), self.morphogen.cpu().numpy()


def run_simulation(
    n_steps: int = 400,
    grid_size: Tuple[int, int] = (512, 512),
    save_interval: int = 20,
    ca_substeps_per_pde_step: int = 10,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[Dict]]:
    """Run full invasion simulation with CA sub-stepping."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[SIM] Running on {device}")
    print(f"[SIM] CA sub-steps per PDE step: {ca_substeps_per_pde_step}")
    
    ca = CellularAutomaton(grid_size=grid_size, device=device)
    
    # Storage
    grid_history = []
    morphogen_history = []
    metrics_history = []
    
    # Initial state
    grid_history.append(ca.grid.cpu().numpy())
    morphogen_history.append(ca.morphogen.cpu().numpy())
    metrics_history.append({
        'step': 0,
        'cell_counts': ca.count_cells(),
        'front_velocity': 0.0,
    })
    
    for step in range(1, n_steps + 1):
        # Execute multiple CA sub-steps per PDE step
        changes = ca.execute_substeps(ca_substeps_per_pde_step)
        
        # Metrics
        front, velocity = ca.get_invasion_front()
        cell_counts = ca.count_cells()
        
        metrics = {
            'step': step,
            'cell_counts': cell_counts,
            'changes': changes,
            'front_velocity': velocity,
            'front_cells': int(front.sum().item()),
        }
        metrics_history.append(metrics)
        
        # Save snapshots
        if step % save_interval == 0:
            grid_history.append(ca.grid.cpu().numpy())
            morphogen_history.append(ca.morphogen.cpu().numpy())
        
        # Progress
        if step % 50 == 0:
            print(f"[SIM] Step {step}/{n_steps} | Cells: {cell_counts} | Front velocity: {velocity:.4f}")
    
    return grid_history, morphogen_history, metrics_history


def main():
    print("=" * 60)
    print("MONTH 3 WEEK 1: STOCHASTIC AGENT-BASED INVASION ENGINE (CALIBRATED)")
    print("=" * 60)
    
    # Run simulation
    grid_hist, morph_hist, metrics = run_simulation(
        n_steps=400,
        grid_size=(512, 512),
        save_interval=20,
    )
    
    # Export
    print("\n[EXPORT] Saving simulation data...")
    Path("output").mkdir(exist_ok=True)
    
    np.save("output/aba_grid_history.npy", np.array(grid_hist, dtype=np.int8))
    np.save("output/aba_morphogen_history.npy", np.array(morph_hist, dtype=np.float32))
    with open("output/aba_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    # Final stats
    final = metrics[-1]
    print(f"\n[SIM] Final cell counts: {final['cell_counts']}")
    print(f"[SIM] Total steps: {len(metrics)}")
    print(f"[SIM] Saved {len(grid_hist)} snapshots")
    
    print("\n[SUCCESS] Month 3 Week 1 Complete: Agent-Based Invasion Engine")
    print("  - Grid history: output/aba_grid_history.npy")
    print("  - Morphogen history: output/aba_morphogen_history.npy")
    print("  - Metrics: output/aba_metrics.json")


if __name__ == "__main__":
    main()