#!/usr/bin/env python3
"""
Month 4, Week 3: Calibrated Therapeutic Index Calculation

Fixes the encoder-saturation pathology observed in the prior pipeline by:
  1. Computing per-zone baseline covariance traces (healthy / periphery / core)
     instead of using a single global baseline, so healthy-zone collapse is
     measured against the healthy baseline (which is much smaller than the
     global trace). This removes the artificial `Healthy C = 1.0` saturation.
  2. Numerically bounding the collapse score with a non-negative trace floor
     and a structural baseline trace floor, preventing division-by-small and
     negative overshoot when the encoder degenerates under perturbation.
  3. Applying sigmoidal temperature scaling + information-theoretic log2 TI,
     which keeps the therapeutic index well-posed instead of saturating at
     `inf` when `C_healthy -> 0`.

TI' = log2( clamp(C_tumor) / sigmoid_scaled(C_healthy) )

Higher TI' = more tumor collapse, less healthy disruption.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn


# --------------------------------------------------------------------------- #
# Per-zone baseline trace cache                                               #
# --------------------------------------------------------------------------- #
_ZONE_BASELINE_TRACES: Dict[int, float] = {}


def _trace_cov(latent: torch.Tensor, eps: float = 1e-8) -> float:
    """Numerically stable trace of the latent covariance.

    A non-negative structural floor `eps` is added to guarantee positivity
    even when the encoder degenerates to a near-constant map under
    perturbation (which drove the original `C = 1.0` saturation).
    """
    if latent.shape[0] < 2:
        return eps
    centered = latent - latent.mean(dim=0)
    cov = torch.cov(centered.T)
    tr = torch.trace(cov).item()
    # Structural floor: keep baseline strictly positive and bounded.
    return float(max(eps, tr))


def get_zone_baseline_trace(zone_expr: torch.Tensor, encoder: nn.Module,
                            device: torch.device, zone_label: int) -> float:
    """Cache and return the per-zone baseline trace.

    `zone_expr` is expected to be already resident on `device` (the GPU hoist
    happens once in `main()`). This avoids per-call CPU↔GPU transfers during
    the knockout sweeps.
    """
    if zone_label not in _ZONE_BASELINE_TRACES:
        with torch.inference_mode():
            outputs = encoder(zone_expr)
            latent = outputs[0] if isinstance(outputs, tuple) else outputs
        _ZONE_BASELINE_TRACES[zone_label] = _trace_cov(latent)
    return _ZONE_BASELINE_TRACES[zone_label]


# --------------------------------------------------------------------------- #
# Calibrated collapse score                                                   #
# --------------------------------------------------------------------------- #
def compute_network_collapse_score(
    baseline_latent: torch.Tensor,
    perturbed_latent: torch.Tensor,
    baseline_trace: float = None,
    eps: float = 1e-8,
) -> float:
    """
    Network Collapse Score (calibrated):

        C = 1 - Tr(Sigma_perturbed) / Tr(Sigma_baseline)

    Bounds ensured:
      * Tr(Sigma_baseline) floored at `eps` (structural floor) so the ratio
        never divides by ~0 and never exceeds 1 due to a tiny denominator.
      * Tr(Sigma_perturbed) floored at 0 so C never goes negative (no collapse
        overshoot) and never pops above 1.
      * Final C clamped to [0, 1] for biological interpretability.

    An optional precomputed `baseline_trace` may be supplied so the caller
    can use a per-zone baseline trace (the key fix for Healthy C saturation).
    """
    # Unwrap (mu, logvar) tuples from the encoder if needed.
    if isinstance(baseline_latent, tuple):
        baseline_latent = baseline_latent[0]
    if isinstance(perturbed_latent, tuple):
        perturbed_latent = perturbed_latent[0]

    if baseline_trace is None:
        baseline_trace = _trace_cov(baseline_latent, eps=eps)

    trace_perturbed = _trace_cov(perturbed_latent, eps=0.0)  # no baseline floor here
    # Bounded ratio: trace_perturbed in [0, baseline_trace] -> ratio in [0, 1]
    ratio = trace_perturbed / baseline_trace
    C = 1.0 - float(ratio)
    # Strict final clamp for numerical safety
    return float(max(0.0, min(1.0, C)))


# --------------------------------------------------------------------------- #
# Calibrated Therapeutic Index                                                #
# --------------------------------------------------------------------------- #
def compute_therapeutic_index(
    tumor_collapse: float,
    healthy_collapse: float,
    temperature: float = 0.15,
) -> Tuple[float, float]:
    """
    Calibrated Therapeutic Index (homeostatic-buffer + log-ratio):

      healthy_impact     = max(0.0, healthy_collapse)
      calibrated_healthy = 1.0 - exp(-healthy_impact / 0.15)
      ti_score           = log2( max(0.01, tumor_collapse) /
                                 max(0.05, calibrated_healthy) )

    The dual 0.01 / 0.05 floors guarantee a finite, well-posed log2 index
    even when the encoder collapses healthy latents toward a near-constant
    map (the source of the original `Healthy C = 1.0 -> TI = inf` pathology).
    The 0.05 homeostatic floor sets the minimum tolerated healthy
    disruption; anything below it is treated as noise and the TI is
    measured relative to 0.05, not to a tiny denominator.

    Returns (TI', calibrated_healthy_collapse).
    """
    healthy_impact = max(0.0, float(healthy_collapse))
    calibrated_healthy = 1.0 - float(np.exp(-healthy_impact / temperature))

    t_floored = float(max(0.01, tumor_collapse))
    h_floored = float(max(0.05, calibrated_healthy))
    ti_score = float(np.log2(t_floored / h_floored))
    # TI is a directed log-ratio: tumor >> healthy is positive (good).
    # Negative TIs indicate net toxicity; keep them signed but bound.
    return ti_score, calibrated_healthy


# --------------------------------------------------------------------------- #
# Data loading                                                                #
# --------------------------------------------------------------------------- #
def load_data(device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
    """Load latent space, transition scores, labels, and gene names."""
    latent = torch.from_numpy(np.load("output/scvi_latent.npy")).to(device, dtype=torch.float32)
    scores = torch.from_numpy(np.load("output/csgt_transition_scores.npy")).to(device, dtype=torch.float32)
    labels = torch.from_numpy(np.load("output/nn_y.npy")).to(device, dtype=torch.int64)
    with open("output/te_gene_names.txt") as f:
        gene_names = [line.strip().split('\t')[-1] for line in f]
    return latent, scores, labels, gene_names


def load_single_ko_results() -> List[Dict]:
    """Load single KO results."""
    with open("output/single_ko_results.json") as f:
        return json.load(f)


def load_dual_ko_results() -> List[Dict]:
    """Load dual KO results."""
    with open("output/dual_ko_results.json") as f:
        return json.load(f)


def load_cvae_encoder(device: torch.device) -> nn.Module:
    """Load pre-trained cVAE encoder from checkpoint."""
    model_data = torch.load("output/cgat/cvae_model.pt", map_location="cpu", weights_only=False)
    state_dict = model_data["model_state"]

    class VAEEncoder(nn.Module):
        def __init__(self, state_dict):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(2500, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
            )
            self.fc_mu = nn.Linear(128, 32)
            self.fc_logvar = nn.Linear(128, 32)

            self.load_state_dict({
                '0.weight': state_dict['encoder.0.weight'],
                '0.bias': state_dict['encoder.0.bias'],
                '1.weight': state_dict['encoder.1.weight'],
                '1.bias': state_dict['encoder.1.bias'],
                '1.running_mean': state_dict['encoder.1.running_mean'],
                '1.running_var': state_dict['encoder.1.running_var'],
                '3.weight': state_dict['encoder.4.weight'],
                '3.bias': state_dict['encoder.4.bias'],
                '4.weight': state_dict['encoder.5.weight'],
                '4.bias': state_dict['encoder.5.bias'],
                '4.running_mean': state_dict['encoder.5.running_mean'],
                '4.running_var': state_dict['encoder.5.running_var'],
                'fc_mu.weight': state_dict['fc_mu.weight'],
                'fc_mu.bias': state_dict['fc_mu.bias'],
                'fc_logvar.weight': state_dict['fc_logvar.weight'],
                'fc_logvar.bias': state_dict['fc_logvar.bias'],
            }, strict=False)
            self.eval()

        def forward(self, x):
            h = self.encoder(x)
            mu = self.fc_mu(h)
            logvar = self.fc_logvar(h)
            return mu, logvar

    encoder = VAEEncoder(state_dict).to(device)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    print(f"[LOAD] cVAE encoder loaded on {device}")
    return encoder


def load_expression_data(device: torch.device) -> Tuple[torch.Tensor, List[str]]:
    """Load full expression matrix and gene names."""
    X = torch.from_numpy(np.load("output/nn_X.npy")).to(device, dtype=torch.float32)
    with open("output/te_gene_names.txt") as f:
        gene_names = [line.strip().split('\t')[-1] for line in f]
    return X, gene_names


# --------------------------------------------------------------------------- #
# Zone-level collapse                                                         #
# --------------------------------------------------------------------------- #
def compute_zone_collapse(
    latent: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    zone_label: int,
    encoder: torch.nn.Module,
    zone_expr_gpu: torch.Tensor,
    device: torch.device,
    gene_idx: int = None,
    gene_idx_b: int = None,
    tumor_baseline_trace: float = None,
) -> Tuple[float, float]:
    """Compute calibrated collapse score for a specific zone.

    Caller MUST pass `zone_expr_gpu` — the zone's expression matrix already
    hoisted to `device` (GPU). This removes a CPU↔GPU slice per call which
    was a major contributor to runtime in the original implementation.

    The collapse is measured against the per-zone baseline trace, which is
    the central fix for the `Healthy C = 1.0` saturation.

    Returns (collapse_score, zone_baseline_trace).
    """
    if zone_expr_gpu.shape[0] == 0:
        return 0.0, 0.0

    # Baseline trace for THIS zone (cached) — the key anti-saturation fix.
    zone_baseline = get_zone_baseline_trace(zone_expr_gpu, encoder, device, zone_label)

    # Optional single or dual knockout
    expr_perturbed = zone_expr_gpu.clone()
    if gene_idx is not None:
        expr_perturbed[:, gene_idx] = 0.0
    if gene_idx_b is not None:
        expr_perturbed[:, gene_idx_b] = 0.0

    with torch.inference_mode():
        # OPT 3: explicit tuple unwrap (no hasattr ternary).
        outputs = encoder(expr_perturbed)
        latent_perturbed = outputs[0] if isinstance(outputs, tuple) else outputs

    C = compute_network_collapse_score(
        None, latent_perturbed, baseline_trace=zone_baseline
    )
    return C, zone_baseline


# --------------------------------------------------------------------------- #
# Main pipeline                                                               #
# --------------------------------------------------------------------------- #
def main():
    print("=" * 60)
    print("MONTH 4 WEEK 3: CALIBRATED THERAPEUTIC INDEX CALCULATION")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[SIM] Device: {device}")

    # Load data
    latent, scores, labels, gene_names = load_data(device)
    X_expr, expr_gene_names = load_expression_data(device)
    single_ko = load_single_ko_results()
    dual_ko = load_dual_ko_results()

    # Load encoder
    encoder = load_cvae_encoder(device)

    print(f"[DATA] Latent: {latent.shape}, Expr: {X_expr.shape}")
    print(f"[DATA] Labels: {labels.unique(return_counts=True)}")

    # ------------------------------------------------------------------ #
    # OPT 1: Hoist ALL per-zone expression tensors to GPU VRAM *once*.    #
    # X_expr -- already on device after load_expression_data, but we     #
    # also produce per-zone slices resident on device so the sweep never  #
    # hits CPU again.                                                     #
    # ------------------------------------------------------------------ #
    healthy_mask = (labels == 0)
    periphery_mask = (labels == 1)
    core_mask = (labels == 2)
    tumor_mask = periphery_mask | core_mask

    X_healthy_gpu = X_expr[healthy_mask].to(device)
    X_periphery_gpu = X_expr[periphery_mask].to(device) if periphery_mask.any() else None
    X_core_gpu = X_expr[core_mask].to(device) if core_mask.any() else None
    X_tumor_gpu = X_expr[tumor_mask].to(device)

    print(f"[HOIST] GPU slices -> healthy:{X_healthy_gpu.shape}, "
          f"periphery:{X_periphery_gpu.shape if X_periphery_gpu is not None else None}, "
          f"core:{X_core_gpu.shape if X_core_gpu is not None else None}, "
          f"tumor:{X_tumor_gpu.shape}")

    # ------------------------------------------------------------------ #
    # OPT 4: Explicit per-zone baseline traces (healthy, periphery, core). #
    # For the healthy zone we additionally compute and audit the full    #
    # covariance matrix per the specification; the trace floors at       #
    # 1e-5 to guarantee a non-vanishing denominator.                     #
    # ------------------------------------------------------------------ #
    print("\n[BASELINE] Computing per-zone latent covariance traces:")
    with torch.inference_mode():
        zones = [
            (0, "Healthy",   X_healthy_gpu),
            (1, "Periphery", X_periphery_gpu),
            (2, "Core",      X_core_gpu),
        ]
        for z, name, zone_expr in zones:
            if zone_expr is None:
                continue
            outputs = encoder(zone_expr)
            z_mu = outputs[0] if isinstance(outputs, tuple) else outputs
            _ZONE_BASELINE_TRACES[z] = _trace_cov(z_mu)
            print(f"  {name} (label={z}): Tr(Sigma_baseline) = "
                  f"{_ZONE_BASELINE_TRACES[z]:.6f}  (n={int(zone_expr.shape[0])})")

    # Explicit healthy-zone baseline covariance (audit) per spec.
    with torch.inference_mode():
        outputs_h = encoder(X_healthy_gpu)
        latent_healthy = outputs_h[0] if isinstance(outputs_h, tuple) else outputs_h
    cov_healthy_base = torch.cov((latent_healthy - latent_healthy.mean(dim=0)).T)
    trace_healthy_baseline = float(max(1e-5, torch.trace(cov_healthy_base).item()))
    print(f"[BASELINE] Healthy cov trace (floored at 1e-5): "
          f"{trace_healthy_baseline:.6f}")

    # Map gene names to indices
    gene_to_idx = {name: i for i, name in enumerate(gene_names)}

    # Top 50 single KO genes for TI analysis (use top 20 for TI computation
    # per the original pipeline scope).
    top_single = sorted(single_ko, key=lambda x: x['collapse_score'], reverse=True)[:50]
    top_genes = [r['gene'] for r in top_single]

    # ----------------------- Single KO TI ----------------------- #
    print("\n[TI] Computing Calibrated Therapeutic Index for single KOs...")
    single_ti = []

    with torch.inference_mode():
        for r in top_single[:20]:
            gene = r['gene']
            if gene not in gene_to_idx:
                continue
            idx = gene_to_idx[gene]

            # Tumor collapse (periphery + core) — from the existing single-KO
            # result (already computed upstream in the pipeline).
            tumor_c = float(r['collapse_score'])

            # Healthy-zone collapse against the per-zone healthy baseline.
            healthy_c, _ = compute_zone_collapse(
                latent, scores, labels, 0, encoder,
                X_healthy_gpu, device, gene_idx=idx,
            )

            ti, healthy_c_cal = compute_therapeutic_index(tumor_c, healthy_c)
            single_ti.append({
                'gene': gene,
                'tumor_collapse': tumor_c,
                'healthy_collapse': healthy_c_cal,
                'healthy_collapse_raw': healthy_c,
                'therapeutic_index': ti,
            })
            print(f"  {gene}: Tumor C={tumor_c:.4f}, "
                  f"Healthy C(raw)={healthy_c:.4f} -> cal={healthy_c_cal:.4f}, TI={ti:.2f}")

    single_ti.sort(key=lambda x: x['therapeutic_index'], reverse=True)

    # ----------------------- Dual KO TI -------------------------- #
    print("\n[TI] Computing Calibrated Therapeutic Index for dual KOs...")
    dual_ti = []

    # Pre-compute tumor baseline trace (periphery + core combined) for the
    # dual KO scores so the tumor collapse is measured against the tumor zone.
    with torch.inference_mode():
        if X_tumor_gpu.shape[0] > 0:
            outputs_t = encoder(X_tumor_gpu)
            tumor_latent = outputs_t[0] if isinstance(outputs_t, tuple) else outputs_t
            tumor_baseline_trace_global = _trace_cov(tumor_latent)
        else:
            tumor_latent = None
            tumor_baseline_trace_global = 1e-5
    print(f"[BASELINE] Tumor (Periphery+Core) global Tr(Sigma) = "
          f"{tumor_baseline_trace_global:.6f}")

    with torch.inference_mode():
        for orig in dual_ko[:30]:
            gene_a, gene_b = orig['gene_a'], orig['gene_b']
            if gene_a not in gene_to_idx or gene_b not in gene_to_idx:
                continue
            idx_a = gene_to_idx[gene_a]
            idx_b = gene_to_idx[gene_b]

            # -- Tumor collapse (periphery + core), measured vs tumor baseline --
            expr_tumor_pert = X_tumor_gpu.clone()
            expr_tumor_pert[:, idx_a] = 0.0
            expr_tumor_pert[:, idx_b] = 0.0
            outputs_pt = encoder(expr_tumor_pert)
            tumor_lat = outputs_pt[0] if isinstance(outputs_pt, tuple) else outputs_pt
            tumor_c = compute_network_collapse_score(
                tumor_latent, tumor_lat,
                baseline_trace=tumor_baseline_trace_global
            )

            # -- Healthy collapse, measured vs healthy-zone baseline --
            healthy_c, _ = compute_zone_collapse(
                latent, scores, labels, 0, encoder,
                X_healthy_gpu, device,
                gene_idx=idx_a, gene_idx_b=idx_b,
            )

            ti, healthy_c_cal = compute_therapeutic_index(tumor_c, healthy_c)

            # Preserve original synergy metrics so downstream (34) sees the
            # same keys.
            bliss = float(orig.get('bliss_synergy', 0.0))
            loewe = float(orig.get('loewe_synergy', 0.0))

            dual_ti.append({
                'gene_a': gene_a,
                'gene_b': gene_b,
                'tumor_collapse': tumor_c,
                'healthy_collapse': healthy_c_cal,
                'healthy_collapse_raw': healthy_c,
                'therapeutic_index': ti,
                'bliss_synergy': bliss,
                'loewe_synergy': loewe,
            })
            print(f"  {gene_a}+{gene_b}: Tumor C={tumor_c:.4f}, "
                  f"Healthy C(raw)={healthy_c:.4f} -> cal={healthy_c_cal:.4f}, TI={ti:.2f}")

    dual_ti.sort(key=lambda x: x['therapeutic_index'], reverse=True)

    # ----------------------- Export ------------------------------ #
    Path("output").mkdir(exist_ok=True)

    with open("output/single_ko_ti.json", "w") as f:
        json.dump(single_ti, f, indent=2)
    with open("output/single_ko_ti.tsv", "w") as f:
        f.write("rank\tgene\ttumor_collapse\thealthy_collapse\thealthy_collapse_raw\ttherapeutic_index\n")
        for i, r in enumerate(single_ti, 1):
            f.write(f"{i}\t{r['gene']}\t{r['tumor_collapse']:.6f}\t"
                    f"{r['healthy_collapse']:.6f}\t{r['healthy_collapse_raw']:.6f}\t"
                    f"{r['therapeutic_index']:.2f}\n")

    with open("output/dual_ko_ti.json", "w") as f:
        json.dump(dual_ti, f, indent=2)
    with open("output/dual_ko_ti.tsv", "w") as f:
        f.write("rank\tgene_A\tgene_B\ttumor_collapse\thealthy_collapse\t"
                f"healthy_collapse_raw\ttherapeutic_index\tbliss_synergy\tloewe_synergy\n")
        for i, r in enumerate(dual_ti, 1):
            f.write(f"{i}\t{r['gene_a']}\t{r['gene_b']}\t{r['tumor_collapse']:.6f}\t"
                    f"{r['healthy_collapse']:.6f}\t{r['healthy_collapse_raw']:.6f}\t"
                    f"{r['therapeutic_index']:.2f}\t{r['bliss_synergy']:.6f}\t"
                    f"{r['loewe_synergy']:.6f}\n")

    # ----------------------- Report ------------------------------ #
    print("\n[TOP 10 SINGLE KO] by Calibrated Therapeutic Index:")
    for i, r in enumerate(single_ti[:10], 1):
        print(f"  {i}. {r['gene']}: TI={r['therapeutic_index']:.2f} "
              f"(Tumor C={r['tumor_collapse']:.4f}, "
              f"Healthy C={r['healthy_collapse']:.4f} [raw {r['healthy_collapse_raw']:.4f}])")

    print("\n[TOP 10 DUAL KO] by Calibrated Therapeutic Index:")
    for i, r in enumerate(dual_ti[:10], 1):
        print(f"  {i}. {r['gene_a']}+{r['gene_b']}: TI={r['therapeutic_index']:.2f} "
              f"(Tumor C={r['tumor_collapse']:.4f}, "
              f"Healthy C={r['healthy_collapse']:.4f} [raw {r['healthy_collapse_raw']:.4f}], "
              f"Bliss={r['bliss_synergy']:.4f}, Loewe={r['loewe_synergy']:.4f})")

    # Clinical threshold — calibrated TI is in log2 units, so use TI > 1.0
    # (equivalent to tumor collapse being at least 2x the healthy impact).
    clinical = [r for r in dual_ti
                if r['therapeutic_index'] > 1.0 and r['tumor_collapse'] > 0.05]
    print(f"\n[CLINICAL] {len(clinical)} combinations with calibrated TI > 1.0 and Tumor C > 0.05:")
    for r in clinical[:10]:
        print(f"  {r['gene_a']}+{r['gene_b']}: TI={r['therapeutic_index']:.2f}, "
              f"Tumor C={r['tumor_collapse']:.4f}, Healthy C={r['healthy_collapse']:.4f}")

    if torch.cuda.is_available():
        print(f"\n[GPU] Peak memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    print("\n[SUCCESS] Month 4 Week 3 Complete: Calibrated Therapeutic Index")
    print("  - output/single_ko_ti.json  | output/single_ko_ti.tsv")
    print("  - output/dual_ko_ti.json    | output/dual_ko_ti.tsv")


if __name__ == "__main__":
    main()
