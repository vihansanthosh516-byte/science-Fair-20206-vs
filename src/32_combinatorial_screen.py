from __future__ import annotations

import importlib.util
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

# Import helper functions from therapeutic_index module.
# We import the calibrated collapse score (already bounded in [0,1] via a
# structural trace floor) plus the per-zone trace helper, so the
# combinatorial screen measures collapse against the TUMOR baseline trace
# instead of a mix of healthy + tumor latents that previously saturated.
spec = importlib.util.spec_from_file_location("therapeutic_index", "src/33_therapeutic_index.py")
therapeutic_index = importlib.util.module_from_spec(spec)
spec.loader.exec_module(therapeutic_index)
compute_network_collapse_score = therapeutic_index.compute_network_collapse_score
compute_therapeutic_index = therapeutic_index.compute_therapeutic_index
_trace_cov = therapeutic_index._trace_cov


class VAEEncoder(nn.Module):
    """cVAE encoder extracted from trained model."""
    
    def __init__(self, state_dict: dict):
        super().__init__()
        # Build encoder architecture matching saved weights
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
        
        # Load weights
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
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu, logvar = self.encode(x)
        return self.reparameterize(mu, logvar)


def load_cvae_encoder(device: torch.device):
    """Load pre-trained cVAE encoder from checkpoint."""
    state_dict = torch.load("output/cgat/cvae_model.pt", map_location="cpu", weights_only=False)["model_state"]
    encoder = VAEEncoder(state_dict).to(device)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    print(f"[LOAD] cVAE encoder loaded on {device}")
    return encoder


def load_data(device: torch.device):
    """Load latent space, transition scores, labels, and gene names."""
    latent = torch.from_numpy(np.load("output/scvi_latent.npy")).to(device, dtype=torch.float32)
    scores = torch.from_numpy(np.load("output/csgt_transition_scores.npy")).to(device, dtype=torch.float32)
    labels = torch.from_numpy(np.load("output/nn_y.npy")).to(device, dtype=torch.int64)
    with open("output/te_gene_names.txt") as f:
        gene_names = [line.strip().split('\t')[-1] for line in f.readlines()]
    return latent, scores, labels, gene_names


def load_single_ko_results() -> List[Dict]:
    """Load single KO results."""
    with open("output/single_ko_results.json") as f:
        return json.load(f)


def load_expression_data(device: torch.device):
    """Load full expression matrix and gene names."""
    X = torch.from_numpy(np.load("output/nn_X.npy")).to(device, dtype=torch.float32)
    with open("output/te_gene_names.txt") as f:
        gene_names = [line.strip().split('\t')[-1] for line in f]
    return X, gene_names





def virtual_knockout(
    encoder: nn.Module,
    expression: torch.Tensor,
    gene_idx: int,
    device: torch.device,
) -> torch.Tensor:
    """Perform virtual knockout by zeroing gene expression and re-encoding."""
    expr_perturbed = expression.clone()
    expr_perturbed[:, gene_idx] = 0.0
    with torch.no_grad():
        mu, _ = encoder.encode(expr_perturbed)
    return mu


def main():
    print("=" * 60)
    print("MONTH 4 WEEK 2: COMBINATORIAL DRUG SCREEN")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[SIM] Device: {device}")

    # Load models and data
    encoder = load_cvae_encoder(device)
    latent, scores, labels, gene_names = load_data(device)
    X_expr, expr_gene_names = load_expression_data(device)
    single_ko = load_single_ko_results()

    # Get top 50 single KO targets for combinatorial screen
    single_ko_sorted = sorted(single_ko, key=lambda x: x['collapse_score'], reverse=True)
    top_genes = [g['gene'] for g in single_ko_sorted[:50]]
    top_indices = [expr_gene_names.index(g) for g in top_genes if g in expr_gene_names]
    print(f"[SCREEN] Top {len(top_indices)} genes for combinatorial screen")

    # Baseline latent and collapse score (global, for reference only).
    baseline_latent = latent
    baseline_trace = torch.trace(torch.cov(latent.T)).item()
    print(f"[BASELINE] Trace(Sigma_global) = {baseline_trace:.4f}")

    # ------------------------------------------------------------------ #
    # OPT 1: Hoist tensors to GPU VRAM *before* the loop.                 #
    # ------------------------------------------------------------------ #
    # Tumor zone (periphery + core) baseline trace — anti-saturation fix.
    # We keep the tumor slice resident on GPU for the duration of the
    # combinatorial sweep so the inner loop does not re-slice from CPU.
    tumor_mask = (labels == 1) | (labels == 2)
    X_expr_tumor = X_expr[tumor_mask].to(device)  # SHARED across all pairs

    # Pre-compute single-KO collapse lookup once (Python dict) so the
    # inner loop avoids an O(n) linear scan from `next(...)`.
    single_ko_lookup = {r['gene']: float(r['collapse_score']) for r in single_ko}

    if tumor_mask.any():
        with torch.inference_mode():
            # OPT 3: Unwrap ternary encoder cleanly.
            outputs = encoder(X_expr_tumor)
            tumor_mu = outputs[0] if isinstance(outputs, tuple) else outputs
        tumor_baseline_trace = _trace_cov(tumor_mu)
        print(f"[BASELINE] Trace(Sigma_tumor) = {tumor_baseline_trace:.4f}  "
              f"(n_tumor={int(tumor_mask.sum())})")
    else:
        tumor_baseline_trace = baseline_trace
        tumor_mu = None

    # Run combinatorial screen
    print("\n[SCREEN] Running dual KO screen...")
    t0 = time.perf_counter()

    results = []
    total_pairs = len(top_indices) * (len(top_indices) - 1) // 2
    pair_count = 0

    # ------------------------------------------------------------------ #
    # OPT 2: Wrap the entire loop block in torch.inference_mode() to      #
    # disable gradient tracking overhead for the whole sweep.            #
    # ------------------------------------------------------------------ #
    with torch.inference_mode():
        for i, idx_i in enumerate(top_indices):
            # Pre-zero gene i once per outer loop iteration; inner loop
            # only needs to additionally zero gene j. This halves the
            # 25M-element column writes inside the inner loop.
            base_i = X_expr_tumor.clone()
            base_i[:, idx_i] = 0.0

            for j, idx_j in enumerate(top_indices[i+1:], i+1):
                pair_count += 1
                gene_i = expr_gene_names[idx_i]
                gene_j = expr_gene_names[idx_j]

                # Dual knockout — perturb only the TUMOR zone. We clone
                # from the already-gene-i-zeroed `base_i` (still on GPU)
                # and additionally zero gene j.
                expr_perturbed = base_i.clone()
                expr_perturbed[:, idx_j] = 0.0

                # OPT 3: explicit tuple unwrap (no hasattr ternary).
                outputs = encoder(expr_perturbed)
                latent_perturbed = outputs[0] if isinstance(outputs, tuple) else outputs

                C = compute_network_collapse_score(
                    tumor_mu, latent_perturbed,
                    baseline_trace=tumor_baseline_trace,
                )

                # Bliss synergy score (O(1) dict lookup, not linear scan).
                C_i = single_ko_lookup.get(gene_i, 0.0)
                C_j = single_ko_lookup.get(gene_j, 0.0)
                bliss_expected = C_i + C_j - C_i * C_j
                bliss_synergy = C - bliss_expected

                # Loewe additivity
                loewe_synergy = C - (C_i + C_j) / 2 if (C_i + C_j) > 0 else 0

                results.append({
                    'gene_a': gene_i,
                    'gene_b': gene_j,
                    'idx_a': int(idx_i),
                    'idx_b': int(idx_j),
                    'collapse_score': float(C),
                    'C_A': float(C_i),
                    'C_B': float(C_j),
                    'bliss_synergy': float(bliss_synergy),
                    'loewe_synergy': float(loewe_synergy),
                })

                if pair_count % 200 == 0:
                    elapsed = time.perf_counter() - t0
                    print(f"  [{pair_count}/{total_pairs}] {gene_i}+{gene_j}: "
                          f"C={C:.4f}, Bliss={bliss_synergy:.4f} ({elapsed:.1f}s)")

    elapsed = time.perf_counter() - t0
    print(f"\n[SCREEN] Completed {len(results)} pairs in {elapsed:.1f}s")

    # Sort by synergy
    results.sort(key=lambda x: x['bliss_synergy'], reverse=True)

    # Export
    Path("output").mkdir(exist_ok=True)
    with open("output/dual_ko_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # TSV summary
    with open("output/dual_ko_summary.tsv", "w") as f:
        f.write("rank\tgene_A\tgene_B\tcollapse_score\tC_A\tC_B\tbliss_synergy\tloewe_synergy\n")
        for i, r in enumerate(results[:50], 1):
            f.write(f"{i}\t{r['gene_a']}\t{r['gene_b']}\t{r['collapse_score']:.6f}\t"
                    f"{r['C_A']:.6f}\t{r['C_B']:.6f}\t{r['bliss_synergy']:.6f}\t{r['loewe_synergy']:.6f}\n")

    # Print top 10
    print("\n[TOP 10] Most synergistic dual knockouts (Bliss):")
    for i, r in enumerate(results[:10], 1):
        print(f"  {i}. {r['gene_a']} + {r['gene_b']}: "
              f"C={r['collapse_score']:.4f}, Bliss={r['bliss_synergy']:.4f}, Loewe={r['loewe_synergy']:.4f}")

    # GPU memory
    if torch.cuda.is_available():
        print(f"\n[GPU] Peak memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    print("\n[SUCCESS] Month 4 Week 2 Complete: Combinatorial Drug Screen")
    print("  - output/dual_ko_results.json")
    print("  - output/dual_ko_summary.tsv")


if __name__ == "__main__":
    import json
    import time
    from pathlib import Path
    from typing import Dict, List, Tuple

    import numpy as np
    import torch
    import torch.nn as nn

    main()