"""
12_gat_build_graph.py
======================
Build k-NN graph in cVAE latent space with categorical edge features.

Uses the 15k subsampled cells (already balanced) from the cVAE latent space.
Metadata (patient, region) loaded from the subsampled h5ad file.
"""
import os, time, resource, gc, json
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import anndata as ad

ROOT = "/mnt/c/Users/vihan/20206 science fair"
OUT_DIR = os.path.join(ROOT, "output", "cgat")

LATENT_PATH = os.path.join(OUT_DIR, "cvae_latent.npy")      # (15000, 32)
ADATA_SUB_PATH = os.path.join(ROOT, "output", "02_adata_subsampled.h5ad")

GRAPH_EDGE_INDEX = os.path.join(OUT_DIR, "gat_edge_index.npy")
GRAPH_EDGE_ATTR = os.path.join(OUT_DIR, "gat_edge_attr.npy")
GRAPH_META = os.path.join(OUT_DIR, "gat_meta.json")

K = 15  # k-NN neighbors

def mem_mb(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}  RSS={mem_mb():.0f}MB", flush=True)

# ---- 1. Load latent embeddings ----
log("Loading latent embeddings...")
latent = np.load(LATENT_PATH).astype(np.float32)  # (15000, 32)

# Load metadata from subsampled h5ad
adata_sub = ad.read_h5ad(ADATA_SUB_PATH)
labels = adata_sub.obs["class"].values.astype(str)  # Core, Periphery, Healthy
patients = adata_sub.obs["ID1"].values.astype(str)
regions = adata_sub.obs["tissue_histology"].values.astype(str)  # core:GBM, peri:GBM, NormalBrain

n_cells, latent_dim = latent.shape
log(f"Loaded {n_cells} cells x {latent_dim} latent dim")
log(f"Class dist: {pd.Series(labels).value_counts().to_dict()}")
log(f"Region dist: {pd.Series(regions).value_counts().to_dict()}")

# Region encoding for edge features
region_order = {"NormalBrain": 0, "peri:GBM": 1, "core:GBM": 2}
region_idx = np.array([region_order[r] for r in regions])

# ---- 2. Build k-NN graph in latent space ----
log(f"Building k-NN graph (k={K})...")
nbrs = NearestNeighbors(n_neighbors=K+1, metric="euclidean", n_jobs=4)
nbrs.fit(latent)
distances, indices = nbrs.kneighbors(latent)

# Exclude self (first neighbor)
distances = distances[:, 1:]
indices = indices[:, 1:]

# ---- 3. Build edge list with attributes ----
log("Building edge attributes...")
rows = []
cols = []
edge_attrs = []

def transition_type(r1, r2):
    if r1 == r2: return 0
    pair = tuple(sorted([r1, r2]))
    if pair == ("core:GBM", "peri:GBM"): return 1
    if pair == ("NormalBrain", "peri:GBM"): return 2
    if pair == ("NormalBrain", "core:GBM"): return 3
    return 4

for i in range(n_cells):
    r_i = regions[i]
    p_i = patients[i]
    r_idx_i = region_idx[i]
    
    for k in range(K):
        j = indices[i, k]
        d = distances[i, k]
        
        rows.append(i)
        cols.append(j)
        
        r_j = regions[j]
        p_j = patients[j]
        r_idx_j = region_idx[j]
        
        same_patient = 1.0 if p_i == p_j else 0.0
        same_region = 1.0 if r_i == r_j else 0.0
        trans_type = transition_type(r_i, r_j)
        region_diff = abs(r_idx_i - r_idx_j)
        
        edge_attrs.append([d, same_patient, same_region, trans_type, region_diff])

edge_index = np.array([rows, cols], dtype=np.int32)  # (2, E)
edge_attr = np.array(edge_attrs, dtype=np.float32)   # (E, 5)

log(f"Graph: {n_cells} nodes, {edge_index.shape[1]} edges")
log(f"Edge attr shape: {edge_attr.shape}")

# ---- 4. Save ----
np.save(GRAPH_EDGE_INDEX, edge_index)
np.save(GRAPH_EDGE_ATTR, edge_attr)

meta = {
    "n_cells": int(n_cells),
    "n_edges": int(edge_index.shape[1]),
    "k": K,
    "latent_dim": latent_dim,
    "edge_attr_names": ["distance", "same_patient", "same_region", "transition_type", "region_diff"],
    "region_order": region_order,
    "transition_types": {
        "0": "same_region",
        "1": "core_periphery",
        "2": "periphery_healthy",
        "3": "core_healthy",
        "4": "other"
    }
}
with open(GRAPH_META, "w") as f:
    json.dump(meta, f, indent=2)

log(f"Saved graph to {OUT_DIR}")
log("DONE.")