"""
15_cgat_build_graph.py
======================
Build k-NN graph on FULL 140k cVAE latent space with edge features.
Memory-efficient: uses sklearn NearestNeighbors with n_jobs.
"""
import os, time, resource, gc, json
import numpy as np
from sklearn.neighbors import NearestNeighbors

ROOT = "/mnt/c/Users/vihan/20206 science fair"
OUT_DIR = os.path.join(ROOT, "output", "cgat")

LATENT_PATH = os.path.join(OUT_DIR, "cvae_latent_full.npy")
LABELS_PATH = os.path.join(OUT_DIR, "cvae_labels_full.npy")
PATIENT_PATH = os.path.join(OUT_DIR, "cvae_patient_full.npy")
REGION_PATH = os.path.join(OUT_DIR, "cvae_region_full.npy")

EDGE_INDEX_PATH = os.path.join(OUT_DIR, "gat_edge_index_full.npy")
EDGE_ATTR_PATH = os.path.join(OUT_DIR, "gat_edge_attr_full.npy")
META_PATH = os.path.join(OUT_DIR, "gat_meta_full.json")

K = 15
DEVICE = "cpu"

def mem_mb(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}  RSS={mem_mb():.0f}MB", flush=True)

# ---- 1. Load ----
log("Loading full latent embeddings...")
latent = np.load(LATENT_PATH).astype(np.float32)  # (140355, 32)
labels = np.load(LABELS_PATH, allow_pickle=True)   # (140355,)
patients = np.load(PATIENT_PATH, allow_pickle=True)
regions = np.load(REGION_PATH, allow_pickle=True)

n_cells, d = latent.shape
log(f"Loaded: {n_cells} cells x {d} dim")

# Region encoding for edge features
region_order = {"NormalBrain": 0, "peri:GBM": 1, "core:GBM": 2}
region_idx = np.array([region_order[r] for r in regions], dtype=np.int8)

# Class encoding
class_order = {"Healthy": 0, "Periphery": 1, "Core": 2}
class_idx = np.array([class_order[l] for l in labels], dtype=np.int8)

# ---- 2. k-NN Graph ----
log(f"Building k-NN graph (k={K})...")
nbrs = NearestNeighbors(n_neighbors=K+1, metric="euclidean", n_jobs=4)
nbrs.fit(latent)
distances, indices = nbrs.kneighbors(latent)

# Exclude self
distances = distances[:, 1:].astype(np.float32)
indices = indices[:, 1:].astype(np.int32)

log(f"Graph: {n_cells} nodes, {n_cells * K} directed edges")

# ---- 3. Edge attributes ----
def transition_type(r1, r2):
    if r1 == r2: return 0
    pair = tuple(sorted([r1, r2]))
    if pair == ("core:GBM", "peri:GBM"): return 1
    if pair == ("NormalBrain", "peri:GBM"): return 2
    if pair == ("NormalBrain", "core:GBM"): return 3
    return 4

log("Computing edge attributes...")
rows = np.repeat(np.arange(n_cells), K)
cols = indices.flatten()

# Vectorized edge attributes
r_i = regions[rows]
r_j = regions[cols]
p_i = patients[rows]
p_j = patients[cols]
ri_idx = region_idx[rows]
rj_idx = region_idx[cols]
di_idx = class_idx[rows]
dj_idx = class_idx[cols]

same_patient = (p_i == p_j).astype(np.float32)
same_region = (r_i == r_j).astype(np.float32)
region_diff = np.abs(ri_idx - rj_idx).astype(np.float32)
class_diff = np.abs(di_idx - dj_idx).astype(np.float32)

trans_type = np.array([transition_type(a, b) for a, b in zip(r_i, r_j)], dtype=np.int8)

edge_attr = np.stack([
    distances.flatten(),      # 0: latent distance
    same_patient,             # 1: same patient
    same_region,              # 2: same region
    trans_type.astype(np.float32),  # 3: transition type
    region_diff,              # 4: region index diff
    class_diff                # 5: class index diff
], axis=1).astype(np.float32)

edge_index = np.stack([rows, cols], axis=0).astype(np.int32)

log(f"Edge index: {edge_index.shape}, Edge attr: {edge_attr.shape}")

# ---- 4. Save ----
np.save(EDGE_INDEX_PATH, edge_index)
np.save(EDGE_ATTR_PATH, edge_attr)

meta = {
    "n_cells": int(n_cells),
    "n_edges": int(edge_index.shape[1]),
    "k": K,
    "latent_dim": d,
    "edge_attr_names": ["distance", "same_patient", "same_region", "transition_type", "region_diff", "class_diff"],
    "region_order": region_order,
    "class_order": class_order,
    "transition_types": {0: "same", 1: "core_peri", 2: "peri_healthy", 3: "core_healthy", 4: "other"}
}
with open(META_PATH, "w") as f:
    json.dump(meta, f, indent=2)

log(f"Saved graph to {OUT_DIR}")
log("DONE.")

if __name__ == "__main__":
    import time
    import sys
    sys.exit(0)