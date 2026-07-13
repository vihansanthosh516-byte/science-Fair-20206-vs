"""
04_export_for_attention_model.py
=================================
Exports the cleaned sparse + dense tensors needed for training the
Attention-Gated Neural Network for the science fair project.

What this produces:
    output/nn_X.npy         -- dense Float32 (N_cells, n_genes) on the SUBSAMPLE
                              (= 15,000 x 2,500 = ~37.5 M floats = ~146 MB)
    output/nn_y.npy         -- int64 class labels (0=Core, 1=Periphery, 2=Healthy)
    output/nn_gene_names.tsv-- gene-name list matching columns of nn_X
    output/nn_class_names.tsv  -- label-to-class-name legend
    output/nn_train_test_split.tsv (idx train/test split reproducible)

We use:
  - HVG of 2,500 genes from the previous step (because raw 28.9k genes slow
    training, but also over-fit for 15k cells)
  - Light per-cell z-score (mean-subtract, divide by std) so gradient flow is
    stable, a standard skill in single-cell DL.
  - Stratified 80/20 train/test split with random_state=42.

Memory footprint: ~146 MB dense + ~150 MB sparse X_sub genator/loader warm +
~600 MB working memory at peak. Total ~1 GB. Well inside WSL budget.

Run:
    cd "/mnt/c/Users/vihan/20206 science fair"
    python src/04_export_for_attention_model.py
"""
import os, gc, time, resource
os.environ["OMP_NUM_THREADS"]        = "2"
os.environ["OPENBLAS_NUM_THREADS"]   = "2"
os.environ["MKL_NUM_THREADS"]        = "2"
os.environ["NUMEXPR_NUM_THREADS"]    = "2"

import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.model_selection import train_test_split

ROOT    = "/mnt/c/Users/vihan/20206 science fair"
OUT_DIR = os.path.join(ROOT, "output")
H5_SUB  = os.path.join(OUT_DIR, "02_adata_subsampled.h5ad")

NPY_X      = os.path.join(OUT_DIR, "nn_X.npy")
NPY_Y      = os.path.join(OUT_DIR, "nn_y.npy")
TSV_GENES  = os.path.join(OUT_DIR, "nn_gene_names.tsv")
TSV_CLASS  = os.path.join(OUT_DIR, "nn_class_names.tsv")
TSV_SPLIT  = os.path.join(OUT_DIR, "nn_train_test_split.tsv")

def mem_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}  RSS={mem_mb():.0f} MB", flush=True)

# ---- 1. Load subsample, extract HVG, export dense matrix -------------------
log("Loading subsample...")
adata_sub = sc.read_h5ad(H5_SUB)
log(f"loaded {adata_sub.shape}")

# Re-derive HVG with same parameters to be deterministic
log("derive HVG (n_top=2500, seurat, batch by class)...")
sc.pp.highly_variable_genes(adata_sub, n_top_genes=2500, flavor="seurat",
                            batch_key="class")
adata_sub.raw = adata_sub
hvg_mask = adata_sub.var["highly_variable"].to_numpy()
gene_names = adata_sub.var_names[hvg_mask].to_numpy()
log(f"selected {hvg_mask.sum()} HVG -- exporting dense Float32")

X = adata_sub.X[:, hvg_mask]
# make dense on the FILTERED columns only -- we keep it memory-friendly
# by subsampling once more if it is too large
# 15000 x 2500 = 37.5M floats = ~146 MB dense -- safe
X_dense = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
X_dense = X_dense.astype(np.float32, copy=False)
log(f"X_dense shape={X_dense.shape}, dtype={X_dense.dtype}")

# ---- 2. z-score per cell (row-wise): standard for DL on single-cell data --
log("z-scaling per cell (row-wise)...")
mu = X_dense.mean(axis=1, keepdims=True)
sigma = X_dense.std(axis=1, keepdims=True) + 1e-6
X_dense = (X_dense - mu) / sigma
X_dense = np.clip(X_dense, -10.0, 10.0).astype(np.float32)
log(f"after row-zscore: mean={X_dense.mean():.3f}, std={X_dense.std():.3f}")

# Save densified, z-scored feature matrix
log(f"saving X -> {NPY_X}")
np.save(NPY_X, X_dense)

# ---- 3. Class labels --------------------------------------------------------
class_names = ["Core", "Periphery", "Healthy"]
obs = adata_sub.obs
cat = obs["class"].astype("category")
# Map to integer class indices
y = np.array([class_names.index(c) for c in cat.to_numpy()], dtype=np.int64)
log(f"label distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
np.save(NPY_Y, y)

# Gne names + class legend
pd.DataFrame({"gene": gene_names}).to_csv(TSV_GENES, sep="\t", index=False)
pd.DataFrame({"class_index": [0, 1, 2],
              "class_name"  : class_names}).to_csv(TSV_CLASS, sep="\t", index=False)

# ---- 4. Stratified train/test split (80/20) -------------------------------
log("train/test split 80/20 stratified...")
idx = np.arange(len(y))
idx_train, idx_test = train_test_split(idx, test_size=0.20,
                                       random_state=42, stratify=y)
log(f"train={len(idx_train)} test={len(idx_test)}")
split = pd.DataFrame({
    "index":  np.concatenate([idx_train, idx_test]),
    "split":  (["train"] * len(idx_train)) + (["test"] * len(idx_test)),
})
split.to_csv(TSV_SPLIT, sep="\t", index=False)

log("DONE.")
log(f"X shape = {X_dense.shape} ; y shape = {y.shape} ; train/test = "
    f"{len(idx_train)}/{len(idx_test)}")
log("Files:")
for f in [NPY_X, NPY_Y, TSV_GENES, TSV_CLASS, TSV_SPLIT]:
    log("  " + f)
