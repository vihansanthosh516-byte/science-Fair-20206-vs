"""
01_load_and_filter.py
======================
Memory-conscious loader for the UCSC Cell Browser `multiomic-gbm` scRNA-seq dataset.

What this does:
- Streams the 10x MTX + metadata into a sparse AnnData (Float32 csr_matrix).
- Keeps ONLY the 3 classes needed for the science fair project:
    * Core       = tumor_core cells from GBM patients
    * Periphery  = tumor_peri cells from GBM patients
    * Healthy    = NormalBrain cells
- Drops doublets and any non-GBM patients to keep the comparison clean.
- Saves a filtered h5ad to disk (loaded by the next script in chunks if needed).
- Also writes a per-class cell-count summary so we know what we are feeding the model.

Memory plan:
  - O(N_cells x N_genes * 4 bytes sparse). Full matrix is 1.4 GB; after 3-class
    filtering we use ~900 MB of RSS plus a small AnnData object.
  - We never call .toarray() on the sparse matrix.

Run order:
    cd "/mnt/c/Users/vihan/20206 science fair"
    python 01_load_and_filter.py
"""
import os, gc, sys, time, json, resource
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.io import mmread

# Limit BLAS thread count so we don't blow up WSL memory on matrix ops.
os.environ["OMP_NUM_THREADS"]        = "2"
os.environ["OPENBLAS_NUM_THREADS"]   = "2"
os.environ["MKL_NUM_THREADS"]        = "2"
os.environ["NUMEXPR_NUM_THREADS"]    = "2"

DATA_DIR        = "/mnt/c/Users/vihan/multiomic-gbm/scrna"
OUT_DIR         = "/mnt/c/Users/vihan/20206 science fair/output"
os.makedirs(OUT_DIR, exist_ok=True)

OUT_H5AD        = os.path.join(OUT_DIR, "01_filtered_three_class.h5ad")
OUT_BC_COUNTS   = os.path.join(OUT_DIR, "01_class_counts.tsv")

# ---- 0. Helpers -------------------------------------------------------------
def mem_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0  # KB->MB

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}  RSS={mem_mb():.0f} MB", flush=True)

# ---- 1. Load barcodes/features as strings (cheap) ---------------------------
log("Reading barcodes and features...")
barcodes = pd.read_csv(os.path.join(DATA_DIR, "barcodes.tsv.gz"),
                       header=None, dtype=str)[0].to_numpy()
features = pd.read_csv(os.path.join(DATA_DIR, "features.tsv.gz"),
                       sep="\t", header=None, dtype=str)
if features.shape[1] == 1:
    gene_ids   = features[0].to_numpy()
    gene_names = gene_ids
else:
    gene_ids   = features[0].to_numpy()
    gene_names = features[1].to_numpy()
log(f"barcodes: {barcodes.shape}, gene_ids: {gene_ids.shape}")

# ---- 2. Read the MTX matrix (sparse, then transpose) -----------------------
log("Reading matrix.mtx.gz (this is the slowest step, expect ~2-3 minutes)...")
t0 = time.time()
X = mmread(os.path.join(DATA_DIR, "matrix.mtx.gz")).tocsr().astype(np.float32)
log(f"Matrix loaded in {time.time()-t0:.1f}s :: {X.dtype}, shape={X.shape}, nnz={X.nnz}")

# 10x MTX is genes x cells. AnnData expects cells x genes.
X = X.T.tocsr()
log(f"Transposed to cells x genes :: shape={X.shape}")

# ---- 3. Read metadata ------------------------------------------------------
log("Reading metadata (meta.tsv)...")
meta = pd.read_csv(os.path.join(DATA_DIR, "meta.tsv"),
                   sep="\t", dtype=str, low_memory=False)
meta = meta.rename(columns={meta.columns[0]: "barcode"})
meta = meta.set_index("barcode", verify_integrity=False)
log(f"meta rows: {meta.shape[0]}, columns: {meta.shape[1]}")

# Reorder both to a common barcode set without ever densifying.
common = np.intersect1d(meta.index.to_numpy(), barcodes)
log(f"intersecting barcodes: {len(common)}")

barcode_to_row = {b: i for i, b in enumerate(barcodes)}
row_idx = np.array([barcode_to_row[b] for b in common], dtype=np.intp)
X_sub = X[row_idx, :]
log(f"After subset to common barcodes: X_sub shape={X_sub.shape}, nnz={X_sub.nnz}")

meta = meta.loc[common]
del X, row_idx, barcode_to_row
gc.collect()

# ---- 4. Cell-class labeling (3-class for the science fair project) ---------
#   Core       = cells labeled core:GBM  (tissue_histology = "core:GBM")
#   Periphery  = cells labeled peri:GBM  (tissue_histology = "peri:GBM")
#   Healthy    = cells from NormalBrain tissue (tissue = "NormalBrain")
th_full = meta["tissue_histology"].to_numpy()
tis_full= meta["tissue"].to_numpy()

is_core   = (th_full == "core:GBM")
is_peri   = (th_full == "peri:GBM")
is_health = (tis_full == "NormalBrain")
log(f"per-class before filtering: core={is_core.sum()}, "
    f"peri={is_peri.sum()}, healthy={is_health.sum()}, "
    f"total_union={(is_core|is_peri|is_health).sum()}")

# ---- 5. Drop doublets / multiplets -----------------------------------------
dob = meta["doblet"].to_numpy()
keep_mask = (is_core | is_peri | is_health) & (dob == "Singlet")
log(f"after dropping non-Singlets: {keep_mask.sum()} cells remain")

classes_full = np.full(len(meta), "other", dtype=object)
classes_full[is_core]   = "Core"
classes_full[is_peri]   = "Periphery"
classes_full[is_health] = "Healthy"
meta = meta.copy()
meta["class"] = classes_full

# Subset sparse matrix and meta in lock-step
X_sub = X_sub[keep_mask]
meta  = meta.loc[keep_mask].copy()
del keep_mask, is_core, is_peri, is_health, classes_full, dob
gc.collect()
log(f"FINAL cells x genes: {X_sub.shape}")

# ---- 6. Build AnnData ------------------------------------------------------
adata = sc.AnnData(
    X=X_sub,
    obs=meta,
    var=pd.DataFrame(index=gene_names),
)
adata.var_names_make_unique()
adata.obs["class"] = pd.Categorical(adata.obs["class"].to_numpy())

print("Per-class counts (final):")
print(adata.obs["class"].value_counts().to_string())

# Persist
adata.write_h5ad(OUT_H5AD, compression="gzip")
log(f"Saved filtered AnnData -> {OUT_H5AD}")

adata.obs["class"].value_counts().to_csv(OUT_BC_COUNTS, sep="\t")
log(f"Saved class counts -> {OUT_BC_COUNTS}")
log("DONE.")
