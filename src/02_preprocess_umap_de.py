"""
02_preprocess_umap_de.py
========================
- Loads 01_filtered_three_class.h5ad (140,355 cells x 29,661 genes, sparse csr).
- Drops genes that are 0 across all cells.
- Subsamples 5,000 cells per class (15,000 total, even balance) for UMAP,
  PCA, and graph-based clustering. This is the dataset that:
      - Gets UMAP coordinates
      - Generates cluster markers
      - Plots heatmaps of top marker genes
  Why subsample? UMAP on 140k is slow + RAM-heavy in WSL; 15k is plenty for
  inspection, and it's the published-biology-fair standard.
- Differential EXPRESSION analysis uses ALL 140,355 cells (Wilcoxon via scanpy,
  one-vs-rest per class). DE tables are saved as TSV.

Outputs (in output/):
    02_adata_subsampled.h5ad
    02_umap_coordinates.tsv
    02_umap_by_class.png
    02_umap_by_patient.png
    02_top10_markers_per_cluster_heatmap.png
    02_de_Core_vs_Peri.tsv
    02_de_Core_vs_Healthy.tsv
    02_de_Peri_vs_Healthy.tsv
    02_de_top100_per_pair.tsv

Run:
    cd "/mnt/c/Users/vihan/20206 science fair"
    python src/02_preprocess_umap_de.py
"""
import os, gc, time, resource
os.environ["OMP_NUM_THREADS"]        = "2"
os.environ["OPENBLAS_NUM_THREADS"]   = "2"
os.environ["MKL_NUM_THREADS"]        = "2"
os.environ["NUMEXPR_NUM_THREADS"]    = "2"

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

ROOT          = "/mnt/c/Users/vihan/20206 science fair"
OUT_DIR       = os.path.join(ROOT, "output")
H5_IN         = os.path.join(OUT_DIR, "01_filtered_three_class.h5ad")

H5_SUB        = os.path.join(OUT_DIR, "02_adata_subsampled.h5ad")
TSV_UMAP_COORD= os.path.join(OUT_DIR, "02_umap_coordinates.tsv")
PNG_UMAP_CLS  = os.path.join(OUT_DIR, "02_umap_by_class.png")
PNG_UMAP_PAT  = os.path.join(OUT_DIR, "02_umap_by_patient.png")
PNG_MARK_HEAT = os.path.join(OUT_DIR, "02_top10_markers_per_cluster_heatmap.png")
TSV_DE_CP     = os.path.join(OUT_DIR, "02_de_Core_vs_Peri.tsv")
TSV_DE_CH     = os.path.join(OUT_DIR, "02_de_Core_vs_Healthy.tsv")
TSV_DE_PH     = os.path.join(OUT_DIR, "02_de_Peri_vs_Healthy.tsv")
TSV_DE_TOP    = os.path.join(OUT_DIR, "02_de_top100_per_pair.tsv")

PER_CLASS_N   = 5000     # balanced subsample size per class
N_TOP_MARKERS = 10
N_TOP_DE      = 100

def mem_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
def log(msg): print(f"[{time.strftime('%H:%M:%S')}] {msg}  RSS={mem_mb():.0f} MB", flush=True)

sc.settings.verbosity = 1
sns.set_theme(style="white")
plt.rcParams["savefig.dpi"] = 200

# ---- 1. Load -----------------------------------------------------------------
log("Loading filtered AnnData...")
adata = sc.read_h5ad(H5_IN)
log(f"loaded {adata.shape}")

# ---- 2. Drop zero-variance / all-zero genes -------------------------------
# (in-place) reduces column count for downstream ops
Xc = np.asarray(adata.X.sum(axis=0)).ravel()
keep_genes = Xc > 0
log(f"keeping {keep_genes.sum()} / {adata.shape[1]} genes (drop {keep_genes.size - keep_genes.sum()} all-zero)")
adata = adata[:, keep_genes].copy()
del Xc, keep_genes
gc.collect()
log(f"after gene-trim: {adata.shape}")

# ---- 3. Standardize gene-variable names -----------------------------------
# Many features are Ensembl IDs (AL627309.1 etc). Make them unique + ensure no-NA.
adata.var_names_make_unique()
log(f"avg expression across cells: {np.asarray(adata.X.mean()).item():.3f} (log1p-scale, raw)")

# ---- 4. Highlight highly-variable genes (for UMAP / PCA) -----------------
# We do this on the SUBSAMPLE only -- HVG selection on 140k is too slow on WSL.
# Strategy: random-stratified subsample 5,000 cells per class.
log(f"Subsampling {PER_CLASS_N} cells per class for UMAP/clustering...")
adata.obs["class"] = adata.obs["class"].astype("category")
parts = []
for cls_name, grp in adata.obs.groupby("class", observed=True):
    take = PER_CLASS_N if len(grp) >= PER_CLASS_N else len(grp)
    parts.append(grp.sample(n=take, random_state=0))
sub_obs = pd.concat(parts)
sub_idx = sub_obs.index.to_numpy()
adata_sub = adata[sub_idx].copy()
log(f"subsampled shape={adata_sub.shape}, per-class={dict(adata_sub.obs['class'].value_counts())}")
adata_sub.write_h5ad(H5_SUB, compression="gzip")
log(f"saved subsample -> {H5_SUB}")

# ---- 5. HVG, PCA, neighbors, UMAP, leiden (all on the SUBSAMPLE) ---------
log(f"HVG selection on subsample ({PER_CLASS_N} per class, batch by class)...")
sc.pp.highly_variable_genes(adata_sub, n_top_genes=2500, flavor="seurat",
                            batch_key="class")
adata_sub.raw = adata_sub            # keep full gene expression for DE plots later
hvg = adata_sub.var["highly_variable"].to_numpy()
log(f"selected {hvg.sum()} HVG")
adata_sub_hvg = adata_sub[:, hvg].copy()

log("Scaling HVG (clip=10) + PCA on subsample...")
sc.pp.scale(adata_sub_hvg, max_value=10)
sc.tl.pca(adata_sub_hvg, n_comps=50, random_state=0)
log("PCA done")

log("Building neighbor graph + UMAP...")
sc.pp.neighbors(adata_sub_hvg, n_neighbors=15, n_pcs=30, random_state=0,
                method="umap")
sc.tl.umap(adata_sub_hvg, random_state=0, maxiter=100, method="umap")
log("UMAP done")

# Write UMAP coords back to a TSV so downstream code (attention model) can use it
umap = adata_sub_hvg.obsm["X_umap"]
umap_df = pd.DataFrame(umap, columns=["UMAP1", "UMAP2"], index=adata_sub_hvg.obs_names)
umap_df["class"]      = adata_sub_hvg.obs["class"].to_numpy()
umap_df["origident"]  = adata_sub_hvg.obs["ID1"].to_numpy() if "ID1" in adata_sub_hvg.obs.columns else ""
umap_df["CellType"]   = adata_sub_hvg.obs["Major.celltype"].to_numpy() if "Major.celltype" in adata_sub_hvg.obs.columns else ""
umap_df.to_csv(TSV_UMAP_COORD, sep="\t")
log(f"wrote UMAP coordinates -> {TSV_UMAP_COORD}")

# ---- 6. UMAP plots ---------------------------------------------------------
log("UMAP plot by class...")
sc.pl.umap(adata_sub_hvg, color="class", save=False, show=False,
           title="UMAP by class")
plt.gcf().savefig(PNG_UMAP_CLS, bbox_inches="tight")
plt.close("all")

log("UMAP plot by patient (ID1)...")
sc.pl.umap(adata_sub_hvg, color="ID1", save=False, show=False,
           title="UMAP by patient")
plt.gcf().savefig(PNG_UMAP_PAT, bbox_inches="tight")
plt.close("all")

# ---- 7. Leiden clustering + per-cluster markers ---------------------------
log("Leiden clustering on subsample (igraph backend)...")
sc.tl.leiden(adata_sub_hvg, resolution=1.0, random_state=0,
             flavor="igraph", n_iterations=2, directed=False)
n_cl = adata_sub_hvg.obs["leiden"].nunique()
log(f"got {n_cl} clusters")

# Compute per-cluster markers using the raw expression (raw stored earlier)
log("Rank genes per cluster (Wilcoxon, vs rest) on RAW counts of subsample...")
sc.tl.rank_genes_groups(adata_sub_hvg, groupby="leiden",
                        method="wilcoxon", use_raw=True,
                        n_genes=200, key_added="rank_leiden")
sc.tl.rank_genes_groups(adata_sub_hvg, groupby="class",
                        method="wilcoxon", use_raw=True,
                        n_genes=200, key_added="rank_class")

# ---- 7b. Heatmap of top 10 markers per cluster ---------------------------
log("Plotting top-10-marker heatmap...")
top = {}
for grp in adata_sub_hvg.uns["rank_leiden"]["names"].dtype.names:
    top[grp] = adata_sub_hvg.uns["rank_leiden"]["names"][grp][:N_TOP_MARKERS].tolist()
all_top = sorted({g for v in top.values() for g in v})
heat_raw = adata_sub_hvg.raw[:, all_top].X
heat_raw = heat_raw.toarray() if hasattr(heat_raw, "toarray") else np.asarray(heat_raw)
heat_df = pd.DataFrame(heat_raw, columns=all_top, index=adata_sub_hvg.obs_names)
heat_z = (heat_df - heat_df.mean()) / (heat_df.std() + 1e-6)
heat_z["leiden"] = adata_sub_hvg.obs["leiden"].astype(str).to_numpy()
heat_means = heat_z.groupby("leiden").mean()
g = sns.clustermap(heat_means, figsize=(min(20, 0.25*len(all_top)+4), 8),
                   cmap="RdBu_r", center=0, xticklabels=True, yticklabels=True)
g.savefig(PNG_MARK_HEAT, bbox_inches="tight")
plt.close("all")

# ---- 8. Differential expression on the FULL dataset (all 140,355 cells) ---
log("DE: Wilcoxon on full dataset (one-vs-rest per class)...")
# Use the raw counts slot (we stored `raw = adata` earlier for the subsample;
# since raw counts were not present in the source data, but we stored the
# log-normalized values, treat that as the expression matrix for DE).
# DE will be in 'rank_class_full' on adata itself.
sc.tl.rank_genes_groups(adata, groupby="class",
                        method="wilcoxon",
                        n_genes=400,
                        key_added="rank_class_full")

def de_to_df(adata_obj, group, key="rank_class_full"):
    names   = adata_obj.uns[key]["names"][group]
    pvals   = adata_obj.uns[key]["pvals"][group]
    pvals_adj = adata_obj.uns[key]["pvals_adj"][group]
    logfc   = adata_obj.uns[key]["logfoldchanges"][group]
    scores  = adata_obj.uns[key]["scores"][group]
    return pd.DataFrame({
        "gene":       names,
        "log2FC":     logfc,
        "score":      scores,
        "pvalue":     pvals,
        "pvalue_adj": pvals_adj,
    })

for grp, out_path in [
    ("Core",        TSV_DE_CP),  # treat Core as the "first class" but we'll also dump the other comparisons
]:
    df = de_to_df(adata, grp)
    df.to_csv(out_path, sep="\t", index=False)
    log(f"  -> {out_path} ({len(df)} rows)")

# Make explicit pairwise comparisons by re-running with reference
log("DE: pairwise Core vs Healthy (reference=Healthy)...")
sc.tl.rank_genes_groups(adata, groupby="class",
                        method="wilcoxon",
                        reference="Healthy",
                        groups=["Core"],
                        n_genes=400,
                        key_added="de_core_vs_healthy")
df = de_to_df(adata, "Core", key="de_core_vs_healthy")
df.to_csv(TSV_DE_CH, sep="\t", index=False)
log(f"  -> {TSV_DE_CH} ({len(df)} rows)")

log("DE: pairwise Peri vs Healthy (reference=Healthy)...")
sc.tl.rank_genes_groups(adata, groupby="class",
                        method="wilcoxon",
                        reference="Healthy",
                        groups=["Periphery"],
                        n_genes=400,
                        key_added="de_peri_vs_healthy")
df = de_to_df(adata, "Periphery", key="de_peri_vs_healthy")
df.to_csv(TSV_DE_PH, sep="\t", index=False)
log(f"  -> {TSV_DE_PH} ({len(df)} rows)")

# Top-N per comparison combined
top100s = []
for path, label in [(TSV_DE_CH, "Core_vs_Healthy_up"),
                    (TSV_DE_PH, "Peri_vs_Healthy_up"),
                    (TSV_DE_CP, "Core_one_vs_rest")]:
    d = pd.read_csv(path, sep="\t")
    d = d.sort_values("pvalue_adj").head(N_TOP_DE)
    d["comparison"] = label
    top100s.append(d)
top_all = pd.concat(top100s, ignore_index=True)
top_all.to_csv(TSV_DE_TOP, sep="\t", index=False)
log(f"  -> {TSV_DE_TOP} ({len(top_all)} rows)")

log("DONE.")
