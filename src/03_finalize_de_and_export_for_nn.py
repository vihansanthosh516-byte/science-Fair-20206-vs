"""
03_finalize_de_and_export_for_nn.py
====================================
Lightweight finalizer that:
  - Re-reads the 3 pairwise DE tables saved by script 02
  - Builds 02_de_top100_per_pair.tsv (top 100 per comparison combined)
  - Builds the per-comparison marker lists (for SHAP / ablation later)
The heavy work is done already; this script is for finishing up neatly.

Run:
    cd "/mnt/c/Users/vihan/20206 science fair"
    python src/03_finalize_de_and_export_for_nn.py
"""
import os, time, resource, gc
import numpy as np
import pandas as pd

ROOT    = "/mnt/c/Users/vihan/20206 science fair"
OUT_DIR = os.path.join(ROOT, "output")
TSV_DE_CP     = os.path.join(OUT_DIR, "02_de_Core_vs_Peri.tsv")
TSV_DE_CH     = os.path.join(OUT_DIR, "02_de_Core_vs_Healthy.tsv")
TSV_DE_PH     = os.path.join(OUT_DIR, "02_de_Peri_vs_Healthy.tsv")
TSV_DE_TOP    = os.path.join(OUT_DIR, "02_de_top100_per_pair.tsv")
TSV_DE_PAPER  = os.path.join(OUT_DIR, "02_paper_key_markers_in_top_de.tsv")

N_TOP = 100

def mem_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}  RSS={mem_mb():.0f} MB", flush=True)

# ---- 1. Read DE tables ------------------------------------------------------
log("loading DE tables...")
cp = pd.read_csv(TSV_DE_CP, sep="\t")
ch = pd.read_csv(TSV_DE_CH, sep="\t")
ph = pd.read_csv(TSV_DE_PH, sep="\t")
log(f"  cp {cp.shape}, ch {ch.shape}, ph {ph.shape}")

# Per comparison, sort by adjusted p and take top 100
def top_n(df, n, label):
    d = df.sort_values(["pvalue_adj","pvalue"], ascending=[True, True]).head(n).copy()
    d["comparison"] = label
    return d

top_all = pd.concat([
    top_n(cp, N_TOP, "Core_vs_Periphery"),
    top_n(ch, N_TOP, "Core_vs_Healthy"),
    top_n(ph, N_TOP, "Periphery_vs_Healthy"),
], ignore_index=True)
top_all.to_csv(TSV_DE_TOP, sep="\t", index=False)
log(f"wrote {TSV_DE_TOP} with {len(top_all)} rows")

# ---- 2. Check overlap with the paper's key Oligo_2_3_2 markers ------------
# Paper reports Oligo_2_3_2 has these upregulated markers:
#   GSN, TUBB2B, HLA-A, ALDOA, CLU, TIMP1, S100A1, SERPINA3, NGFR
# And downregulated: OPALIN, KIF19, ALDOC, PCDH9, DOCK9
PAPER_UP   = ["GSN","TUBB2B","HLA-A","ALDOA","CLU","TIMP1","S100A1","SERPINA3","NGFR"]
PAPER_DOWN = ["OPALIN","KIF19","ALDOC","PCDH9","DOCK9"]
PAPER_ALL  = set(PAPER_UP) | set(PAPER_DOWN)
log("Paper Oligo_2_3_2 marker set (up): " + ",".join(PAPER_UP))
log("Paper Oligo_2_3_2 marker set (down): " + ",".join(PAPER_DOWN))

in_paper = []
for df, label in [(ch, "Core_vs_Healthy"),
                  (ph, "Periphery_vs_Healthy"),
                  (cp, "Core_vs_Periphery")]:
    df = df.sort_values("pvalue_adj")
    for g in PAPER_ALL:
        row = df[df["gene"] == g]
        if len(row):
            r = row.iloc[0]
            in_paper.append({
                "comparison": label,
                "gene": g,
                "direction_in_paper": "UP_in_Oligo_2_3_2" if g in PAPER_UP else "DOWN_in_Oligo_2_3_2",
                "log2FC_here": r["log2FC"],
                "score_here" : r["score"],
                "pvalue_adj_here": r["pvalue_adj"],
            })

in_paper = pd.DataFrame(in_paper)
in_paper.to_csv(TSV_DE_PAPER, sep="\t", index=False)
log(f"wrote {TSV_DE_PAPER} with {len(in_paper)} rows (paper Oligo_2_3_2 markers checked in DE)")
log("DONE.")
