"""
11_cvae_extract_latent.py
==========================
Extract frozen cVAE latent embeddings for ALL 140k cells (not just the 15k subsample).
Uses the trained cVAE encoder on the full dataset in memory-efficient batches.
"""
import os, time, gc, resource
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"

import numpy as np
import pandas as pd
import anndata as ad
import torch
from torch.utils.data import DataLoader, TensorDataset
from scipy.sparse import issparse

ROOT = "/mnt/c/Users/vihan/20206 science fair"
FULL_H5AD = os.path.join(ROOT, "output", "01_filtered_three_class.h5ad")
CVAE_MODEL = os.path.join(ROOT, "output", "cgat", "cvae_model.pt")
OUT_DIR = os.path.join(ROOT, "output", "cgat")

LATENT_FULL_PATH = os.path.join(OUT_DIR, "cvae_latent_full.npy")
LABELS_FULL_PATH = os.path.join(OUT_DIR, "cvae_labels_full.npy")
PATIENT_FULL_PATH = os.path.join(OUT_DIR, "cvae_patient_full.npy")
REGION_FULL_PATH = os.path.join(OUT_DIR, "cvae_region_full.npy")

BATCH_SIZE = 2048
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def mem_mb(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}  RSS={mem_mb():.0f}MB", flush=True)

# ---- 1. Load trained cVAE model ----
log(f"Loading cVAE model from {CVAE_MODEL}...")
checkpoint = torch.load(CVAE_MODEL, map_location=DEVICE)
config = checkpoint["config"]
n_genes = config["n_genes"]
latent_dim = config["latent_dim"]
hidden_dims = config["hidden_dims"]
gene_names = config["gene_names"]
dropout = 0.1

class ContrastiveVAE(torch.nn.Module):
    def __init__(self, n_genes, latent_dim=32, hidden_dims=[256, 128], dropout=0.1):
        super().__init__()
        enc = []
        in_dim = n_genes
        for h in hidden_dims:
            enc += [torch.nn.Linear(in_dim, h), torch.nn.BatchNorm1d(h), torch.nn.ReLU(), torch.nn.Dropout(dropout)]
            in_dim = h
        self.encoder = torch.nn.Sequential(*enc)
        self.fc_mu = torch.nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = torch.nn.Linear(hidden_dims[-1], latent_dim)
        
        # Projection head (for contrastive loss) - needed for loading state dict
        self.proj = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, 64), torch.nn.ReLU(), torch.nn.Linear(64, 32)
        )
        
        dec = []
        in_dim = latent_dim
        for h in reversed(hidden_dims):
            dec += [torch.nn.Linear(in_dim, h), torch.nn.BatchNorm1d(h), torch.nn.ReLU(), torch.nn.Dropout(dropout)]
            in_dim = h
        self.decoder = torch.nn.Sequential(*dec)
        self.dec_out = torch.nn.Linear(hidden_dims[0], n_genes)
        
    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)
    
    def get_latent(self, x):
        mu, _ = self.encode(x)
        return mu

model = ContrastiveVAE(n_genes, latent_dim, hidden_dims, dropout).to(DEVICE)
model.load_state_dict(checkpoint["model_state"])
model.eval()
log(f"Model loaded: {n_genes} genes -> {latent_dim} latent dim")

# ---- 2. Load FULL dataset in sparse format ----
log(f"Loading full AnnData from {FULL_H5AD}...")
adata = ad.read_h5ad(FULL_H5AD)
n_cells, n_genes_full = adata.shape
log(f"Full data: {n_cells} cells x {n_genes_full} genes")

# The cVAE was trained on 2500 HVGs (subset of genes)
# We need to subset to the same genes used during training
# The gene names in config should match the first 2500 genes of nn_X.npy
# Let's load the gene names from the subsampled data to get the exact order
import numpy as np
sub_gene_names = pd.read_csv(os.path.join(ROOT, "output", "nn_gene_names.tsv"), sep="\t")["gene"].tolist()
log(f"Subsampled genes: {len(sub_gene_names)}")

# Map gene names to column indices in full data
full_gene_names = adata.var_names.tolist()
gene_to_idx = {g: i for i, g in enumerate(full_gene_names)}
sub_idx = [gene_to_idx[g] for g in sub_gene_names]
log(f"Subset to {len(sub_idx)} HVGs")

# Get metadata
patient = adata.obs["ID1"].values.astype(str)
region = adata.obs["tissue_histology"].values.astype(str)
class_labels = adata.obs["class"].values.astype(str)

# ---- 3. Extract latents in batches ----
log("Extracting latent embeddings in batches...")
X_sparse = adata.X
all_latents = []
all_labels = []
all_patients = []
all_regions = []

with torch.no_grad():
    for start in range(0, n_cells, BATCH_SIZE):
        end = min(start + BATCH_SIZE, n_cells)
        
        # Get sparse batch, convert to dense (only 2500 genes)
        X_batch = X_sparse[start:end, sub_idx]
        if issparse(X_batch):
            X_batch = X_batch.toarray()
        X_batch = X_batch.astype(np.float32)
        
        X_tensor = torch.from_numpy(X_batch).to(DEVICE)
        z = model.get_latent(X_tensor).cpu().numpy()
        
        all_latents.append(z)
        all_labels.append(class_labels[start:end])
        all_patients.append(patient[start:end])
        all_regions.append(region[start:end])
        
        if start % 50000 == 0:
            log(f"  processed {end}/{n_cells}")

latent_full = np.concatenate(all_latents, axis=0)
labels_full = np.concatenate(all_labels, axis=0)
patients_full = np.concatenate(all_patients, axis=0)
regions_full = np.concatenate(all_regions, axis=0)

log(f"Latent shape: {latent_full.shape}")
log(f"Labels: {np.unique(labels_full, return_counts=True)}")

# ---- 4. Save ----
np.save(LATENT_FULL_PATH, latent_full)
np.save(LABELS_FULL_PATH, labels_full)
np.save(PATIENT_FULL_PATH, patients_full)
np.save(REGION_FULL_PATH, regions_full)

log(f"Saved:")
log(f"  {LATENT_FULL_PATH}  ({latent_full.shape})")
log(f"  {LABELS_FULL_PATH}  ({labels_full.shape})")
log(f"  {PATIENT_FULL_PATH}  ({patients_full.shape})")
log(f"  {REGION_FULL_PATH}  ({regions_full.shape})")
log("DONE.")