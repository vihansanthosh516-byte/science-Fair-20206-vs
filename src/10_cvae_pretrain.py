"""
10_cvae_pretrain.py
====================
Contrastive-VAE (cVAE) using pre-processed subsampled data (15k cells x 2500 HVGs).
Uses the already-computed nn_X.npy which is row-z-scored and memory-efficient.
"""
import os, sys, time, gc, resource, json
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

ROOT = "/mnt/c/Users/vihan/20206 science fair"
OUT_DIR = os.path.join(ROOT, "output", "cgat")
os.makedirs(OUT_DIR, exist_ok=True)

MODEL_PATH = os.path.join(OUT_DIR, "cvae_model.pt")
METRICS_PATH = os.path.join(OUT_DIR, "cvae_metrics.json")
LATENT_PATH = os.path.join(OUT_DIR, "cvae_latent.npy")

# Load pre-processed data
NPY_X = os.path.join(ROOT, "output", "nn_X.npy")           # (15000, 2500) float32
NPY_Y = os.path.join(ROOT, "output", "nn_y.npy")           # (15000,) int64
TSV_GENE = os.path.join(ROOT, "output", "nn_gene_names.tsv")
TSV_CLS = os.path.join(ROOT, "output", "nn_class_names.tsv")
TSV_SPLIT = os.path.join(ROOT, "output", "nn_train_test_split.tsv")

# Hyperparameters
LATENT_DIM = 32
HIDDEN_DIMS = [256, 128]
BATCH_SIZE = 512
EPOCHS = 60
LR = 2e-3
WEIGHT_DECAY = 1e-5
BETA_KL = 1.0
LAMBDA_CONTRAST = 0.5
TEMPERATURE = 0.1
DROPOUT_RATE = 0.1
AUG_DROPOUT = 0.1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def mem_mb(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}  RSS={mem_mb():.0f}MB", flush=True)

# ---- 1. Load pre-processed data ----
log("Loading pre-processed data...")
X = np.load(NPY_X).astype(np.float32)   # (15000, 2500) - already row-z-scored
y = np.load(NPY_Y)                       # (15000,)
gene_names = pd.read_csv(TSV_GENE, sep="\t")["gene"].tolist()
class_names = pd.read_csv(TSV_CLS, sep="\t")["class_name"].tolist()
split_df = pd.read_csv(TSV_SPLIT, sep="\t")
split_map = dict(zip(split_df["index"], split_df["split"]))

n_cells, n_genes = X.shape
log(f"Data: {n_cells} cells x {n_genes} genes")

# Build positive pair indices: same class + same patient
# We need patient metadata - load from subsampled h5ad
import anndata as ad
adata_sub = ad.read_h5ad(os.path.join(ROOT, "output", "02_adata_subsampled.h5ad"))
patient_ids = adata_sub.obs["ID1"].values.astype(str)
regions = adata_sub.obs["class"].values.astype(str)  # Core, Periphery, Healthy

# Build (patient, region) -> indices mapping
pair_to_idx = {}
for i in range(n_cells):
    key = (patient_ids[i], regions[i])
    pair_to_idx.setdefault(key, []).append(i)

valid_keys = [k for k, v in pair_to_idx.items() if len(v) >= 2]
log(f"Valid (patient,region) groups for bio positives: {len(valid_keys)}")

# Convert to tensors
X_tensor = torch.from_numpy(X)
y_tensor = torch.from_numpy(y).long()

# Train/test split
train_mask = np.array([split_map[i] == "train" for i in range(n_cells)])
test_mask = ~train_mask
log(f"Train: {train_mask.sum()}, Test: {test_mask.sum()}")

# ---- 2. Model ----
class ContrastiveVAE(nn.Module):
    def __init__(self, n_genes, latent_dim=32, hidden_dims=[256, 128], dropout=0.1):
        super().__init__()
        self.n_genes = n_genes
        self.latent_dim = latent_dim
        
        # Encoder
        enc = []
        in_dim = n_genes
        for h in hidden_dims:
            enc += [nn.Linear(in_dim, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        self.encoder = nn.Sequential(*enc)
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)
        
        # Contrastive projection
        self.proj = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.ReLU(), nn.Linear(64, 32)
        )
        
        # Decoder (MSE for continuous data)
        dec = []
        in_dim = latent_dim
        for h in reversed(hidden_dims):
            dec += [nn.Linear(in_dim, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        self.decoder = nn.Sequential(*dec)
        self.dec_out = nn.Linear(hidden_dims[0], n_genes)
        
    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.dec_out(self.decoder(z))
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        z_proj = F.normalize(self.proj(z), dim=-1)
        return recon, mu, logvar, z_proj
    
    def get_latent(self, x):
        mu, _ = self.encode(x)
        return mu

# ---- 3. Losses ----
def mse_loss(recon, x):
    return F.mse_loss(recon, x, reduction="mean")

def kl_div(mu, logvar):
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

def info_nce(z_i, z_j, temp=0.1):
    B = z_i.size(0)
    z_i = F.normalize(z_i, dim=-1)
    z_j = F.normalize(z_j, dim=-1)
    z = torch.cat([z_i, z_j], dim=0)  # (2B, D)
    sim = torch.mm(z, z.t()) / temp   # (2B, 2B)
    mask = torch.eye(2*B, dtype=torch.bool, device=z.device)
    sim.masked_fill_(mask, -9e15)
    labels = torch.cat([torch.arange(B, device=z.device)+B, torch.arange(B, device=z.device)])
    return F.cross_entropy(sim, labels)

# ---- 4. DataLoader with positive pair sampling ----
# Pre-compute positive indices for training set
train_indices = np.where(train_mask)[0]
test_indices = np.where(test_mask)[0]

# Build positive pair mapping for train set
train_patient = patient_ids[train_mask]
train_region = regions[train_mask]
train_pair_to_local = {}
for local_idx, global_idx in enumerate(train_indices):
    key = (train_patient[local_idx], train_region[local_idx])
    train_pair_to_local.setdefault(key, []).append(local_idx)

# Dataset returns (x, bio_positive_x, tech_positive_x)
class PairedDataset(torch.utils.data.Dataset):
    def __init__(self, X, indices, pair_to_local):
        self.X = X
        self.indices = indices
        self.pair_to_local = pair_to_local
        self.global_to_local = {g: l for l, g in enumerate(indices)}
        
    def __len__(self): return len(self.indices)
    
    def __getitem__(self, local_idx):
        x = self.X[local_idx]
        global_idx = self.indices[local_idx]
        key = (patient_ids[global_idx], regions[global_idx])
        candidates = self.pair_to_local[key]
        if len(candidates) > 1:
            bio_local = np.random.choice([c for c in candidates if c != local_idx])
        else:
            bio_local = local_idx
        x_bio = self.X[bio_local]
        # Technical: dropout
        mask = torch.rand_like(x) > AUG_DROPOUT
        x_tech = x * mask
        return x, x_bio, x_tech

train_dataset = PairedDataset(X_tensor, train_indices, train_pair_to_local)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

# Test loader (no pairs needed, just for eval)
test_dataset = TensorDataset(X_tensor[test_mask], y_tensor[test_mask])
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ---- 5. Training ----
log(f"Device: {DEVICE}")
model = ContrastiveVAE(n_genes, LATENT_DIM, HIDDEN_DIMS, DROPOUT_RATE).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

log(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

history = {"epoch": [], "loss": [], "mse": [], "kl": [], "contrast": []}

for epoch in range(1, EPOCHS + 1):
    model.train()
    metrics = {"total": 0, "mse": 0, "kl": 0, "contrast": 0}
    t0 = time.time()
    
    for x, x_bio, x_tech in train_loader:
        x = x.to(DEVICE)
        x_bio = x_bio.to(DEVICE)
        x_tech = x_tech.to(DEVICE)
        
        # Original
        recon, mu, logvar, z_proj = model(x)
        loss_mse = mse_loss(recon, x)
        loss_kl = kl_div(mu, logvar)
        
        # Technical contrast
        _, _, _, z_tech = model(x_tech)
        loss_cont_tech = info_nce(z_proj, z_tech, TEMPERATURE)
        
        # Biological contrast
        _, _, _, z_bio = model(x_bio)
        loss_cont_bio = info_nce(z_proj, z_bio, TEMPERATURE)
        
        loss_contrast = (loss_cont_tech + loss_cont_bio) * 0.5
        loss = loss_mse + BETA_KL * loss_kl + LAMBDA_CONTRAST * loss_contrast
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        metrics["total"] += loss.item()
        metrics["mse"] += loss_mse.item()
        metrics["kl"] += loss_kl.item()
        metrics["contrast"] += loss_contrast.item()
    
    scheduler.step()
    
    for k in metrics: metrics[k] /= len(train_loader)
    history["epoch"].append(epoch)
    history["loss"].append(metrics["total"])
    history["mse"].append(metrics["mse"])
    history["kl"].append(metrics["kl"])
    history["contrast"].append(metrics["contrast"])
    
    log(f"Epoch {epoch:3d}/{EPOCHS}  Total={metrics['total']:.4f}  MSE={metrics['mse']:.4f}  KL={metrics['kl']:.4f}  Contrast={metrics['contrast']:.4f}  Time={time.time()-t0:.1f}s")
    
    # Save checkpoint
    if epoch % 10 == 0 or epoch == EPOCHS:
        torch.save({
            "model_state": model.state_dict(),
            "config": {"n_genes": n_genes, "latent_dim": LATENT_DIM, 
                       "hidden_dims": HIDDEN_DIMS, "gene_names": gene_names},
            "history": history
        }, MODEL_PATH)
        with open(METRICS_PATH, "w") as f:
            json.dump(history, f, indent=2)

# ---- 6. Extract latent for all 15k cells ----
log("Extracting latent embeddings...")
model.eval()
with torch.no_grad():
    latent = model.get_latent(X_tensor.to(DEVICE)).cpu().numpy()
np.save(LATENT_PATH, latent)
log(f"Saved latent: {latent.shape} -> {LATENT_PATH}")

# Save test set latent for downstream evaluation
np.save(os.path.join(OUT_DIR, "cvae_latent_test.npy"), latent[test_mask])
np.save(os.path.join(OUT_DIR, "cvae_labels_test.npy"), y[test_mask])
log("DONE.")