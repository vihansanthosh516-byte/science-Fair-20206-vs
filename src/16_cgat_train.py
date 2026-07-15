"""
16_cgat_train.py
================
Train Edge-Aware GAT on FULL 140k-cell graph using full-batch training.
No neighbor sampling needed - full graph fits in memory.
"""
import os, time, resource, gc, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv
from sklearn.metrics import (accuracy_score, f1_score, precision_recall_fscore_support,
                             confusion_matrix, classification_report, roc_auc_score)
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = "/mnt/c/Users/vihan/20206 science fair"
OUT_DIR = os.path.join(ROOT, "output", "cgat")

EDGE_INDEX_PATH = os.path.join(OUT_DIR, "gat_edge_index_full.npy")
EDGE_ATTR_PATH = os.path.join(OUT_DIR, "gat_edge_attr_full.npy")
LATENT_PATH = os.path.join(OUT_DIR, "cvae_latent_full.npy")
LABELS_PATH = os.path.join(OUT_DIR, "cvae_labels_full.npy")

MODEL_PATH = os.path.join(OUT_DIR, "gat_full_model.pt")
METRICS_PATH = os.path.join(OUT_DIR, "gat_full_metrics.json")
PRED_PATH = os.path.join(OUT_DIR, "gat_full_predictions.tsv")
PNG_CM = os.path.join(OUT_DIR, "gat_full_confusion.png")
PNG_LOSS = os.path.join(OUT_DIR, "gat_full_training_loss.png")

# Hyperparameters
HIDDEN_DIM = 64
HEADS = 8
N_LAYERS = 2
DROPOUT = 0.3
LR = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 50
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def mem_mb(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}  RSS={mem_mb():.0f}MB  DEV={DEVICE}", flush=True)

# ---- 1. Load Data ----
log("Loading full graph...")
edge_index = np.load(EDGE_INDEX_PATH).astype(np.int64)
edge_attr = np.load(EDGE_ATTR_PATH).astype(np.float32)
latent = np.load(LATENT_PATH).astype(np.float32)
labels = np.load(LABELS_PATH, allow_pickle=True)

class_names = ["Healthy", "Periphery", "Core"]
class_to_idx = {c: i for i, c in enumerate(class_names)}
y = np.array([class_to_idx[c] for c in labels], dtype=np.int64)

n_cells, in_dim = latent.shape
log(f"Data: {n_cells} cells, {in_dim} features, {edge_index.shape[1]} edges")
log(f"Class dist: {np.bincount(y)}")

# Convert to tensors
x = torch.from_numpy(latent).float()
edge_index_t = torch.from_numpy(edge_index).long()
edge_attr_t = torch.from_numpy(edge_attr).float()
y_t = torch.from_numpy(y).long()

# Train/test split (stratified 80/20)
train_idx, test_idx = train_test_split(np.arange(n_cells), test_size=0.2, random_state=42, stratify=y)
train_mask = torch.zeros(n_cells, dtype=torch.bool)
train_mask[train_idx] = True
test_mask = torch.zeros(n_cells, dtype=torch.bool)
test_mask[test_idx] = True

log(f"Train: {train_mask.sum().item()}, Test: {test_mask.sum().item()}")

data = Data(x=x, edge_index=edge_index_t, edge_attr=edge_attr_t, y=y_t)
data = data.to(DEVICE)

# ---- 2. Edge-Aware GAT ----
class EdgeAwareGAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, heads=8, n_layers=2, dropout=0.3, edge_dim=6):
        super().__init__()
        self.n_layers = n_layers
        self.dropout = dropout
        
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        # Layer 1
        self.convs.append(GATv2Conv(in_dim, hidden_dim, heads=heads, dropout=dropout,
                                    edge_dim=edge_dim, add_self_loops=True, concat=True))
        self.bns.append(nn.BatchNorm1d(hidden_dim * heads))
        
        # Hidden layers
        for _ in range(n_layers - 2):
            self.convs.append(GATv2Conv(hidden_dim * heads, hidden_dim, heads=heads, dropout=dropout,
                                        edge_dim=edge_dim, add_self_loops=True, concat=True))
            self.bns.append(nn.BatchNorm1d(hidden_dim * heads))
        
        # Output layer (single head, no concat)
        self.convs.append(GATv2Conv(hidden_dim * heads, out_dim, heads=1, dropout=dropout,
                                    edge_dim=edge_dim, add_self_loops=True, concat=False))
    
    def forward(self, x, edge_index, edge_attr):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_attr=edge_attr)
            if i < len(self.convs) - 1:
                x = F.elu(x)
                x = self.bns[i](x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x

model = EdgeAwareGAT(
    in_dim=in_dim,
    hidden_dim=HIDDEN_DIM,
    out_dim=len(class_names),
    heads=HEADS,
    n_layers=N_LAYERS,
    dropout=DROPOUT,
    edge_dim=edge_attr.shape[1]
).to(DEVICE)

log(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
criterion = nn.CrossEntropyLoss()

# ---- 3. Training (Full-Batch) ----
log("Training (full-batch)...")
train_losses = []
test_losses = []
test_accs = []
best_acc = 0

for epoch in range(1, EPOCHS + 1):
    model.train()
    t0 = time.time()
    
    optimizer.zero_grad()
    out = model(data.x, data.edge_index, data.edge_attr)
    loss = criterion(out[train_mask], data.y[train_mask])
    loss.backward()
    optimizer.step()
    scheduler.step()
    train_loss = loss.item()
    train_losses.append(train_loss)
    
    # Validation
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index, data.edge_attr)
        test_loss = criterion(out[test_mask], data.y[test_mask]).item()
        test_losses.append(test_loss)
        
        pred = out[test_mask].argmax(dim=-1)
        acc = (pred == data.y[test_mask]).float().mean().item()
        test_accs.append(acc)
        
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), MODEL_PATH)
    
    log(f"Epoch {epoch:3d}/{EPOCHS}  TrainLoss={train_loss:.4f}  ValLoss={test_loss:.4f}  ValAcc={acc:.4f}  Best={best_acc:.4f}  ({time.time()-t0:.1f}s)")

# ---- 4. Final Evaluation ----
log("Final evaluation...")
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()
with torch.no_grad():
    out = model(data.x, data.edge_index, data.edge_attr)
    probs = F.softmax(out[test_mask], dim=-1).cpu().numpy()
    pred = probs.argmax(axis=1)
    true = data.y[test_mask].cpu().numpy()

# Metrics
acc = accuracy_score(true, pred)
macro_f1 = f1_score(true, pred, average="macro")
weighted_f1 = f1_score(true, pred, average="weighted")
prec, rec, _, _ = precision_recall_fscore_support(true, pred, average="macro")
y_true_bin = label_binarize(true, classes=[0,1,2])
auc = roc_auc_score(y_true_bin, probs, average="macro", multi_class="ovr")
cm = confusion_matrix(true, pred, labels=[0,1,2])
report = classification_report(true, pred, target_names=class_names, output_dict=True)

metrics = {
    "method": "C-GAT (Full Graph, Full-Batch)",
    "accuracy": float(acc),
    "macro_f1": float(macro_f1),
    "weighted_f1": float(weighted_f1),
    "macro_precision": float(prec),
    "macro_recall": float(rec),
    "macro_auc_ovr": float(auc),
    "confusion_matrix": cm.tolist(),
    "per_class": report,
    "best_val_acc": best_acc,
    "final_test_acc": float(acc),
    "n_params": sum(p.numel() for p in model.parameters()),
    "n_cells": int(n_cells),
    "n_edges": int(edge_index.shape[1]),
    "epochs": EPOCHS
}

log(f"C-GAT Full: acc={acc:.4f}, macro_f1={macro_f1:.4f}, AUC={auc:.4f}")

# Save metrics
with open(METRICS_PATH, "w") as f:
    json.dump(metrics, f, indent=2)

# Save predictions
import pandas as pd
pred_df = pd.DataFrame({
    "true": [class_names[i] for i in true],
    "pred": [class_names[i] for i in pred],
    **{f"prob_{cn}": probs[:, i] for i, cn in enumerate(class_names)}
})
pred_df.to_csv(PRED_PATH, sep="\t", index=False)

# Plots
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].plot(train_losses, label="train")
axes[0].plot(test_losses, label="val")
axes[0].set_title("Loss"); axes[0].legend()
axes[1].plot(test_accs, label="val acc", color="green")
axes[1].set_title("Val Accuracy"); axes[1].legend()
fig.tight_layout()
fig.savefig(PNG_LOSS, dpi=180); plt.close(fig)

fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names, ax=ax)
ax.set_title("C-GAT Full Graph"); ax.set_xlabel("Predicted"); ax.set_ylabel("True")
fig.tight_layout()
fig.savefig(PNG_CM, dpi=180); plt.close(fig)

log(f"Saved: {METRICS_PATH}, {PRED_PATH}, {PNG_CM}, {PNG_LOSS}")
log("DONE.")

if __name__ == "__main__":
    import sys
    sys.exit(0)