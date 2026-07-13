"""
06_method1_classical_baseline.py
=================================
Classical ML baseline: Logistic Regression (multinomial) + Random Forest
on 2,500 HVGs from the 15k balanced subsample.

- Uses the pre-saved nn_X.npy / nn_y.npy (already HVG-subset, row-z-scored)
- Stratified 80/20 split (same as deep model for fair comparison)
- Outputs: metrics, per-class report, confusion matrices, feature importance
"""
import os, time, resource, json, gc
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, f1_score, precision_recall_fscore_support,
                             confusion_matrix, classification_report, roc_auc_score)
from sklearn.preprocessing import label_binarize
import joblib

ROOT    = "/mnt/c/Users/vihan/20206 science fair"
OUT_DIR = os.path.join(ROOT, "output")
NPY_X   = os.path.join(OUT_DIR, "nn_X.npy")
NPY_Y   = os.path.join(OUT_DIR, "nn_y.npy")
TSV_GENE_NM = os.path.join(OUT_DIR, "nn_gene_names.tsv")
TSV_CLS_NM  = os.path.join(OUT_DIR, "nn_class_names.tsv")
TSV_SPLIT   = os.path.join(OUT_DIR, "nn_train_test_split.tsv")

PNG_CM_LR = os.path.join(OUT_DIR, "method1_lr_confusion.png")
PNG_CM_RF = os.path.join(OUT_DIR, "method1_rf_confusion.png")
TSV_METRICS = os.path.join(OUT_DIR, "method1_metrics.json")
TSV_IMP = os.path.join(OUT_DIR, "method1_lr_coefficients.tsv")
TSV_IMP_RF = os.path.join(OUT_DIR, "method1_rf_importance.tsv")

def mem_mb(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}  RSS={mem_mb():.0f}MB", flush=True)

# ---- 1. Load data (already filtered, HVG-subset, row-z-scored) --------------
log("Loading dense tensors...")
X = np.load(NPY_X)  # (15000, 2500) float32
y = np.load(NPY_Y)  # (15000,) int64
gene_names = pd.read_csv(TSV_GENE_NM, sep="\t")["gene"].tolist()
class_df   = pd.read_csv(TSV_CLS_NM, sep="\t")
class_names = class_df["class_name"].tolist()
split_df   = pd.read_csv(TSV_SPLIT, sep="\t")
split_map = dict(zip(split_df["index"], split_df["split"]))
log(f"X {X.shape}, y {y.shape}, classes={class_names}")

train_mask = np.array([split_map[i] == "train" for i in range(len(y))])
test_mask  = ~train_mask
X_train, X_test = X[train_mask], X[test_mask]
y_train, y_test = y[train_mask], y[test_mask]
log(f"train {X_train.shape}, test {X_test.shape}")

# ---- 2. Logistic Regression (multinomial, L2) -------------------------------
log("Training Logistic Regression (multinomial, l2, max_iter=1000)...")
lr = LogisticRegression(
    solver="lbfgs",
    C=1.0,
    max_iter=1000,
    n_jobs=2,
    random_state=42,
    class_weight="balanced"
)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
y_proba_lr = lr.predict_proba(X_test)

# ---- 3. Random Forest -------------------------------------------------------
log("Training Random Forest (n_estimators=500, max_depth=None)...")
rf = RandomForestClassifier(
    n_estimators=500,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    n_jobs=2,
    random_state=42,
    class_weight="balanced"
)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
y_proba_rf = rf.predict_proba(X_test)

# ---- 4. Metrics -------------------------------------------------------------
def compute_metrics(y_true, y_pred, y_proba, name):
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    weighted_f1 = f1_score(y_true, y_pred, average="weighted")
    precision, recall, _, _ = precision_recall_fscore_support(y_true, y_pred, average="macro")
    # AUC one-vs-rest
    y_true_bin = label_binarize(y_true, classes=[0,1,2])
    auc = roc_auc_score(y_true_bin, y_proba, average="macro", multi_class="ovr")
    cm = confusion_matrix(y_true, y_pred, labels=[0,1,2])
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    return {
        "method": name,
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_auc_ovr": float(auc),
        "confusion_matrix": cm.tolist(),
        "per_class": report
    }

metrics_lr = compute_metrics(y_test, y_pred_lr, y_proba_lr, "LogisticRegression")
metrics_rf = compute_metrics(y_test, y_pred_rf, y_proba_rf, "RandomForest")

log(f"LR: acc={metrics_lr['accuracy']:.4f}, macro_f1={metrics_lr['macro_f1']:.4f}, AUC={metrics_lr['macro_auc_ovr']:.4f}")
log(f"RF: acc={metrics_rf['accuracy']:.4f}, macro_f1={metrics_rf['macro_f1']:.4f}, AUC={metrics_rf['macro_auc_ovr']:.4f}")

# ---- 5. Save confusion matrices (plot) -------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
def plot_cm(cm, title, path):
    fig, ax = plt.subplots(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names,
                yticklabels=class_names, cmap="Blues", ax=ax)
    ax.set_title(title); ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)

plot_cm(np.array(metrics_lr["confusion_matrix"]),
        "Logistic Regression", PNG_CM_LR)
plot_cm(np.array(metrics_rf["confusion_matrix"]),
        "Random Forest", PNG_CM_RF)
log(f"Saved {PNG_CM_LR}, {PNG_CM_RF}")

# ---- 6. Save coefficients / feature importance ------------------------------
# LR: coefficient per class (3 x 2500)
lr_coef = pd.DataFrame(lr.coef_, columns=gene_names, index=class_names)
lr_coef.to_csv(TSV_IMP, sep="\t")
log(f"Saved LR coefficients -> {TSV_IMP}")

# RF: mean decrease impurity per feature
rf_imp = pd.DataFrame({
    "gene": gene_names,
    "importance": rf.feature_importances_
}).sort_values("importance", ascending=False)
rf_imp.to_csv(TSV_IMP_RF, sep="\t", index=False)
log(f"Saved RF importance -> {TSV_IMP_RF}")

# ---- 7. Save metrics JSON ---------------------------------------------------
with open(TSV_METRICS, "w") as f:
    json.dump({"LogisticRegression": metrics_lr, "RandomForest": metrics_rf}, f, indent=2)
log(f"Saved metrics -> {TSV_METRICS}")

# ---- 8. Save models ---------------------------------------------------------
joblib.dump(lr, os.path.join(OUT_DIR, "method1_lr.joblib"))
joblib.dump(rf, os.path.join(OUT_DIR, "method1_rf.joblib"))
log("Saved joblib models. DONE.")