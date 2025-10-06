# validation.py
# Leak-free nested CV for MIAS-style HDF5 with keys: scan, SEVERITY, CLASS, REFNUM

import os, json, argparse, warnings
from typing import Tuple, Dict, Any
import numpy as np
import h5py
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, roc_curve, precision_recall_curve,
    confusion_matrix, accuracy_score, precision_score, recall_score, f1_score,
    brier_score_loss
)
from sklearn.calibration import CalibrationDisplay

warnings.filterwarnings("ignore")
np.random.seed(42)

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def bootstrap_ci(values: np.ndarray, n_boot: int = 2000, alpha: float = 0.05):
    rng = np.random.default_rng(42)
    vals = np.asarray(values, float)
    boots = [rng.choice(vals, size=vals.size, replace=True).mean() for _ in range(n_boot)]
    return float(vals.mean()), float(np.percentile(boots, 100*(alpha/2))), float(np.percentile(boots, 100*(1-alpha/2)))

def parse_args():
    ap = argparse.ArgumentParser(description="Nested CV validation on MIAS-like HDF5.")
    # path is relative to AI-Model-Code-Files by default
    ap.add_argument("--h5", default=r"..\Cancer-Dataset-Files\all_mias_scans.h5",
                    help="HDF5 with datasets: scan, SEVERITY, CLASS, REFNUM")
    ap.add_argument("--task", default="benign_vs_malignant",
                    choices=["benign_vs_malignant", "abnormal_vs_normal"])
    ap.add_argument("--reports", default="validation_reports")
    ap.add_argument("--n_splits", type=int, default=5)
    ap.add_argument("--inner_splits", type=int, default=3)
    return ap.parse_args()

# ----------------------------
# HDF5 LOADING (tailored to your file)
# ----------------------------
def _dec(x):
    return x.decode() if isinstance(x, (bytes, np.bytes_)) else str(x)

def load_from_h5(path: str):
    """
    Expects keys in H5:
      - scan      : (N, H, W) uint8 images
      - SEVERITY  : (N,) bytes strings b'B', b'M', or b'nan' for normals
      - CLASS     : (N,) bytes strings e.g. b'NORM', b'CIRC', ...
      - REFNUM    : (N,) bytes strings like b'mdb001'
    """
    with h5py.File(path, "r") as f:
        if "scan" not in f:
            raise KeyError("HDF5 must contain dataset 'scan' (images)")
        X = f["scan"][...]                                # (N,H,W) uint8
        severity = np.array([_dec(v) for v in f["SEVERITY"][...]])
        lesion_cls = np.array([_dec(v) for v in f["CLASS"][...]])
        refs = np.array([_dec(v) for v in f["REFNUM"][...]])

    # Normalize to [0,1], add channel axis
    X = X.astype("float32") / 255.0
    if X.ndim == 3:
        X = X[..., None]  # (N,H,W,1)

    return X, severity, lesion_cls, refs

def infer_patient_groups(refs: np.ndarray) -> np.ndarray:
    """
    MIAS naming: mdb001, mdb002 are same patient (L/R).
    """
    ids = []
    for r in refs:
        digits = "".join(ch for ch in r if ch.isdigit())
        if digits == "": ids.append(hash(r)); continue
        num = int(digits)
        ids.append((num - 1) // 2)  # 1,2 -> 0; 3,4 -> 1; ...
    return np.array(ids)

def build_binary_labels(task: str, severity: np.ndarray, lesion_cls: np.ndarray):
    """
    For your file:
      - severity: 'B', 'M', or 'nan' (normal)
      - lesion_cls: 'NORM' or other lesion class names

    Tasks:
      * benign_vs_malignant:
          Use only benign/malignant cases (drop normals).
          y = 1 if 'M', 0 if 'B'
          Returns (y, mask) where mask selects B/M only.
      * abnormal_vs_normal:
          y = 1 if CLASS != 'NORM', else 0
          Returns (y, None)
    """
    sev = np.array([s.upper() for s in severity])
    cls = np.array([c.upper() for c in lesion_cls])

    if task == "benign_vs_malignant":
        mask = (sev == "B") | (sev == "M")
        y = (sev[mask] == "M").astype(int)
        return y, mask
    elif task == "abnormal_vs_normal":
        y = (cls != "NORM").astype(int)
        return y, None
    else:
        raise ValueError("Unknown task")

# ----------------------------
# MODELS & METRICS
# ----------------------------
def get_models_and_grids():
    models = {
        "logreg": Pipeline([
            ("scaler", StandardScaler(with_mean=False)),
            ("clf", LogisticRegression(max_iter=300, class_weight="balanced"))
        ]),
        "rf": RandomForestClassifier(class_weight="balanced_subsample", random_state=42, n_jobs=-1)
    }
    grids = {
        "logreg": {"clf__C": [0.1, 1.0, 3.0]},
        "rf": {"n_estimators": [200, 400], "max_depth": [None, 12, 20], "min_samples_split": [2, 5]}
    }
    return models, grids

def flatten(X: np.ndarray) -> np.ndarray:
    return X.reshape(len(X), -1)

def metric_dict(y_true, y_prob, y_pred) -> Dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auroc": roc_auc_score(y_true, y_prob),
        "auprc": average_precision_score(y_true, y_prob),
        "brier": brier_score_loss(y_true, y_prob),
    }

# ----------------------------
# PLOTS
# ----------------------------
def plot_roc_pr(y_true, y_prob, tag, outdir):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auroc = roc_auc_score(y_true, y_prob)
    plt.figure(); plt.plot(fpr, tpr, label=f"AUC={auroc:.3f}"); plt.plot([0,1],[0,1],"k--")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title(f"ROC - {tag}"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"roc_{tag}.png")); plt.close()

    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    plt.figure(); plt.plot(rec, prec, label=f"AP={ap:.3f}")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title(f"PR - {tag}"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"pr_{tag}.png")); plt.close()

def plot_cm(y_true, y_pred, tag, outdir):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(); plt.imshow(cm, cmap="Blues"); plt.title(f"Confusion Matrix - {tag}")
    plt.xlabel("Predicted"); plt.ylabel("True")
    for (i,j),v in np.ndenumerate(cm): plt.text(j,i,str(v),ha="center",va="center")
    plt.colorbar(); plt.tight_layout(); plt.savefig(os.path.join(outdir, f"cm_{tag}.png")); plt.close()

def plot_calibration(y_true, y_prob, tag, outdir):
    plt.figure(); CalibrationDisplay.from_predictions(y_true, y_prob, n_bins=10)
    plt.title(f"Calibration - {tag}"); plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"calibration_{tag}.png")); plt.close()

# ----------------------------
# NESTED CV
# ----------------------------
def nested_cv_binary(X, y, groups, outdir, task_name, n_splits=5, inner_splits=3) -> Dict[str, Any]:
    Xf = flatten(X)
    models, grids = get_models_and_grids()
    outer = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    all_results = {}

    for model_name, base in models.items():
        per_fold = []; y_all=[]; p_all=[]; z_all=[]

        for k, (tr, te) in enumerate(outer.split(Xf, y, groups)):
            X_tr, X_te = Xf[tr], Xf[te]
            y_tr, y_te = y[tr], y[te]

            inner = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=k)
            gs = GridSearchCV(base, grids[model_name], scoring="f1", cv=inner, n_jobs=-1)
            gs.fit(X_tr, y_tr)

            best = gs.best_estimator_
            y_prob = best.predict_proba(X_te)[:,1]
            y_pred = (y_prob >= 0.5).astype(int)

            m = metric_dict(y_te, y_prob, y_pred); per_fold.append(m)
            tag = f"{task_name}_{model_name}_fold{k+1}"
            plot_roc_pr(y_te, y_prob, tag, outdir); plot_cm(y_te, y_pred, tag, outdir); plot_calibration(y_te, y_prob, tag, outdir)
            y_all.append(y_te); p_all.append(y_prob); z_all.append(y_pred)

        stack = {k: np.array([f[k] for f in per_fold]) for k in per_fold[0].keys()}
        summary = {}
        for mname, vals in stack.items():
            mean, lo, hi = bootstrap_ci(vals)
            summary[mname] = {"mean": float(vals.mean()), "std": float(vals.std()), "ci95": [lo, hi]}
        all_results[model_name] = summary

        y_cat = np.concatenate(y_all); p_cat = np.concatenate(p_all); z_cat = np.concatenate(z_all)
        plot_roc_pr(y_cat, p_cat, f"{task_name}_{model_name}_ALL", outdir)
        plot_cm(y_cat, z_cat, f"{task_name}_{model_name}_ALL", outdir)
        plot_calibration(y_cat, p_cat, f"{task_name}_{model_name}_ALL", outdir)

    out_json = os.path.join(outdir, f"results_{task_name}.json")
    with open(out_json, "w") as f: json.dump(all_results, f, indent=2)
    print(f"\nSaved results → {out_json}")
    return all_results

# ----------------------------
# MAIN
# ----------------------------
def main():
    args = parse_args()
    ensure_dir(args.reports)

    print(f"Loading: {args.h5}")
    X, severity, lesion_cls, refs = load_from_h5(args.h5)
    groups = infer_patient_groups(refs)

    # label build per task
    y, mask = build_binary_labels(args.task, severity, lesion_cls)
    if mask is not None:
        X = X[mask]; groups = groups[mask]

    pos = int(y.sum())
    print(f"Task: {args.task} | Samples: {len(y)} | Positives: {pos} | Negatives: {len(y)-pos}")

    nested_cv_binary(X, y, groups, args.reports, args.task, args.n_splits, args.inner_splits)
    print(f"All figures saved in: {args.reports}/")

if __name__ == "__main__":
    main()
