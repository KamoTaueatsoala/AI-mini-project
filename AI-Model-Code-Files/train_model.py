import os
import h5py
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# ------------------------------
# CONFIG
# ------------------------------
H5_PATH = "all_mias_scans.h5"  # update path
MODEL_SAVE_PATH = "abnormality_detector.pkl"
TEST_SIZE = 0.2
VAL_SIZE = 0.2  # fraction of train for validation
RANDOM_STATE = 42

# ------------------------------
# LOAD HDF5
# ------------------------------
def _decode(x):
    return x.decode() if isinstance(x, (bytes, np.bytes_)) else str(x)

with h5py.File(H5_PATH, "r") as f:
    X = f["scan"][...].astype("float32") / 255.0  # normalize to [0,1]
    classes = np.array([_decode(c) for c in f["CLASS"][...]])

# Binary labels: 0 = NORM, 1 = abnormal
y = (classes != "NORM").astype(int)

# Flatten images for scikit-learn classifiers
X_flat = X.reshape(len(X), -1)

# ------------------------------
# SPLIT: TRAIN / VALIDATION / TEST
# ------------------------------
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X_flat, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=VAL_SIZE, random_state=RANDOM_STATE, stratify=y_trainval
)

print(f"Train: {len(y_train)}, Val: {len(y_val)}, Test: {len(y_test)}")
print(f"Positive cases in train: {y_train.sum()}, test: {y_test.sum()}")

# ------------------------------
# STANDARDIZE
# ------------------------------
scaler = StandardScaler(with_mean=False)  # without_mean for sparse/flattened images
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# ------------------------------
# MODEL TRAINING: Logistic Regression & Random Forest
# ------------------------------
# Logistic Regression (baseline)
lr_model = LogisticRegression(max_iter=300, class_weight="balanced")
lr_model.fit(X_train, y_train)

# Random Forest with hyperparameter tuning
rf_param_grid = {
    'n_estimators': [100, 150],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5]
}
grid_search = GridSearchCV(RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE),
                           rf_param_grid, cv=3, scoring="f1", n_jobs=-1)
grid_search.fit(X_train, y_train)
rf_model = grid_search.best_estimator_

# ------------------------------
# EVALUATION FUNCTION
# ------------------------------
def evaluate_model(name, model, X, y):
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else None
    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred)
    rec = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    auc = roc_auc_score(y, y_prob) if y_prob is not None else None
    print(f"\n{name} Metrics:")
    print(f"Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "auc": auc}

print("\nValidation Metrics:")
evaluate_model("Logistic Regression", lr_model, X_val, y_val)
evaluate_model("Random Forest", rf_model, X_val, y_val)

# ------------------------------
# SAVE BEST MODEL
# ------------------------------
# For simplicity, we save the Random Forest as the final model
joblib.dump({"model": rf_model, "scaler": scaler}, MODEL_SAVE_PATH)
print(f"\nSaved model to {MODEL_SAVE_PATH}")
