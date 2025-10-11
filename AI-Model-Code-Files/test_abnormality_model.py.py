import h5py
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# ------------------------------
# CONFIG
# ------------------------------
H5_PATH = "all_mias_scans.h5"  # path to test dataset
MODEL_PATH = "abnormality_detector.pkl"

# ------------------------------
# LOAD MODEL
# ------------------------------
saved = joblib.load(MODEL_PATH)
model = saved["model"]
scaler = saved["scaler"]

# ------------------------------
# LOAD TEST DATA
# ------------------------------
def _decode(x):
    return x.decode() if isinstance(x, (bytes, np.bytes_)) else str(x)

with h5py.File(H5_PATH, "r") as f:
    X = f["scan"][...].astype("float32") / 255.0
    classes = np.array([_decode(c) for c in f["CLASS"][...]])

y = (classes != "NORM").astype(int)
X_flat = X.reshape(len(X), -1)
X_scaled = scaler.transform(X_flat)

# ------------------------------
# PREDICTIONS & METRICS
# ------------------------------
y_pred = model.predict(X_scaled)
y_prob = model.predict_proba(X_scaled)[:,1]

acc = accuracy_score(y, y_pred)
prec = precision_score(y, y_pred)
rec = recall_score(y, y_pred)
f1 = f1_score(y, y_pred)
auc = roc_auc_score(y, y_prob)

print("\nTest Metrics:")
print(f"Accuracy: {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall: {rec:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"AUC: {auc:.4f}")

# Optional: Predict single image
def predict_single_image(img_array):
    img_flat = img_array.reshape(1, -1)
    img_scaled = scaler.transform(img_flat)
    pred = model.predict(img_scaled)[0]
    prob = model.predict_proba(img_scaled)[0,1]
    return pred, prob
