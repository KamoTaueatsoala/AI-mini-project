# demo_show.py
# Show ONE normal and ONE abnormal mammogram from the testing set (outer fold 1)
# Saves: demo_pair.png + individual PNGs
#
# Usage examples (run from AI-Model-Code-Files):
#   python demo_show.py
#   python demo_show.py --normal mdb011 --abnormal mdb001
#   python demo_show.py --h5 "..\Cancer-Dataset-Files\all_mias_scans.h5"

import argparse, os, sys
import numpy as np
import h5py

# Headless-safe plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedGroupKFold

def _dec(x):
    return x.decode() if isinstance(x, (bytes, np.bytes_)) else str(x)

def find_h5(user_path: str | None = None) -> str:
    # Priority: CLI arg → common project locations (both variants seen in your repo/zip)
    candidates = []
    if user_path:
        candidates.append(user_path)
    candidates += [
        r"..\Cancer-Dataset-Files\all_mias_scans.h5",
        r".\all_mias_scans.h5",
        r"..\AI-Model-Code-Files\all_mias_scans.h5",
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(
        "Could not find all_mias_scans.h5. "
        "Pass --h5 <path> or place it in AI-Model-Code-Files or Cancer-Dataset-Files."
    )

def groups_from_refs(refs: np.ndarray) -> np.ndarray:
    """MIAS: mdb001/mdb002 same patient."""
    ids = []
    for r in refs:
        d = "".join(ch for ch in r if ch.isdigit())
        ids.append((int(d) - 1) // 2 if d else hash(r))
    return np.array(ids)

def to2d(a: np.ndarray) -> np.ndarray:
    return a[..., 0] if a.ndim == 3 and a.shape[-1] == 1 else a

def save_png(img: np.ndarray, title: str, out_path: str, max_side: int = 1280):
    img2 = to2d(img)
    H, W = img2.shape
    long_side = max(H, W)
    scale = min(1.0, max_side / long_side)
    figw, figh = (W * scale) / 100.0, (H * scale) / 100.0
    plt.figure(figsize=(figw, figh), dpi=100)
    plt.imshow(img2, cmap="gray", vmin=0.0, vmax=1.0)
    plt.axis("off")
    plt.title(title, fontsize=10)
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close()

def main():
    ap = argparse.ArgumentParser(description="Export one normal + one abnormal from testing set (outer fold 1).")
    ap.add_argument("--h5", default=None, help="Path to all_mias_scans.h5")
    ap.add_argument("--normal", default=None, help="REFNUM for desired normal (must be in test set)")
    ap.add_argument("--abnormal", default=None, help="REFNUM for desired abnormal (must be in test set)")
    ap.add_argument("--outdir", default="demo_outputs", help="Directory to save images")
    args = ap.parse_args()

    h5_path = find_h5(args.h5)
    os.makedirs(args.outdir, exist_ok=True)

    print(f"[LOAD] {h5_path}")
    with h5py.File(h5_path, "r") as f:
        X = f["scan"][...]                # (N,H,W) uint8
        CLASS = np.array([_dec(v) for v in f["CLASS"][...]])
        SEVERITY = np.array([_dec(v) for v in f["SEVERITY"][...]])
        REFNUM = np.array([_dec(v) for v in f["REFNUM"][...]])

    # Normalise to [0,1] and add channel if needed
    X = X.astype(np.float32) / 255.0
    if X.ndim == 3:
        X = X[..., None]

    # Labels for abnormal_vs_normal
    y = (np.char.upper(CLASS) != "NORM").astype(int)
    groups = groups_from_refs(REFNUM)

    # Recreate the exact testing set (outer fold 1)
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    splits = list(cv.split(X.reshape(len(X), -1), y, groups))
    _, test_idx = splits[0]

    # Helper: REFNUM -> index
    ref_index = {ref: i for i, ref in enumerate(REFNUM)}

    # Pick the two examples
    if args.normal is not None:
        assert args.normal in ref_index, f"{args.normal} not found in REFNUM"
        i_norm = ref_index[args.normal]
        assert i_norm in test_idx, f"{args.normal} is not in the testing set"
        assert y[i_norm] == 0, f"{args.normal} is not normal (label={y[i_norm]})"
    else:
        # first normal from the test set
        i_norm = next(i for i in test_idx if y[i] == 0)

    if args.abnormal is not None:
        assert args.abnormal in ref_index, f"{args.abnormal} not found in REFNUM"
        i_abn = ref_index[args.abnormal]
        assert i_abn in test_idx, f"{args.abnormal} is not in the testing set"
        assert y[i_abn] == 1, f"{args.abnormal} is not abnormal (label={y[i_abn]})"
    else:
        # first abnormal from the test set
        i_abn = next(i for i in test_idx if y[i] == 1)

    # Save individual images
    norm_title = f"{REFNUM[i_norm]} • CLASS={CLASS[i_norm]} • Normal (0)"
    abn_title  = f"{REFNUM[i_abn]}  • CLASS={CLASS[i_abn]}  • Abnormal (1)"

    norm_path = os.path.join(args.outdir, f"{REFNUM[i_norm]}_normal.png")
    abn_path  = os.path.join(args.outdir,  f"{REFNUM[i_abn]}_abnormal.png")

    save_png(X[i_norm], norm_title, norm_path)
    save_png(X[i_abn], abn_title, abn_path)

    # Save side-by-side pair
    fig = plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1); plt.imshow(to2d(X[i_norm]), cmap="gray", vmin=0, vmax=1); plt.axis("off"); plt.title(norm_title)
    plt.subplot(1, 2, 2); plt.imshow(to2d(X[i_abn]),  cmap="gray", vmin=0, vmax=1); plt.axis("off"); plt.title(abn_title)
    plt.suptitle("Testing Set (Outer Fold 1) — StratifiedGroupKFold(n=5, rs=42), grouped by patient", fontsize=10)
    pair_path = os.path.join(args.outdir, "demo_pair.png")
    plt.tight_layout()
    plt.savefig(pair_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Print a tiny manifest for your slides/notes
    print("\n[DEMO READY]")
    print(f" Normal:   {REFNUM[i_norm]}  | CLASS={CLASS[i_norm]}  | saved → {norm_path}")
    print(f" Abnormal: {REFNUM[i_abn]}   | CLASS={CLASS[i_abn]}   | saved → {abn_path}")
    print(f" Pair figure saved → {pair_path}")
    print("\nQuote in your demo:")
    print("  Testing set = outer fold 1 of StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42), grouped by patient (mdb001/mdb002).")

if __name__ == "__main__":
    main()
