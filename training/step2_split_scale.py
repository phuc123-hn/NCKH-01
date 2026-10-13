"""
Step 2: Load .dat, loc label, chia train/val/test, StandardScaler
Version tiet kiem RAM - dung memmap, khong load toan bo vao memory
Input : data/ember2018/X_train.dat, y_train.dat, X_test.dat, y_test.dat
Output: data/processed/X_train.npy, X_val.npy, X_test.npy
        data/processed/y_train.npy, y_val.npy, y_test.npy
        data/processed/scaler.pkl
"""
import sys
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


EMBER_DIR    = Path("data/ember2018")
OUT_DIR      = Path("data/processed")
N_FEATURES   = 2381
RANDOM_STATE = 42


OUT_DIR.mkdir(parents=True, exist_ok=True)

def load_via_memmap(ember_dir: Path, n_features: int):
    """
    Doc .dat bang memmap - KHONG load vao RAM.
    Memmap chi doc tu disk khi can, tiet kiem RAM toi da.
    """
    print("Load bang memmap (khong vao RAM)...")

    y_tr = np.memmap(ember_dir / "y_train.dat", dtype=np.float32, mode="r")
    X_tr = np.memmap(ember_dir / "X_train.dat", dtype=np.float32, mode="r")
    X_tr = X_tr.reshape(len(y_tr), n_features)

    y_te = np.memmap(ember_dir / "y_test.dat",  dtype=np.float32, mode="r")
    X_te = np.memmap(ember_dir / "X_test.dat",  dtype=np.float32, mode="r")
    X_te = X_te.reshape(len(y_te), n_features)

    print(f"  X_train: {X_tr.shape} (tren disk, chua vao RAM)")
    print(f"  X_test : {X_te.shape} (tren disk, chua vao RAM)")
    return X_tr, y_tr.astype(np.int32), X_te, y_te.astype(np.int32)

def get_labeled_indices(y: np.ndarray, name: str):
    """Tra ve chi so cua cac mau co label 0 hoac 1."""
    idx = np.where(y != -1)[0]
    counts = dict(zip(*np.unique(y[idx], return_counts=True)))
    print(f"  {name}: {len(y):,} -> {len(idx):,} labeled")
    print(f"    Benign : {counts.get(0, 0):,}")
    print(f"    Malware: {counts.get(1, 0):,}")
    return idx

def fit_scaler_batched(X_memmap: np.ndarray, indices: np.ndarray,
                        batch_size: int = 50000) -> StandardScaler:
    """
    Fit StandardScaler theo tung batch de tiet kiem RAM.
    Dung partial_fit thay vi fit toan bo mot luc.
    """
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    n = len(indices)
    n_batches = (n + batch_size - 1) // batch_size

    print(f"  Fitting scaler theo {n_batches} batch (moi batch {batch_size:,} mau)...")
    for i in range(n_batches):
        batch_idx = indices[i*batch_size : (i+1)*batch_size]
        X_batch = np.array(X_memmap[batch_idx], dtype=np.float32)
        scaler.partial_fit(X_batch)
        print(f"    Batch {i+1}/{n_batches} done", end="\r")
    print()
    return scaler

def transform_and_save(X_memmap: np.ndarray, indices: np.ndarray,
                        scaler: StandardScaler, out_path: Path,
                        batch_size: int = 50000):
    """
    Transform va luu truc tiep ra file .npy theo tung batch.
    Khong can load toan bo vao RAM cung mot luc.
    """
    n = len(indices)
    n_features = X_memmap.shape[1]
    n_batches = (n + batch_size - 1) // batch_size

    # Tao file npy truoc (dung memmap de ghi)
    out = np.lib.format.open_memmap(
        str(out_path), mode="w+", dtype=np.float32, shape=(n, n_features)
    )

    print(f"  Transform + luu {out_path.name} ({n_batches} batch)...")
    for i in range(n_batches):
        batch_idx = indices[i*batch_size : (i+1)*batch_size]
        local_idx = slice(i*batch_size, min((i+1)*batch_size, n))
        X_batch = np.array(X_memmap[batch_idx], dtype=np.float32)
        out[local_idx] = scaler.transform(X_batch)
        print(f"    Batch {i+1}/{n_batches} done", end="\r")
    print()
    del out  # flush to disk

def main():
    print("=" * 55)
    print("STEP 2: Split + Scale (RAM-efficient)")
    print("=" * 55)

    # Kiem tra file dau vao
    for f in ["X_train.dat", "y_train.dat", "X_test.dat", "y_test.dat"]:
        if not (EMBER_DIR / f).exists():
            print(f"[LOI] Khong tim thay {EMBER_DIR/f}")
            sys.exit(1)

    # Load bang memmap
    X_tr_mm, y_tr, X_te_mm, y_te = load_via_memmap(EMBER_DIR, N_FEATURES)

    # Kiem tra chat luong (chi doc 1000 mau dau)
    print("\nKiem tra chat luong (1000 mau dau):")
    sample = np.array(X_tr_mm[:1000])
    if sample.max() == 0.0:
        print("[LOI] Data toan so 0!")
        sys.exit(1)
    print(f"  OK: min={sample.min():.4f} max={sample.max():.4f} mean={sample.mean():.4f}")

    # Lay chi so labeled
    print("\nLoc unlabeled:")
    tr_idx = get_labeled_indices(y_tr, "Train")
    te_idx = get_labeled_indices(y_te, "Test ")

    # Split train -> train + val
    print("\nChia train/val (85/15):")
    y_tr_labeled = y_tr[tr_idx]
    tr_sub_idx, val_sub_idx = train_test_split(
        np.arange(len(tr_idx)),
        test_size=0.15,
        random_state=RANDOM_STATE,
        stratify=y_tr_labeled
    )
    train_idx = tr_idx[tr_sub_idx]
    val_idx   = tr_idx[val_sub_idx]

    print(f"  Train: {len(train_idx):,} mau")
    print(f"  Val  : {len(val_idx):,} mau")
    print(f"  Test : {len(te_idx):,} mau")

    # Luu labels truoc (nhe, load duoc het)
    np.save(OUT_DIR / "y_train.npy", y_tr[train_idx])
    np.save(OUT_DIR / "y_val.npy",   y_tr[val_idx])
    np.save(OUT_DIR / "y_test.npy",  y_te[te_idx])
    print("\nDa luu y_train, y_val, y_test")

    # Fit scaler theo batch tren train set
    print("\nFitting StandardScaler (batch mode):")
    scaler = fit_scaler_batched(X_tr_mm, train_idx)
    joblib.dump(scaler, OUT_DIR / "scaler.pkl")
    print("  Da luu scaler.pkl")

    # Transform va luu X theo batch
    print("\nTransform + luu X files:")
    transform_and_save(X_tr_mm, train_idx, scaler, OUT_DIR / "X_train.npy")
    transform_and_save(X_tr_mm, val_idx,   scaler, OUT_DIR / "X_val.npy")
    transform_and_save(X_te_mm, te_idx,    scaler, OUT_DIR / "X_test.npy")

    # Xac nhan ket qua
    print("\nXac nhan file da luu:")
    for fname in ["X_train.npy","X_val.npy","X_test.npy",
                  "y_train.npy","y_val.npy","y_test.npy","scaler.pkl"]:
        path = OUT_DIR / fname
        mb = path.stat().st_size / 1024 / 1024
        print(f"  {fname}: {mb:.0f} MB")

    # Kiem tra nhanh ket qua scale
    print("\nKiem tra X_train sau scale (1000 mau dau):")
    X_check = np.load(OUT_DIR / "X_train.npy", mmap_mode="r")
    sample = np.array(X_check[:1000])
    print(f"  min={sample.min():.4f} max={sample.max():.4f} mean={sample.mean():.4f}")
    if abs(sample.mean()) < 0.1:
        print("  [OK] Scale dung - mean gan 0")
    else:
        print("  [WARN] Mean xa 0 - kiem tra lai")

    print("\n[XONG] Chay tiep: python training/step3_train.py")

if __name__ == "__main__":
    main()