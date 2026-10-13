"""
Step 1: Vectorize EMBER 2018 JSONL -> .dat files
Fix loi multiprocessing tren Windows + Python 3.13
Tai tao dung logic cua ember.vectorize_subset nhung single-thread
"""
import os
import sys
import json
import numpy as np
from pathlib import Path


# CONFIG - chi sua day neu can

DATA_DIR   = Path("data/ember2018")
N_FEATURES = 2351


def get_extractor():
    import ember.features as ef
    return ef.PEFeatureExtractor(feature_version=2)

def count_lines(file_paths):
    """Dem tong so dong de cap phat memmap."""
    total = 0
    for fp in file_paths:
        with open(fp, "r", encoding="utf-8") as f:
            total += sum(1 for line in f if line.strip())
    return total

def vectorize_single_thread(X_path, y_path, file_paths, extractor):
    """
    Tai tao ember.vectorize_subset nhung KHONG dung multiprocessing.
    Dung np.memmap de tiet kiem RAM - khong load toan bo vao memory.
    """
    print(f"  Dang dem so mau...")
    nrows = count_lines(file_paths)
    print(f"  Tong so mau: {nrows:,}")

    # Cap phat file tren disk (memmap)
    X = np.memmap(X_path, dtype=np.float32, mode="w+", shape=(nrows, extractor.dim))
    y = np.memmap(y_path, dtype=np.float32, mode="w+", shape=nrows)
    del X, y  # Dong de ghi lai sau

    # Xu ly tung dong
    irow = 0
    errors = 0

    try:
        from tqdm import tqdm
        use_tqdm = True
    except ImportError:
        use_tqdm = False

    for fpath in file_paths:
        print(f"  [{irow:,}/{nrows:,}] Xu ly: {Path(fpath).name}")
        with open(fpath, "r", encoding="utf-8") as f:
            lines = f.readlines()

        iter_lines = tqdm(lines, ncols=70) if use_tqdm else lines

        for line in iter_lines:
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
                feat = extractor.process_raw_features(raw)
                label = raw.get("label", -1)

                # Ghi vao memmap
                X = np.memmap(X_path, dtype=np.float32, mode="r+",
                               shape=(nrows, extractor.dim))
                X[irow] = feat
                del X

                y = np.memmap(y_path, dtype=np.float32, mode="r+", shape=nrows)
                y[irow] = float(label)
                del y

                irow += 1
            except Exception:
                errors += 1
                irow += 1  # giu index dong bo

    if errors > 0:
        print(f"  [WARN] {errors} dong loi da bi bo qua")

    print(f"  -> Hoan tat: {irow:,} mau")
    return irow

def verify(data_dir: Path) -> bool:
    """Xac nhan .dat doc duoc va co du lieu that."""
    print("\n=== Xac nhan .dat files ===")
    X = np.memmap(data_dir / "X_train.dat", dtype=np.float32, mode="r")
    y = np.memmap(data_dir / "y_train.dat", dtype=np.float32, mode="r")

    n = len(y)
    X = X.reshape(n, N_FEATURES)

    print(f"  X_train shape : {X.shape}")
    print(f"  min={X.min():.4f}  max={X.max():.4f}  mean={X.mean():.4f}")
    print(f"  Row[0] (5feat): {X[0, :5]}")

    labels = dict(zip(*np.unique(y.astype(int), return_counts=True)))
    print(f"  Labels        : {labels}")

    if X.max() == 0.0 and X.min() == 0.0:
        print("  [LOI] Toan so 0 - vectorize that bai!")
        return False

    print("  [OK] Data hop le!")
    return True

def main():
    print("=" * 55)
    print("STEP 1: Vectorize EMBER 2018 (single-thread, safe)")
    print("=" * 55)
    print(f"DATA_DIR: {DATA_DIR.resolve()}\n")

    if not DATA_DIR.exists():
        print(f"[LOI] Khong tim thay: {DATA_DIR}")
        sys.exit(1)

    # Neu da co .dat roi
    out_files = ["X_train.dat", "y_train.dat", "X_test.dat", "y_test.dat"]
    if all((DATA_DIR / f).exists() for f in out_files):
        print("[OK] Da co .dat files.")
        ok = verify(DATA_DIR)
        if ok:
            print("\nChay tiep: python training/step2_split_scale.py")
        sys.exit(0 if ok else 1)

    # Kiem tra JSONL
    train_paths = sorted(DATA_DIR.glob("train_features_*.jsonl"))
    test_path   = DATA_DIR / "test_features.jsonl"

    if not train_paths:
        print("[LOI] Khong tim thay train_features_*.jsonl")
        sys.exit(1)
    if not test_path.exists():
        print("[LOI] Khong tim thay test_features.jsonl")
        sys.exit(1)

    print(f"Tim thay {len(train_paths)} file train + 1 file test")

    # Load extractor
    print("\nLoad PEFeatureExtractor...")
    extractor = get_extractor()
    print(f"[OK] dim = {extractor.dim}")

    # --- TRAIN ---
    print("\n=== Vectorize TRAIN SET ===")
    print("(mat khoang 20-40 phut, khong tat terminal)")
    vectorize_single_thread(
        str(DATA_DIR / "X_train.dat"),
        str(DATA_DIR / "y_train.dat"),
        [str(p) for p in train_paths],
        extractor
    )

    # --- TEST ---
    print("\n=== Vectorize TEST SET ===")
    vectorize_single_thread(
        str(DATA_DIR / "X_test.dat"),
        str(DATA_DIR / "y_test.dat"),
        [str(test_path)],
        extractor
    )

    # Kich thuoc file
    print("\n=== Kich thuoc file output ===")
    for fname in out_files:
        path = DATA_DIR / fname
        if path.exists():
            mb = path.stat().st_size / 1024 / 1024
            print(f"  {fname}: {mb:.0f} MB")

    # Xac nhan
    ok = verify(DATA_DIR)
    if ok:
        print("\n[XONG] Chay tiep: python training/step2_split_scale.py")
    else:
        print("\n[LOI] Kiem tra lai.")

if __name__ == "__main__":
    main()