"""
Step 3: Train XGBoost, LightGBM, RandomForest voi 5-fold CV
Version tiet kiem RAM - doc data bang mmap_mode, fold thu cong
Input : data/processed/X_train.npy, y_train.npy
Output: models/artifacts/xgb.pkl, lgb.pkl, rf.pkl, metadata.json
        results/reports/cv_results.csv
"""
import sys
import time
import json
import numpy as np
import joblib
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier


DATA_DIR     = Path("data/processed")
MODEL_DIR    = Path("models/artifacts")
RES_DIR      = Path("results/reports")
RANDOM_STATE = 42
N_SPLITS     = 5


MODEL_DIR.mkdir(parents=True, exist_ok=True)
RES_DIR.mkdir(parents=True, exist_ok=True)

def load_data():
    """
    Load bang mmap_mode='r' - khong copy vao RAM.
    Chi chuyen sang array khi thuc su can (trong tung fold).
    """
    print("Load data (mmap mode)...")
    X = np.load(DATA_DIR / "X_train.npy", mmap_mode="r")
    y = np.load(DATA_DIR / "y_train.npy")   # y nho, load het cung duoc
    print(f"  X: {X.shape} | y: {y.shape}")
    print(f"  Benign: {(y==0).sum():,} | Malware: {(y==1).sum():,}")

    if np.array(X[:100]).max() == 0.0:
        print("[LOI] Data toan so 0!")
        sys.exit(1)
    return X, y

def run_cv(name: str, model, X_mmap, y: np.ndarray):
    """
    5-fold CV thu cong.
    Moi fold: chi load X_train_fold va X_val_fold vao RAM, xong giai phong.
    """
    print(f"\n{'='*52}")
    print(f"Training: {name}")
    print(f"{'='*52}")

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True,
                         random_state=RANDOM_STATE)
    f1_list, auc_list = [], []
    t0 = time.time()

    for fold, (tr_idx, val_idx) in enumerate(cv.split(y, y), 1):
        # Chi load fold nay vao RAM, xong la giai phong
        X_tr  = np.array(X_mmap[tr_idx],  dtype=np.float32)
        X_val = np.array(X_mmap[val_idx], dtype=np.float32)
        y_tr  = y[tr_idx]
        y_val = y[val_idx]

        model.fit(X_tr, y_tr)
        del X_tr  # giai phong RAM ngay

        y_pred  = model.predict(X_val)
        y_proba = model.predict_proba(X_val)[:, 1]
        del X_val  # giai phong RAM ngay

        f1  = f1_score(y_val, y_pred)
        auc = roc_auc_score(y_val, y_proba)
        f1_list.append(f1)
        auc_list.append(auc)
        print(f"  Fold {fold}/{N_SPLITS}: F1={f1:.4f}  AUC={auc:.4f}")

    elapsed = time.time() - t0
    f1_arr  = np.array(f1_list)
    auc_arr = np.array(auc_list)

    print(f"\n  CV F1 : {f1_arr.mean():.4f} +/- {f1_arr.std():.4f}")
    print(f"  CV AUC: {auc_arr.mean():.4f} +/- {auc_arr.std():.4f}")
    print(f"  Time  : {elapsed:.0f}s ({elapsed/60:.1f} min)")

    # Train tren toan bo train set
    print(f"  Train final model tren toan bo {len(y):,} mau...")
    X_full = np.array(X_mmap, dtype=np.float32)
    model.fit(X_full, y)
    del X_full

    return {
        "model"      : name,
        "cv_f1_mean" : round(float(f1_arr.mean()),  4),
        "cv_f1_std"  : round(float(f1_arr.std()),   4),
        "cv_auc_mean": round(float(auc_arr.mean()), 4),
        "cv_auc_std" : round(float(auc_arr.std()),  4),
        "time_min"   : round(elapsed / 60, 1),
    }, model

def main():
    print("=" * 52)
    print("STEP 3: Train Models (5-fold CV, RAM-efficient)")
    print("=" * 52)

    X, y = load_data()
    all_results = []

    # ---- 1. LightGBM ----
    lgb_model = lgb.LGBMClassifier(
        n_estimators    = 500,
        num_leaves      = 63,
        max_depth       = -1,
        learning_rate   = 0.05,
        subsample       = 0.8,
        colsample_bytree= 0.8,
        min_child_samples=20,
        random_state    = RANDOM_STATE,
        n_jobs          = -1,
        verbose         = -1,
    )
    res, lgb_fit = run_cv("LightGBM", lgb_model, X, y)
    all_results.append(res)
    joblib.dump(lgb_fit, MODEL_DIR / "lgb.pkl")
    print(f"  [SAVED] lgb.pkl")

    # ---- 2. XGBoost ----
    xgb_model = xgb.XGBClassifier(
        n_estimators    = 500,
        max_depth       = 8,
        learning_rate   = 0.05,
        subsample       = 0.8,
        colsample_bytree= 0.8,
        eval_metric     = "logloss",
        random_state    = RANDOM_STATE,
        n_jobs          = -1,
        verbosity       = 0,
        device          = "cpu",
    )
    res, xgb_fit = run_cv("XGBoost", xgb_model, X, y)
    all_results.append(res)
    joblib.dump(xgb_fit, MODEL_DIR / "xgb.pkl")
    print(f"  [SAVED] xgb.pkl")

    # ---- 3. Random Forest ----
    rf_model = RandomForestClassifier(
        n_estimators    = 300,
        max_depth       = 20,
        min_samples_leaf= 5,
        max_features    = "sqrt",
        random_state    = RANDOM_STATE,
        n_jobs          = -1,
    )
    res, rf_fit = run_cv("RandomForest", rf_model, X, y)
    all_results.append(res)
    joblib.dump(rf_fit, MODEL_DIR / "rf.pkl")
    print(f"  [SAVED] rf.pkl")

    # Luu metadata cho web
    metadata = {
        "n_features"     : int(X.shape[1]),
        "feature_version": 2,
        "lief_version"   : "0.17.6",
        "models"         : ["lgb", "xgb", "rf"],
        "scaler_path"    : "data/processed/scaler.pkl",
        "random_state"   : RANDOM_STATE,
    }
    with open(MODEL_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print("\n  [SAVED] metadata.json")

    # Luu CV results
    df = pd.DataFrame(all_results)
    df.to_csv(RES_DIR / "cv_results.csv", index=False)

    print(f"\n{'='*52}")
    print("CV RESULTS SUMMARY")
    print(f"{'='*52}")
    print(df[["model","cv_f1_mean","cv_f1_std",
              "cv_auc_mean","time_min"]].to_string(index=False))
    print(f"\n[XONG] Chay tiep: python training/step4_evaluate.py")

if __name__ == "__main__":
    main()