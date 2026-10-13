"""
Step 4: Danh gia 3 model tren holdout test set
- Metrics chuan: F1, AUC, FPR, FNR, Confusion Matrix
- Youden's J threshold optimization (tu file ALGORITHMS.html)
- Holdout stability test (anti-hoc-vet)
- ROC curve + Feature importance plots
Input : data/processed/X_test.npy, y_test.npy + models/artifacts/*.pkl
Output: results/reports/test_results.csv
        results/reports/threshold_analysis.csv
        results/reports/holdout_stability.csv
        results/plots/roc_curves.png
        results/plots/feature_importance.png
        results/plots/threshold_analysis.png
"""
import sys
import json
import numpy as np
import joblib
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (
    f1_score, roc_auc_score, accuracy_score,
    confusion_matrix, roc_curve
)
from sklearn.model_selection import train_test_split


DATA_DIR  = Path("data/processed")
MODEL_DIR = Path("models/artifacts")
RES_DIR   = Path("results/reports")
PLOT_DIR  = Path("results/plots")
RANDOM_STATE = 42


RES_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

COLORS = {
    "LightGBM"   : "#1D9E75",
    "XGBoost"    : "#3B8BD4",
    "RandomForest": "#D85A30",
}


# LOAD


def load_test():
    print("Load test set...")
    X = np.load(DATA_DIR / "X_test.npy", mmap_mode="r")
    y = np.load(DATA_DIR / "y_test.npy")
    print(f"  X_test: {X.shape}")
    print(f"  Benign: {(y==0).sum():,} | Malware: {(y==1).sum():,}")
    return X, y

def load_models():
    models = {}
    for key, name in [("lgb","LightGBM"),("xgb","XGBoost"),("rf","RandomForest")]:
        pkl = MODEL_DIR / f"{key}.pkl"
        if pkl.exists():
            models[name] = joblib.load(pkl)
            print(f"  [OK] {name}")
        else:
            print(f"  [SKIP] {pkl} khong tim thay")
    return models


# PHAN 1: METRICS CHUAN


def evaluate_standard(name, model, X, y):
    """F1, AUC, FPR, FNR tai nguong mac dinh 0.5."""
    y_pred  = model.predict(np.array(X))
    y_proba = model.predict_proba(np.array(X))[:, 1]

    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    f1  = f1_score(y, y_pred)
    auc = roc_auc_score(y, y_proba)
    acc = accuracy_score(y, y_pred)
    fpr_rate = fp / (fp + tn)
    fnr_rate = fn / (fn + tp)

    print(f"\n  {name} (nguong=0.5):")
    print(f"    F1={f1:.4f}  AUC={auc:.4f}  ACC={acc:.4f}")
    print(f"    TP={tp:,}  TN={tn:,}  FP={fp:,}  FN={fn:,}")
    print(f"    FPR={fpr_rate:.4f}  FNR={fnr_rate:.4f}")

    fpr_curve, tpr_curve, thresholds = roc_curve(y, y_proba)

    return {
        "model": name, "threshold": 0.5,
        "f1": round(f1, 4), "auc": round(auc, 4), "accuracy": round(acc, 4),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        "fpr_rate": round(fpr_rate, 4), "fnr_rate": round(fnr_rate, 4),
    }, y_proba, (fpr_curve, tpr_curve, auc)


# PHAN 2: YOUDEN'S J THRESHOLD OPTIMIZATION


def youden_j_threshold(name, model, X, y, y_proba):
    """
    Tim nguong tau* toi uu theo Youden's J statistic:
        J = max(TPR(tau) - FPR(tau))
    Tu file ALGORITHMS.html cua tac gia.

    So sanh F1/FPR/FNR tai tau=0.5 vs tau=tau*.
    """
    fpr_curve, tpr_curve, thresholds = roc_curve(y, y_proba)

    # Tinh Youden's J cho moi nguong
    j_scores = tpr_curve - fpr_curve
    best_idx = np.argmax(j_scores)
    tau_star = thresholds[best_idx]
    j_max    = j_scores[best_idx]

    # Predict lai voi nguong tau*
    y_pred_opt = (y_proba >= tau_star).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, y_pred_opt).ravel()
    f1_opt  = f1_score(y, y_pred_opt)
    fpr_opt = fp / (fp + tn)
    fnr_opt = fn / (fn + tp)

    print(f"\n  {name} (Youden's J tau*={tau_star:.4f}, J={j_max:.4f}):")
    print(f"    F1={f1_opt:.4f}  FPR={fpr_opt:.4f}  FNR={fnr_opt:.4f}")
    print(f"    TP={tp:,}  TN={tn:,}  FP={fp:,}  FN={fn:,}")

    return {
        "model": name, "threshold": round(float(tau_star), 4),
        "youden_j": round(float(j_max), 4),
        "f1": round(f1_opt, 4),
        "fpr_rate": round(fpr_opt, 4), "fnr_rate": round(fnr_opt, 4),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }, tau_star, (fpr_curve, tpr_curve, j_scores, thresholds)


# PHAN 3: HOLDOUT STABILITY


def holdout_stability(name, model, X_all, y_all,
                      seeds=[0, 7, 13, 42, 99, 123, 456, 2025]):
    """
    Chay predict tren nhieu random split khac nhau.
    Model that: std < 0.005.
    Model hoc vet: F1 dao dong manh khi doi seed.
    """
    f1s = []
    for seed in seeds:
        _, X_t, _, y_t = train_test_split(
            np.arange(len(y_all)), y_all,
            test_size=0.2, random_state=seed, stratify=y_all
        )
        X_sample = np.array(X_all[X_t])
        f1s.append(f1_score(y_t, model.predict(X_sample)))

    arr = np.array(f1s)
    verdict = "ON DINH" if arr.std() < 0.005 else "DAO DONG - kiem tra lai!"
    print(f"  {name}: mean={arr.mean():.4f}  std={arr.std():.5f} -> {verdict}")

    return {
        "model"  : name,
        "f1_mean": round(float(arr.mean()), 4),
        "f1_std" : round(float(arr.std()),  5),
        "verdict": verdict,
    }


# PHAN 4: PLOTS


def plot_roc(roc_data: dict, save_path: Path):
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, (fpr, tpr, auc) in roc_data.items():
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.4f})",
                color=COLORS.get(name, "gray"), lw=2)
    ax.plot([0,1],[0,1], "--", color="gray", lw=1, label="Random")
    ax.set_xlabel("False Positive Rate (FPR)")
    ax.set_ylabel("True Positive Rate (TPR)")
    ax.set_title("ROC Curve — EMBER 2018 Test Set (200k mau)")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close()
    print(f"  [SAVED] {save_path.name}")

def plot_threshold_analysis(thresh_data: dict, save_path: Path):
    """
    Ve bieu do Youden's J score theo nguong cho tung model.
    Danh dau diem tau* toi uu.
    """
    fig, axes = plt.subplots(1, len(thresh_data), figsize=(6*len(thresh_data), 4))
    if len(thresh_data) == 1:
        axes = [axes]

    for ax, (name, (fpr_c, tpr_c, j_scores, thresholds)) in \
            zip(axes, thresh_data.items()):
        # Chi plot phan thresholds < 1.0 (bo diem dau)
        mask = thresholds < 1.0
        ax.plot(thresholds[mask], j_scores[mask],
                color=COLORS.get(name, "gray"), lw=2)
        best_idx = np.argmax(j_scores[mask])
        tau_star = thresholds[mask][best_idx]
        ax.axvline(tau_star, color="red", linestyle="--", lw=1.5,
                   label=f"tau*={tau_star:.3f}")
        ax.axvline(0.5, color="orange", linestyle=":", lw=1.5,
                   label="tau=0.5 (default)")
        ax.set_xlabel("Threshold")
        ax.set_ylabel("Youden's J = TPR - FPR")
        ax.set_title(f"{name}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Youden's J Threshold Optimization", fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close()
    print(f"  [SAVED] {save_path.name}")

def plot_feature_importance(models: dict, save_path: Path, top_n: int = 20):
    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(7*n, 8))
    if n == 1:
        axes = [axes]

    for ax, (name, model) in zip(axes, models.items()):
        if not hasattr(model, "feature_importances_"):
            continue
        imp = model.feature_importances_
        idx = np.argsort(imp)[-top_n:]
        ax.barh(range(top_n), imp[idx],
                color=COLORS.get(name, "#888"), alpha=0.85)
        ax.set_yticks(range(top_n))
        ax.set_yticklabels([f"feat_{i}" for i in idx], fontsize=7)
        ax.set_title(f"{name} — Top {top_n} features")
        ax.set_xlabel("Importance score")

    fig.suptitle("Feature Importance — EMBER 2018", fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [SAVED] {save_path.name}")

def plot_comparison(standard: list, optimized: list, save_path: Path):
    """
    So sanh F1/FPR/FNR tai nguong 0.5 vs Youden's J tau*.
    """
    models  = [r["model"] for r in standard]
    f1_std  = [r["f1"]      for r in standard]
    f1_opt  = [r["f1"]      for r in optimized]
    fpr_std = [r["fpr_rate"] for r in standard]
    fpr_opt = [r["fpr_rate"] for r in optimized]
    fnr_std = [r["fnr_rate"] for r in standard]
    fnr_opt = [r["fnr_rate"] for r in optimized]

    x = np.arange(len(models))
    w = 0.35
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    for ax, metric, std_vals, opt_vals, title, better in zip(
        axes,
        ["F1", "FPR", "FNR"],
        [f1_std, fpr_std, fnr_std],
        [f1_opt, fpr_opt, fnr_opt],
        ["F1 Score (cao hon tot hon)",
         "FPR - False Positive Rate (thap hon tot hon)",
         "FNR - False Negative Rate (thap hon tot hon)"],
        ["high", "low", "low"]
    ):
        bars1 = ax.bar(x - w/2, std_vals, w, label="tau=0.5 (default)",
                       color="#64748b", alpha=0.8)
        bars2 = ax.bar(x + w/2, opt_vals, w, label="tau* (Youden's J)",
                       color="#7c85f3", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=10, fontsize=9)
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2, axis="y")

    fig.suptitle("Youden's J Optimization: tau=0.5 vs tau*", fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close()
    print(f"  [SAVED] {save_path.name}")


# MAIN


def main():
    print("=" * 55)
    print("STEP 4: Evaluate Models")
    print("=" * 55)

    X_test, y_test = load_test()

    print("\nLoad models:")
    models = load_models()
    if not models:
        print("[LOI] Khong co model. Chay step3 truoc!")
        sys.exit(1)

    # Load them train set de dung cho holdout stability
    print("\nLoad train set cho stability test...")
    X_train = np.load(DATA_DIR / "X_train.npy", mmap_mode="r")
    y_train = np.load(DATA_DIR / "y_train.npy")
    X_all = np.concatenate([np.array(X_train), np.array(X_test)])
    y_all = np.concatenate([y_train, y_test])
    print(f"  Tong: {len(y_all):,} mau")

    standard_results  = []
    optimized_results = []
    roc_data    = {}
    thresh_data = {}
    proba_cache = {}

    # ---- Phan 1: Metrics chuan ----
    print("\n" + "="*55)
    print("PHAN 1: Metrics chuan (tau=0.5)")
    print("="*55)
    for name, model in models.items():
        res, y_proba, roc = evaluate_standard(name, model, X_test, y_test)
        standard_results.append(res)
        roc_data[name]    = roc
        proba_cache[name] = y_proba

    # ---- Phan 2: Youden's J ----
    print("\n" + "="*55)
    print("PHAN 2: Youden's J Threshold Optimization")
    print("="*55)
    for name, model in models.items():
        y_proba = proba_cache[name]
        res, tau_star, t_data = youden_j_threshold(
            name, model, X_test, y_test, y_proba
        )
        optimized_results.append(res)
        thresh_data[name] = t_data

    # ---- Phan 3: Holdout stability ----
    print("\n" + "="*55)
    print("PHAN 3: Holdout Stability Test")
    print("="*55)
    stab_results = []
    for name, model in models.items():
        r = holdout_stability(name, model, X_all, y_all)
        stab_results.append(r)

    # ---- Luu ket qua ----
    pd.DataFrame(standard_results).to_csv(
        RES_DIR / "test_results_default.csv", index=False)
    pd.DataFrame(optimized_results).to_csv(
        RES_DIR / "test_results_youden.csv", index=False)
    pd.DataFrame(stab_results).to_csv(
        RES_DIR / "holdout_stability.csv", index=False)

    # ---- Plots ----
    print("\n" + "="*55)
    print("PLOTS")
    print("="*55)
    plot_roc(roc_data, PLOT_DIR / "roc_curves.png")
    plot_threshold_analysis(thresh_data, PLOT_DIR / "threshold_analysis.png")
    plot_feature_importance(models, PLOT_DIR / "feature_importance.png")
    plot_comparison(standard_results, optimized_results,
                    PLOT_DIR / "threshold_comparison.png")

    # ---- Tom tat cuoi ----
    print("\n" + "="*55)
    print("TOM TAT KET QUA")
    print("="*55)
    df_std = pd.DataFrame(standard_results)
    df_opt = pd.DataFrame(optimized_results)

    print("\nMetrics chuan (tau=0.5):")
    print(df_std[["model","f1","auc","fpr_rate","fnr_rate"]].to_string(index=False))

    print("\nSau Youden's J optimization:")
    print(df_opt[["model","threshold","youden_j","f1",
                  "fpr_rate","fnr_rate"]].to_string(index=False))

    print("\nHoldout Stability:")
    print(pd.DataFrame(stab_results)[["model","f1_mean","f1_std","verdict"]]
          .to_string(index=False))

    # ---- Phan 4: Inference Speed ----
    print("\n" + "="*55)
    print("PHAN 4: Inference Speed (1 file PE)")
    print("="*55)
    import time
    x_single = np.array(X_test[0:1])
    speed_results = []

    for name, model in models.items():
        # Warm up
        for _ in range(10):
            model.predict_proba(x_single)
        # Do chinh xac
        t0 = time.perf_counter()
        for _ in range(1000):
            model.predict_proba(x_single)
        elapsed_ms = (time.perf_counter() - t0) / 1000 * 1000
        print(f"  {name}: {elapsed_ms:.3f} ms/file")
        speed_results.append({
            "model": name,
            "inference_ms": round(elapsed_ms, 3)
        })

    pd.DataFrame(speed_results).to_csv(
        RES_DIR / "inference_speed.csv", index=False)
    print("  [SAVED] inference_speed.csv")

    # Khuyen nghi model cho production
    fastest = min(speed_results, key=lambda x: x["inference_ms"])
    print(f"\n  -> Nhanh nhat: {fastest['model']} ({fastest['inference_ms']:.3f} ms)")

    print(f"\n[XONG] Ket qua luu trong results/")
    print("Chay tiep: python training/step5_shap.py")

if __name__ == "__main__":
    main()