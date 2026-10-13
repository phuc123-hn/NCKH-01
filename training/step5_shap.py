"""
Step 5: SHAP Analysis + Composite Malware Score (CMS)
- SHAP TreeExplainer cho 3 model
- Tim top features overlap giua 3 model
- De xuat cong thuc CMS tu SHAP weights
- So sanh CMS vs full features
- Luu SHAP values cho web su dung
Input : data/processed/X_test.npy, y_test.npy + models/artifacts/*.pkl
Output: results/reports/shap_top_features.csv
        results/reports/cms_results.csv
        results/reports/shap_values_xgb.npy  (dung cho web)
        results/plots/shap_summary.png
        results/plots/shap_waterfall_malware.png
        results/plots/shap_waterfall_benign.png
        results/plots/cms_comparison.png
        models/artifacts/cms_config.json
"""
import sys
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import joblib
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression


DATA_DIR  = Path("data/processed")
MODEL_DIR = Path("models/artifacts")
RES_DIR   = Path("results/reports")
PLOT_DIR  = Path("results/plots")

# So mau dung cho SHAP (du dai dien, khong qua ton RAM)
N_SHAP_SAMPLES = 1000
RANDOM_STATE   = 42
TOP_N_FEATURES = 50   # lay top 50 de phan tich
CMS_N_FEATURES = 20   # dung top 20 de xay CMS


RES_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

COLORS = {
    "LightGBM"   : "#1D9E75",
    "XGBoost"    : "#3B8BD4",
    "RandomForest": "#D85A30",
}


# KIEM TRA SHAP

def check_shap():
    try:
        import shap
        print(f"[OK] shap {shap.__version__}")
        return shap
    except ImportError:
        print("[LOI] Chua cai shap. Chay: pip install shap")
        sys.exit(1)


# PHAN 1: SHAP ANALYSIS

def compute_shap(shap, name, model, X_sample, y_sample):
    """
    Tinh SHAP values bang TreeExplainer.
    Chi dung N_SHAP_SAMPLES mau de tiet kiem RAM + thoi gian.
    """
    print(f"\n  {name}: tinh SHAP tren {len(X_sample)} mau...")
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # Xu ly nhieu kieu output khac nhau:
    # - LightGBM: list [benign_shap, malware_shap] moi cai (n, feat)
    # - XGBoost : ndarray (n, feat)
    # - RF      : ndarray (n, feat, 2) hoac list
    if isinstance(shap_values, list):
        sv = shap_values[1]          # lay class malware
    elif shap_values.ndim == 3:
        sv = shap_values[:, :, 1]    # (n, feat, 2) -> lay class 1
    else:
        sv = shap_values             # (n, feat) binh thuong

    # Mean absolute SHAP value cho tung feature
    mean_abs = np.abs(sv).mean(axis=0)

    print(f"    SHAP shape: {sv.shape}")
    print(f"    Top 5 features: {np.argsort(mean_abs)[-5:][::-1].tolist()}")

    return sv, mean_abs

def get_top_features(mean_abs_dict: dict, top_n: int) -> pd.DataFrame:
    """
    Tong hop top features tu ca 3 model.
    Feature xuat hien trong nhieu model = discriminative that su.
    """
    rows = []
    for name, mean_abs in mean_abs_dict.items():
        idx_sorted = np.argsort(mean_abs)[::-1][:top_n]
        for rank, idx in enumerate(idx_sorted, 1):
            rows.append({
                "model"      : name,
                "feature_idx": int(idx),
                "feature_name": f"feat_{idx}",
                "mean_abs_shap": round(float(mean_abs[idx]), 6),
                "rank"       : rank,
            })
    return pd.DataFrame(rows)

def find_overlap_features(df_shap: pd.DataFrame, top_n: int) -> list:
    """
    Tim features xuat hien trong top_n cua ca 3 model.
    Day la features thuc su quan trong, khong phu thuoc vao model.
    """
    sets = []
    for model_name in df_shap["model"].unique():
        top_idx = set(
            df_shap[df_shap["model"] == model_name]
            .nsmallest(top_n, "rank")["feature_idx"]
            .tolist()
        )
        sets.append(top_idx)

    overlap = set.intersection(*sets)
    print(f"\n  Features overlap trong top-{top_n} cua ca 3 model: {len(overlap)}")
    print(f"  Indices: {sorted(overlap)}")
    return sorted(overlap)


# PHAN 2: COMPOSITE MALWARE SCORE (CMS)


def build_cms(shap_values_xgb: np.ndarray, X_sample: np.ndarray,
              y_sample: np.ndarray, top_n: int = CMS_N_FEATURES):
    """
    Xay dung Composite Malware Score tu SHAP weights cua XGBoost.

    Cong thuc:
        CMS(x) = sum_i [ w_i * x_i ]  voi i trong top-N features
        w_i = mean_abs_SHAP_i / sum(mean_abs_SHAP)  (normalized weight)

    Sau do dung Logistic Regression tren CMS features de so sanh voi full model.
    """
    print(f"\n  Xay CMS tu top-{top_n} SHAP features cua XGBoost...")

    # Tinh SHAP weights
    mean_abs = np.abs(shap_values_xgb).mean(axis=0)
    top_idx  = np.argsort(mean_abs)[::-1][:top_n]
    weights  = mean_abs[top_idx]
    weights  = weights / weights.sum()  # normalize

    # CMS feature vector: chi dung top_n features
    X_cms = X_sample[:, top_idx]

    # Train Logistic Regression tren CMS features
    # (nhe, nhanh, de giai thich hon full model)
    lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    lr.fit(X_cms, y_sample)
    y_pred_cms  = lr.predict(X_cms)
    y_proba_cms = lr.predict_proba(X_cms)[:, 1]

    f1_cms  = f1_score(y_sample, y_pred_cms)
    auc_cms = roc_auc_score(y_sample, y_proba_cms)

    print(f"    CMS (LR, {top_n} features): F1={f1_cms:.4f}  AUC={auc_cms:.4f}")

    # Luu config cho web
    cms_config = {
        "top_feature_indices": top_idx.tolist(),
        "shap_weights"       : weights.tolist(),
        "n_features"         : int(top_n),
        "model_used"         : "XGBoost",
        "description"        : (
            f"CMS = weighted sum of top-{top_n} SHAP features from XGBoost. "
            "Weights normalized from mean absolute SHAP values."
        )
    }

    return {
        "f1_cms" : round(f1_cms,  4),
        "auc_cms": round(auc_cms, 4),
        "top_idx": top_idx,
        "weights": weights,
    }, cms_config


# PHAN 3: PLOTS


def plot_shap_summary(shap_module, shap_values: np.ndarray,
                      X_sample: np.ndarray, name: str,
                      save_path: Path, top_n: int = 20):
    """SHAP summary bar plot — top features theo mean |SHAP|."""
    fig, ax = plt.subplots(figsize=(8, 7))
    mean_abs = np.abs(shap_values).mean(axis=0)
    idx = np.argsort(mean_abs)[-top_n:]

    ax.barh(range(top_n), mean_abs[idx],
            color=COLORS.get(name, "#888"), alpha=0.85)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([f"feat_{i}" for i in idx], fontsize=8)
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(f"SHAP Feature Importance — {name}\nTop {top_n} features")
    ax.grid(True, alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close()
    print(f"    [SAVED] {save_path.name}")

def plot_waterfall_sample(shap_values: np.ndarray, X_sample: np.ndarray,
                           y_sample: np.ndarray, label: int,
                           name: str, save_path: Path, top_n: int = 15):
    """
    Waterfall plot cho 1 mau cu the (malware hoac benign).
    Giai thich tai sao model phan loai nhu vay.
    """
    # Tim mau dau tien co label mong muon
    indices = np.where(y_sample == label)[0]
    if len(indices) == 0:
        return
    idx = indices[0]

    sv    = shap_values[idx]
    x_val = X_sample[idx]

    # Top features theo |SHAP| cua mau nay
    top_feat = np.argsort(np.abs(sv))[-top_n:][::-1]
    feat_names  = [f"feat_{i}" for i in top_feat]
    feat_shap   = sv[top_feat]
    feat_values = x_val[top_feat]

    colors = ["#ef4444" if s > 0 else "#3B8BD4" for s in feat_shap]

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.barh(range(top_n), feat_shap[::-1],
                   color=colors[::-1], alpha=0.85)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(
        [f"{feat_names[::-1][i]} (val={feat_values[::-1][i]:.3f})"
         for i in range(top_n)],
        fontsize=8
    )
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("SHAP value (do= do tac dong den quyet dinh)")
    label_str = "MALWARE" if label == 1 else "BENIGN"
    ax.set_title(
        f"SHAP Waterfall — {name} — Mau {label_str}\n"
        f"Do: (+) day ve malware | (-) day ve benign"
    )
    ax.grid(True, alpha=0.2, axis="x")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close()
    print(f"    [SAVED] {save_path.name}")

def plot_cms_comparison(full_results: dict, cms_result: dict,
                         save_path: Path):
    """So sanh F1/AUC: Full features vs CMS top-N features."""
    labels   = ["Full (2381 feat)", f"CMS (top-{CMS_N_FEATURES} feat)"]
    f1_vals  = [full_results["f1"],  cms_result["f1_cms"]]
    auc_vals = [full_results["auc"], cms_result["auc_cms"]]

    x = np.arange(2)
    w = 0.35
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))

    for ax, vals, title, color in zip(
        axes,
        [f1_vals, auc_vals],
        ["F1 Score", "AUC-ROC"],
        ["#3B8BD4", "#1D9E75"]
    ):
        bars = ax.bar(x, vals, w*2, color=[color, "#a78bfa"], alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylim(min(vals)*0.98, 1.0)
        ax.set_title(title)
        ax.grid(True, alpha=0.2, axis="y")
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f"{val:.4f}", ha="center", va="bottom", fontsize=9)

    fig.suptitle(
        f"CMS vs Full Features (XGBoost)\n"
        f"CMS dung top-{CMS_N_FEATURES} SHAP-weighted features thay vi 2381",
        fontsize=11
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close()
    print(f"    [SAVED] {save_path.name}")

def plot_overlap_features(df_shap: pd.DataFrame, overlap: list,
                           save_path: Path, top_n: int = 20):
    """Heatmap: SHAP rank cua tung feature overlap trong 3 model."""
    if not overlap:
        return

    models = df_shap["model"].unique().tolist()
    rank_data = {}
    for feat_idx in overlap[:top_n]:
        row = {}
        for m in models:
            sub = df_shap[
                (df_shap["model"] == m) &
                (df_shap["feature_idx"] == feat_idx)
            ]
            row[m] = int(sub["rank"].values[0]) if len(sub) > 0 else 999
        rank_data[f"feat_{feat_idx}"] = row

    df_heat = pd.DataFrame(rank_data).T
    fig, ax = plt.subplots(figsize=(7, max(4, len(rank_data)*0.4 + 2)))
    im = ax.imshow(df_heat.values, cmap="YlOrRd_r", aspect="auto")
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, fontsize=9)
    ax.set_yticks(range(len(df_heat)))
    ax.set_yticklabels(df_heat.index.tolist(), fontsize=8)
    for i in range(len(df_heat)):
        for j in range(len(models)):
            ax.text(j, i, str(df_heat.values[i, j]),
                    ha="center", va="center", fontsize=7, color="black")
    plt.colorbar(im, ax=ax, label="Rank (thap = quan trong hon)")
    ax.set_title("Rank cua overlap features trong 3 model\n(mau xanh = rank cao = quan trong)")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close()
    print(f"    [SAVED] {save_path.name}")


# MAIN


def main():
    # Ghi log ra file
    import sys
    log_path = RES_DIR / "step5_output.txt"
    class Tee:
        def __init__(self, *files):
            self.files = files
        def write(self, obj):
            for f in self.files:
                f.write(obj); f.flush()
        def flush(self):
            for f in self.files: f.flush()
    log_file = open(log_path, "w", encoding="utf-8")
    sys.stdout = Tee(sys.__stdout__, log_file)

    print("=" * 55)
    print("STEP 5: SHAP Analysis + Composite Malware Score")
    print("=" * 55)
    print(f"Log: {log_path}\n")

    shap = check_shap()

    # Load data
    print("Load data...")
    X_test = np.load(DATA_DIR / "X_test.npy", mmap_mode="r")
    y_test = np.load(DATA_DIR / "y_test.npy")

    # Lay mau balanced (500 malware + 500 benign)
    np.random.seed(RANDOM_STATE)
    mal_idx = np.where(y_test == 1)[0]
    ben_idx = np.where(y_test == 0)[0]
    n_each  = N_SHAP_SAMPLES // 2
    sample_idx = np.concatenate([
        np.random.choice(mal_idx, n_each, replace=False),
        np.random.choice(ben_idx, n_each, replace=False),
    ])
    np.random.shuffle(sample_idx)

    X_sample = np.array(X_test[sample_idx], dtype=np.float32)
    y_sample = y_test[sample_idx]
    print(f"  SHAP sample: {X_sample.shape}")
    print(f"  Malware: {(y_sample==1).sum()} | Benign: {(y_sample==0).sum()}")

    # Load models
    print("\nLoad models...")
    models = {}
    for key, name in [("lgb","LightGBM"),("xgb","XGBoost"),("rf","RandomForest")]:
        pkl = MODEL_DIR / f"{key}.pkl"
        if pkl.exists():
            models[name] = joblib.load(pkl)
            print(f"  [OK] {name}")

    # ---- Phan 1: SHAP ----
    print("\n" + "="*55)
    print("PHAN 1: Tinh SHAP values")
    print("="*55)

    shap_values_dict = {}
    mean_abs_dict    = {}

    for name, model in models.items():
        sv, mean_abs = compute_shap(shap, name, model, X_sample, y_sample)
        shap_values_dict[name] = sv
        mean_abs_dict[name]    = mean_abs

        # Ve SHAP summary cho tung model
        plot_shap_summary(
            shap, sv, X_sample, name,
            PLOT_DIR / f"shap_summary_{name.lower()}.png"
        )
        # Ve waterfall cho 1 malware va 1 benign
        plot_waterfall_sample(
            sv, X_sample, y_sample, label=1, name=name,
            save_path=PLOT_DIR / f"shap_waterfall_{name.lower()}_malware.png"
        )
        plot_waterfall_sample(
            sv, X_sample, y_sample, label=0, name=name,
            save_path=PLOT_DIR / f"shap_waterfall_{name.lower()}_benign.png"
        )

    # Luu SHAP values cua XGBoost cho web
    np.save(RES_DIR / "shap_values_xgb.npy", shap_values_dict["XGBoost"])
    print("\n  [SAVED] shap_values_xgb.npy")

    # Top features moi model
    df_shap = get_top_features(mean_abs_dict, TOP_N_FEATURES)
    df_shap.to_csv(RES_DIR / "shap_top_features.csv", index=False)
    print("  [SAVED] shap_top_features.csv")

    # Overlap features
    overlap = find_overlap_features(df_shap, top_n=20)
    plot_overlap_features(df_shap, overlap,
                          PLOT_DIR / "shap_overlap_features.png")

    # ---- Phan 2: CMS ----
    print("\n" + "="*55)
    print("PHAN 2: Composite Malware Score (CMS)")
    print("="*55)

    cms_result, cms_config = build_cms(
        shap_values_dict["XGBoost"], X_sample, y_sample, top_n=CMS_N_FEATURES
    )

    # Full XGBoost result tren cung mau
    xgb_model = models["XGBoost"]
    y_pred_full  = xgb_model.predict(X_sample)
    y_proba_full = xgb_model.predict_proba(X_sample)[:, 1]
    full_result = {
        "f1" : round(f1_score(y_sample, y_pred_full), 4),
        "auc": round(roc_auc_score(y_sample, y_proba_full), 4),
    }
    print(f"  XGBoost full (2381 feat): F1={full_result['f1']:.4f} AUC={full_result['auc']:.4f}")

    # So sanh
    df_cms = pd.DataFrame([
        {"method": f"XGBoost full (2381)",
         "f1": full_result["f1"], "auc": full_result["auc"]},
        {"method": f"CMS top-{CMS_N_FEATURES} SHAP",
         "f1": cms_result["f1_cms"], "auc": cms_result["auc_cms"]},
    ])
    df_cms.to_csv(RES_DIR / "cms_results.csv", index=False)
    print("  [SAVED] cms_results.csv")

    # Luu CMS config cho web
    with open(MODEL_DIR / "cms_config.json", "w") as f:
        json.dump(cms_config, f, indent=2)
    print("  [SAVED] cms_config.json")

    # Plot CMS comparison
    print("\n  Plots:")
    plot_cms_comparison(full_result, cms_result,
                        PLOT_DIR / "cms_comparison.png")

    # ---- Tom tat ----
    print("\n" + "="*55)
    print("TOM TAT STEP 5")
    print("="*55)

    print("\nTop-10 features quan trong nhat (overlap 3 model):")
    if overlap:
        for feat_idx in overlap[:10]:
            row = df_shap[df_shap["feature_idx"] == feat_idx]
            print(f"  feat_{feat_idx}:")
            for _, r in row.iterrows():
                print(f"    {r['model']:15s} rank={r['rank']:3d}  "
                      f"mean_shap={r['mean_abs_shap']:.5f}")

    print(f"\nCMS vs Full XGBoost:")
    print(f"  Full (2381 feat): F1={full_result['f1']:.4f}  AUC={full_result['auc']:.4f}")
    print(f"  CMS  ({CMS_N_FEATURES:4d} feat): F1={cms_result['f1_cms']:.4f}  AUC={cms_result['auc_cms']:.4f}")

    gap_f1 = full_result['f1'] - cms_result['f1_cms']
    if abs(gap_f1) < 0.01:
        print(f"  -> CMS dat {(1-abs(gap_f1)/full_result['f1'])*100:.1f}% hieu suat chi voi {CMS_N_FEATURES} features")
    else:
        print(f"  -> Full model tot hon {gap_f1:.4f} F1 — CMS la approximation hop ly")

    print(f"\n[XONG] Tat ca luu trong results/")
    print("Buoc tiep theo: xay dung web/")

if __name__ == "__main__":
    main()