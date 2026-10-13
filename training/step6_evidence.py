"""
Step 6: Evidence Aggregation Framework — Explainability Layer

Kiến trúc ĐÚNG:
  ML Score (XGBoost) = PHÁN QUYẾT CHÍNH   → verdict + confidence
  Evidence Engine    = GIẢI THÍCH TẠI SAO  → cái gì khiến ML nói vậy
  SHAP top features  = GIẢI THÍCH CHI TIẾT → feature nào đóng góp bao nhiêu

KHÔNG có Final = alpha*ML + beta*Evidence nữa.
Evidence không ganh với ML, Evidence phục vụ ML.

Pipeline output cho mỗi file PE:
  {
    "verdict"    : "MALWARE",          # ML >= tau_star
    "confidence" : 0.947,              # ML score thô
    "evidence"   : {
        "packing_evidence" : 0.92,     # giải thích ngữ nghĩa
        "tls_evidence"     : 0.85,
        "import_evidence"  : 0.78,
        "header_evidence"  : 0.91,
        "loadcfg_evidence" : 0.60,
    },
    "dominant_evidence": "Entropy cao + RX sections (packing/encryption)",
    "shap_top"   : [                   # SHAP contribution
        {"feature": "feat_637", "contribution": +0.50},
        ...
    ]
  }

Phần đánh giá step 6 đo:
  - Robustness: F1 theo Gaussian noise (ML model, không phải Evidence)
  - SHAP Stability: variance của SHAP qua 5 subset
  - Evidence Alignment: Evidence score có khớp chiều với SHAP không?
    (nếu SHAP nói feat_2359 quan trọng → tls_evidence phải cao)

Input : data/processed/*.npy + models/artifacts/*.pkl
Output: results/reports/evidence_results.csv
        results/reports/robustness_results.csv
        results/reports/shap_stability.csv
        results/reports/evidence_alignment.csv
        results/plots/evidence_radar.png
        results/plots/robustness_curve.png
        results/plots/shap_stability.png
        models/artifacts/evidence_config.json
"""
import sys, json, warnings
warnings.filterwarnings("ignore")

import numpy as np
import joblib
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix
from sklearn.metrics import roc_curve


# CONFIG

DATA_DIR     = Path("data/processed")
MODEL_DIR    = Path("models/artifacts")
RES_DIR      = Path("results/reports")
PLOT_DIR     = Path("results/plots")
RANDOM_STATE = 42
NOISE_LEVELS = [0.0, 0.1, 0.5, 1.0, 1.5, 2.0]
TAU_STAR     = 0.627   # Youden's J từ step 4 — KHÔNG thay đổi

COLORS = {
    "LightGBM"   : "#1D9E75",
    "XGBoost"    : "#3B8BD4",
    "RandomForest": "#D85A30",
}

# Evidence → SHAP feature mapping (dùng để đo alignment)
# feat nào trong SHAP tương ứng evidence nào
EVIDENCE_FEATURE_MAP = {
    "packing_evidence" : [503, 691],   # entropy + RX sections
    "tls_evidence"     : [2359],       # TLS data directory
    "import_evidence"  : [2364],       # Delay Import Descriptor
    "header_evidence"  : [637, 626],   # COFF characteristics + timestamp
    "loadcfg_evidence" : [2360],       # Load Config Table
}

EVIDENCE_WEIGHTS = {
    "packing_evidence" : 0.25,   # dựa trên SHAP rank step 5
    "tls_evidence"     : 0.25,
    "import_evidence"  : 0.15,
    "header_evidence"  : 0.25,
    "loadcfg_evidence" : 0.10,
}

RES_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)



# LOGGING


def setup_log():
    log_path = RES_DIR / "step6_output.txt"
    class Tee:
        def __init__(self, *f): self.files = f
        def write(self, o):
            for f in self.files: f.write(o); f.flush()
        def flush(self):
            for f in self.files: f.flush()
    lf = open(log_path, "w", encoding="utf-8")
    sys.stdout = Tee(sys.__stdout__, lf)
    return log_path



# TẦNG 1+2: PE SIGNALS → EVIDENCE  (giải thích, không phán quyết)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def extract_evidence(X: np.ndarray) -> dict:
    """
    Chuyển PE features (đã z-score) thành Evidence vector.

    Mỗi Evidence value ∈ [0, 1] — đo mức độ hiện diện của
    một dấu hiệu bảo mật cụ thể. KHÔNG phải xác suất malware.

    Lưu ý quan trọng về threshold cho binary PE fields:
    - feat_2359 (TLS), feat_2364 (Delay Import), feat_2360 (Load Config)
      là binary/sparse fields — hầu hết file PE có giá trị = 0
    - Sau StandardScaler: z-score ≈ 0 cho ~95% file
    - Dùng threshold = 0.0 (mean distribution) thay vì 1.5
      để sigmoid phân biệt được "có mặt" vs "vắng mặt"
    - Scale nhỏ hơn (0.3) để tăng độ nhạy với sự khác biệt nhỏ

    Returns dict: name → ndarray(n_samples,)
    """
    # --- Packing Evidence ---
    # Entropy cao + nhiều RX section → dấu hiệu packed/encrypted
    # feat_503 = byte entropy histogram vùng cao (continuous)
    # feat_691 = số RX sections (continuous)
    # Threshold 1.5 hợp lý vì cả hai là continuous feature
    entropy_sig    = X[:, 503]
    rx_section_sig = X[:, 691]
    packing = (sigmoid((entropy_sig    - 1.5) / 0.3) * 0.7 +
               sigmoid((rx_section_sig - 0.5) / 0.3) * 0.3)

    # --- TLS Evidence ---
    # TLS Table CÓ MẶT = anti-analysis technique (TLS callback)
    # feat_2359 = TLS data directory (binary/sparse field)
    # Threshold = 0.0: z > 0 nghĩa là TLS size trên trung bình → có mặt
    # Scale = 0.3: nhạy hơn với sự khác biệt nhỏ sau StandardScaler
    tls = sigmoid(X[:, 2359] / 0.3)

    # --- Import Evidence ---
    # Delay Import DESCRIPTOR CÓ MẶT = ẩn API calls khỏi static analysis
    # feat_2364 = Delay Import Descriptor (binary/sparse field)
    # Threshold = 0.0: z > 0 = Delay Import có mặt
    imp = sigmoid(X[:, 2364] / 0.3)

    # --- Header Evidence ---
    # Timestamp bất hợp lệ + COFF characteristics bất thường
    # feat_626 = timestamp (continuous — z thấp = timestamp nhỏ/giả)
    # feat_637 = COFF characteristics hash (continuous — z cao = flags lạ)
    # Timestamp: z-score thấp (timestamp nhỏ, epoch gần 0) → đáng ngờ
    ts_ev   = sigmoid((-X[:, 626] - 1.5) / 0.3)
    # COFF characteristics: z-score cao → flags bất thường
    coff_ev = sigmoid(( X[:, 637] - 1.0) / 0.3)
    header  = (ts_ev + coff_ev) / 2.0

    # --- Load Config Evidence ---
    # Load Config VẮNG MẶT = thiếu security compilation flags (CFG, ASLR...)
    # feat_2360 = Load Config Table (binary/sparse field)
    # z < 0 = Load Config không có → đáng ngờ (dùng -z để flip direction)
    loadcfg = sigmoid(-X[:, 2360] / 0.3)

    return {
        "packing_evidence" : packing,
        "tls_evidence"     : tls,
        "import_evidence"  : imp,
        "header_evidence"  : header,
        "loadcfg_evidence" : loadcfg,
    }


def evidence_descriptions() -> dict:
    """Mô tả ngữ nghĩa của từng evidence — dùng cho output web."""
    return {
        "packing_evidence" : "Entropy cao + nhiều RX section → file bị packed/encrypted",
        "tls_evidence"     : "TLS callback có mặt → kỹ thuật anti-analysis",
        "import_evidence"  : "Delay Import → ẩn API calls khỏi static analysis",
        "header_evidence"  : "Timestamp bất hợp lệ + COFF characteristics bất thường",
        "loadcfg_evidence" : "Load Config vắng mặt → thiếu security compilation flags",
    }


def get_dominant_evidence(ev_dict: dict, idx: int) -> str:
    """Trả về tên evidence có giá trị cao nhất cho sample idx."""
    scores = {k: float(v[idx]) for k, v in ev_dict.items()}
    dominant = max(scores, key=scores.get)
    desc = evidence_descriptions()
    return f"{dominant} ({desc[dominant]})"



# PREDICT: ML QUYẾT ĐỊNH, EVIDENCE GIẢI THÍCH


def predict_with_explanation(
    model,
    X: np.ndarray,
    shap_values: np.ndarray,
    ev_dict: dict,
    tau: float = TAU_STAR,
    top_k_shap: int = 5,
) -> list[dict]:
    """
    Output chuẩn cho mỗi sample — dùng cho web API.

    verdict    : "MALWARE" / "BENIGN"   ← ML quyết định
    confidence : float [0,1]             ← ML score
    evidence   : dict name→value         ← giải thích ngữ nghĩa
    shap_top   : list of (feature, contribution) ← feature quan trọng nhất
    """
    ml_scores = model.predict_proba(X)[:, 1]
    results   = []

    for i in range(len(X)):
        score = float(ml_scores[i])

        # Evidence vector
        ev = {name: float(vals[i]) for name, vals in ev_dict.items()}

        # SHAP top-k
        sv = shap_values[i]
        top_idx = np.argsort(np.abs(sv))[-top_k_shap:][::-1]
        shap_top = [
            {"feature"     : f"feat_{j}",
             "contribution": round(float(sv[j]), 4)}
            for j in top_idx
        ]

        results.append({
            "verdict"          : "MALWARE" if score >= tau else "BENIGN",
            "confidence"       : round(score, 4),
            "evidence"         : {k: round(v, 4) for k, v in ev.items()},
            "dominant_evidence": get_dominant_evidence(ev_dict, i),
            "shap_top"         : shap_top,
        })

    return results



# ROBUSTNESS TEST — đo độ bền ML model, không phải Evidence


def robustness_test(models: dict,
                    X_test: np.ndarray,
                    y_test: np.ndarray) -> pd.DataFrame:
    print("\n" + "=" * 55)
    print("ROBUSTNESS TEST — F1 theo Gaussian Noise")
    print("Mục đích: đo ML model có bền với malware biến thể không")
    print("=" * 55)
    np.random.seed(RANDOM_STATE)
    rows = []

    for name, model in models.items():
        print(f"\n  {name}:")
        row = {"model": name}
        for sigma in NOISE_LEVELS:
            if sigma > 0:
                noise = np.random.normal(0, sigma, X_test.shape).astype(np.float32)
                X_n   = X_test + noise
            else:
                X_n = X_test.copy()

            y_pred = model.predict(X_n)
            f1     = f1_score(y_test, y_pred)
            row[f"f1_s{sigma}"] = round(f1, 4)
            print(f"    sigma={sigma:.1f}: F1={f1:.4f}")

        f1_0   = row["f1_s0.0"]
        f1_max = row[f"f1_s{NOISE_LEVELS[-1]}"]
        drop   = (f1_0 - f1_max) / f1_0 * 100
        row["f1_baseline"] = f1_0
        row["drop_pct"]    = round(drop, 2)
        print(f"    Drop tại sigma={NOISE_LEVELS[-1]}: {drop:.2f}%")
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(RES_DIR / "robustness_results.csv", index=False)
    print("\n  [SAVED] robustness_results.csv")
    return df


def plot_robustness(df: pd.DataFrame, save_path: Path):
    fig, ax = plt.subplots(figsize=(8, 5))
    for _, row in df.iterrows():
        name = row["model"]
        f1s  = [row[f"f1_s{s}"] for s in NOISE_LEVELS]
        ax.plot(NOISE_LEVELS, f1s, "o-",
                color=COLORS.get(name, "gray"), lw=2, label=name)
        ax.annotate(f"{f1s[-1]:.3f}",
                    (NOISE_LEVELS[-1], f1s[-1]),
                    xytext=(5, 0), textcoords="offset points",
                    fontsize=9, color=COLORS.get(name, "gray"))

    ax.set_xlabel("Gaussian Noise Level (σ)")
    ax.set_ylabel("F1 Score")
    ax.set_title("Robustness — F1 theo Noise Level\n"
                 "Mô phỏng malware biến thể (obfuscation/packing)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.5, 1.01)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close()
    print(f"  [SAVED] {save_path.name}")



# SHAP STABILITY — đo SHAP ổn định qua các subset


def shap_stability(model, X_test: np.ndarray,
                   n_subsets: int = 5,
                   n_samples: int = 200) -> tuple[np.ndarray, np.ndarray]:
    import shap

    print(f"\n  SHAP Stability: {n_subsets} subsets × {n_samples} mẫu...")
    explainer    = shap.TreeExplainer(model)
    subset_means = []
    np.random.seed(RANDOM_STATE)

    for i in range(n_subsets):
        idx = np.random.choice(len(X_test), n_samples, replace=False)
        sv  = explainer.shap_values(X_test[idx])
        if isinstance(sv, list): sv = sv[1]
        elif sv.ndim == 3:       sv = sv[:, :, 1]
        subset_means.append(np.abs(sv).mean(axis=0))
        print(f"    Subset {i + 1}/{n_subsets} done")

    arr       = np.array(subset_means)          # (n_subsets, n_features)
    mean_abs  = arr.mean(axis=0)
    std_abs   = arr.std(axis=0)
    stability = np.clip(1 - std_abs / (mean_abs + 1e-8), 0, 1)

    top_idx = np.argsort(mean_abs)[-50:][::-1]
    df_stab = pd.DataFrame({
        "feature_idx"    : top_idx,
        "feature_name"   : [f"feat_{i}" for i in top_idx],
        "mean_abs_shap"  : mean_abs[top_idx].round(5),
        "std_abs_shap"   : std_abs[top_idx].round(5),
        "stability_score": stability[top_idx].round(4),
    })
    df_stab.to_csv(RES_DIR / "shap_stability.csv", index=False)
    print("  [SAVED] shap_stability.csv")

    print("\n  Top-10 theo mean |SHAP|:")
    print(f"  {'Feature':12s} {'Mean|SHAP|':>10} {'Stability':>10} {'Status':>10}")
    print(f"  {'-' * 48}")
    for _, r in df_stab.head(10).iterrows():
        status = ("STABLE"   if r.stability_score > 0.7 else
                  "OK"       if r.stability_score > 0.4 else
                  "UNSTABLE")
        print(f"  {r.feature_name:12s} {r.mean_abs_shap:>10.5f}"
              f" {r.stability_score:>10.4f} {status:>10}")

    return mean_abs, stability, df_stab


def plot_shap_stability(df_stab: pd.DataFrame, save_path: Path):
    top10 = df_stab.head(10).copy()
    colors = [("#1D9E75" if s > 0.7 else "#f59e0b" if s > 0.4 else "#ef4444")
              for s in top10.stability_score]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(top10.feature_name[::-1], top10.mean_abs_shap[::-1],
                   color=colors[::-1], alpha=0.85)
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("SHAP Stability — Top 10 Features\n"
                 "Màu: Xanh=Stable, Vàng=OK, Đỏ=Unstable")

    for bar, s in zip(bars, top10.stability_score[::-1]):
        ax.text(bar.get_width() + 1e-4, bar.get_y() + bar.get_height() / 2,
                f"stab={s:.2f}", va="center", fontsize=8)
    ax.grid(True, alpha=0.2, axis="x")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close()
    print(f"  [SAVED] {save_path.name}")



# EVIDENCE ALIGNMENT — Evidence có khớp với SHAP không?


def evidence_alignment(ev_dict: dict,
                       mean_abs_shap: np.ndarray,
                       y_test: np.ndarray) -> pd.DataFrame:
    """
    Kiểm tra: evidence nào liên kết với feature SHAP quan trọng nhất.
    Nếu Evidence và SHAP nhất quán → giải thích đáng tin cậy.

    Metric: với mỗi evidence, tính correlation giữa evidence score
    và SHAP contribution của feature tương ứng (trên malware samples).
    """
    print("\n" + "=" * 55)
    print("EVIDENCE ALIGNMENT — Kiểm tra độ nhất quán với SHAP")
    print("=" * 55)

    mal_idx = np.where(y_test == 1)[0]
    rows = []

    for ev_name, feat_indices in EVIDENCE_FEATURE_MAP.items():
        # SHAP importance của các feature liên quan
        shap_importance = mean_abs_shap[feat_indices].mean()
        # Evidence score trên malware samples
        ev_scores_mal   = ev_dict[ev_name][mal_idx].mean()

        # Rank trong toàn bộ feature set (0=nhất)
        shap_rank = int(np.sum(mean_abs_shap > shap_importance))

        row = {
            "evidence"        : ev_name,
            "features"        : str(feat_indices),
            "shap_importance" : round(float(shap_importance), 5),
            "shap_rank"       : shap_rank,
            "ev_mean_malware" : round(float(ev_scores_mal), 4),
            "weight"          : EVIDENCE_WEIGHTS[ev_name],
        }
        rows.append(row)
        print(f"  {ev_name:22s}: SHAP rank={shap_rank:4d}"
              f"  EV(mal)={ev_scores_mal:.3f}"
              f"  SHAP_imp={shap_importance:.5f}")

    df = pd.DataFrame(rows).sort_values("shap_rank")
    df.to_csv(RES_DIR / "evidence_alignment.csv", index=False)
    print("  [SAVED] evidence_alignment.csv")
    return df



# EVALUATE ML MODEL  (chỉ đo ML, không combine)


def evaluate_ml(name: str, y_score: np.ndarray,
                y_test: np.ndarray, threshold: float) -> dict:
    y_pred = (y_score >= threshold).astype(int)
    f1     = f1_score(y_test, y_pred)
    try:
        auc = roc_auc_score(y_test, y_score)
    except Exception:
        auc = 0.0
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    fpr = fp / (fp + tn)
    fnr = fn / (fn + tp)
    print(f"  {name:28s}: F1={f1:.4f}  AUC={auc:.4f}"
          f"  FPR={fpr:.4f}  FNR={fnr:.4f}")
    return {"method": name, "f1": f1, "auc": auc,
            "fpr": fpr, "fnr": fnr, "threshold": threshold}



# PLOTS


def plot_evidence_radar(ev_dict: dict, risk_scores: np.ndarray,
                        y_test: np.ndarray, save_path: Path):
    """Radar chart so sánh Evidence vector của 1 malware và 1 benign."""
    mal_idx = np.where(y_test == 1)[0][0]
    ben_idx = np.where(y_test == 0)[0][0]

    names  = list(ev_dict.keys())
    labels = [n.replace("_evidence", "").replace("_", "\n").title()
              for n in names]
    N      = len(names)
    angles = [n / float(N) * 2 * 3.14159 for n in range(N)]
    angles += angles[:1]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5),
                             subplot_kw=dict(polar=True))

    for ax, idx, title, color in zip(
        axes,
        [mal_idx, ben_idx],
        ["Mẫu MALWARE", "Mẫu BENIGN"],
        ["#ef4444", "#1D9E75"]
    ):
        vals = [float(ev_dict[n][idx]) for n in names] + \
               [float(ev_dict[names[0]][idx])]  # close loop

        ax.plot(angles, vals, color=color, lw=2)
        ax.fill(angles, vals, color=color, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylim(0, 1)
        # Risk score hiển thị để context, không phải để phán quyết
        conf = risk_scores[idx]  # dùng ML confidence ở đây nếu truyền vào
        ax.set_title(f"{title}\nML confidence={conf:.3f}",
                     pad=15, fontsize=11, color=color)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Evidence Vector — So sánh Malware vs Benign\n"
        "Evidence giải thích tại sao ML phán quyết vậy, không phán quyết thay ML",
        fontsize=12
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close()
    print(f"  [SAVED] {save_path.name}")



# SAVE CONFIG


def save_evidence_config(tau_star: float) -> dict:
    config = {
        "architecture": {
            "decision"     : "XGBoost ML score >= tau_star",
            "explanation"  : "Evidence Vector (5 dimensions)",
            "detail"       : "SHAP top features per sample",
        },
        "tau_star"          : tau_star,
        "evidence_weights"  : EVIDENCE_WEIGHTS,
        "evidence_features" : {k: v for k, v in EVIDENCE_FEATURE_MAP.items()},
        "evidence_descriptions": evidence_descriptions(),
        "note": (
            "Evidence không thay thế ML — Evidence giải thích ML. "
            "Verdict = ML >= tau_star. Confidence = ML score. "
            "Evidence = tại sao. SHAP = feature nào."
        ),
    }
    out = MODEL_DIR / "evidence_config.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"  [SAVED] evidence_config.json")
    return config



# MAIN


def main():
    log_path = setup_log()
    print("=" * 55)
    print("STEP 6: Evidence Aggregation Framework")
    print("ML quyết định — Evidence giải thích")
    print("=" * 55)
    print(f"Log: {log_path}\n")

    # --- Load data ---
    print("Load data...")
    X_test = np.array(np.load(DATA_DIR / "X_test.npy", mmap_mode="r"),
                      dtype=np.float32)
    y_test = np.load(DATA_DIR / "y_test.npy")
    X_val  = np.array(np.load(DATA_DIR / "X_val.npy",  mmap_mode="r"),
                      dtype=np.float32)
    y_val  = np.load(DATA_DIR / "y_val.npy")
    print(f"  X_test: {X_test.shape} | X_val: {X_val.shape}")

    # --- Load models ---
    print("\nLoad models...")
    models = {}
    for key, name in [("lgb", "LightGBM"),
                      ("xgb", "XGBoost"),
                      ("rf",  "RandomForest")]:
        pkl = MODEL_DIR / f"{key}.pkl"
        if pkl.exists():
            models[name] = joblib.load(pkl)
            print(f"  [OK] {name}")

    xgb_model = models["XGBoost"]

    
    # A. ROBUSTNESS TEST — ML model có bền không?
    
    df_robust = robustness_test(models, X_test, y_test)
    plot_robustness(df_robust, PLOT_DIR / "robustness_curve.png")

    
    # B. SHAP STABILITY — SHAP có ổn định không?
    
    print("\n" + "=" * 55)
    print("SHAP STABILITY SCORE (XGBoost)")
    print("=" * 55)
    mean_abs, stability, df_stab = shap_stability(xgb_model, X_test)
    plot_shap_stability(df_stab, PLOT_DIR / "shap_stability.png")

    
    # C. EVIDENCE EXTRACTION
    
    print("\n" + "=" * 55)
    print("EVIDENCE EXTRACTION")
    print("=" * 55)
    ev_test = extract_evidence(X_test)
    ev_val  = extract_evidence(X_val)

    # Summary trên malware vs benign
    mal_idx = np.where(y_test == 1)[0]
    ben_idx = np.where(y_test == 0)[0]
    print(f"\n  {'Evidence':22s}  {'Malware mean':>12}  {'Benign mean':>11}")
    print(f"  {'-' * 50}")
    rows_ev = []
    for name, vals in ev_test.items():
        m = float(vals[mal_idx].mean())
        b = float(vals[ben_idx].mean())
        diff = m - b
        print(f"  {name:22s}  {m:>12.4f}  {b:>11.4f}  Δ={diff:+.4f}")
        rows_ev.append({"evidence": name,
                        "malware_mean": round(m, 4),
                        "benign_mean": round(b, 4),
                        "delta": round(diff, 4)})

    pd.DataFrame(rows_ev).to_csv(RES_DIR / "evidence_results.csv", index=False)
    print("  [SAVED] evidence_results.csv")

    # Calibration check — Δ > 0.05 = evidence phân biệt được malware/benign
    print("\n  Calibration check (Δ > 0.05 = phân biệt được):")
    any_weak = False
    for r in rows_ev:
        ok     = abs(r["delta"]) > 0.05
        status = "OK  " if ok else "WEAK"
        note   = "" if ok else "  <-- SHAP sẽ là giải thích chính trên web"
        print(f"    {r['evidence']:22s}: Δ={r['delta']:+.4f}  [{status}]{note}")
        if not ok:
            any_weak = True
    if any_weak:
        print("\n  [WARN] Evidence yếu vẫn hiển thị nhưng web sẽ ưu tiên SHAP explain.")
    else:
        print("\n  [OK] Tất cả evidence phân biệt tốt.")

    
    # D. EVIDENCE ALIGNMENT — nhất quán với SHAP không?
    
    ev_alignment = evidence_alignment(ev_test, mean_abs, y_test)

    
    # E. ML MODEL EVALUATION (vẫn dùng tau_star từ step 4)
    
    print("\n" + "=" * 55)
    print("ML MODEL EVALUATION  (tham chiếu step 4)")
    print("=" * 55)
    ml_scores_test = xgb_model.predict_proba(X_test)[:, 1]
    ml_result      = evaluate_ml("XGBoost (tau* step4)",
                                 ml_scores_test, y_test, TAU_STAR)
    pd.DataFrame([ml_result]).to_csv(
        RES_DIR / "final_score_results.csv", index=False)
    print("  [SAVED] final_score_results.csv")

    
    # F. SAMPLE OUTPUT — ví dụ output cho 3 malware đầu tiên
    
    print("\n" + "=" * 55)
    print("SAMPLE OUTPUT (3 malware mẫu đầu tiên)")
    print("Đây là output mà web API sẽ trả về")
    print("=" * 55)

    import shap
    explainer  = shap.TreeExplainer(xgb_model)
    sv_test    = explainer.shap_values(X_test[:100])  # chỉ lấy 100 mẫu
    if isinstance(sv_test, list): sv_test = sv_test[1]
    elif sv_test.ndim == 3:       sv_test = sv_test[:, :, 1]

    ev_test_small = {k: v[:100] for k, v in ev_test.items()}
    predictions   = predict_with_explanation(
        xgb_model, X_test[:100], sv_test, ev_test_small,
        tau=TAU_STAR, top_k_shap=5
    )

    # In 3 malware mẫu
    shown = 0
    for i, pred in enumerate(predictions):
        if pred["verdict"] == "MALWARE" and shown < 3:
            print(f"\n  Sample {i}:")
            print(f"    verdict     : {pred['verdict']}")
            print(f"    confidence  : {pred['confidence']:.4f}")
            print(f"    evidence    :")
            for k, v in pred["evidence"].items():
                print(f"      {k:22s}: {v:.4f}")
            print(f"    dominant    : {pred['dominant_evidence']}")
            print(f"    shap top 5  :")
            for s in pred["shap_top"]:
                sign = "+" if s["contribution"] > 0 else ""
                print(f"      {s['feature']:12s}: {sign}{s['contribution']:.4f}")
            shown += 1

    
    # G. PLOTS
    
    print("\nPlots...")
    ml_conf_test = xgb_model.predict_proba(X_test)[:, 1]
    plot_evidence_radar(ev_test, ml_conf_test, y_test,
                        PLOT_DIR / "evidence_radar.png")

    
    # H. SAVE CONFIG
    
    config = save_evidence_config(TAU_STAR)

    
    # TÓM TẮT
    
    print("\n" + "=" * 55)
    print("TÓM TẮT STEP 6")
    print("=" * 55)

    print("\nRobustness drop tại sigma=2.0:")
    for _, r in df_robust.iterrows():
        print(f"  {r.model:15s}: F1_baseline={r.f1_baseline:.4f}"
              f"  drop={r.drop_pct:.2f}%")

    print("\nEvidence Alignment (SHAP rank thấp = quan trọng hơn):")
    for _, r in ev_alignment.sort_values("shap_rank").iterrows():
        print(f"  {r.evidence:22s}: SHAP rank={r.shap_rank:4d}"
              f"  EV(mal)={r.ev_mean_malware:.3f}")

    print(f"\nML Decision threshold (tau*): {TAU_STAR}")
    print(f"ML F1 trên test set: {ml_result['f1']:.4f}")
    print(f"ML AUC trên test set: {ml_result['auc']:.4f}")

    print("\nKiến trúc output web API:")
    print("  verdict     ← ML >= tau_star  (quyết định)")
    print("  confidence  ← ML score        (tin cậy)")
    print("  evidence    ← 5 giá trị       (giải thích ngữ nghĩa)")
    print("  shap_top    ← top features    (giải thích kỹ thuật)")

    print(f"\n[XONG] Bước tiếp: xây dựng web/ API dùng evidence_config.json")


if __name__ == "__main__":
    main()