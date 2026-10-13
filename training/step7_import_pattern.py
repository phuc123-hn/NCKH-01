"""
Step 7: Import Pattern Evidence Engine

Vấn đề step 6 giải quyết:
  Evidence Vector (tls, import, header...) dựa trên feature hash
  → không phân biệt được malware/benign ở population mean (Δ ≈ 0)
  → vì sparse binary fields bị làm phẳng sau StandardScaler

Giải pháp step 7:
  Dùng lief đọc import table THẬT từ file PE
  → map tên API về threat category có ngữ nghĩa bảo mật
  → tạo Import Pattern Score độc lập với feature hash

Kiến trúc output cho web API (bổ sung vào step 6):
  verdict          ← ML >= tau_star           (quyết định)
  confidence       ← ML score                 (tin cậy)
  evidence         ← 5 giá trị sigmoid        (giải thích tổng quát)
  shap_top         ← top features per-sample  (giải thích kỹ thuật)
  import_patterns  ← tên API thật + category  (MỚI — giải thích cụ thể)
  threat_hint      ← chuỗi mô tả mối đe dọa  (MỚI — dễ hiểu cho người dùng)

Quan trọng:
  Import Pattern không phán quyết — chỉ cung cấp thêm ngữ cảnh.
  Verdict vẫn do XGBoost quyết định.

Input : data/processed/*.npy + models/artifacts/*.pkl
        data/ember2018/train_features_*.jsonl (để đánh giá trên EMBER)
Output: results/reports/import_pattern_results.csv
        results/reports/step7_output.txt
        results/plots/import_pattern_coverage.png
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
from collections import defaultdict


# CONFIG

MODEL_DIR = Path("models/artifacts")
DATA_DIR  = Path("data/processed")
RES_DIR   = Path("results/reports")
PLOT_DIR  = Path("results/plots")
EMBER_DIR = Path("data/ember2018")
TAU_STAR  = 0.627

RES_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)


# IMPORT PATTERN DICTIONARY
# Dựa trên:
#   - Kết quả debug_imports.py: APIs thật trong EMBER malware
#   - MITRE ATT&CK technique mapping
#   - Kết quả test_real_pe.py: APIs trong Windows system files
#
# Mỗi category có weight phản ánh mức độ nguy hiểm:
#   weight cao = ít xuất hiện trong benign, nhiều trong malware

IMPORT_PATTERNS = {
    # ----------------------------------------------------------------
    # Tier 1 — Rất đặc trưng cho malware, hiếm trong benign hợp lệ
    # Các API này có mục đích rất cụ thể, benign app hầu như không dùng
    # ----------------------------------------------------------------
    "process_injection": {
        "weight"  : 0.95,
        "apis"    : ["VirtualAllocEx", "WriteProcessMemory",
                     "CreateRemoteThread", "NtUnmapViewOfSection",
                     "ZwUnmapViewOfSection", "RtlCreateUserThread",
                     "NtCreateThreadEx"],
        "desc"    : "Process injection — chèn code vào process khác",
        "mitre"   : "T1055",
        # Loại bỏ: không có API nào generic ở đây
    },
    "keylogger": {
        "weight"  : 0.90,
        "apis"    : ["SetWindowsHookEx", "GetAsyncKeyState",
                     "CallNextHookEx"],
        "desc"    : "Keylogger — hook keyboard để ghi lại phím bấm",
        "mitre"   : "T1056.001",
        # Loại bỏ: GetKeyboardState (dùng trong UI apps), RegisterHotKey
    },
    "network_download": {
        "weight"  : 0.85,
        "apis"    : ["URLDownloadToFile", "URLDownloadToCacheFile",
                     "WinHttpSendRequest", "InternetOpenUrl",
                     "InternetReadFile", "HttpSendRequest"],
        "desc"    : "Network download — tải payload từ internet",
        "mitre"   : "T1105",
        # Loại bỏ: InternetConnectW/A (quá generic, dùng trong browser)
    },
    "anti_analysis": {
        "weight"  : 0.80,
        "apis"    : ["CheckRemoteDebuggerPresent",
                     "NtQueryInformationProcess",
                     "ZwQueryInformationProcess"],
        "desc"    : "Anti-analysis — kiểm tra debugger/sandbox chủ động",
        "mitre"   : "T1622",
        # Loại bỏ hoàn toàn:
        #   IsDebuggerPresent  → có trong hầu hết Windows app (CRT runtime)
        #   OutputDebugString  → dùng trong logging của benign app
        #   FindWindow         → dùng trong UI app thông thường
        #   GetTickCount       → dùng để đo thời gian, cực kỳ phổ biến
        # Chỉ giữ API thực sự chỉ malware mới gọi trực tiếp
    },
    "persistence_service": {
        "weight"  : 0.75,
        "apis"    : ["CreateService", "StartServiceCtrlDispatcher",
                     "AddMandatoryAce", "SchtasksCreate"],
        "desc"    : "Persistence via service — tạo Windows service để tồn tại",
        "mitre"   : "T1543.003",
        # Tách RegSetValueEx thành category riêng vì quá generic
        # CreateService/SchtasksCreate đặc trưng hơn nhiều
    },
    "persistence_registry": {
        "weight"  : 0.45,
        "apis"    : ["RegSetValueExA", "RegSetValueExW"],
        "desc"    : "Registry modification — ghi vào registry (cần context thêm)",
        "mitre"   : "T1547.001",
        # Weight thấp vì RegSetValueEx có trong benign (alg.exe, notepad.exe)
        # Chỉ có ý nghĩa khi kết hợp với API khác
        # Dùng exact name để tránh match quá rộng
    },
    "crypto_ransomware": {
        "weight"  : 0.80,
        "apis"    : ["CryptEncrypt", "CryptDecrypt",
                     "BCryptEncrypt", "BCryptDecrypt",
                     "CryptGenRandom"],
        "desc"    : "Crypto operation — mã hóa dữ liệu (ransomware pattern)",
        "mitre"   : "T1486",
        # Loại bỏ: CryptAcquireContext (dùng trong TLS/HTTPS của benign)
        # Loại bỏ: CryptBinaryToString (dùng trong base64 conversion)
    },
    "socket_raw": {
        "weight"  : 0.65,
        "apis"    : ["WSAConnect", "WSASend", "WSARecv",
                     "WSAStartup"],
        "desc"    : "Raw socket — kết nối mạng cấp thấp qua Winsock",
        "mitre"   : "T1095",
        # Loại bỏ: socket, connect, send, recv, bind, listen, accept
        # Những tên đó quá ngắn → match nhầm vào tên API khác chứa substring
        # WSA* cụ thể hơn và đặc trưng hơn
    },
}



# IMPORT PATTERN SCORING


def compute_import_pattern_score(api_names: list[str]) -> dict:
    """
    Tính Import Pattern Score từ danh sách tên API thật.

    Returns:
      score      : float [0, 1] — tổng hợp có trọng số
      hits       : dict category → list API names matched
      threat_hint: str — mô tả mối đe dọa chính
      n_categories: int — số threat categories bị kích hoạt
    """
    if not api_names:
        return {
            "score"        : 0.0,
            "hits"         : {},
            "threat_hint"  : "Không có import table",
            "n_categories" : 0,
        }

    api_lower = [a.lower() for a in api_names]
    hits      = {}

    for category, cfg in IMPORT_PATTERNS.items():
        matched = [
            api for api in api_names
            if any(p.lower() in api.lower() for p in cfg["apis"])
        ]
        if matched:
            hits[category] = matched[:4]

    if not hits:
        return {
            "score"        : 0.0,
            "hits"         : {},
            "threat_hint"  : "Không phát hiện suspicious API pattern",
            "n_categories" : 0,
        }

    # Score = weighted sum, chuẩn hóa về [0,1]
    # Dùng max weight trong các category bị hit làm dominant
    # Cộng thêm bonus cho multi-category (nhiều category = phức tạp hơn)
    max_weight   = max(IMPORT_PATTERNS[c]["weight"] for c in hits)
    n_cats       = len(hits)
    combo_bonus  = min(0.15, (n_cats - 1) * 0.05)  # tối đa +0.15
    score        = min(1.0, max_weight + combo_bonus)

    # Threat hint — mô tả category nguy hiểm nhất
    dominant_cat = max(hits, key=lambda c: IMPORT_PATTERNS[c]["weight"])
    cfg_dominant = IMPORT_PATTERNS[dominant_cat]

    if n_cats == 1:
        threat_hint = cfg_dominant["desc"]
    else:
        other_cats = [c for c in hits if c != dominant_cat]
        other_desc = ", ".join(other_cats[:2])
        threat_hint = f"{cfg_dominant['desc']} + {other_desc}"

    return {
        "score"        : round(score, 4),
        "hits"         : hits,
        "threat_hint"  : threat_hint,
        "n_categories" : n_cats,
        "dominant"     : dominant_cat,
        "mitre"        : cfg_dominant["mitre"],
    }



# EVALUATE TRÊN EMBER JSONL (không cần file PE thật)
# Đọc imports từ JSONL đã có sẵn trong data/ember2018/


def evaluate_on_ember(n_samples: int = 5000) -> pd.DataFrame:
    """
    Đánh giá Import Pattern Score trên EMBER dataset.
    Đọc trực tiếp từ JSONL — không cần lief.

    Mục đích:
      - Đo tỷ lệ malware vs benign có suspicious APIs
      - So sánh Import Pattern Score giữa 2 class
      - Xác nhận pattern có discriminative value
    """
    print(f"\n  Đọc {n_samples} mẫu từ EMBER JSONL...")
    rows   = []
    counts = defaultdict(int)

    jsonl_files = sorted(EMBER_DIR.glob("train_features_*.jsonl"))
    if not jsonl_files:
        print("  [WARN] Không tìm thấy JSONL files trong data/ember2018/")
        return pd.DataFrame()

    # Bỏ qua file 0 (train_features_0.jsonl chỉ có 50k unlabeled/benign)
    # Bắt đầu từ file 1 trở đi để có cả malware và benign
    jsonl_files = [f for f in jsonl_files
                   if f.name != "train_features_0.jsonl"]
    print(f"  Đọc từ {len(jsonl_files)} file (bỏ file 0 toàn unlabeled)...")

    seen = 0
    for jsonl_path in jsonl_files:
        if seen >= n_samples:
            break
        with open(jsonl_path, "r", errors="replace") as f:
            for line in f:
                if seen >= n_samples:
                    break
                try:
                    obj    = json.loads(line.strip())
                    label  = int(obj.get("label", -1))
                    if label not in (0, 1):
                        continue

                    # Flatten import functions
                    imports    = obj.get("imports", {})
                    api_names  = []
                    for dll, funcs in imports.items():
                        if isinstance(funcs, list):
                            api_names.extend(funcs)

                    result = compute_import_pattern_score(api_names)

                    rows.append({
                        "sha256"       : obj.get("sha256", "")[:16],
                        "label"        : label,
                        "n_apis"       : len(api_names),
                        "ip_score"     : result["score"],
                        "n_categories" : result["n_categories"],
                        "dominant"     : result.get("dominant", "none"),
                        "threat_hint"  : result["threat_hint"],
                    })
                    counts[label] += 1
                    seen += 1

                except Exception:
                    continue

    df = pd.DataFrame(rows)
    print(f"  Đọc được: {len(df)} mẫu "
          f"(malware={counts[1]}, benign={counts[0]})")
    return df


def analyze_ember_results(df: pd.DataFrame) -> dict:
    """Phân tích kết quả trên EMBER — tính Δ và coverage."""
    if df.empty:
        return {}

    mal = df[df.label == 1]
    ben = df[df.label == 0]

    # Import Pattern Score mean
    ip_mal  = mal.ip_score.mean()
    ip_ben  = ben.ip_score.mean()
    delta   = ip_mal - ip_ben

    # Coverage: % mẫu có ít nhất 1 suspicious category
    cov_mal = (mal.n_categories > 0).mean()
    cov_ben = (ben.n_categories > 0).mean()

    # Top categories trong malware
    top_cats = (mal[mal.dominant != "none"]
                .dominant.value_counts().head(5))

    print(f"\n  Import Pattern Score:")
    print(f"    Malware mean : {ip_mal:.4f}")
    print(f"    Benign  mean : {ip_ben:.4f}")
    print(f"    Δ            : {delta:+.4f}  "
          f"({'OK >0.05' if delta > 0.05 else 'WEAK'})")

    print(f"\n  Coverage (có suspicious APIs):")
    print(f"    Malware : {cov_mal*100:.1f}%")
    print(f"    Benign  : {cov_ben*100:.1f}%")

    print(f"\n  Top threat categories trong malware:")
    for cat, cnt in top_cats.items():
        pct = cnt / len(mal) * 100
        print(f"    {cat:25s}: {cnt:5d} ({pct:.1f}%)")

    return {
        "ip_score_malware" : round(ip_mal,  4),
        "ip_score_benign"  : round(ip_ben,  4),
        "delta"            : round(delta,   4),
        "coverage_malware" : round(cov_mal, 4),
        "coverage_benign"  : round(cov_ben, 4),
    }



# PLOT


def plot_import_pattern_coverage(df: pd.DataFrame,
                                 stats: dict,
                                 save_path: Path):
    """
    Biểu đồ so sánh Import Pattern giữa malware và benign.
    2 subplot: score distribution + category breakdown.
    """
    if df.empty:
        return

    mal = df[df.label == 1]
    ben = df[df.label == 0]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # --- Subplot 1: Score distribution ---
    ax = axes[0]
    bins = np.linspace(0, 1, 25)
    ax.hist(ben.ip_score, bins=bins, alpha=0.6,
            color="#1D9E75", label=f"Benign (n={len(ben):,})")
    ax.hist(mal.ip_score, bins=bins, alpha=0.6,
            color="#ef4444", label=f"Malware (n={len(mal):,})")
    ax.axvline(stats.get("ip_score_malware", 0),
               color="#ef4444", ls="--", lw=1.5,
               label=f"Mean malware={stats.get('ip_score_malware',0):.3f}")
    ax.axvline(stats.get("ip_score_benign", 0),
               color="#1D9E75", ls="--", lw=1.5,
               label=f"Mean benign={stats.get('ip_score_benign',0):.3f}")
    ax.set_xlabel("Import Pattern Score")
    ax.set_ylabel("Số mẫu")
    ax.set_title(f"Import Pattern Score Distribution\n"
                 f"Δ = {stats.get('delta',0):+.4f}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Subplot 2: Category coverage ---
    ax = axes[1]
    categories = list(IMPORT_PATTERNS.keys())
    cov_mal_list = []
    cov_ben_list = []

    for cat in categories:
        has_cat_mal = mal.dominant.str.contains(cat, regex=False).mean()
        has_cat_ben = ben.dominant.str.contains(cat, regex=False).mean()
        cov_mal_list.append(has_cat_mal * 100)
        cov_ben_list.append(has_cat_ben * 100)

    x     = np.arange(len(categories))
    width = 0.35
    bars1 = ax.bar(x - width/2, cov_mal_list, width,
                   color="#ef4444", alpha=0.8, label="Malware")
    bars2 = ax.bar(x + width/2, cov_ben_list, width,
                   color="#1D9E75", alpha=0.8, label="Benign")

    ax.set_xticks(x)
    ax.set_xticklabels(
        [c.replace("_", "\n") for c in categories],
        fontsize=7, rotation=0
    )
    ax.set_ylabel("% mẫu có API category này")
    ax.set_title("Threat Category Coverage\nMalware vs Benign")
    ax.legend()
    ax.grid(True, alpha=0.2, axis="y")

    fig.suptitle("Import Pattern Evidence — Step 7", fontsize=13)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close()
    print(f"  [SAVED] {save_path.name}")



# FULL PIPELINE OUTPUT (tích hợp với step 6)


def full_output_with_import_pattern(
    verdict    : str,
    confidence : float,
    evidence   : dict,
    shap_top   : list,
    api_names  : list[str],
) -> dict:
    """
    Output hoàn chỉnh cho web API — kết hợp step 6 + step 7.

    api_names: list tên function từ lief import table của file PE upload
    """
    ip_result = compute_import_pattern_score(api_names)

    return {
        # Step 6 output (không thay đổi)
        "verdict"          : verdict,
        "confidence"       : confidence,
        "evidence"         : evidence,
        "shap_top"         : shap_top,

        # Step 7 additions
        "import_pattern"   : {
            "score"        : ip_result["score"],
            "n_categories" : ip_result["n_categories"],
            "hits"         : ip_result["hits"],
            "threat_hint"  : ip_result["threat_hint"],
            "mitre"        : ip_result.get("mitre", ""),
        },
    }



# TEST TRÊN WINDOWS PE FILES THẬT


def test_on_real_pe_files() -> list[dict]:
    """
    Test Import Pattern trên file PE thật từ Windows.
    Không cần chạy full pipeline — chỉ test phần import pattern.
    """
    import lief

    test_files = [
        r"C:\Windows\System32\alg.exe",
        r"C:\Windows\System32\calc.exe",
        r"C:\Windows\notepad.exe",
        r"C:\Windows\System32\AggregatorHost.exe",
        r"C:\Windows\System32\agentactivationruntimestarter.exe",
    ]

    results = []
    for path in test_files:
        p = Path(path)
        if not p.exists():
            continue

        try:
            pe = lief.parse(str(p))
            api_names = []
            for imp in pe.imports:
                for entry in imp.entries:
                    if entry.name:
                        api_names.append(entry.name)

            result = compute_import_pattern_score(api_names)
            results.append({
                "file"       : p.name,
                "size_kb"    : round(p.stat().st_size / 1024, 1),
                "n_apis"     : len(api_names),
                "ip_score"   : result["score"],
                "n_cats"     : result["n_categories"],
                "hits"       : result["hits"],
                "threat_hint": result["threat_hint"],
            })
        except Exception as e:
            results.append({"file": p.name, "error": str(e)})

    return results



# MAIN


def setup_log():
    log_path = RES_DIR / "step7_output.txt"
    class Tee:
        def __init__(self, *f): self.files = f
        def write(self, o):
            for f in self.files: f.write(o); f.flush()
        def flush(self):
            for f in self.files: f.flush()
    lf = open(log_path, "w", encoding="utf-8")
    sys.stdout = Tee(sys.__stdout__, lf)
    return log_path


def main():
    log_path = setup_log()
    print("=" * 55)
    print("STEP 7: Import Pattern Evidence Engine")
    print("=" * 55)
    print(f"Log: {log_path}\n")

    
    # A. KIỂM TRA IMPORT PATTERN DICTIONARY
    
    print("Import Pattern Dictionary:")
    print(f"  {'Category':25s}  {'Weight':>7}  {'#APIs':>5}  MITRE")
    print(f"  {'-'*55}")
    for cat, cfg in IMPORT_PATTERNS.items():
        print(f"  {cat:25s}  {cfg['weight']:>7.2f}  "
              f"{len(cfg['apis']):>5}  {cfg['mitre']}")

    
    # B. ĐÁNH GIÁ TRÊN EMBER JSONL
    
    print("\n" + "=" * 55)
    print("ĐÁNH GIÁ TRÊN EMBER DATASET (5000 mẫu)")
    print("=" * 55)

    df_ember = evaluate_on_ember(n_samples=5000)

    if not df_ember.empty:
        stats = analyze_ember_results(df_ember)

        df_ember.to_csv(RES_DIR / "import_pattern_results.csv", index=False)
        print("  [SAVED] import_pattern_results.csv")

        plot_import_pattern_coverage(
            df_ember, stats,
            PLOT_DIR / "import_pattern_coverage.png"
        )

        # Nhận xét calibration
        delta = stats.get("delta", 0)
        print(f"\n  Calibration so với Evidence step 6:")
        print(f"    Evidence tls_evidence Δ     = +0.0000  [WEAK]")
        print(f"    Import Pattern Score  Δ     = {delta:+.4f}  "
              f"[{'OK' if delta > 0.05 else 'WEAK'}]")
        if delta > 0.05:
            print(f"    → Import Pattern TỐTHƠN tls_evidence")
            print(f"      Giải thích API cụ thể có discriminative value")
        else:
            print(f"    → Import Pattern chưa đủ mạnh để phân biệt")
            print(f"      Đây là limitation của static import analysis")
    else:
        stats = {}

    
    # C. TEST TRÊN WINDOWS PE THẬT
    
    print("\n" + "=" * 55)
    print("TEST TRÊN WINDOWS SYSTEM FILES")
    print("Kỳ vọng: system files có score thấp hoặc trung bình")
    print("=" * 55)

    real_results = test_on_real_pe_files()

    for r in real_results:
        if "error" in r:
            print(f"\n  {r['file']}: ERROR — {r['error']}")
            continue
        hits_str = ", ".join(r["hits"].keys()) if r["hits"] else "none"
        print(f"\n  {r['file']}  ({r['size_kb']} KB)")
        print(f"    n_apis    : {r['n_apis']}")
        print(f"    ip_score  : {r['ip_score']:.4f}")
        print(f"    categories: {hits_str}")
        print(f"    hint      : {r['threat_hint']}")

    
    # D. SAMPLE FULL OUTPUT (step 6 + step 7 kết hợp)
    
    print("\n" + "=" * 55)
    print("SAMPLE: Full output cho web API (step 6 + step 7)")
    print("=" * 55)

    # Giả lập output của 1 malware mẫu từ step 6
    sample_output = full_output_with_import_pattern(
        verdict    = "MALWARE",
        confidence = 0.9998,
        evidence   = {
            "packing_evidence" : 0.6935,
            "tls_evidence"     : 0.4918,
            "import_evidence"  : 0.4902,
            "header_evidence"  : 0.1035,
            "loadcfg_evidence" : 0.5278,
        },
        shap_top   = [
            {"feature": "feat_691",  "contribution": +0.9424},
            {"feature": "feat_801",  "contribution": +0.6280},
            {"feature": "feat_637",  "contribution": +0.3950},
            {"feature": "feat_2359", "contribution": +0.3318},
        ],
        api_names  = [
            # Giả lập import table của malware mẫu
            "VirtualAllocEx", "WriteProcessMemory",
            "CreateRemoteThread", "RegSetValueExW",
            "IsDebuggerPresent", "CreateProcessA",
        ],
    )

    print(f"\n  verdict          : {sample_output['verdict']}")
    print(f"  confidence       : {sample_output['confidence']}")
    print(f"\n  evidence (step 6):")
    for k, v in sample_output["evidence"].items():
        print(f"    {k:22s}: {v}")
    print(f"\n  import_pattern (step 7):")
    ip = sample_output["import_pattern"]
    print(f"    score          : {ip['score']}")
    print(f"    n_categories   : {ip['n_categories']}")
    print(f"    threat_hint    : {ip['threat_hint']}")
    print(f"    mitre          : {ip['mitre']}")
    print(f"    hits           :")
    for cat, apis in ip["hits"].items():
        print(f"      [{cat}] {apis}")

    
    # E. LƯU CONFIG
    
    import json as json_module
    config = {
        "version"  : "step7",
        "tau_star" : TAU_STAR,
        "import_patterns": {
            cat: {
                "weight": cfg["weight"],
                "desc"  : cfg["desc"],
                "mitre" : cfg["mitre"],
                "apis"  : cfg["apis"],
            }
            for cat, cfg in IMPORT_PATTERNS.items()
        },
        "scoring_note": (
            "score = max_weight(hit_categories) + combo_bonus. "
            "Import Pattern không phán quyết — chỉ giải thích. "
            "Verdict vẫn do XGBoost quyết định."
        ),
    }

    config_path = MODEL_DIR / "import_pattern_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json_module.dump(config, f, indent=2, ensure_ascii=False)
    print(f"\n  [SAVED] import_pattern_config.json")

    
    # TÓM TẮT
    
    print("\n" + "=" * 55)
    print("TÓM TẮT STEP 7")
    print("=" * 55)

    print(f"\nImport Pattern Dictionary:")
    print(f"  {len(IMPORT_PATTERNS)} threat categories")
    print(f"  {sum(len(c['apis']) for c in IMPORT_PATTERNS.values())} APIs")
    total_mitre = len(set(c['mitre'] for c in IMPORT_PATTERNS.values()))
    print(f"  {total_mitre} MITRE ATT&CK techniques")

    if stats:
        print(f"\nEMBER evaluation (5000 mẫu):")
        print(f"  Import Pattern Δ = {stats.get('delta',0):+.4f}")
        print(f"  Coverage malware = {stats.get('coverage_malware',0)*100:.1f}%")
        print(f"  Coverage benign  = {stats.get('coverage_benign',0)*100:.1f}%")

    print(f"\nKiến trúc web API (step 6 + step 7):")
    print(f"  verdict         ← ML >= tau_star")
    print(f"  confidence      ← ML score")
    print(f"  evidence        ← 5 sigmoid values (step 6)")
    print(f"  shap_top        ← per-sample SHAP (step 6)")
    print(f"  import_pattern  ← lief API names + threat category (step 7)")
    print(f"  threat_hint     ← human-readable description (step 7)")

    print(f"\n[XONG] Bước tiếp: xây dựng web/ với Flask")
    print(f"       Dùng import_pattern_config.json + evidence_config.json")


if __name__ == "__main__":
    main()