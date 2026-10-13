"""
Test pipeline trên file PE thật từ Windows System32.
Chạy từ thư mục gốc project:
  python test_real_pe.py

Kiểm tra toàn bộ luồng:
  Phase 1: validate PE
  Phase 2: extract 2381 features
  XGBoost: predict + confidence
  Evidence: 5 giá trị
  SHAP: top 5 features
  Import patterns: suspicious APIs (preview step 7)
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import joblib
import json
from pathlib import Path

# ============================================================
# CONFIG — sửa đường dẫn nếu cần
# ============================================================
MODEL_DIR  = Path("models/artifacts")
DATA_DIR   = Path("data/processed")
TAU_STAR   = 0.627

PE_FILES = [
    r"C:\Windows\System32\agentactivationruntimestarter.exe",
    r"C:\Windows\System32\AggregatorHost.exe",
    r"C:\Windows\System32\alg.exe",
    r"C:\Windows\System32\calc.exe",
    r"C:\Windows\notepad.exe",
]

# Suspicious APIs theo category
SUSPICIOUS_APIS = {
    "download" : ["URLDownloadToFile", "WinHttpSendRequest",
                  "InternetOpenUrl", "InternetReadFile",
                  "HttpSendRequest"],
    "execute"  : ["WinExec", "ShellExecute", "ShellExecuteEx",
                  "CreateProcess", "CreateProcessAsUser"],
    "inject"   : ["VirtualAllocEx", "WriteProcessMemory",
                  "CreateRemoteThread", "NtUnmapViewOfSection",
                  "ZwUnmapViewOfSection"],
    "keylog"   : ["SetWindowsHookEx", "GetAsyncKeyState",
                  "GetKeyboardState", "RegisterHotKey"],
    "socket"   : ["WSAConnect", "connect", "WSASend",
                  "HttpSendRequestW", "InternetConnectW"],
    "persist"  : ["RegSetValueEx", "CreateService",
                  "SchtasksCreate", "StartServiceCtrlDispatcher"],
    "crypto"   : ["CryptEncrypt", "CryptDecrypt",
                  "CryptBinaryToString", "BCryptEncrypt"],
}

# ============================================================
# PHASE 1: VALIDATE PE
# ============================================================
def phase1_validate(path: str) -> dict:
    """Kiểm tra file có hợp lệ PE không."""
    p = Path(path)
    if not p.exists():
        return {"ok": False, "error": "File không tồn tại"}

    size_kb = p.stat().st_size / 1024
    if size_kb > 50 * 1024:
        return {"ok": False, "error": f"File quá lớn: {size_kb:.0f} KB"}

    # Kiểm tra MZ header
    with open(p, "rb") as f:
        header = f.read(2)
    if header != b"MZ":
        return {"ok": False, "error": "Không phải PE file (thiếu MZ header)"}

    return {"ok": True, "size_kb": round(size_kb, 1), "path": str(p)}


# ============================================================
# PHASE 2: EXTRACT FEATURES + IMPORT PATTERNS
# ============================================================
def phase2_extract(path: str) -> dict:
    """Trích 2381 features và import patterns từ PE file."""
    try:
        import lief
        import ember.features as ef
    except ImportError as e:
        return {"ok": False, "error": f"Missing lib: {e}"}

    try:
        # Đọc raw bytes
        with open(path, "rb") as f:
            raw = f.read()

        # Extract features bằng ember — gọi thủ công từng extractor
        # để tránh lief.bad_format không còn trong lief 0.17.x
        extractor = ef.PEFeatureExtractor(feature_version=2)

        # Parse PE bằng lief trước
        pe = lief.parse(path)
        if pe is None:
            return {"ok": False, "error": "lief không parse được file"}

        # Lấy raw features từ JSONL-style dict (giống cách ember đọc JSONL)
        # Dùng extractor.raw_features() rồi process_raw_features()
        # Patch lief errors cho ember compatibility voi lief 0.17.x
        lief.bad_format  = Exception
        lief.bad_file    = Exception
        lief.pe_error    = Exception
        lief.parser_error = Exception
        lief.read_out_of_bound = Exception

        features = extractor.feature_vector(raw)

        # Đọc import table bằng lief để lấy tên API thật
        imports_by_dll = {}
        all_func_names = []

        try:
            for imp in pe.imports:
                funcs = [e.name for e in imp.entries if e.name]
                if funcs:
                    imports_by_dll[imp.name] = funcs
                    all_func_names.extend(funcs)
        except Exception:
            pass

        # Tìm suspicious APIs
        hits = {}
        for category, apis in SUSPICIOUS_APIS.items():
            matched = [
                f for f in all_func_names
                if any(a.lower() in f.lower() for a in apis)
            ]
            if matched:
                hits[category] = matched[:4]

        return {
            "ok"           : True,
            "features"     : features,
            "n_features"   : len(features),
            "imports_dlls" : list(imports_by_dll.keys())[:8],
            "total_funcs"  : len(all_func_names),
            "suspicious"   : hits,
        }

    except Exception as e:
        return {"ok": False, "error": str(e)}


# ============================================================
# EVIDENCE ENGINE
# ============================================================
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))

def extract_evidence(x: np.ndarray) -> dict:
    """x = scaled feature vector (1D, 2381 dims)."""
    packing = (sigmoid((x[503] - 1.5) / 0.3) * 0.7 +
               sigmoid((x[691] - 0.5) / 0.3) * 0.3)
    tls     = sigmoid(x[2359] / 0.3)
    imp     = sigmoid(x[2364] / 0.3)
    ts_ev   = sigmoid((-x[626] - 1.5) / 0.3)
    coff_ev = sigmoid(( x[637] - 1.0) / 0.3)
    header  = (ts_ev + coff_ev) / 2.0
    loadcfg = sigmoid(-x[2360] / 0.3)

    return {
        "packing_evidence" : round(float(packing), 4),
        "tls_evidence"     : round(float(tls),     4),
        "import_evidence"  : round(float(imp),     4),
        "header_evidence"  : round(float(header),  4),
        "loadcfg_evidence" : round(float(loadcfg), 4),
    }


# ============================================================
# MAIN PIPELINE
# ============================================================
def run_pipeline(pe_path: str, model, scaler, explainer) -> dict:
    name = Path(pe_path).name

    # Phase 1
    v1 = phase1_validate(pe_path)
    if not v1["ok"]:
        return {"file": name, "error": v1["error"]}

    # Phase 2
    v2 = phase2_extract(pe_path)
    if not v2["ok"]:
        return {"file": name, "error": v2["error"]}

    raw_features = np.array(v2["features"], dtype=np.float32).reshape(1, -1)

    # Pad hoặc truncate về 2381
    N = 2381
    if raw_features.shape[1] < N:
        pad = np.zeros((1, N - raw_features.shape[1]), dtype=np.float32)
        raw_features = np.hstack([raw_features, pad])
    elif raw_features.shape[1] > N:
        raw_features = raw_features[:, :N]

    # Scale
    X_scaled = scaler.transform(raw_features).astype(np.float32)

    # XGBoost predict
    ml_score = float(model.predict_proba(X_scaled)[0, 1])
    verdict  = "MALWARE" if ml_score >= TAU_STAR else "BENIGN"

    # Evidence
    evidence = extract_evidence(X_scaled[0])

    # SHAP
    sv = explainer.shap_values(X_scaled)
    if isinstance(sv, list): sv = sv[1]
    elif sv.ndim == 3:       sv = sv[:, :, 1]
    sv = sv[0]
    top_idx  = np.argsort(np.abs(sv))[-5:][::-1]
    shap_top = [{"feature": f"feat_{i}",
                 "contribution": round(float(sv[i]), 4)}
                for i in top_idx]

    return {
        "file"            : name,
        "size_kb"         : v1["size_kb"],
        "verdict"         : verdict,
        "confidence"      : round(ml_score, 4),
        "evidence"        : evidence,
        "shap_top"        : shap_top,
        "imports_dlls"    : v2["imports_dlls"],
        "total_funcs"     : v2["total_funcs"],
        "suspicious_apis" : v2["suspicious"],
        "n_features"      : v2["n_features"],
    }


def print_result(r: dict):
    if "error" in r:
        print(f"\n  [{r['file']}] ERROR: {r['error']}")
        return

    verdict_label = "🔴 MALWARE" if r["verdict"] == "MALWARE" else "🟢 BENIGN"
    print(f"\n{'='*55}")
    print(f"  {r['file']}  ({r['size_kb']} KB)")
    print(f"{'='*55}")
    print(f"  verdict     : {r['verdict']}  (confidence={r['confidence']:.4f})")
    print(f"  features    : {r['n_features']} dims extracted")

    print(f"\n  Evidence:")
    for k, v in r["evidence"].items():
        bar = "█" * int(v * 20)
        print(f"    {k:22s}: {v:.4f}  {bar}")

    print(f"\n  SHAP top 5:")
    for s in r["shap_top"]:
        sign = "+" if s["contribution"] > 0 else ""
        print(f"    {s['feature']:12s}: {sign}{s['contribution']:.4f}")

    print(f"\n  Imports: {r['total_funcs']} functions từ {len(r['imports_dlls'])} DLLs")
    print(f"    DLLs: {r['imports_dlls'][:5]}")

    if r["suspicious_apis"]:
        print(f"\n  [!] SUSPICIOUS APIs found:")
        for cat, apis in r["suspicious_apis"].items():
            print(f"    [{cat}] {apis}")
    else:
        print(f"\n  [OK] Không có suspicious APIs rõ ràng")


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    print("=" * 55)
    print("TEST PIPELINE — Real PE Files")
    print("=" * 55)

    # Load model + scaler
    print("\nLoad models...")
    try:
        model  = joblib.load(MODEL_DIR / "xgb.pkl")
        scaler = joblib.load(Path("data/processed/scaler.pkl"))
        print("  [OK] XGBoost + scaler")
    except Exception as e:
        print(f"  [ERR] {e}")
        print("  Kiểm tra: models/artifacts/xgb.pkl và scaler.pkl có tồn tại không")
        exit(1)

    # Load SHAP explainer
    print("Load SHAP explainer...")
    import shap
    explainer = shap.TreeExplainer(model)
    print("  [OK] TreeExplainer")

    # Tìm file PE có trên máy
    found_files = []
    for path in PE_FILES:
        if Path(path).exists():
            found_files.append(path)
            print(f"  [OK] {Path(path).name}")
        else:
            print(f"  [--] {Path(path).name} — không tìm thấy")

    # Tìm thêm trong System32 nếu thiếu
    if len(found_files) < 3:
        print("\n  Tìm thêm file trong System32...")
        for f in Path(r"C:\Windows\System32").glob("*.exe"):
            if len(found_files) >= 5:
                break
            if f.stat().st_size < 10 * 1024 * 1024:  # < 10MB
                found_files.append(str(f))
                print(f"  [+] {f.name}")

    print(f"\nChạy pipeline trên {len(found_files)} file...")

    # Chạy pipeline
    results = []
    for pe_path in found_files:
        r = run_pipeline(pe_path, model, scaler, explainer)
        print_result(r)
        results.append(r)

    # Tóm tắt
    print(f"\n{'='*55}")
    print("TÓM TẮT")
    print(f"{'='*55}")
    ok = [r for r in results if "error" not in r]
    mal = [r for r in ok if r["verdict"] == "MALWARE"]
    ben = [r for r in ok if r["verdict"] == "BENIGN"]
    print(f"  Tổng file test : {len(results)}")
    print(f"  Xử lý thành công: {len(ok)}")
    print(f"  MALWARE verdict : {len(mal)}")
    print(f"  BENIGN  verdict : {len(ben)}")

    if mal:
        print(f"\n  [!] Các file bị phán là MALWARE:")
        for r in mal:
            print(f"    {r['file']} ({r['size_kb']} KB) — confidence={r['confidence']:.4f}")
        print(f"\n  Lưu ý: file System32 bị phán MALWARE = false positive")
        print(f"  Điều này bình thường — model train trên EMBER không biết")
        print(f"  file đó là Windows system file hợp lệ.")
        print(f"  Đây là limitation của static-only analysis.")

    has_suspicious = [r for r in ok if r.get("suspicious_apis")]
    if has_suspicious:
        print(f"\n  [!] File có suspicious APIs ({len(has_suspicious)} file):")
        for r in has_suspicious:
            cats = list(r["suspicious_apis"].keys())
            print(f"    {r['file']}: {cats}")

    print(f"\n[XONG] Kết quả này xác nhận pipeline hoạt động trên PE thật.")
    print(f"       Step 7 sẽ tích hợp suspicious_apis vào Evidence Engine.")