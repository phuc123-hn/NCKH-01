"""
web/app.py — Flask backend
Malware Detection System

Routes:
  GET  /                → upload page
  POST /scan            → scan file PE, trả về JSON
  GET  /stream/<job_id> → SSE stream từng bước chat log
"""

import os
import json
import time
import uuid
import threading
from pathlib import Path
from flask import (Flask, render_template, request,
                   jsonify, Response, stream_with_context)



# Custom JSON encoder — xử lý numpy types và ndarray

class SafeEncoder(json.JSONEncoder):
    def default(self, obj):
        try:
            import numpy as np
            if isinstance(obj, np.integer):  return int(obj)
            if isinstance(obj, np.floating): return float(obj)
            if isinstance(obj, np.ndarray):  return obj.tolist()
        except ImportError:
            pass
        return super().default(obj)


def safe_json_dumps(obj) -> str:
    """json.dumps với SafeEncoder — không bao giờ throw TypeError cho numpy."""
    return json.dumps(obj, ensure_ascii=False, cls=SafeEncoder)


# App setup

BASE_DIR   = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50MB max

# In-memory job store (production: dùng Redis)
jobs: dict[str, dict] = {}


# Load pipeline (lazy — chỉ load khi cần)

_pipeline = None

def get_pipeline():
    global _pipeline
    if _pipeline is None:
        _pipeline = load_pipeline()
    return _pipeline


def load_pipeline():
    """
    Load model, scaler, config từ models/artifacts/.
    Trả về dict chứa tất cả thứ cần cho inference.
    """
    import joblib
    import json as json_lib

    artifacts = BASE_DIR.parent / "models" / "artifacts"
    processed  = BASE_DIR.parent / "data" / "processed"

    pipeline = {}

    # XGBoost model
    xgb_path = artifacts / "xgb.pkl"
    if xgb_path.exists():
        pipeline["model"]  = joblib.load(xgb_path)
        pipeline["model_name"] = "XGBoost"
    else:
        pipeline["model"]  = None
        pipeline["model_name"] = "mock"

    # Scaler
    scaler_path = processed / "scaler.pkl"
    if scaler_path.exists():
        pipeline["scaler"] = joblib.load(scaler_path)
    else:
        pipeline["scaler"] = None

    # Evidence config
    ev_cfg_path = artifacts / "evidence_config.json"
    if ev_cfg_path.exists():
        with open(ev_cfg_path, encoding="utf-8") as f:
            pipeline["evidence_config"] = json_lib.load(f)
    else:
        pipeline["evidence_config"] = {}

    # Import pattern config
    ip_cfg_path = artifacts / "import_pattern_config.json"
    if ip_cfg_path.exists():
        with open(ip_cfg_path, encoding="utf-8") as f:
            pipeline["import_config"] = json_lib.load(f)
    else:
        pipeline["import_config"] = {}

    # SHAP explainer (lazy — khởi tạo khi predict lần đầu)
    pipeline["shap_explainer"] = None

    return pipeline



# Core inference functions


def phase1_validate(filepath: str) -> dict:
    """Kiểm tra file PE hợp lệ."""
    p = Path(filepath)
    if not p.exists():
        return {"ok": False, "error": "File không tồn tại"}

    size_kb = p.stat().st_size / 1024
    if size_kb > 50 * 1024:
        return {"ok": False, "error": f"File quá lớn ({size_kb:.0f} KB, tối đa 50MB)"}

    with open(p, "rb") as f:
        header = f.read(2)
    if header != b"MZ":
        return {"ok": False, "error": "Không phải file PE (thiếu MZ header)"}

    return {
        "ok"      : True,
        "size_kb" : round(size_kb, 1),
        "filename": p.name,
    }


def phase2_extract(filepath: str) -> dict:
    """Trích 2381 features từ file PE."""
    try:
        import lief
        import numpy as np
        import builtins
        import ember.features as ef
        from sklearn.feature_extraction import FeatureHasher

        # ── Patch 1: lief 0.17.x compatibility ──
        for attr in ("bad_format", "bad_file", "pe_error",
                     "parser_error", "read_out_of_bound", "not_found"):
            if not hasattr(lief, attr):
                setattr(lief, attr, Exception)

        # ── Patch 2: numpy >= 1.24 removed deprecated type aliases ──
        # Dùng getattr thay vì hasattr để tránh FutureWarning khi alias sắp bị redefined
        import warnings as _warnings
        for _alias in ("int", "float", "complex", "bool", "object", "str"):
            with _warnings.catch_warnings():
                _warnings.simplefilter("ignore")
                _has = getattr(np, _alias, None) is not None
            if not _has:
                setattr(np, _alias, getattr(builtins, _alias))

        # ── Patch 3: sklearn >= 1.6 FeatureHasher rejects bare string ──
        _orig_transform = FeatureHasher.transform
        def _safe_transform(self, X, **kw):
            if self.input_type == "string":
                X = [[item] if isinstance(item, str) else item for item in X]
            return _orig_transform(self, X, **kw)
        FeatureHasher.transform = _safe_transform

        with open(filepath, "rb") as f:
            raw = f.read()

        extractor = ef.PEFeatureExtractor(feature_version=2)
        features  = extractor.feature_vector(raw)

        # Đọc import table bằng lief (raw bytes, không dùng path string)
        pe = lief.parse(raw)
        imports_by_dll = {}
        all_funcs      = []

        if pe and hasattr(pe, "imports"):
            for imp in pe.imports:
                funcs = [e.name for e in imp.entries if e.name]
                if funcs:
                    imports_by_dll[imp.name] = funcs
                    all_funcs.extend(funcs)

        return {
            "ok"         : True,
            "features"   : features,
            "n_features" : len(features),
            "dlls"       : list(imports_by_dll.keys())[:8],
            "n_funcs"    : len(all_funcs),
            "all_funcs"  : all_funcs,
        }

    except Exception as e:
        import traceback
        return {"ok": False, "error": str(e), "traceback": traceback.format_exc()}


def run_ml(features, pipeline: dict) -> dict:
    """Scale features và predict bằng XGBoost."""
    import numpy as np

    raw = np.array(features, dtype=np.float32).reshape(1, -1)
    N   = 2381
    if raw.shape[1] < N:
        pad = np.zeros((1, N - raw.shape[1]), dtype=np.float32)
        raw = np.hstack([raw, pad])
    elif raw.shape[1] > N:
        raw = raw[:, :N]

    model  = pipeline.get("model")
    scaler = pipeline.get("scaler")

    if model is None or scaler is None:
        # Mock khi không có model
        import random
        score = random.uniform(0.05, 0.99)
        return {
            "ok"       : True,
            "score"    : round(score, 4),
            "verdict"  : "MALWARE" if score >= 0.627 else "BENIGN",
            "mock"     : True,
        }

    X = scaler.transform(raw).astype(np.float32)
    score = float(model.predict_proba(X)[0, 1])

    # SHAP (lazy init)
    shap_top = []
    try:
        import shap
        if pipeline["shap_explainer"] is None:
            pipeline["shap_explainer"] = shap.TreeExplainer(model)
        sv = pipeline["shap_explainer"].shap_values(X)
        if isinstance(sv, list): sv = sv[1]
        elif sv.ndim == 3:       sv = sv[:, :, 1]
        sv = sv[0]
        top_idx = __import__("numpy").argsort(
            __import__("numpy").abs(sv))[-5:][::-1]
        shap_top = [
            {"feature": f"feat_{i}", "contribution": round(float(sv[i]), 4)}
            for i in top_idx
        ]
    except Exception:
        pass

    return {
        "ok"      : True,
        "score"   : round(score, 4),
        "verdict" : "MALWARE" if score >= 0.627 else "BENIGN",
        "X_scaled": X,          # numpy array — chỉ dùng nội bộ, KHÔNG push vào log
        "shap_top": shap_top,   # list of dict — safe để serialize
        "mock"    : False,
    }


def _sigmoid(x):
    import math
    return 1.0 / (1.0 + math.exp(-max(-50, min(50, x))))


def extract_evidence(X_scaled) -> dict:
    """Tính Evidence Vector từ X_scaled."""
    import numpy as np
    x = X_scaled[0]
    packing = (_sigmoid((x[503] - 1.5) / 0.3) * 0.7 +
               _sigmoid((x[691] - 0.5) / 0.3) * 0.3)
    return {
        "packing_evidence" : round(float(packing), 4),
        "tls_evidence"     : round(float(_sigmoid(x[2359] / 0.3)), 4),
        "import_evidence"  : round(float(_sigmoid(x[2364] / 0.3)), 4),
        "header_evidence"  : round(float(
            (_sigmoid((-x[626] - 1.5) / 0.3) +
             _sigmoid(( x[637] - 1.0) / 0.3)) / 2), 4),
        "loadcfg_evidence" : round(float(_sigmoid(-x[2360] / 0.3)), 4),
    }


SUSPICIOUS_APIS = {
    "process_injection" : {
        "weight": 0.95,
        "apis"  : ["VirtualAllocEx","WriteProcessMemory","CreateRemoteThread",
                   "NtUnmapViewOfSection","ZwUnmapViewOfSection","NtCreateThreadEx"],
        "desc"  : "Process injection — chèn code vào process khác",
        "mitre" : "T1055",
    },
    "keylogger" : {
        "weight": 0.90,
        "apis"  : ["SetWindowsHookEx","GetAsyncKeyState","CallNextHookEx"],
        "desc"  : "Keylogger — hook keyboard ghi lại phím bấm",
        "mitre" : "T1056.001",
    },
    "network_download" : {
        "weight": 0.85,
        "apis"  : ["URLDownloadToFile","WinHttpSendRequest","InternetOpenUrl",
                   "InternetReadFile","HttpSendRequest"],
        "desc"  : "Network download — tải payload từ internet",
        "mitre" : "T1105",
    },
    "anti_analysis" : {
        "weight": 0.80,
        "apis"  : ["CheckRemoteDebuggerPresent","NtQueryInformationProcess",
                   "ZwQueryInformationProcess"],
        "desc"  : "Anti-analysis — phát hiện debugger/sandbox",
        "mitre" : "T1622",
    },
    "crypto_ransomware" : {
        "weight": 0.80,
        "apis"  : ["CryptEncrypt","CryptDecrypt","BCryptEncrypt",
                   "BCryptDecrypt","CryptGenRandom"],
        "desc"  : "Crypto — mã hóa dữ liệu (ransomware pattern)",
        "mitre" : "T1486",
    },
    "persistence_service" : {
        "weight": 0.75,
        "apis"  : ["CreateService","StartServiceCtrlDispatcher","SchtasksCreate"],
        "desc"  : "Persistence via service",
        "mitre" : "T1543.003",
    },
    "socket_raw" : {
        "weight": 0.65,
        "apis"  : ["WSAConnect","WSASend","WSARecv","WSAStartup"],
        "desc"  : "Raw socket — kết nối mạng cấp thấp",
        "mitre" : "T1095",
    },
    "persistence_registry" : {
        "weight": 0.45,
        "apis"  : ["RegSetValueExA","RegSetValueExW"],
        "desc"  : "Registry modification",
        "mitre" : "T1547.001",
    },
}


def compute_import_pattern(all_funcs: list) -> dict:
    hits = {}
    for cat, cfg in SUSPICIOUS_APIS.items():
        matched = [f for f in all_funcs
                   if any(a.lower() in f.lower() for a in cfg["apis"])]
        if matched:
            hits[cat] = matched[:3]

    if not hits:
        return {"score": 0.0, "hits": {}, "threat_hint": "Không phát hiện suspicious API", "n_categories": 0}

    max_w    = max(SUSPICIOUS_APIS[c]["weight"] for c in hits)
    n_cats   = len(hits)
    score    = min(1.0, max_w + min(0.15, (n_cats - 1) * 0.05))
    dominant = max(hits, key=lambda c: SUSPICIOUS_APIS[c]["weight"])
    others   = [c for c in hits if c != dominant]
    hint     = SUSPICIOUS_APIS[dominant]["desc"]
    if others:
        hint += " + " + ", ".join(others[:2])

    return {
        "score"      : round(score, 4),
        "hits"       : hits,
        "threat_hint": hint,
        "n_categories": n_cats,
        "mitre"      : SUSPICIOUS_APIS[dominant]["mitre"],
    }


def mock_ai_interpretation(result: dict) -> str:
    """
    Tạo giải thích ngôn ngữ tự nhiên từ kết quả scan.
    Khi có Anthropic API key → thay bằng API call thật.
    """
    verdict    = result.get("verdict", "UNKNOWN")
    confidence = result.get("confidence", 0)
    evidence   = result.get("evidence", {})
    ip         = result.get("import_pattern", {})

    if verdict == "MALWARE":
        level = "rất cao" if confidence > 0.95 else "cao" if confidence > 0.80 else "trung bình"
        lines = [
            f"File này có khả năng {level} là mã độc (độ tin cậy {confidence*100:.1f}%).",
            "",
        ]

        # Packing
        pack = evidence.get("packing_evidence", 0)
        if pack > 0.6:
            lines.append(f"Entropy byte cao bất thường ({pack:.2f}/1.0) cho thấy "
                         "nội dung file bị nén hoặc mã hóa — kỹ thuật phổ biến để "
                         "ẩn code độc hại khỏi antivirus truyền thống.")

        # TLS — đã loại khỏi UI: tls_evidence không phân biệt malware/benign
        # ở mức population (Δ≈0, xem báo cáo Step 6). Per-sample signal thật
        # của TLS callback (feat_2359) đã được phản ánh qua SHAP/shap_top bên dưới.

        # Import pattern
        if ip.get("hits"):
            for cat, apis in list(ip["hits"].items())[:2]:
                desc = SUSPICIOUS_APIS.get(cat, {}).get("desc", cat)
                api_list = ", ".join(apis[:2])
                lines.append(f"{desc}: phát hiện {api_list}.")

        # Header — đã loại khỏi UI cùng lý do với TLS: header_evidence
        # có Δ ≈ -0.0034 (báo cáo Step 6), không phân biệt malware/benign.

        lines += [
            "",
            "Khuyến nghị: Không chạy file này. "
            "Cách ly và tiến hành phân tích sâu hơn nếu cần thiết.",
        ]

    else:
        lines = [
            f"File này có vẻ an toàn (độ tin cậy {(1-confidence)*100:.1f}% là benign).",
            "",
            "Không phát hiện dấu hiệu đáng ngờ rõ ràng trong quá trình phân tích tĩnh.",
        ]
        if ip.get("hits"):
            cats = list(ip["hits"].keys())
            lines.append(f"Lưu ý: phát hiện một số API có thể đáng ngờ "
                         f"({', '.join(cats[:2])}) nhưng model đánh giá là "
                         "không đủ để kết luận là mã độc.")
        lines += [
            "",
            "Lưu ý: Phân tích tĩnh có giới hạn. "
            "File nhỏ tải payload từ mạng (stager/dropper) "
            "có thể không bị phát hiện bởi phương pháp này.",
        ]

    return "\n".join(lines)



# Background job runner


def run_scan_job(job_id: str, filepath: str):
    """Chạy scan trong background thread, ghi kết quả vào jobs dict."""
    job = jobs[job_id]

    def push(step: str, msg: str, data=None):
        """Thêm bước vào job log."""
        job["log"].append({
            "step": step,
            "msg" : msg,
            "data": data,
            "ts"  : time.time(),
        })

    try:
        pipeline = get_pipeline()

        # Phase 1
        push("phase1", "Đang kiểm tra cấu trúc file PE...")
        time.sleep(0.3)
        v1 = phase1_validate(filepath)
        if not v1["ok"]:
            push("error", v1["error"])
            job["status"] = "error"
            return
        push("phase1", f"File hợp lệ — {v1['size_kb']} KB", v1)

        # Phase 2
        push("phase2", "Đang trích xuất đặc trưng (2381 features)...")
        time.sleep(0.3)
        v2 = phase2_extract(filepath)
        if not v2["ok"]:
            push("error", v2["error"])
            job["status"] = "error"
            return
        push("phase2",
             f"Trích xuất xong — {v2['n_features']} features, "
             f"{v2['n_funcs']} import functions",
             {"n_features": v2["n_features"], "n_funcs": v2["n_funcs"],
              "dlls": v2["dlls"]})

        # ML
        push("model", f"Đang phân tích bằng {pipeline['model_name']}...")
        time.sleep(0.5)
        ml = run_ml(v2["features"], pipeline)
        if not ml["ok"]:
            push("error", ml.get("error", "Lỗi model"))
            job["status"] = "error"
            return
        push("model",
             f"Kết quả: {ml['verdict']} (confidence {ml['score']*100:.1f}%)",
             {"verdict": ml["verdict"], "confidence": ml["score"],
              "mock": ml.get("mock", False)})

        # Evidence
        push("evidence", "Đang đánh giá Evidence Vector...")
        time.sleep(0.3)
        evidence = {}
        if ml.get("X_scaled") is not None:
            evidence = extract_evidence(ml["X_scaled"])
        else:
            # Mock evidence khi không có model thật
            import random
            evidence = {
                "packing_evidence" : round(random.uniform(0.1, 0.9), 4),
                "tls_evidence"     : round(random.uniform(0.3, 0.6), 4),
                "import_evidence"  : round(random.uniform(0.3, 0.6), 4),
                "header_evidence"  : round(random.uniform(0.1, 0.3), 4),
                "loadcfg_evidence" : round(random.uniform(0.3, 0.7), 4),
            }
        push("evidence", "Evidence Vector đã tính xong", evidence)

        # Import Pattern
        push("import", "Đang kiểm tra Import Pattern (MITRE ATT&CK)...")
        time.sleep(0.3)
        ip_result = compute_import_pattern(v2.get("all_funcs", []))
        if ip_result["n_categories"] > 0:
            push("import",
                 f"Phát hiện {ip_result['n_categories']} threat categor{'y' if ip_result['n_categories']==1 else 'ies'}: "
                 f"{ip_result['threat_hint']}",
                 ip_result)
        else:
            push("import", "Không phát hiện suspicious API pattern rõ ràng", ip_result)

        # AI interpretation
        push("ai", "Đang tổng hợp giải thích...")
        time.sleep(0.4)

        full_result = {
            "verdict"        : ml["verdict"],
            "confidence"     : ml["score"],
            "evidence"       : evidence,
            "shap_top"       : ml.get("shap_top", []),
            "import_pattern" : ip_result,
            "file_info"      : v1,
        }

        interpretation = mock_ai_interpretation(full_result)

        # Stream AI interpretation từng dòng
        for line in interpretation.split("\n"):
            push("ai_stream", line)
            time.sleep(0.08)

        push("done", "Phân tích hoàn tất")
        job["result"] = full_result
        job["interpretation"] = interpretation
        job["status"] = "done"

    except Exception as e:
        job["log"].append({"step": "error", "msg": str(e), "data": None, "ts": time.time()})
        job["status"] = "error"

    finally:
        # Cleanup file upload
        try:
            Path(filepath).unlink(missing_ok=True)
        except Exception:
            pass



# Routes
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/scan", methods=["POST"])
def scan():
    """Nhận file upload, tạo job, trả về job_id."""
    if "file" not in request.files:
        return jsonify({"error": "Không có file"}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "Tên file trống"}), 400

    # Lưu file tạm — sanitize tên để tránh lỗi path có dấu cách/ký tự đặc biệt
    job_id    = str(uuid.uuid4())
    # Chỉ giữ extension gốc, tên file dùng job_id để tránh path issues
    orig_name = file.filename or "upload"
    ext       = Path(orig_name).suffix.lower() or ".exe"
    safe_name = f"{job_id}{ext}"
    filepath  = str(UPLOAD_DIR / safe_name)
    file.save(filepath)

    # Tạo job
    jobs[job_id] = {
        "status"        : "running",
        "log"           : [],
        "result"        : None,
        "interpretation": None,
        "filename"      : file.filename,
        "cursor"        : 0,
    }

    # Chạy trong background thread
    t = threading.Thread(target=run_scan_job, args=(job_id, filepath), daemon=True)
    t.start()

    return jsonify({"job_id": job_id})


@app.route("/stream/<job_id>")
def stream(job_id: str):
    """
    Server-Sent Events — gửi từng bước scan về frontend.
    Frontend nhận event và append vào chat log.
    """
    if job_id not in jobs:
        return jsonify({"error": "Job không tồn tại"}), 404

    def generate():
        job = jobs[job_id]
        cursor = 0
        while True:
            log = job["log"]
            while cursor < len(log):
                entry = log[cursor]
                cursor += 1
                data   = safe_json_dumps(entry)
                yield f"data: {data}\n\n"

            if job["status"] in ("done", "error"):
                # Gửi event kết thúc
                yield f"data: {json.dumps({'step': 'end', 'status': job['status']})}\n\n"
                break

            time.sleep(0.1)

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control"              : "no-cache",
            "X-Accel-Buffering"          : "no",
            "Access-Control-Allow-Origin": "*",
        },
    )


@app.route("/result/<job_id>")
def result(job_id: str):
    """Trả về kết quả đầy đủ sau khi job xong."""
    if job_id not in jobs:
        return jsonify({"error": "Job không tồn tại"}), 404
    job = jobs[job_id]
    return jsonify({
        "status"        : job["status"],
        "result"        : job["result"],
        "interpretation": job["interpretation"],
        "filename"      : job["filename"],
    })


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000, threaded=True)