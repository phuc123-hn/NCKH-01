# Kiến trúc và Ý tưởng Hệ thống

## Câu hỏi nghiên cứu

"Liệu một framework kết hợp ML phát hiện malware, SHAP-stability feature analysis, Evidence Aggregation và Import Pattern Engine có cải thiện đồng thời độ chính xác, khả năng giải thích và độ bền vững so với các phương pháp đơn lẻ hiện có không?"

---

## Xuất phát điểm

Hầu hết nghiên cứu hiện tại chỉ làm một trong ba việc:

- **So sánh model:** train XGBoost/LightGBM trên EMBER, báo F1 rồi kết luận — không giải thích được tại sao.
- **SHAP explain:** dùng mean |SHAP| để chọn feature quan trọng — không đo độ ổn định qua các subset.
- **Rule-based detection:** kiểm tra PE structure theo ngưỡng cứng — không kết hợp được với ML.

Hệ thống này kết hợp cả ba và thêm hai yếu tố chưa ai làm đồng thời trên EMBER: Youden's J threshold optimization và SHAP Stability Score.

---

## Kiến trúc tổng thể

```
File PE (raw bytes)
        |
        v
[Phase 1] Validate
  - Kiểm tra MZ header, PE signature
  - Kiểm tra kích thước file
        |
        v
[Phase 2] Extract Features
  - lief parse PE structure
  - ember extractor tính 2381 features
  - StandardScaler normalize
        |
        +---------------------------+
        |                           |
        v                           v
[Lớp ML]                    [Lớp Explanation]
XGBoost predict             Evidence Engine
confidence = 0.947          + SHAP per-sample
        |                   + Import Pattern
        v                           |
  verdict = MALWARE                 |
  (ML >= tau* = 0.627)              |
        |                           |
        +---------------------------+
                    |
                    v
            Output cho web:
            - verdict
            - confidence
            - evidence (5 giá trị)
            - shap_top (5 features)
            - import_patterns (API names)
            - threat_hint (mô tả ngắn)
```

---

## Ý tưởng cốt lõi: ML quyết định, Evidence giải thích

Phân biệt rõ hai vai trò:

**ML (XGBoost):** Học pattern phi tuyến từ 2381 features — đưa ra verdict. Không cần giải thích được.

**Evidence Engine:** Không cạnh tranh với ML. Chỉ trả lời câu hỏi "tại sao ML nói vậy?" bằng ngôn ngữ bảo mật mà người dùng hiểu được.

Lý do tách biệt: Nếu cho ML học cả Evidence flags thì ML có thể bỏ qua hoặc gán trọng số thấp. Khi Evidence là lớp riêng độc lập, nó luôn được hiển thị bất kể ML nghĩ gì.

---

## Năm thành phần chính

### 1. XGBoost với Youden's J threshold

Thay vì ngưỡng mặc định 0.5, tìm tau* tối ưu hóa J = max(TPR - FPR).

Với XGBoost: tau* = 0.627, giảm FPR từ 3.59% xuống 2.05% (giảm 43%) trong khi F1 tăng từ 0.9613 lên 0.9682.

Ý nghĩa thực tế: Với 1 triệu file scan mỗi ngày, giảm 43% false positive = giảm 15,400 cảnh báo nhầm mỗi ngày.

### 2. SHAP Stability Score

Thay vì chỉ dùng mean |SHAP|, tính SHAP trên 5 random subset và đo variance per feature.

```
stability_score = 1 - std(SHAP) / (mean(SHAP) + epsilon)
```

Feature với stability > 0.7 được đánh là STABLE — không phụ thuộc distribution cụ thể của EMBER 2018, có giá trị khi gặp malware mới.

Kết quả: Tất cả top-10 features đều STABLE (0.89–0.98). feat_637 (COFF characteristics) và feat_2359 (TLS Table) ổn định nhất với 0.94.

### 3. Evidence Vector (5 dimensions)

Mỗi Evidence value = sigmoid của z-score PE feature tương ứng, đặt tên theo ngữ nghĩa bảo mật:

| Evidence | Feature nguồn | Ý nghĩa |
|----------|---------------|---------|
| packing_evidence | feat_503, feat_691 | Entropy cao + nhiều RX section |
| tls_evidence | feat_2359 | TLS callback — anti-analysis |
| import_evidence | feat_2364 | Delay Import — ẩn API calls |
| header_evidence | feat_637, feat_626 | COFF flags + timestamp bất thường |
| loadcfg_evidence | feat_2360 | Load Config vắng mặt |

Điểm khác biệt so với feature vector: Evidence là kết quả của phép suy luận có ngữ nghĩa bảo mật, không phải số liệu thô. Người không chuyên cũng hiểu "Packing Evidence = 0.92" nghĩa là file có khả năng bị packed/encrypted cao.

### 4. Import Pattern Engine

Đọc import table thật bằng lief (không dùng feature hash), map tên API về threat category theo MITRE ATT&CK.

```
file PE → lief parse → tên API thật → so khớp dictionary → threat category
```

Điểm khác biệt: Evidence Vector dùng import hash (feat_943-2222) — không thể đảo ngược về tên API. Import Pattern Engine đọc tên API thật — giải thích được cụ thể: "File có VirtualAllocEx + WriteProcessMemory → process injection (T1055)".

Coverage trên EMBER: 45.1% malware có ít nhất 1 suspicious category, 24.3% benign — Δ = +0.1685.

### 5. Robustness Test

Thêm Gaussian noise vào X_test ở các mức σ = 0.1, 0.5, 1.0, 1.5, 2.0 để mô phỏng malware biến thể (obfuscation, packing mới).

Kết quả xác nhận XGBoost là lựa chọn đúng cho production — drop 49.75% tại σ=2.0, tốt hơn LightGBM (67.34%) và RandomForest (76.28%).

---

## Hạn chế của hệ thống

**Hạn chế 1 — Stager/Dropper nhỏ:** File PE < 50KB tải payload từ mạng thường có entropy thấp, import ít, features bình thường — static analysis khó phát hiện. Đây là hạn chế của static analysis nói chung. Hướng mở rộng: kết hợp với dynamic analysis (sandbox).

**Hạn chế 2 — Evidence calibration:** tls_evidence, import_evidence, header_evidence có Δ ≈ 0 ở population mean do đặc tính phân phối của sparse binary fields sau StandardScaler. Evidence có giá trị ở mức per-sample nhưng không phân biệt ở mức thống kê tổng thể.

**Hạn chế 3 — Dataset EMBER 2018:** Data từ 2018. Malware landscape sau 2018 (ransomware fileless, LoLBin attacks) có thể có pattern khác. Robustness test với noise là proxy không hoàn hảo cho domain shift thực tế.

**Hạn chế 4 — Import Pattern coverage:** 54.9% malware không có API trong danh sách suspicious — hoặc dùng API khác, hoặc import qua runtime loading (LoadLibrary/GetProcAddress) thay vì import table tĩnh.

---

## Điểm đóng góp so với literature

| Phương pháp | Literature hiện tại | Đề tài này |
|-------------|--------------------|-----------:|
| ML trên EMBER | Có (Anderson 2018, nhiều paper 2022-2024) | Có + Youden's J |
| SHAP explain | Có (IEEE 2024) | Có + Stability Score |
| Threshold optimization | Không | Có — Youden's J |
| SHAP Stability Score | Phân tích riêng lẻ (MDPI 2026) | Tích hợp vào pipeline |
| PE Anomaly + ML kết hợp | Không trên EMBER | Có — Evidence Framework |
| Import Pattern + MITRE ATT&CK | Không | Có — 8 categories |

---

## Output chuẩn cho web API

```json
{
  "verdict": "MALWARE",
  "confidence": 0.9998,
  "evidence": {
    "packing_evidence": 0.6935,
    "tls_evidence": 0.4918,
    "import_evidence": 0.4902,
    "header_evidence": 0.1035,
    "loadcfg_evidence": 0.5278
  },
  "shap_top": [
    {"feature": "feat_691", "contribution": +0.9424},
    {"feature": "feat_801", "contribution": +0.6280},
    {"feature": "feat_637", "contribution": +0.3950},
    {"feature": "feat_2359", "contribution": +0.3318}
  ],
  "import_pattern": {
    "score": 1.0,
    "n_categories": 2,
    "threat_hint": "Process injection + persistence",
    "mitre": "T1055",
    "hits": {
      "process_injection": ["VirtualAllocEx", "WriteProcessMemory"],
      "persistence_registry": ["RegSetValueExW"]
    }
  }
}
```
