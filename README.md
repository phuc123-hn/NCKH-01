# PE Malware Detector

Hệ thống hỗ trợ phân tích và phát hiện mã độc dạng file PE (Portable Executable) dựa trên **đặc trưng tĩnh** và **học máy**, triển khai dưới dạng ứng dụng web tương tác có khả năng **giải thích quyết định phân loại**.

> Đề tài nghiên cứu khoa học sinh viên năm học 2025–2026, Viện Công nghệ số, Trường Đại học Thủ Dầu Một.
> Nhóm tác giả: Ngô Hồng Phúc, Nguyễn Hoàng Sơn. Giảng viên hướng dẫn: ThS. Trần Bá Minh Sơn.

---

## Điểm nổi bật

| Thành phần | Mô tả |
|---|---|
| **Mô hình XGBoost** | F1 = 0,9682 và AUC = 0,9938 trên tập kiểm thử độc lập 200.000 mẫu |
| **Youden's J Threshold** | Ngưỡng tối ưu τ* = 0,627, giảm FPR từ 3,59% xuống 2,05% (giảm 43% cảnh báo nhầm) |
| **Evidence Aggregation Framework** | 5 chỉ số bằng chứng liên tục (Packing, TLS, Import, Header, LoadCfg) kết hợp SHAP per-sample |
| **Import Pattern Engine** | Phân loại API import theo 8 danh mục MITRE ATT&CK, coverage 45,1% trên mẫu mã độc |
| **Ứng dụng web Flask** | Chat log real-time qua Server-Sent Events, giải thích ngôn ngữ tự nhiên qua `mock_ai_interpretation()` |

Nguyên tắc thiết kế: **ML quyết định, Evidence giải thích.** XGBoost là lớp phán quyết duy nhất; Evidence Vector và SHAP chỉ phục vụ giải thích.

---

## Dữ liệu và đặc trưng

- Bộ dữ liệu: [EMBER 2018](https://arxiv.org/abs/1804.04637), dùng 600.000 mẫu có nhãn cân bằng (300.000 benign và 300.000 malware), bỏ 400.000 mẫu unlabeled.
- Chia dữ liệu: train 510.000 / validation 90.000 (85/15 stratified), test độc lập 200.000 mẫu.
- Trích xuất **2.381 đặc trưng tĩnh** bằng `ember` (feature_version=2) và `lief 0.17.6`:

| Nhóm đặc trưng | Số chiều |
|---|---|
| Byte Histogram | 256 |
| Byte Entropy Histogram | 256 |
| String Extraction | 104 |
| General File Info | 10 |
| Header File Info | 62 |
| Section Info | 255 |
| Imports Info | 1.280 |
| Exports Info | 128 |
| Data Directories | 30 |

---

## Kết quả

### Cross-Validation 5-Fold (tập train)

| Mô hình | CV F1 | CV AUC | Thời gian train | Inference/file |
|---|---|---|---|---|
| LightGBM | 0,9755 ± 0,0005 | 0,9972 | 28,3 phút | 6,36 ms |
| **XGBoost** | **0,9760 ± 0,0007** | 0,9972 | 65,6 phút | **0,677 ms** |
| Random Forest | 0,9558 ± 0,0005 | 0,9933 | 48,2 phút | 36,5 ms |

### Tập kiểm thử độc lập (200.000 mẫu)

| Mô hình | F1 | AUC | FPR (τ=0,5) | FNR (τ=0,5) | τ* Youden | FPR (τ*) | FNR (τ*) |
|---|---|---|---|---|---|---|---|
| XGBoost | 0,9682 | 0,9938 | 3,59% | 3,17% | 0,627 | 2,05% | 4,24% |
| LightGBM | 0,9675 | 0,9938 | 2,49% | 3,96% | 0,576 | 2,49% | 3,96% |
| Random Forest | 0,9395 | 0,9860 | 6,44% | 5,69% | 0,523 | 5,75% | 6,30% |

Holdout Stability Test qua 8 random seed (0, 7, 13, 42, 99, 123, 456, 2025) cho độ lệch chuẩn F1 = 0,00023.

---

## Cấu trúc thư mục

```
malware_detection_new/
├── data/              # Dữ liệu EMBER 2018 và dữ liệu đã xử lý (không đưa lên git)
├── training/          # Pipeline huấn luyện: step1 đến step7, config.py
├── models/            # xgb / lgb / rf .pkl, scaler.pkl, metadata.json (không đưa lên git)
├── results/
│   ├── reports/       # File CSV, TXT output của từng bước
│   └── plots/         # Biểu đồ PNG (ROC, SHAP, robustness, evidence, import)
├── web/               # Ứng dụng Flask: app.py, api/routes.py, templates/, static/
├── Biểu đồ/           # Biểu đồ dùng trong báo cáo
├── docx/              # Tài liệu báo cáo
├── debug_imports.py   # Script kiểm tra import
└── patch_ember.py     # Vá thư viện ember cho NumPy 2.x
```

> **Lưu ý:** `data/`, `data_01/`, `data_test_malware/`, `models/` rất nặng (hàng GB) nên không nằm trong repo. Cần tải dữ liệu và tự huấn luyện lại model theo hướng dẫn bên dưới.

---

## Cài đặt

Yêu cầu: Python 3.13, khuyến nghị RAM từ 16 GB trở lên (tối thiểu khoảng 8 GB nhờ cơ chế `numpy.memmap`).

```bash
git clone https://github.com/phuc123-hn/NCKH-01.git
cd NCKH-01

python -m venv venv
# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate

pip install -r web/requirements.txt
```

Các thư viện chính:

| Thư viện | Phiên bản |
|---|---|
| xgboost | 3.2.0 |
| lightgbm | 4.6.0 |
| scikit-learn | 1.8.0 |
| numpy | 2.4.4 |
| lief | 0.17.6 |
| shap | 0.52.0 |
| Flask | 3.1.3 |
| pandas | 3.0.3 |
| joblib | 1.5.3 |

### Vá thư viện ember cho NumPy 2.x

Thư viện `ember` gốc không tương thích NumPy 2.0 (`np.int` đã bị xóa) và có lỗi `FeatureHasher` trong `SectionInfo`. Chạy script vá trước khi trích xuất đặc trưng:

```bash
python patch_ember.py
```

---

## Huấn luyện mô hình

1. Tải EMBER 2018 và đặt vào `data/ember2018/`.
2. Chạy lần lượt các bước trong `training/`. Mỗi bước lưu output ra file (`.dat`, `.npy`, `.pkl`) nên có thể chạy lại từ bất kỳ bước nào.

| Bước | Nội dung |
|---|---|
| 1. Vectorize | Chuyển JSONL sang numpy memmap `.dat` (đơn luồng, do `multiprocessing.Pool` không tương thích Python 3.13 trên Windows) |
| 2. Split & Scale | Lọc unlabeled, chia 85/15, fit `StandardScaler` chỉ trên tập train |
| 3. Train | 5-Fold Stratified CV thủ công cho XGBoost, LightGBM, Random Forest |
| 4. Evaluate | F1, AUC, FPR/FNR, Youden's J, Holdout Stability, đo tốc độ inference |
| 5. SHAP | TreeExplainer và SHAP Stability Score |
| 6. Evidence | Evidence Vector và phân tích theo nhóm TP/TN/FP/FN |
| 7. Import Pattern | Dictionary 8 danh mục MITRE ATT&CK và đánh giá coverage |

Model sau huấn luyện được lưu vào `models/`.

---

## Chạy ứng dụng web

```bash
cd web
python app.py
```

Mở trình duyệt tại `http://127.0.0.1:5000`, kéo thả file `.exe` hoặc `.dll` (tối đa 50 MB) và nhấn **Phân tích file**.

### Pipeline phân tích

1. **Phase 1**: kiểm tra tính hợp lệ PE (MZ header `0x4D5A`, PE signature `0x50450000`, kích thước tối thiểu).
2. **Phase 2**: trích xuất 2.381 đặc trưng và chuẩn hóa bằng scaler đã huấn luyện.
3. **Phase 3**: XGBoost dự đoán với ngưỡng τ* = 0,627 và tính SHAP top features.
4. **Phase 4**: tính Evidence Vector (5 thành phần liên tục).
5. **Phase 5**: nhận dạng Import Pattern theo MITRE ATT&CK.

### Ví dụ JSON trả về

```json
{
  "verdict": "MALWARE",
  "confidence": 0.9998,
  "tau_star": 0.627,
  "evidence": {
    "packing_evidence": 0.69,
    "tls_evidence": 0.49,
    "import_evidence": 0.51,
    "header_evidence": 0.72,
    "loadcfg_evidence": 0.83
  },
  "shap_top": [
    {"feature": "feat_637", "value": 0.514, "direction": "malware"},
    {"feature": "feat_2359", "value": 0.510, "direction": "malware"}
  ],
  "import_pattern": {
    "score": 0.85,
    "categories": ["process_injection", "anti_analysis"],
    "apis_found": ["VirtualAllocEx", "CheckRemoteDebuggerPresent"],
    "mitre": ["T1055", "T1497"]
  }
}
```

---

## Import Pattern Engine

| Danh mục | Weight | MITRE | API mẫu |
|---|---|---|---|
| Process Injection | 0,95 | T1055 | VirtualAllocEx, WriteProcessMemory, CreateRemoteThread |
| Keylogger | 0,90 | T1056.001 | SetWindowsHookEx, GetAsyncKeyState |
| Network Download | 0,85 | T1105 | URLDownloadToFile, WinHttpSendRequest |
| Crypto/Ransomware | 0,80 | T1486 | CryptEncrypt, BCryptEncrypt |
| Anti-Analysis | 0,75 | T1497 | CheckRemoteDebuggerPresent, NtQueryInformationProcess |
| Persistence - Service | 0,75 | T1543.003 | CreateService, StartService |
| Persistence - Registry | 0,45 | T1547.001 | RegSetValueEx, RegCreateKeyEx |
| Process Execute | 0,40 | T1059 | CreateProcess, ShellExecute |

---

## Hạn chế

- **Stager/dropper nhỏ**: entropy thấp, import ít nên trông giống phần mềm hợp lệ; XGBoost thường có confidence thấp (0,50–0,75).
- **Không phân tích động**: không phát hiện được fileless malware hay polymorphic malware.
- **Domain shift**: EMBER 2018 thu thập năm 2018, cần retrain định kỳ với dữ liệu mới (ví dụ EMBER 2024).
- **Robustness**: F1 giảm mạnh khi thêm nhiễu Gaussian từ σ = 0,1 trở lên ở cả ba mô hình (XGBoost giảm ít nhất, 49,75% tại σ = 2,0).
- **Evidence Vector yếu ở mức population**: 4/5 chỉ số có Δ gần 0 do ngưỡng sigmoid đặt tay trên z-score sau StandardScaler. Vì vậy Evidence chỉ dùng để giải thích, còn giải thích per-sample bằng SHAP là phần hoạt động hiệu quả.
- **Import Pattern**: 24,3% file benign bị flag ít nhất một API đáng ngờ, và 54,9% malware không có API nào trong dictionary. Engine chỉ xét tên API, không xét ngữ cảnh sử dụng.
- **Ràng buộc phần cứng**: ma trận đặc trưng khoảng 7,2 GB nên pipeline dùng `numpy.memmap` và batch 50.000 mẫu.

---

## Hướng phát triển

- Tích hợp phân tích động (Cuckoo Sandbox, Any.run API).
- Retrain với EMBER 2024 hoặc mẫu từ MalwareBazaar.
- Calibrate ngưỡng sigmoid của Evidence Vector từ dữ liệu thay vì đặt tay.
- Mở rộng Import Pattern Engine (Heaven's Gate, Direct Syscall, rootkit, bootkit).
- Triển khai production: Gunicorn/uWSGI, authentication, rate limiting, audit logging.

---

## Tài liệu tham khảo

1. Anderson, H. S., & Roth, P. (2018). *EMBER: An Open Dataset for Training Static PE Malware Machine Learning Models.* arXiv:1804.04637.
2. Lundberg, S. M., & Lee, S.-I. (2017). *A Unified Approach to Interpreting Model Predictions.* NeurIPS 30.
3. Chen, T., & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System.* KDD '16.
4. Ke, G. et al. (2017). *LightGBM: A Highly Efficient Gradient Boosting Decision Tree.* NeurIPS 30.
5. Breiman, L. (2001). *Random Forests.* Machine Learning, 45(1).
6. Youden, W. J. (1950). *Index for rating diagnostic tests.* Cancer, 3(1).
7. FutureComputing4AI. *EMBER 2024.* https://github.com/FutureComputing4AI/EMBER2024
8. MITRE Corporation. *MITRE ATT&CK.* https://attack.mitre.org/

---

## Lưu ý sử dụng

Công cụ phục vụ **nghiên cứu và đào tạo**. Kết quả chỉ mang tính hỗ trợ phân tích, không thay thế giải pháp bảo mật chuyên nghiệp. Không thực thi các file mẫu nghi ngờ trên máy thật.
