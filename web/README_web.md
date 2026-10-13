# Web — Hướng dẫn cài đặt và chạy

## Cấu trúc file

Đặt file theo đúng cấu trúc này:

```
malware_detection_new/
├── models/artifacts/
│   ├── xgb.pkl
│   ├── evidence_config.json
│   └── import_pattern_config.json
├── data/processed/
│   └── scaler.pkl
└── web/
    ├── app.py              ← backend Flask
    ├── uploads/            ← tạo tự động khi chạy
    ├── templates/
    │   └── index.html      ← giao diện chính
    ├── requirements.txt
    └── README.md
```

## Cài đặt

```bash
cd web
pip install -r requirements.txt
```

## Chạy

```bash
# Từ thư mục web/
python app.py
```

Mở trình duyệt: http://localhost:5000

## Khi không có model (demo UI)

Nếu chưa có `xgb.pkl` hoặc `scaler.pkl`, server tự động dùng mock data
để test giao diện — confidence random, evidence random. Mọi bước chat
log vẫn chạy đầy đủ.

## Thêm Anthropic API (thông dịch trung gian thật)

Khi có API key, thay hàm `mock_ai_interpretation()` trong `app.py`:

```python
import anthropic

def ai_interpretation(result: dict) -> str:
    client = anthropic.Anthropic(api_key="sk-ant-...")

    prompt = f"""
Bạn là chuyên gia bảo mật. Phân tích kết quả scan file PE sau
và giải thích bằng ngôn ngữ tự nhiên, dễ hiểu cho người dùng cuối:

Verdict: {result['verdict']}
Confidence: {result['confidence']*100:.1f}%
Evidence: {result['evidence']}
Import Pattern: {result['import_pattern']}
SHAP top features: {result['shap_top']}

Giải thích ngắn gọn, rõ ràng, không dùng thuật ngữ kỹ thuật quá nhiều.
Nêu rõ tại sao file này bị phán là malware (hoặc benign).
Cuối cùng đưa ra khuyến nghị cụ thể.
"""

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=600,
        messages=[{"role": "user", "content": prompt}]
    )
    return message.content[0].text
```

## Các endpoint

| Method | Route | Mô tả |
|--------|-------|-------|
| GET  | `/` | Trang upload |
| POST | `/scan` | Upload file, tạo job → trả `job_id` |
| GET  | `/stream/<job_id>` | SSE stream từng bước |
| GET  | `/result/<job_id>` | Kết quả đầy đủ (JSON) |