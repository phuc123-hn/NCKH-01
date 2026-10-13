"""
Chạy từ thư mục gốc project:
  python debug_imports.py

Mục đích: xem import table của malware trong EMBER JSONL
để biết có URLDownloadToFile, SetWindowsHookEx, VirtualAllocEx không
"""
import json
from pathlib import Path

DATA_DIR = Path("data/ember2018")

# Thử lần lượt các file train
jsonl_files = sorted(DATA_DIR.glob("train_features_*.jsonl"))
print(f"Tìm thấy {len(jsonl_files)} file JSONL\n")

found = 0
target = 3  # Tìm 3 malware có imports

# Suspicious APIs quan tâm
SUSPICIOUS = {
    "download"  : ["URLDownloadToFile", "WinHttpSendRequest",
                   "InternetOpenUrl", "InternetReadFile"],
    "execute"   : ["WinExec", "ShellExecute", "CreateProcess",
                   "ShellExecuteEx"],
    "inject"    : ["VirtualAllocEx", "WriteProcessMemory",
                   "CreateRemoteThread", "NtUnmapViewOfSection"],
    "keylog"    : ["SetWindowsHookEx", "GetAsyncKeyState",
                   "GetKeyboardState"],
    "socket"    : ["WSAConnect", "connect", "send", "recv"],
    "persist"   : ["RegSetValueEx", "CreateService", "SchtasksCreate"],
}

for jsonl_path in jsonl_files:
    if found >= target:
        break
    print(f"=== Scanning {jsonl_path.name} ===")

    with open(jsonl_path, 'r', errors='replace') as f:
        for i, line in enumerate(f):
            if found >= target:
                break
            try:
                obj = json.loads(line.strip())
            except Exception:
                continue

            label   = obj.get('label', -1)
            imports = obj.get('imports', {})

            if label == 1 and imports:
                # Flatten tất cả function names
                all_funcs = []
                for dll, funcs in imports.items():
                    if isinstance(funcs, list):
                        all_funcs.extend(funcs)

                if not all_funcs:
                    continue

                # Tìm suspicious APIs
                hits = {}
                for category, apis in SUSPICIOUS.items():
                    matched = [f for f in all_funcs
                               if any(a.lower() in f.lower() for a in apis)]
                    if matched:
                        hits[category] = matched[:3]

                found += 1
                print(f"\n  Malware #{found} (line {i}):")
                print(f"  sha256   : {obj.get('sha256','')[:20]}...")
                print(f"  DLLs     : {list(imports.keys())[:6]}")
                print(f"  Total funcs: {len(all_funcs)}")

                if hits:
                    print(f"  SUSPICIOUS APIs found:")
                    for cat, apis in hits.items():
                        print(f"    [{cat}] {apis}")
                else:
                    print(f"  No suspicious APIs in top list")
                    print(f"  Sample funcs: {all_funcs[:8]}")

# Thống kê nhanh: bao nhiêu % malware có imports
print("\n\n=== THỐNG KÊ NHANH (1000 mẫu đầu file 0) ===")
counts = {'total': 0, 'malware': 0, 'mal_with_imports': 0,
          'mal_with_suspicious': 0}

with open(jsonl_files[0], 'r', errors='replace') as f:
    for i, line in enumerate(f):
        if i >= 1000:
            break
        try:
            obj = json.loads(line.strip())
        except Exception:
            continue

        counts['total'] += 1
        label   = obj.get('label', -1)
        imports = obj.get('imports', {})

        if label == 1:
            counts['malware'] += 1
            all_funcs = []
            for dll, funcs in imports.items():
                if isinstance(funcs, list):
                    all_funcs.extend(funcs)

            if all_funcs:
                counts['mal_with_imports'] += 1

                for category, apis in SUSPICIOUS.items():
                    if any(a.lower() in f.lower()
                           for f in all_funcs for a in apis):
                        counts['mal_with_suspicious'] += 1
                        break

print(f"  Tổng mẫu     : {counts['total']}")
print(f"  Malware      : {counts['malware']}")
print(f"  Mal + imports: {counts['mal_with_imports']}"
      f" ({counts['mal_with_imports']/max(counts['malware'],1)*100:.1f}%)")
print(f"  Mal + suspicious APIs: {counts['mal_with_suspicious']}"
      f" ({counts['mal_with_suspicious']/max(counts['malware'],1)*100:.1f}%)")

print("\n[XONG] Dùng kết quả này để quyết định có làm Import Pattern Evidence không")