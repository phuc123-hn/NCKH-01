"""
Patch ember/features.py de tuong thich voi:
- numpy >= 2.0 (bo np.int)
- lief >= 0.12 (thay doi API)
Chay 1 lan, sau do import ember binh thuong.
"""
import sys
import subprocess

def find_ember_features():
    """Tim duong dan den ember/features.py."""
    import ember.features as ef
    import inspect
    return inspect.getfile(ef)

def patch_file(path: str):
    print(f"Patching: {path}")
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    original = content

    # Fix 1: np.int -> np.int64 (numpy 2.0 bo np.int)
    content = content.replace(
        "dtype=np.int)",
        "dtype=np.int64)"
    )

    # Fix 2: lief.not_found -> lief.lief_errors.not_found (lief >= 0.12)
    # Trong lief moi, exception la lief.lief_errors.not_found
    # Nhung process_raw_features chi dung voi dict (khong can lief)
    # nen khong can fix phan nay

    if content == original:
        print("  [OK] Khong can patch (da duoc fix hoac khac version)")
    else:
        # Backup
        backup = path + ".backup"
        with open(backup, "w", encoding="utf-8") as f:
            f.write(original)
        print(f"  Backup luu tai: {backup}")

        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        print("  [OK] Da patch thanh cong")

def verify_patch():
    """Kiem tra process_raw_features chay duoc voi JSONL dict."""
    import importlib
    import ember.features as ef
    importlib.reload(ef)

    import json
    extractor = ef.PEFeatureExtractor(feature_version=2)

    # Test voi mot dong JSONL that
    import sys
    jsonl_path = sys.argv[1] if len(sys.argv) > 1 else "data/ember2018/train_features_0.jsonl"

    with open(jsonl_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                obj = json.loads(line)
                break

    errors = 0
    for fe in extractor.features:
        name = fe.name
        if name not in obj:
            print(f"  SKIP {name}: key khong co trong JSONL")
            continue
        try:
            vec = fe.process_raw_features(obj[name])
            print(f"  OK   {name}: shape={vec.shape}, mean={vec.mean():.4f}")
        except Exception as e:
            print(f"  LOI  {name}: {e}")
            errors += 1

    return errors == 0

if __name__ == "__main__":
    path = find_ember_features()
    patch_file(path)

    print("\n=== Kiem tra sau patch ===")
    ok = verify_patch()
    if ok:
        print("\n[OK] Ember san sang - chay lai step1_vectorize.py")
    else:
        print("\n[WARN] Van con loi - xem chi tiet o tren")