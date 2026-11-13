"""
ตรวจสอบไฟล์ label (YOLO format) ที่ตีกรอบไว้คู่กับรูปใน data/raw/
ใช้หลังตีกรอบเสร็จแต่ละชุด (หรือทั้งหมด) เพื่อจับข้อผิดพลาดก่อนเข้า Phase 3

วิธีใช้:
  python src/data_prep/verify_labels.py
"""

from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw"
LABELS_DIR = REPO_ROOT / "data" / "labels"
CLASSES_PATH = REPO_ROOT / "data" / "classes.txt"

MIN_BOX_AREA_RATIO = 0.0005  # กรอบเล็กกว่านี้ของพื้นที่รูป น่าจะผิดพลาด (คลิกพลาด)
MAX_BOX_AREA_RATIO = 0.5     # กรอบใหญ่กว่านี้ น่าจะผิดพลาด (ลากครอบทั้งรถ)


def load_num_classes() -> int:
    if not CLASSES_PATH.exists():
        print(f"[เตือน] ไม่พบ {CLASSES_PATH} จะข้ามการเช็ค class id")
        return -1
    with open(CLASSES_PATH, encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def main():
    num_classes = load_num_classes()
    image_files = sorted(p for p in RAW_DIR.glob("*.jpg"))
    print(f"รูปใน data/raw/: {len(image_files)} ไฟล์")

    missing_labels = []
    empty_labels = []
    bad_format = []
    bad_box_size = []
    multi_box = []
    ok_count = 0

    for img_path in image_files:
        label_path = LABELS_DIR / (img_path.stem + ".txt")
        if not label_path.exists():
            missing_labels.append(img_path.name)
            continue

        lines = [l.strip() for l in label_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        if not lines:
            empty_labels.append(img_path.name)
            continue

        if len(lines) > 1:
            multi_box.append((img_path.name, len(lines)))

        try:
            im = Image.open(img_path)
            img_w, img_h = im.size
        except Exception:
            img_w, img_h = None, None

        file_ok = True
        for line in lines:
            parts = line.split()
            if len(parts) != 5:
                bad_format.append((label_path.name, line))
                file_ok = False
                continue
            try:
                cls_id = int(parts[0])
                x, y, w, h = (float(v) for v in parts[1:])
            except ValueError:
                bad_format.append((label_path.name, line))
                file_ok = False
                continue

            if num_classes > 0 and not (0 <= cls_id < num_classes):
                bad_format.append((label_path.name, f"class id ผิดช่วง: {line}"))
                file_ok = False

            if not all(0.0 <= v <= 1.0 for v in (x, y, w, h)):
                bad_format.append((label_path.name, f"ค่านอกช่วง 0-1: {line}"))
                file_ok = False

            if img_w and img_h:
                area_ratio = w * h
                if area_ratio < MIN_BOX_AREA_RATIO:
                    bad_box_size.append((label_path.name, f"กรอบเล็กผิดปกติ ({area_ratio:.5f} ของพื้นที่รูป)"))
                elif area_ratio > MAX_BOX_AREA_RATIO:
                    bad_box_size.append((label_path.name, f"กรอบใหญ่ผิดปกติ ({area_ratio:.3f} ของพื้นที่รูป)"))

        if file_ok:
            ok_count += 1

    print()
    print(f"ปกติ: {ok_count} / {len(image_files)}")
    print(f"ไม่มีไฟล์ label: {len(missing_labels)}")
    for f in missing_labels[:20]:
        print(f"  - {f}")
    if len(missing_labels) > 20:
        print(f"  ... อีก {len(missing_labels) - 20} ไฟล์")

    print(f"\nไฟล์ label ว่างเปล่า: {len(empty_labels)}")
    for f in empty_labels[:20]:
        print(f"  - {f}")

    print(f"\nรูปแบบผิด (ไม่ใช่ 5 ค่า / class ผิด / ค่านอกช่วง): {len(bad_format)}")
    for f, detail in bad_format[:20]:
        print(f"  - {f}: {detail}")

    print(f"\nขนาดกรอบผิดปกติ: {len(bad_box_size)}")
    for f, detail in bad_box_size[:20]:
        print(f"  - {f}: {detail}")

    print(f"\nรูปที่มีมากกว่า 1 กรอบ (ตรวจว่าตั้งใจหรือพลาด): {len(multi_box)}")
    for f, n in multi_box[:20]:
        print(f"  - {f}: {n} กรอบ")

    total_issues = len(missing_labels) + len(empty_labels) + len(bad_format) + len(bad_box_size)
    print()
    if total_issues == 0:
        print("✅ ผ่านการตรวจสอบทั้งหมด พร้อมเข้า Phase 3 ได้")
    else:
        print(f"⚠️  พบปัญหารวม {total_issues} รายการ ต้องแก้ก่อนเข้า Phase 3")


if __name__ == "__main__":
    main()
