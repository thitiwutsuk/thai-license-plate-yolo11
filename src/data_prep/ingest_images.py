"""
รับรูปใหม่ (ถ่ายเองหรือโหลดจากเน็ต) เข้าสู่ data/raw/ แบบมาตรฐานเดียวกัน:
  - resize ด้านยาวสุดไม่เกิน MAX_SIDE
  - แปลงเป็น JPEG คุณภาพ JPEG_QUALITY
  - ตั้งชื่อไฟล์ต่อเนื่องจากไฟล์ล่าสุดใน data/raw/
  - กันไฟล์ซ้ำด้วย sha256 hash
  - บันทึก/อัปเดต data/raw/dataset_manifest.csv

วิธีใช้:
  python src/data_prep/ingest_images.py --input path/to/new_photos_folder --source self
  python src/data_prep/ingest_images.py --input path/to/downloaded_folder --source "Wikimedia Commons" --license "CC BY-SA 4.0" --author "ชื่อผู้ถ่าย" --url "https://..."
"""

import argparse
import csv
import hashlib
import re
from pathlib import Path

from PIL import Image

MAX_SIDE = 1280
JPEG_QUALITY = 85

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw"
MANIFEST_PATH = RAW_DIR / "dataset_manifest.csv"
MANIFEST_FIELDS = ["filename", "source", "author", "license", "source_url", "sha256", "width", "height", "status"]


def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def load_manifest() -> list[dict]:
    if not MANIFEST_PATH.exists():
        return []
    with open(MANIFEST_PATH, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def save_manifest(rows: list[dict]) -> None:
    with open(MANIFEST_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def next_index(existing_rows: list[dict]) -> int:
    max_idx = 0
    pattern = re.compile(r"raw_(\d+)_")
    for row in existing_rows:
        m = pattern.match(row["filename"])
        if m:
            max_idx = max(max_idx, int(m.group(1)))
    for p in RAW_DIR.glob("raw_*"):
        m = pattern.match(p.name)
        if m:
            max_idx = max(max_idx, int(m.group(1)))
    return max_idx + 1


def resize_and_save(src_path: Path, dst_path: Path) -> tuple[int, int]:
    im = Image.open(src_path).convert("RGB")
    w, h = im.size
    scale = MAX_SIDE / max(w, h)
    if scale < 1:
        im = im.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    im.save(dst_path, "JPEG", quality=JPEG_QUALITY, optimize=True)
    return im.size


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="โฟลเดอร์ที่มีรูปใหม่รอ ingest")
    parser.add_argument("--source", required=True, help='แหล่งที่มา เช่น "self" หรือ "Wikimedia Commons"')
    parser.add_argument("--author", default="", help="ผู้ถ่าย/เจ้าของลิขสิทธิ์ (ถ้ามี)")
    parser.add_argument("--license", default="", help='สัญญาอนุญาต เช่น "CC BY-SA 4.0" (ใส่ "self" ถ้าถ่ายเอง)')
    parser.add_argument("--url", default="", help="ลิงก์อ้างอิงต้นทาง (ถ้ามี)")
    args = parser.parse_args()

    input_dir = Path(args.input)
    if not input_dir.is_dir():
        raise SystemExit(f"ไม่พบโฟลเดอร์: {input_dir}")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_manifest()
    known_hashes = {row["sha256"] for row in rows}

    idx = next_index(rows)
    added, skipped_dupe, skipped_error = 0, 0, 0

    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    candidates = sorted(p for p in input_dir.iterdir() if p.suffix.lower() in exts)

    for src in candidates:
        try:
            file_hash = sha256_of_file(src)
        except OSError:
            print(f"[ข้าม] อ่านไฟล์ไม่ได้: {src.name}")
            skipped_error += 1
            continue

        if file_hash in known_hashes:
            print(f"[ข้าม] ซ้ำกับรูปที่มีอยู่แล้ว: {src.name}")
            skipped_dupe += 1
            continue

        stem = re.sub(r"[^\w\-]+", "_", src.stem, flags=re.UNICODE)[:60]
        dst_name = f"raw_{idx:03d}_{stem}.jpg"
        dst_path = RAW_DIR / dst_name

        try:
            w, h = resize_and_save(src, dst_path)
        except Exception as e:
            print(f"[ข้าม] resize ไม่สำเร็จ: {src.name} ({e})")
            skipped_error += 1
            continue

        rows.append({
            "filename": dst_name,
            "source": args.source,
            "author": args.author,
            "license": args.license,
            "source_url": args.url,
            "sha256": file_hash,
            "width": w,
            "height": h,
            "status": "pending_review",
        })
        known_hashes.add(file_hash)
        print(f"[เพิ่ม] {src.name} -> {dst_name} ({w}x{h})")
        idx += 1
        added += 1

    save_manifest(rows)

    passed = sum(1 for r in rows if r["status"] == "passed")
    print()
    print(f"เพิ่มใหม่ {added} รูป | ข้ามเพราะซ้ำ {skipped_dupe} | ข้ามเพราะ error {skipped_error}")
    print(f"รวมทั้งหมดใน manifest: {len(rows)} รูป (ผ่านตรวจแล้ว {passed} / 250)")
    print(f'หมายเหตุ: รูปใหม่ถูกตั้งสถานะเป็น "pending_review" — ต้องตรวจเงื่อนไข (หน้าตรง/ระยะ/เห็นป้ายครบ) แล้วแก้คอลัมน์ status เป็น "passed" หรือ "rejected" เองใน dataset_manifest.csv')


if __name__ == "__main__":
    main()
