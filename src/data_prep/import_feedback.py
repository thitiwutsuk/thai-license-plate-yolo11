"""
ตรวจสอบและนำเข้า feedback ที่เก็บจากเว็ปแอป (outputs/feedback/) เข้าสู่ data/raw/ จริง
สำหรับรอบเทรนถัดไป — ต้องผ่านการตรวจด้วยตาของ maintainer ก่อนเสมอ ไม่มีการนำเข้าอัตโนมัติ

ขั้นตอนใช้งาน:
  1. python src/data_prep/import_feedback.py --review
     สร้าง contact sheet ที่ outputs/feedback/review/ ให้ดูด้วยตาว่ากรอบที่ผู้ใช้ยืนยัน/วาด
     ถูกต้องจริงไหม (คล้าย visualize_labels.py)

  2. python src/data_prep/import_feedback.py --import fb_20261106..,fb_20261106..
     นำเข้าเฉพาะ image_id ที่ตรวจแล้วว่าใช้ได้ (คอมม่าคั่น) เข้า data/raw/ + data/labels/
     พร้อมอัปเดต dataset_manifest.csv และทำ marker กันนำเข้าซ้ำใน feedback_log.csv

  3. python src/data_prep/import_feedback.py --import all
     นำเข้าทุกแถวที่ยังไม่เคย import (ใช้เมื่อตรวจสอบจาก --review แล้วว่าทุกรูปโอเค)
"""

import argparse
import csv
import re
import shutil
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[2]
FEEDBACK_DIR = REPO_ROOT / "outputs" / "feedback"
FB_IMAGES_DIR = FEEDBACK_DIR / "images"
FB_LABELS_DIR = FEEDBACK_DIR / "labels"
LOG_PATH = FEEDBACK_DIR / "feedback_log.csv"
REVIEW_DIR = FEEDBACK_DIR / "review"

RAW_DIR = REPO_ROOT / "data" / "raw"
LABELS_DIR = REPO_ROOT / "data" / "labels"
MANIFEST_PATH = RAW_DIR / "dataset_manifest.csv"
MANIFEST_FIELDS = ["filename", "source", "author", "license", "source_url", "sha256", "width", "height", "status"]

# verdict ที่มี label YOLO พร้อมใช้เทรน (ดู src/app/feedback_utils.py)
LABELED_VERDICTS = {"correct", "corrected", "found_missed"}

THUMB = 320
COLS, ROWS = 4, 3
LABEL_H = 20
PER_SHEET = COLS * ROWS


def load_log() -> list[dict]:
    if not LOG_PATH.exists():
        return []
    with open(LOG_PATH, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def save_log(rows: list[dict]) -> None:
    fieldnames = list(rows[0].keys()) if rows else [
        "timestamp", "image_id", "source_filename", "model_box_xyxy",
        "model_confidence", "verdict", "corrected_yolo_box", "note", "imported",
    ]
    with open(LOG_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def pending_rows(rows: list[dict]) -> list[dict]:
    """แถวที่ยังไม่เคย import และมี label ใช้ได้จริง"""
    out = []
    for row in rows:
        if row.get("imported"):
            continue
        if row.get("verdict") not in LABELED_VERDICTS:
            continue
        label_path = FB_LABELS_DIR / f"{row['image_id']}.txt"
        image_path = FB_IMAGES_DIR / f"{row['image_id']}.jpg"
        if label_path.exists() and image_path.exists():
            out.append(row)
    return out


def draw_boxes(img_path: Path, label_path: Path) -> Image.Image:
    im = Image.open(img_path).convert("RGB")
    w, h = im.size
    draw = ImageDraw.Draw(im)
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        _, cx, cy, bw, bh = parts
        cx, cy, bw, bh = float(cx) * w, float(cy) * h, float(bw) * w, float(bh) * h
        x0, y0 = cx - bw / 2, cy - bh / 2
        x1, y1 = cx + bw / 2, cy + bh / 2
        draw.rectangle([x0, y0, x1, y1], outline=(248, 189, 56), width=max(2, w // 300))
    return im


def make_review_sheets(rows: list[dict]) -> None:
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
    except Exception:
        font = ImageFont.load_default()

    for sheet_idx in range(0, len(rows), PER_SHEET):
        batch = rows[sheet_idx:sheet_idx + PER_SHEET]
        sheet = Image.new("RGB", (COLS * THUMB, ROWS * (THUMB + LABEL_H)), (30, 30, 30))
        draw = ImageDraw.Draw(sheet)
        for i, row in enumerate(batch):
            image_id = row["image_id"]
            img_path = FB_IMAGES_DIR / f"{image_id}.jpg"
            label_path = FB_LABELS_DIR / f"{image_id}.txt"
            annotated = draw_boxes(img_path, label_path)
            annotated.thumbnail((THUMB, THUMB))
            x = (i % COLS) * THUMB
            y = (i // COLS) * (THUMB + LABEL_H)
            sheet.paste(annotated, (x + (THUMB - annotated.width) // 2, y + (THUMB - annotated.height) // 2))
            draw.rectangle([x, y + THUMB, x + THUMB, y + THUMB + LABEL_H], fill=(0, 0, 0))
            caption = f"{image_id} [{row['verdict']}]"
            draw.text((x + 4, y + THUMB + 2), caption[:30], fill=(255, 255, 0), font=font)

        out_path = REVIEW_DIR / f"feedback_review_{sheet_idx:03d}.jpg"
        sheet.save(out_path, quality=85)
        print(f"บันทึก {out_path}")


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


def next_raw_index(existing_rows: list[dict]) -> int:
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


def do_import(rows_to_import: list[dict]) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    manifest_rows = load_manifest()
    idx = next_raw_index(manifest_rows)

    for row in rows_to_import:
        image_id = row["image_id"]
        src_img = FB_IMAGES_DIR / f"{image_id}.jpg"
        src_label = FB_LABELS_DIR / f"{image_id}.txt"

        dst_name = f"raw_{idx:03d}_feedback_{image_id}.jpg"
        dst_img = RAW_DIR / dst_name
        dst_label = LABELS_DIR / f"{Path(dst_name).stem}.txt"

        shutil.copy2(src_img, dst_img)
        shutil.copy2(src_label, dst_label)

        with Image.open(dst_img) as im:
            w, h = im.size

        manifest_rows.append({
            "filename": dst_name,
            "source": "User feedback",
            "author": "",
            "license": "",
            "source_url": row.get("source_filename", ""),
            "sha256": "",
            "width": w,
            "height": h,
            # ผ่านการตรวจของ maintainer แล้วผ่าน --review ก่อนเรียก --import เสมอ จึงตั้ง passed ได้เลย
            "status": "passed",
        })
        print(f"[นำเข้า] {image_id} -> {dst_name}")
        idx += 1

    save_manifest(manifest_rows)

    # mark ใน feedback_log.csv กันนำเข้าซ้ำ
    all_rows = load_log()
    imported_ids = {r["image_id"] for r in rows_to_import}
    for row in all_rows:
        if row["image_id"] in imported_ids:
            row["imported"] = "yes"
    save_log(all_rows)
    print(f"\nนำเข้าทั้งหมด {len(rows_to_import)} รูป เข้า data/raw/ + data/labels/ แล้ว (status=passed)")
    print("ขั้นตอนถัดไป: รัน src/data_prep/split_folds.py ใหม่เพื่อรวมรูปเหล่านี้เข้ารอบเทรนถัดไป")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--review", action="store_true", help="สร้าง contact sheet ให้ตรวจด้วยตา (ไม่นำเข้าอะไร)")
    parser.add_argument("--import", dest="import_ids", default=None,
                         help='image_id ที่จะนำเข้า คั่นด้วยคอมม่า หรือ "all" สำหรับทุกแถวที่ค้างอยู่')
    args = parser.parse_args()

    all_rows = load_log()
    candidates = pending_rows(all_rows)

    if not candidates:
        print("ไม่มี feedback ใหม่ที่พร้อมนำเข้า (ต้องมี verdict ที่มี label และยังไม่เคย import)")
        return

    if args.review:
        print(f"พบ {len(candidates)} รายการที่รอตรวจ กำลังสร้าง contact sheet...")
        make_review_sheets(candidates)
        print(f"\nตรวจดูที่ {REVIEW_DIR} แล้วค่อยรัน --import ตาม image_id ที่โอเค")
        return

    if args.import_ids:
        if args.import_ids.strip().lower() == "all":
            selected = candidates
        else:
            wanted = {s.strip() for s in args.import_ids.split(",") if s.strip()}
            selected = [r for r in candidates if r["image_id"] in wanted]
            missing = wanted - {r["image_id"] for r in selected}
            if missing:
                print(f"หมายเหตุ: ไม่พบหรือ import ไปแล้ว: {', '.join(sorted(missing))}")
        if not selected:
            print("ไม่มีรายการที่ตรงกับ image_id ที่ระบุ")
            return
        do_import(selected)
        return

    parser.print_help()


if __name__ == "__main__":
    main()
