"""
วาดกรอบ (YOLO format) ทับรูปจริง แล้วรวมเป็น contact sheet เพื่อตรวจด้วยตา
ว่าตีกรอบตรงตำแหน่งป้ายจริงหรือไม่ (ก่อนเข้า Phase 3)

วิธีใช้:
  python src/data_prep/visualize_labels.py                 # สุ่มตรวจ 36 รูป
  python src/data_prep/visualize_labels.py --count 60       # สุ่มตรวจ 60 รูป
  python src/data_prep/visualize_labels.py --all            # ตรวจทุกรูปที่มี label (หลายแผ่น)
"""

import argparse
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw"
LABELS_DIR = REPO_ROOT / "data" / "labels"
OUT_DIR = REPO_ROOT / "outputs" / "label_review"

THUMB = 320
COLS, ROWS = 4, 3
LABEL_H = 20
PER_SHEET = COLS * ROWS


def draw_boxes(img_path: Path, label_path: Path) -> Image.Image:
    im = Image.open(img_path).convert("RGB")
    w, h = im.size
    draw = ImageDraw.Draw(im)
    if label_path.exists():
        for line in label_path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            _, cx, cy, bw, bh = parts
            cx, cy, bw, bh = float(cx) * w, float(cy) * h, float(bw) * w, float(bh) * h
            x0, y0 = cx - bw / 2, cy - bh / 2
            x1, y1 = cx + bw / 2, cy + bh / 2
            draw.rectangle([x0, y0, x1, y1], outline=(255, 0, 0), width=max(2, w // 300))
    else:
        draw.text((10, 10), "NO LABEL", fill=(255, 0, 0))
    return im


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=36)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    image_files = sorted(RAW_DIR.glob("*.jpg"))

    if args.all:
        selected = image_files
    else:
        random.seed(args.seed)
        selected = random.sample(image_files, min(args.count, len(image_files)))
        selected.sort()

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
    except Exception:
        font = ImageFont.load_default()

    for sheet_idx in range(0, len(selected), PER_SHEET):
        batch = selected[sheet_idx:sheet_idx + PER_SHEET]
        sheet = Image.new("RGB", (COLS * THUMB, ROWS * (THUMB + LABEL_H)), (30, 30, 30))
        draw = ImageDraw.Draw(sheet)
        for i, img_path in enumerate(batch):
            label_path = LABELS_DIR / (img_path.stem + ".txt")
            annotated = draw_boxes(img_path, label_path)
            annotated.thumbnail((THUMB, THUMB))
            x = (i % COLS) * THUMB
            y = (i // COLS) * (THUMB + LABEL_H)
            sheet.paste(annotated, (x + (THUMB - annotated.width) // 2, y + (THUMB - annotated.height) // 2))
            draw.rectangle([x, y + THUMB, x + THUMB, y + THUMB + LABEL_H], fill=(0, 0, 0))
            draw.text((x + 4, y + THUMB + 2), img_path.stem[:28], fill=(255, 255, 0), font=font)

        out_path = OUT_DIR / f"label_check_{sheet_idx:03d}.jpg"
        sheet.save(out_path, quality=85)
        print(f"บันทึก {out_path}")

    print(f"\nรวม {len(selected)} รูป ({(len(selected) + PER_SHEET - 1) // PER_SHEET} แผ่น) ที่ {OUT_DIR}")


if __name__ == "__main__":
    main()
