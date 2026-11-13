"""
หาค่า confidence threshold ที่เหมาะสมกว่าค่าคงที่ 0.5 เดิม
โดยรันโมเดลครั้งเดียวที่ conf ต่ำมาก (เก็บทุกกรอบที่เป็นไปได้) แล้วกรองด้วย Python
ในแต่ละค่า threshold ที่จะลอง — ไม่ต้องรันโมเดลซ้ำหลายรอบ

สำหรับแต่ละรูปในชุดทดสอบ (data/test/, 50 รูป, 1 กรอบจริงต่อรูป):
  ที่ threshold หนึ่งๆ ถ้ามีกรอบที่ conf >= threshold:
    เลือกกรอบที่ใหญ่ที่สุด (ตามกติกาเดียวกับ predict_and_crop.py)
    ถ้า IoU กับกรอบจริง >= 0.5 -> "ครอปสำเร็จ"
    ถ้า IoU < 0.5             -> "ตำแหน่งผิด" (ครอปพลาด)
  ถ้าไม่มีกรอบเลยที่ conf >= threshold -> "ไม่พบป้าย" (ครอปพลาด)

วิธีใช้:
  python src/train/tune_threshold.py
  python src/train/tune_threshold.py --weights models/best.pt --device mps
"""

import argparse
from pathlib import Path

from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parents[2]
TEST_DIR = REPO_ROOT / "data" / "test"
OUT_DIR = REPO_ROOT / "outputs" / "predictions"

THRESHOLDS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]


def load_gt_box(label_path: Path):
    if not label_path.exists():
        return None
    lines = [l.strip() for l in label_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not lines:
        return None
    _, cx, cy, w, h = lines[0].split()
    return float(cx), float(cy), float(w), float(h)


def to_xyxy(cx, cy, w, h):
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def iou(box_a, box_b):
    ax0, ay0, ax1, ay1 = box_a
    bx0, by0, bx1, by1 = box_b
    inter_x0, inter_y0 = max(ax0, bx0), max(ay0, by0)
    inter_x1, inter_y1 = min(ax1, bx1), min(ay1, by1)
    inter_w, inter_h = max(0.0, inter_x1 - inter_x0), max(0.0, inter_y1 - inter_y0)
    inter_area = inter_w * inter_h
    area_a = (ax1 - ax0) * (ay1 - ay0)
    area_b = (bx1 - bx0) * (by1 - by0)
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default=str(REPO_ROOT / "models" / "best.pt"))
    parser.add_argument("--device", default="mps")
    parser.add_argument("--iou-thresh", type=float, default=0.5)
    args = parser.parse_args()

    weights_path = Path(args.weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"ไม่พบไฟล์โมเดล {weights_path}")

    model = YOLO(str(weights_path))

    image_files = sorted((TEST_DIR / "images").glob("*.jpg"))
    total = len(image_files)

    # รันครั้งเดียวที่ conf ต่ำมาก เก็บกรอบ+conf ทั้งหมดต่อรูปไว้ก่อน
    per_image = []  # list of (gt_xyxy, [(conf, area, xyxy), ...])
    for img_path in image_files:
        label_path = TEST_DIR / "labels" / (img_path.stem + ".txt")
        gt = load_gt_box(label_path)
        if gt is None:
            continue
        gt_xyxy = to_xyxy(*gt)

        results = model.predict(source=str(img_path), conf=0.01, device=args.device, verbose=False)
        boxes = results[0].boxes
        candidates = []
        if boxes is not None and len(boxes) > 0:
            for i in range(len(boxes)):
                xywhn = boxes.xywhn[i].tolist()
                conf = float(boxes.conf[i])
                area = xywhn[2] * xywhn[3]
                candidates.append((conf, area, to_xyxy(*xywhn)))
        per_image.append((gt_xyxy, candidates))

    print("=" * 70)
    print(f"ไล่ทดสอบ confidence threshold บนชุดทดสอบ {total} รูป")
    print("=" * 70)
    print(f"{'threshold':>10} | {'ครอปสำเร็จ':>12} | {'ไม่พบป้าย':>10} | {'ตำแหน่งผิด':>10}")
    print("-" * 70)

    rows = []
    for t in THRESHOLDS:
        success = 0
        no_detection = 0
        wrong_location = 0
        for gt_xyxy, candidates in per_image:
            filtered = [c for c in candidates if c[0] >= t]
            if not filtered:
                no_detection += 1
                continue
            # เลือกกรอบใหญ่ที่สุดในบรรดากรอบที่ผ่านเกณฑ์ (ตามกติกาเดียวกับ predict_and_crop.py)
            best = max(filtered, key=lambda c: c[1])
            score = iou(best[2], gt_xyxy)
            if score >= args.iou_thresh:
                success += 1
            else:
                wrong_location += 1

        rate = success / total if total else 0.0
        rows.append((t, success, no_detection, wrong_location, rate))
        print(f"{t:>10.2f} | {success:>5}/{total} ({rate*100:5.1f}%) | {no_detection:>10} | {wrong_location:>10}")

    best_row = max(rows, key=lambda r: r[4])
    print("-" * 70)
    print(f"\nค่า threshold ที่ให้อัตราครอปสำเร็จสูงสุดบนชุดทดสอบนี้: {best_row[0]:.2f} "
          f"({best_row[1]}/{total} = {best_row[4]*100:.1f}%)")
    print("\nหมายเหตุ: ชุดทดสอบนี้ทุกรูปมีป้ายจริงอยู่แล้ว (ไม่มีรูปที่ไม่ควรเจอป้ายเลย)")
    print("จึงวัด false positive (เช่น กรณีกระดาษเอกสารที่เคยเจอ) ไม่ได้จากชุดนี้โดยตรง")
    print("การขยับ threshold สูงขึ้นเล็กน้อยจากจุดนี้ช่วยลดความเสี่ยง false positive ในโลกจริง")
    print("โดยแลกกับอัตราครอปสำเร็จที่ลดลงเล็กน้อยตามตารางด้านบน")

    summary_path = OUT_DIR / "threshold_sweep.csv"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("threshold,crop_success_count,no_detection_count,wrong_location_count,crop_success_rate\n")
        for t, success, no_det, wrong, rate in rows:
            f.write(f"{t:.2f},{success},{no_det},{wrong},{rate:.4f}\n")
    print(f"\nบันทึกตารางเต็มที่: {summary_path}")


if __name__ == "__main__":
    main()
