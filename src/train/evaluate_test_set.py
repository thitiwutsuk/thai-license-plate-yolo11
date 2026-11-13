"""
ประเมินผลโมเดล (models/best.pt) กับชุดทดสอบ 50 รูปใน data/test/
ซึ่งโมเดลไม่เคยเห็นระหว่างเทรน/validate มาก่อนเลย (กันไว้ตั้งแต่ split_folds.py)

คำนวณ:
  - Precision, Recall, mAP@0.5 (มาตรฐาน object detection)
  - อัตราครอปสำเร็จ: เทียบกรอบที่โมเดลทำนาย (confidence >= 0.5, คัด NMS, เลือกกรอบใหญ่สุดถ้าเจอหลายกรอบ)
    กับกรอบจริงที่ตีไว้ (ground truth) ด้วยค่า IoU >= 0.5 ถือว่า "ครอปได้ป้ายถูกตำแหน่ง"

วิธีใช้:
  python src/train/evaluate_test_set.py
  python src/train/evaluate_test_set.py --weights models/best.pt --conf 0.5 --iou-thresh 0.5
"""

import argparse
from pathlib import Path

from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parents[2]
TEST_DIR = REPO_ROOT / "data" / "test"
OUT_DIR = REPO_ROOT / "outputs" / "predictions"


def load_gt_box(label_path: Path):
    """อ่านกรอบจริง (YOLO format, คาดว่ามี 1 กรอบต่อรูป) คืนค่า (cx, cy, w, h) แบบ normalized"""
    if not label_path.exists():
        return None
    lines = [l.strip() for l in label_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not lines:
        return None
    parts = lines[0].split()
    _, cx, cy, w, h = parts
    return float(cx), float(cy), float(w), float(h)


def to_xyxy(cx, cy, w, h):
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def iou(box_a, box_b):
    ax0, ay0, ax1, ay1 = box_a
    bx0, by0, bx1, by1 = box_b
    inter_x0 = max(ax0, bx0)
    inter_y0 = max(ay0, by0)
    inter_x1 = min(ax1, bx1)
    inter_y1 = min(ay1, by1)
    inter_w = max(0.0, inter_x1 - inter_x0)
    inter_h = max(0.0, inter_y1 - inter_y0)
    inter_area = inter_w * inter_h
    area_a = (ax1 - ax0) * (ay1 - ay0)
    area_b = (bx1 - bx0) * (by1 - by0)
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default=str(REPO_ROOT / "models" / "best.pt"))
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--iou-thresh", type=float, default=0.5, help="เกณฑ์ IoU ที่ถือว่าครอปสำเร็จ")
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()

    weights_path = Path(args.weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"ไม่พบไฟล์โมเดล {weights_path} — ต้องเทรนให้เสร็จก่อน (Phase 3)")

    model = YOLO(str(weights_path))

    # 1) ตัวชี้วัดมาตรฐาน object detection ผ่าน model.val()
    data_yaml = TEST_DIR / "data.yaml"
    print("=" * 60)
    print("ประเมินผลมาตรฐาน (Precision / Recall / mAP@0.5)")
    print("=" * 60)
    metrics = model.val(
        data=str(data_yaml),
        device=args.device,
        project=str(OUT_DIR / "runs"),
        name="test_set_eval",
        conf=args.conf,
        plots=True,
    )
    precision = float(metrics.box.mp)
    recall = float(metrics.box.mr)
    map50 = float(metrics.box.map50)

    # 2) อัตราครอปสำเร็จ: เทียบกรอบทำนายกับกรอบจริงทีละรูป
    print("\n" + "=" * 60)
    print("คำนวณอัตราครอปสำเร็จ (IoU >= {:.2f})".format(args.iou_thresh))
    print("=" * 60)

    image_files = sorted((TEST_DIR / "images").glob("*.jpg"))
    total = len(image_files)
    success = 0
    no_detection = []
    low_iou = []

    for img_path in image_files:
        label_path = TEST_DIR / "labels" / (img_path.stem + ".txt")
        gt = load_gt_box(label_path)

        results = model.predict(source=str(img_path), conf=args.conf, device=args.device, verbose=False)
        boxes = results[0].boxes

        if gt is None:
            continue  # ไม่ควรเกิดขึ้น (ทุกรูปควรมี label) แต่กันไว้

        gt_xyxy = to_xyxy(*gt)

        if boxes is None or len(boxes) == 0:
            no_detection.append(img_path.name)
            continue

        # เลือกกรอบใหญ่ที่สุด (ตามกติกาเลือกรถคันหลัก)
        areas = (boxes.xywhn[:, 2] * boxes.xywhn[:, 3]).tolist()
        best_idx = areas.index(max(areas))
        pred_xywhn = boxes.xywhn[best_idx].tolist()
        pred_xyxy = to_xyxy(*pred_xywhn)

        score = iou(pred_xyxy, gt_xyxy)
        if score >= args.iou_thresh:
            success += 1
        else:
            low_iou.append((img_path.name, round(score, 3)))

    crop_success_rate = success / total if total else 0.0

    print(f"\nรูปทดสอบทั้งหมด: {total}")
    print(f"ครอปสำเร็จ (IoU >= {args.iou_thresh}): {success} ({crop_success_rate*100:.1f}%)")
    print(f"ไม่พบป้ายเลย: {len(no_detection)}")
    for f in no_detection:
        print(f"  - {f}")
    print(f"เจอป้ายแต่ตำแหน่งไม่ตรงพอ (IoU ต่ำ): {len(low_iou)}")
    for f, s in low_iou:
        print(f"  - {f}: IoU={s}")

    # บันทึกสรุป
    summary_path = OUT_DIR / "test_set_summary.csv"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("metric,value\n")
        f.write(f"precision,{precision:.4f}\n")
        f.write(f"recall,{recall:.4f}\n")
        f.write(f"map50,{map50:.4f}\n")
        f.write(f"crop_success_rate,{crop_success_rate:.4f}\n")
        f.write(f"crop_success_count,{success}\n")
        f.write(f"total_test_images,{total}\n")

    print(f"\nบันทึกสรุปที่: {summary_path}")
    print(f"\nสรุป: Precision={precision:.4f} Recall={recall:.4f} mAP@0.5={map50:.4f} | อัตราครอปสำเร็จ={crop_success_rate*100:.1f}%")


if __name__ == "__main__":
    main()
