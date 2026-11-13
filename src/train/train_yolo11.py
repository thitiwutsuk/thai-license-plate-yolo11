"""
ฝึกโมเดล YOLO11n แบบ 5-Fold Cross-Validation
ใช้ข้อมูลจาก data/cv/fold_1..5/ ที่สร้างไว้โดย src/data_prep/split_folds.py

วิธีใช้:
  python src/train/train_yolo11.py                    # รันครบ 5 fold เต็ม (100 epoch/fold)
  python src/train/train_yolo11.py --epochs 3 --smoke  # ทดสอบเร็ว (smoke test) ก่อนรันจริง
"""

import argparse
import csv
import shutil
from pathlib import Path

from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parents[2]
CV_DIR = REPO_ROOT / "data" / "cv"
MODELS_DIR = REPO_ROOT / "models"
OUT_DIR = REPO_ROOT / "outputs" / "predictions"
N_FOLDS = 5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--device", default="mps", help='"mps" (Apple GPU, เร็วกว่า), "cpu", หรือ "0" สำหรับ CUDA GPU')
    parser.add_argument("--smoke", action="store_true", help="รันแบบเร็วเพื่อตรวจว่า pipeline ทำงานได้ ไม่ใช่ผลจริง")
    parser.add_argument("--folds", type=str, default=None, help='ระบุ fold ที่จะรัน เช่น "1" หรือ "1,3" (ค่าเริ่มต้น: รันครบ 1-5)')
    args = parser.parse_args()

    folds_to_run = [int(x) for x in args.folds.split(",")] if args.folds else list(range(1, N_FOLDS + 1))

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # โหลดผลจาก fold ก่อนหน้า (ถ้ามี) เพื่อไม่ให้รันทีละ fold แล้วผลเก่าหาย
    summary_path = OUT_DIR / "cv_summary.csv"
    results_by_fold = {}
    if summary_path.exists():
        with open(summary_path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row["fold"] == "average":
                    continue
                results_by_fold[int(row["fold"])] = {
                    "fold": int(row["fold"]),
                    "precision": float(row["precision"]),
                    "recall": float(row["recall"]),
                    "map50": float(row["map50"]),
                    "weights": row["weights"],
                }
        print(f"พบผลเดิมจาก fold ที่เคยรันแล้ว: {sorted(results_by_fold.keys())}")

    for fold in folds_to_run:
        data_yaml = CV_DIR / f"fold_{fold}" / "data.yaml"
        if not data_yaml.exists():
            raise FileNotFoundError(f"ไม่พบ {data_yaml} — รัน src/data_prep/split_folds.py ก่อน")

        print(f"\n{'='*60}\nFold {fold}/{N_FOLDS}\n{'='*60}")

        model = YOLO(args.model)
        run_name = f"fold_{fold}"
        train_result = model.train(
            data=str(data_yaml),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            project=str(OUT_DIR / "runs"),
            name=run_name,
            exist_ok=True,
            verbose=not args.smoke,
            plots=not args.smoke,
        )

        metrics = model.val(
            data=str(data_yaml),
            device=args.device,
            project=str(OUT_DIR / "runs"),
            name=f"{run_name}_val",
            plots=not args.smoke,
        )
        precision = float(metrics.box.mp)
        recall = float(metrics.box.mr)
        map50 = float(metrics.box.map50)

        print(f"Fold {fold}: Precision={precision:.4f} Recall={recall:.4f} mAP@0.5={map50:.4f}")

        weights_path = OUT_DIR / "runs" / run_name / "weights" / "best.pt"
        results_by_fold[fold] = {
            "fold": fold,
            "precision": precision,
            "recall": recall,
            "map50": map50,
            "weights": str(weights_path),
        }

    # รวมผลทุก fold ที่เคยรันมา (ไม่ใช่แค่รอบนี้) เพื่อหาค่าเฉลี่ย+fold ที่ดีที่สุดจากทั้งหมด
    all_results = [results_by_fold[k] for k in sorted(results_by_fold.keys())]
    avg_precision = sum(r["precision"] for r in all_results) / len(all_results)
    avg_recall = sum(r["recall"] for r in all_results) / len(all_results)
    avg_map50 = sum(r["map50"] for r in all_results) / len(all_results)

    best_result = max(all_results, key=lambda r: r["map50"])
    best_fold = best_result["fold"]
    best_map = best_result["map50"]
    best_weights_path = Path(best_result["weights"])

    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["fold", "precision", "recall", "map50", "weights"])
        writer.writeheader()
        writer.writerows(all_results)
        writer.writerow({
            "fold": "average",
            "precision": round(avg_precision, 4),
            "recall": round(avg_recall, 4),
            "map50": round(avg_map50, 4),
            "weights": "",
        })

    print(f"\n{'='*60}\nสรุปผล Cross-Validation (จากทุก fold ที่เคยรันมา: {sorted(results_by_fold.keys())})\n{'='*60}")
    print(f"ค่าเฉลี่ย: Precision={avg_precision:.4f} Recall={avg_recall:.4f} mAP@0.5={avg_map50:.4f}")
    print(f"Fold ที่ดีที่สุด: fold_{best_fold} (mAP@0.5={best_map:.4f})")
    print(f"บันทึกสรุปที่: {summary_path}")

    if best_weights_path and best_weights_path.exists():
        dest = MODELS_DIR / "best.pt"
        shutil.copy(best_weights_path, dest)
        print(f"คัดลอกโมเดลที่ดีที่สุดไปที่: {dest}")
    else:
        print("[เตือน] ไม่พบไฟล์ weights ของ fold ที่ดีที่สุด ตรวจสอบ outputs/predictions/runs/ เอง")


if __name__ == "__main__":
    main()
