"""
Pipeline ตรวจจับและครอปป้ายทะเบียนไทย — รับรูปรถ 1 คัน คืนรูปป้ายทะเบียนที่ครอปแล้ว

ขั้นตอน (ตามที่รายงานโครงงานอธิบายไว้):
  1. รับรูป
  2. รัน YOLO11 หากรอบป้าย + ค่าความมั่นใจ (ultralytics จัดการ resize preprocessing ให้อัตโนมัติ)
  3. คัดกรอง: ทิ้งกรอบ confidence < 60%, ใช้ NMS ตัดกรอบซ้อนกัน (ทำผ่าน conf/iou ของ predict())
  4. ถ้าไม่เหลือกรอบ -> แจ้ง "ไม่พบป้ายทะเบียน"
  5. ถ้าเจอหลายป้าย -> เลือกกรอบใหญ่ที่สุด (รถคันหลัก)
  6. ครอปด้วย OpenCV ให้ชิดขอบป้าย (src/crop/crop_utils.py)
  7. ปรับขนาดผลลัพธ์เป็น 400x200 พิกเซล -> บันทึกที่ outputs/crops/

วิธีใช้ (Command Line):
  python src/inference/predict_and_crop.py                       # โหมดโต้ตอบ: โปรแกรมถาม path รูปเอง (พิมพ์ q เพื่อออก)
  python src/inference/predict_and_crop.py --image car.jpg       # รูปเดียว
  python src/inference/predict_and_crop.py --image a.jpg b.jpg   # หลายรูป
  python src/inference/predict_and_crop.py --image samples/       # ทั้งโฟลเดอร์
  python src/inference/predict_and_crop.py --image car.jpg --conf 0.5 --device cpu
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from crop.crop_utils import crop_plate  # noqa: E402

from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "outputs" / "crops"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def resolve_device(device: str = "auto") -> str:
    """เลือกอุปกรณ์ประมวลผลอัตโนมัติ: CUDA -> MPS (Mac) -> CPU (ถ้าระบุเองมาแล้วใช้ตามที่ระบุ)"""
    if device and device != "auto":
        return device
    import torch

    if torch.cuda.is_available():
        return "0"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def _read_image(image_path: Path):
    """อ่านรูปด้วย cv2.imread ตามเดิม ถ้าอ่านไม่ได้ (เช่น path ภาษาไทยบน Windows) ลองอ่านผ่าน bytes แทน"""
    image = cv2.imread(str(image_path))
    if image is None:
        try:
            image = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR)
        except OSError:
            image = None
    return image


def _write_image(path: Path, image) -> bool:
    """บันทึกรูปด้วย cv2.imwrite ตามเดิม ถ้าไม่สำเร็จ (path ภาษาไทยบน Windows) บันทึกผ่าน bytes แทน"""
    if cv2.imwrite(str(path), image):
        return True
    ok, buf = cv2.imencode(path.suffix or ".jpg", image)
    if ok:
        buf.tofile(str(path))
    return bool(ok)


def predict_and_crop(image_path: Path, weights_path: Path, conf: float = 0.6, iou: float = 0.45, device: str = "auto", model=None):
    """
    คืนค่า (cropped_image_or_None, message, info)
    info เป็น dict เสมอ มีคีย์: num_detections, box_xyxy, confidence, elapsed_ms
    (box_xyxy/confidence เป็น None ถ้าไม่พบป้าย) — ไม่กระทบตรรกะการตรวจจับ/ครอปเดิมเลย
    แค่คืนค่าข้อมูลที่คำนวณอยู่แล้วเพิ่มเติม สำหรับใช้แสดงผล (เช่นวาดกรอบทับรูปต้นฉบับ)

    model: ส่งโมเดลที่โหลดไว้แล้วเข้ามาได้ (ใช้ตอนตรวจหลายรูปต่อเนื่อง จะได้ไม่ต้องโหลดซ้ำทุกรูป)
    """
    t0 = time.perf_counter()
    if model is None:
        model = YOLO(str(weights_path))
    results = model.predict(source=str(image_path), conf=conf, iou=iou, device=resolve_device(device), verbose=False)
    boxes = results[0].boxes
    elapsed_ms = (time.perf_counter() - t0) * 1000

    if boxes is None or len(boxes) == 0:
        return None, "ไม่พบป้ายทะเบียน", {
            "num_detections": 0, "box_xyxy": None, "confidence": None, "elapsed_ms": elapsed_ms,
        }

    # เลือกกรอบใหญ่ที่สุด (รถคันหลัก) ถ้าเจอหลายป้าย
    areas = (boxes.xyxy[:, 2] - boxes.xyxy[:, 0]) * (boxes.xyxy[:, 3] - boxes.xyxy[:, 1])
    best_idx = int(areas.argmax())
    best_box = boxes.xyxy[best_idx].tolist()
    best_conf = float(boxes.conf[best_idx])

    image = _read_image(image_path)
    if image is None:
        return None, f"เปิดรูปไม่ได้: {image_path}", {
            "num_detections": len(boxes), "box_xyxy": best_box, "confidence": best_conf, "elapsed_ms": elapsed_ms,
        }

    cropped = crop_plate(image, tuple(best_box))
    info = {
        "num_detections": len(boxes),
        "box_xyxy": best_box,
        "confidence": best_conf,
        "elapsed_ms": elapsed_ms,
    }
    return cropped, f"พบป้ายทะเบียน (confidence={best_conf:.2f}, จำนวนกรอบที่เจอ={len(boxes)})", info


def collect_images(paths: list[str]) -> list[Path]:
    """รับ path ของรูปหรือโฟลเดอร์ (ได้หลายรายการ) แล้วคืนรายการไฟล์รูปที่พบ"""
    found: list[Path] = []
    for raw in paths:
        p = Path(raw).expanduser()
        if p.is_dir():
            found.extend(sorted(f for f in p.iterdir() if f.suffix.lower() in IMAGE_EXTS))
        elif p.is_file():
            found.append(p)
        else:
            print(f"[ข้าม] ไม่พบไฟล์หรือโฟลเดอร์: {p}")
    return found


def clean_typed_path(text: str) -> str:
    """ทำความสะอาด path ที่พิมพ์/ลากไฟล์วางในเทอร์มินัล (ตัดเครื่องหมายคำพูด และ \\ หน้าช่องว่างแบบ macOS)"""
    text = text.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        text = text[1:-1]
    return text.replace("\\ ", " ")


def process_image(image_path: Path, model, args, out_path: Path | None = None) -> bool:
    print(f"\nรูป: {image_path}")
    cropped, message, info = predict_and_crop(image_path, Path(args.weights), args.conf, args.iou, args.device, model=model)
    print(message)
    print(f"ใช้เวลาตรวจจับ: {info['elapsed_ms']:.0f} ms")
    if cropped is None:
        return False

    if out_path is None:
        out_dir = Path(args.out_dir) if args.out_dir else OUTPUT_DIR
        out_path = out_dir / f"{image_path.stem}_plate.jpg"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if _write_image(out_path, cropped):
        print(f"บันทึกผลลัพธ์ที่: {out_path}")
        return True
    print(f"[เตือน] บันทึกไฟล์ไม่สำเร็จ: {out_path}")
    return False


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # กันข้อความไทยพังในเทอร์มินัล Windows
    except (AttributeError, OSError):
        pass

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image", nargs="*", default=None,
                        help="path ของรูปรถ (ใส่ได้หลายรูป หรือเป็นโฟลเดอร์) ถ้าไม่ใส่ โปรแกรมจะถาม path รูปเอง")
    parser.add_argument("--weights", default=str(REPO_ROOT / "models" / "best.pt"))
    parser.add_argument("--conf", type=float, default=0.6, help="ค่าความมั่นใจขั้นต่ำ (ทิ้งกรอบที่ต่ำกว่านี้)")
    parser.add_argument("--iou", type=float, default=0.45, help="เกณฑ์ IoU สำหรับ NMS (ตัดกรอบซ้อนกัน)")
    parser.add_argument("--device", default="auto", help='"auto" (ค่าเริ่มต้น: CUDA -> MPS -> CPU), "cpu", "mps" หรือ "0" (CUDA)')
    parser.add_argument("--out", default=None, help="path ไฟล์ผลลัพธ์ (ใช้ได้เมื่อตรวจรูปเดียว; ค่าเริ่มต้น: outputs/crops/<ชื่อไฟล์>_plate.jpg)")
    parser.add_argument("--out-dir", default=None, help="โฟลเดอร์เก็บผลลัพธ์ (ค่าเริ่มต้น: outputs/crops/)")
    args = parser.parse_args()

    weights_path = Path(args.weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"ไม่พบโมเดล: {weights_path} — ต้องมีไฟล์ models/best.pt")

    args.device = resolve_device(args.device)
    print(f"โหลดโมเดล: {weights_path.name} | อุปกรณ์ประมวลผล: {args.device}")
    model = YOLO(str(weights_path))

    # ---- โหมดระบุรูปมากับคำสั่ง ----
    if args.image:
        images = collect_images(args.image)
        if not images:
            raise SystemExit("ไม่พบรูปที่จะตรวจจับ")
        if args.out and len(images) > 1:
            raise SystemExit("--out ใช้ได้กับรูปเดียวเท่านั้น (ถ้าหลายรูปให้ใช้ --out-dir)")
        ok = 0
        for img in images:
            if process_image(img, model, args, Path(args.out) if args.out else None):
                ok += 1
        print(f"\nสรุป: ตรวจรูปทั้งหมด {len(images)} รูป พบป้ายและครอปสำเร็จ {ok} รูป")
        return

    # ---- โหมดโต้ตอบ: ถาม path รูปเอง ----
    print("\nโหมดโต้ตอบ — พิมพ์ path ของรูปรถ (หรือโฟลเดอร์) แล้วกด Enter | พิมพ์ q เพื่อออก")
    while True:
        try:
            typed = input("\nใส่ path รูป: ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        typed = clean_typed_path(typed)
        if typed.lower() in {"q", "quit", "exit"}:
            break
        if not typed:
            continue
        images = collect_images([typed])
        for img in images:
            process_image(img, model, args)
    print("จบการทำงาน")


if __name__ == "__main__":
    main()
