"""
เก็บ feedback จากผู้ใช้จริงของเว็ปแอป (ยืนยันผลถูก/ผิด + วาดกรอบแก้ไขเอง) ไว้เป็นวัตถุดิบ
สำหรับรอบเทรนโมเดลถัดไป — ไม่กระทบโมเดลที่ใช้งานจริงเลย (ดู docs/ ส่วน feedback)

ทุก record ที่บันทึกต้องผ่านการตรวจด้วยตาของ maintainer ผ่าน src/data_prep/import_feedback.py
ก่อนถูกนำเข้า data/raw/ จริง ไม่มีเส้นทางไหนเข้า dataset แบบอัตโนมัติ
"""

import csv
import uuid
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
FEEDBACK_DIR = REPO_ROOT / "outputs" / "feedback"
IMAGES_DIR = FEEDBACK_DIR / "images"
LABELS_DIR = FEEDBACK_DIR / "labels"
LOG_PATH = FEEDBACK_DIR / "feedback_log.csv"

LOG_FIELDS = [
    "timestamp",
    "image_id",
    "source_filename",
    "model_box_xyxy",
    "model_confidence",
    "verdict",
    "corrected_yolo_box",
    "note",
    "imported",
]


def canvas_rect_to_yolo(rect: dict, canvas_w: int, canvas_h: int, img_w: int, img_h: int):
    """
    แปลงกรอบสี่เหลี่ยมที่วาดบน st_canvas (fabric.js object) ให้เป็น YOLO format
    (cx, cy, w, h) normalized เทียบขนาดรูปต้นฉบับจริง (img_w, img_h)

    fabric.js เก็บตำแหน่ง/ขนาดกรอบเป็น left, top, width, height ของตอนวาดครั้งแรก
    แล้วถ้าผู้ใช้ลาก-ปรับขนาดภายหลัง ขนาดจริงจะคูณด้วย scaleX/scaleY เพิ่มเข้ามา
    ต้องคูณกลับก่อนถึงจะได้ขนาดกรอบที่แสดงผลจริงบน canvas
    """
    left = rect["left"]
    top = rect["top"]
    width = rect["width"] * rect.get("scaleX", 1)
    height = rect["height"] * rect.get("scaleY", 1)

    # canvas อาจแสดงรูปย่อ/ขยายจากขนาดจริง ต้องคูณ scale กลับเป็นพิกัดรูปต้นฉบับก่อน
    scale_x = img_w / canvas_w
    scale_y = img_h / canvas_h

    x0 = left * scale_x
    y0 = top * scale_y
    x1 = (left + width) * scale_x
    y1 = (top + height) * scale_y

    # กันกรอบเกินขอบรูป (เผื่อผู้ใช้ลากทะลุขอบ canvas)
    x0, x1 = sorted((max(0, min(x0, img_w)), max(0, min(x1, img_w))))
    y0, y1 = sorted((max(0, min(y0, img_h)), max(0, min(y1, img_h))))

    cx = (x0 + x1) / 2 / img_w
    cy = (y0 + y1) / 2 / img_h
    w = (x1 - x0) / img_w
    h = (y1 - y0) / img_h
    return cx, cy, w, h


def _next_image_id() -> str:
    return f"fb_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:6]}"


def xyxy_to_yolo(box_xyxy, img_w: int, img_h: int):
    """แปลงกรอบ pixel xyxy (ของโมเดลที่ predict_and_crop คืนมา) เป็น YOLO format normalized"""
    x0, y0, x1, y1 = box_xyxy
    cx = (x0 + x1) / 2 / img_w
    cy = (y0 + y1) / 2 / img_h
    w = (x1 - x0) / img_w
    h = (y1 - y0) / img_h
    return cx, cy, w, h


def save_feedback(
    image_bytes: bytes,
    verdict: str,
    source_filename: str,
    img_w: int,
    img_h: int,
    model_box_xyxy=None,
    model_confidence=None,
    corrected_yolo_box=None,
    note: str = "",
) -> str:
    """
    บันทึก feedback หนึ่งรายการ: เซฟรูป (เสมอ) + label YOLO (ถ้ามีกรอบ ไม่ว่าจะเป็นกรอบเดิมที่
    มนุษย์ยืนยันว่าถูก หรือกรอบที่วาดแก้ไข/วาดใหม่) + append แถวลง feedback_log.csv

    verdict: "correct" | "wrong_location" | "no_plate" | "found_missed"
    img_w, img_h: ขนาดรูปต้นฉบับ (pixel) ใช้แปลง model_box_xyxy เป็น YOLO เมื่อ verdict == "correct"
    corrected_yolo_box: (cx, cy, w, h) normalized ถ้ามี — ใช้เป็น label แทนกรอบเดิมของโมเดล
                        (มาจากการวาดกรอบใหม่/แก้ไขผ่าน canvas_rect_to_yolo())
    คืนค่า image_id ที่ใช้อ้างอิง
    """
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    LABELS_DIR.mkdir(parents=True, exist_ok=True)

    image_id = _next_image_id()
    img_path = IMAGES_DIR / f"{image_id}.jpg"
    img_path.write_bytes(image_bytes)

    yolo_box = corrected_yolo_box
    if yolo_box is None and verdict == "correct" and model_box_xyxy is not None:
        # มนุษย์ยืนยันว่ากรอบเดิมถูก -> ใช้กรอบเดิมของโมเดลเป็น label ได้เลย
        yolo_box = xyxy_to_yolo(model_box_xyxy, img_w, img_h)

    if yolo_box is not None:
        label_path = LABELS_DIR / f"{image_id}.txt"
        cx, cy, w, h = yolo_box
        label_path.write_text(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n", encoding="utf-8")

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    is_new = not LOG_PATH.exists()
    with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "image_id": image_id,
            "source_filename": source_filename,
            "model_box_xyxy": model_box_xyxy,
            "model_confidence": model_confidence,
            "verdict": verdict,
            "corrected_yolo_box": corrected_yolo_box,
            "note": note,
            "imported": "",
        })

    return image_id
