"""
ฟังก์ชันช่วยครอปรูปป้ายทะเบียนด้วย OpenCV จากกรอบที่ YOLO ตรวจจับได้
"""

import cv2
import numpy as np

PAD_PX = 3  # ขยายขอบกรอบเล็กน้อยให้ครอปชิดป้ายพอดี ไม่ตัดขอบป้ายขาด
OUTPUT_SIZE = (400, 200)  # (width, height)


def crop_plate(image: np.ndarray, xyxy_box: tuple[float, float, float, float]) -> np.ndarray:
    """
    ครอปป้ายทะเบียนจากรูปต้นฉบับตามกรอบที่ตรวจจับได้ (พิกเซลจริง ไม่ใช่ normalized)
    ขยายขอบ PAD_PX พิกเซล แต่ไม่เกินขอบรูปต้นฉบับ แล้ว resize เป็น OUTPUT_SIZE
    """
    h, w = image.shape[:2]
    x0, y0, x1, y1 = xyxy_box

    x0 = max(0, int(round(x0)) - PAD_PX)
    y0 = max(0, int(round(y0)) - PAD_PX)
    x1 = min(w, int(round(x1)) + PAD_PX)
    y1 = min(h, int(round(y1)) + PAD_PX)

    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"กรอบไม่ถูกต้อง: ({x0},{y0},{x1},{y1}) บนรูปขนาด {w}x{h}")

    cropped = image[y0:y1, x0:x1]
    resized = cv2.resize(cropped, OUTPUT_SIZE, interpolation=cv2.INTER_CUBIC)
    return resized
