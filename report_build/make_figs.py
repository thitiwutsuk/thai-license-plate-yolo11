import csv
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch, Polygon
import numpy as np

matplotlib.rcParams["font.family"] = ["Arial Unicode MS", "Sukhumvit Set"]
matplotlib.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "fig"
OUT.mkdir(exist_ok=True)

NAVY = "#1e3a8a"
INDIGO = "#4f46e5"
AMBER = "#f59e0b"
GREEN = "#16a34a"
RED = "#dc2626"
GRAY = "#64748b"
LIGHT = "#eef2ff"


def save(fig, name, dpi=170):
    fig.savefig(OUT / name, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", name)


def box(ax, x, y, w, h, text, fc="#ffffff", ec=NAVY, tc="#0f172a", fs=10, lw=1.6, bold=False, r=0.12):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.02,rounding_size={r}", fc=fc, ec=ec, lw=lw)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc,
            fontweight="bold" if bold else "normal", linespacing=1.35)


def arrow(ax, x0, y0, x1, y1, color=GRAY, lw=1.8, style="-|>", ls="-"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=14, color=color, lw=lw, linestyle=ls))


def load_img(name):
    im = cv2.imread(str(ROOT / "data" / "raw" / name))
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


def load_label(name):
    p = ROOT / "data" / "labels" / (Path(name).stem + ".txt")
    cx, cy, w, h = [float(v) for v in p.read_text().split()[1:5]]
    return cx, cy, w, h


# ---------------------------------------------------------------- 1) project pipeline
def fig_project_pipeline():
    fig, ax = plt.subplots(figsize=(11, 6.6))
    ax.set_xlim(0, 11); ax.set_ylim(0, 6.6); ax.axis("off")
    steps = [
        ("1  เก็บรูปรถ\nWikimedia Commons\n416 รูป", "#dbeafe"),
        ("2  ตรวจคัดกรอง + Ingest\nresize 1280px · sha256\nผ่าน 250 / ไม่ผ่าน 166", "#dbeafe"),
        ("3  ตีกรอบป้าย (Label)\nmakesense.ai\nYOLO format", "#e0e7ff"),
        ("4  ตรวจ Label\nverify_labels.py\n+ contact sheet", "#e0e7ff"),
        ("5  แบ่งข้อมูล\nTest 50 + Train 200\n5-Fold (กันรถซ้ำ)", "#fef3c7"),
        ("6  ฝึกโมเดล YOLO11n\n100 epoch / fold\nTransfer Learning", "#fef3c7"),
        ("7  เลือกโมเดลที่ดีที่สุด\nfold 3 → models/best.pt", "#dcfce7"),
        ("8  ประเมินชุด Test 50 รูป\nP · R · mAP · Crop 98%", "#dcfce7"),
    ]
    xs = [0.2, 2.95, 5.7, 8.45]
    pos = []
    for i, (t, c) in enumerate(steps):
        row = 0 if i < 4 else 1
        col = i if i < 4 else 7 - i
        x = xs[col]; y = 4.7 - row * 2.15
        box(ax, x, y, 2.35, 1.35, t, fc=c, fs=9.6)
        pos.append((x, y))
    for i in range(3):
        arrow(ax, pos[i][0] + 2.4, pos[i][1] + 0.68, pos[i + 1][0] - 0.05, pos[i + 1][1] + 0.68)
    arrow(ax, pos[3][0] + 1.18, pos[3][1] - 0.02, pos[4][0] + 1.18, pos[4][1] + 1.4)
    for i in range(4, 7):
        arrow(ax, pos[i][0] - 0.05, pos[i][1] + 0.68, pos[i + 1][0] + 2.4, pos[i + 1][1] + 0.68)
    # phase 5-6
    box(ax, 0.2, 0.55, 5.2, 1.15, "9  Inference: predict_and_crop.py (CLI)  ·  Streamlit Web App\nรับรูปรถ → ตรวจจับ → ครอป 400×200 → ดาวน์โหลด", fc="#fce7f3", ec="#be185d", fs=9.6)
    box(ax, 5.7, 0.55, 5.1, 1.15, "10  Human-in-the-Loop Feedback\nผู้ใช้ยืนยัน/วาดกรอบใหม่ → maintainer ตรวจ → เทรนรอบถัดไป", fc="#f3e8ff", ec="#7e22ce", fs=9.6)
    arrow(ax, pos[4][0] + 1.18, pos[4][1] - 0.02, 2.8, 1.75, color=GRAY)
    arrow(ax, 5.4, 1.12, 5.65, 1.12, color=GRAY)
    ax.text(5.5, 6.4, "ภาพรวมขั้นตอนของทั้งโครงงาน (Project Pipeline)", ha="center", fontsize=13, fontweight="bold", color=NAVY)
    save(fig, "fig_project_pipeline.png")


# ---------------------------------------------------------------- 2) inference pipeline
def fig_inference_pipeline():
    fig, ax = plt.subplots(figsize=(11.5, 5.6))
    ax.set_xlim(0, 11.5); ax.set_ylim(0, 5.6); ax.axis("off")
    box(ax, 0.1, 3.3, 1.55, 1.1, "รับรูปรถ\n1 คัน\n(JPG/PNG)", fc="#dbeafe", fs=9.5)
    box(ax, 2.0, 3.3, 1.75, 1.1, "Preprocess\nletterbox\n640×640", fc="#dbeafe", fs=9.5)
    box(ax, 4.1, 3.3, 1.85, 1.1, "YOLO11n\nInference\nboxes + conf", fc="#e0e7ff", fs=9.5)
    # diamond
    d = Polygon([(7.6, 4.55), (8.75, 3.85), (7.6, 3.15), (6.45, 3.85)], closed=True, fc="#fef3c7", ec=AMBER, lw=1.8)
    ax.add_patch(d)
    ax.text(7.6, 3.85, "conf ≥\nthreshold?", ha="center", va="center", fontsize=9.3, fontweight="bold", color="#78350f")
    box(ax, 9.35, 3.3, 2.0, 1.1, "Filter + NMS\nIoU 0.45\nเลือกกรอบใหญ่สุด", fc="#fef3c7", ec=AMBER, fs=9.3)
    box(ax, 9.35, 1.55, 2.0, 1.0, "OpenCV Crop\n+3px → 400×200", fc="#dcfce7", ec=GREEN, fs=9.5)
    box(ax, 9.35, 0.15, 2.0, 0.95, "Output\nplate.png", fc="#bbf7d0", ec=GREEN, fs=10, bold=True)
    arrow(ax, 1.68, 3.85, 1.97, 3.85); arrow(ax, 3.78, 3.85, 4.07, 3.85); arrow(ax, 5.98, 3.85, 6.42, 3.85)
    arrow(ax, 8.78, 3.85, 9.32, 3.85, color=GREEN)
    ax.text(9.0, 4.05, "ใช่", color=GREEN, fontsize=9.5, fontweight="bold", ha="center")
    arrow(ax, 10.35, 3.27, 10.35, 2.6, color=GREEN); arrow(ax, 10.35, 1.52, 10.35, 1.15, color=GREEN)
    # no branch
    arrow(ax, 7.6, 3.12, 7.6, 2.55, color=RED)
    ax.text(7.7, 2.8, "ไม่", color=RED, fontsize=9.5, fontweight="bold")
    box(ax, 6.0, 1.45, 3.2, 1.05, "ไม่พบป้ายทะเบียน\nแจ้งผู้ใช้ให้ถ่ายใหม่ หรือวาดกรอบเอง", fc="#fee2e2", ec=RED, fs=9.3)
    arrow(ax, 7.6, 1.42, 7.6, 1.0, color=RED)
    box(ax, 5.55, 0.1, 3.5, 0.88, "เก็บกรอบที่ผู้ใช้วาด → maintainer ตรวจ\n→ เทรนรอบถัดไป", fc="#f3e8ff", ec="#7e22ce", fs=8.8)
    ax.text(5.75, 5.3, "Inference Pipeline: จากรูปรถ ถึงป้ายที่ครอปแล้ว", ha="center", fontsize=13, fontweight="bold", color=NAVY)
    save(fig, "fig_inference_pipeline.png")


# ---------------------------------------------------------------- 3) YOLO11 architecture
def fig_yolo_arch():
    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    ax.set_xlim(0, 11.5); ax.set_ylim(0, 5.8); ax.axis("off")
    # input
    box(ax, 0.1, 2.3, 1.1, 1.2, "Input\n640×640\n×3", fc="#e2e8f0", ec=GRAY, fs=9.5)
    # backbone
    ax.add_patch(Rectangle((1.5, 0.55), 3.55, 4.55, fc="#eff6ff", ec="#93c5fd", lw=1.4, ls="--"))
    ax.text(3.27, 4.85, "BACKBONE  (สกัดคุณลักษณะ)", ha="center", fontsize=10.5, fontweight="bold", color=NAVY)
    bb = ["Conv (stem)", "Conv + C3k2", "Conv + C3k2  → P3", "Conv + C3k2  → P4", "Conv + C3k2", "SPPF  →  C2PSA  → P5"]
    for i, t in enumerate(bb):
        box(ax, 1.7, 4.15 - i * 0.68, 3.15, 0.52, t, fc="#ffffff", ec="#60a5fa", fs=8.8)
    # neck
    ax.add_patch(Rectangle((5.35, 0.55), 2.75, 4.55, fc="#fefce8", ec="#fcd34d", lw=1.4, ls="--"))
    ax.text(6.72, 4.85, "NECK  (รวมหลายสเกล)", ha="center", fontsize=10.5, fontweight="bold", color="#92400e")
    nk = ["Upsample + Concat\n(P5 → P4 → P3)", "C3k2  (top-down)", "Downsample + Concat\n(P3 → P4 → P5)", "C3k2  (bottom-up)"]
    for i, t in enumerate(nk):
        box(ax, 5.52, 4.0 - i * 1.0, 2.4, 0.78, t, fc="#ffffff", ec="#fbbf24", fs=8.8)
    # head
    ax.add_patch(Rectangle((8.4, 0.55), 2.95, 4.55, fc="#f0fdf4", ec="#86efac", lw=1.4, ls="--"))
    ax.text(9.87, 4.85, "HEAD  (ทำนายผล)", ha="center", fontsize=10.5, fontweight="bold", color="#166534")
    box(ax, 8.6, 3.55, 2.55, 0.9, "Detect @ P3 (80×80)\nวัตถุเล็ก", fc="#ffffff", ec="#4ade80", fs=8.8)
    box(ax, 8.6, 2.45, 2.55, 0.9, "Detect @ P4 (40×40)\nวัตถุกลาง", fc="#ffffff", ec="#4ade80", fs=8.8)
    box(ax, 8.6, 1.35, 2.55, 0.9, "Detect @ P5 (20×20)\nวัตถุใหญ่", fc="#ffffff", ec="#4ade80", fs=8.8)
    ax.text(9.87, 0.85, "Anchor-free · Decoupled head\n(box + class แยกสาขา)", ha="center", va="center", fontsize=8.6, color="#166534")
    arrow(ax, 1.22, 2.9, 1.68, 2.9, color=NAVY); arrow(ax, 5.05, 2.9, 5.5, 2.9, color=NAVY); arrow(ax, 8.12, 2.9, 8.58, 2.9, color=NAVY)
    ax.text(5.75, 5.55, "สถาปัตยกรรม YOLO11 (ภาพรวม)", ha="center", fontsize=13, fontweight="bold", color=NAVY)
    ax.text(5.75, 0.12, "ผลลัพธ์: กรอบ (x, y, w, h) + ความมั่นใจ + class  →  NMS  →  กรอบสุดท้าย   |   yolo11n: 2,604,883 พารามิเตอร์ · 6.6 GFLOPs",
            ha="center", fontsize=8.8, color=GRAY)
    save(fig, "fig_yolo_arch.png")


# ---------------------------------------------------------------- 4) IoU
def fig_iou():
    fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.4))
    cases = [((1, 1, 4, 3), (1.4, 1.2, 4, 3), "IoU สูง (≈0.75)"), ((1, 1, 4, 3), (3, 2.2, 4, 3), "IoU ปานกลาง (≈0.18)"), ((0.5, 0.5, 3, 2.4), (5, 3, 3, 2.4), "IoU = 0")]
    for ax, (a, b, title) in zip(axs, cases):
        ax.set_xlim(0, 9); ax.set_ylim(0, 6.5); ax.set_aspect("equal"); ax.axis("off")
        ax.add_patch(Rectangle(a[:2], a[2], a[3], fc="#3b82f6", alpha=0.35, ec=NAVY, lw=2))
        ax.add_patch(Rectangle(b[:2], b[2], b[3], fc="#f59e0b", alpha=0.35, ec="#b45309", lw=2))
        x0, y0 = max(a[0], b[0]), max(a[1], b[1]); x1, y1 = min(a[0] + a[2], b[0] + b[2]), min(a[1] + a[3], b[1] + b[3])
        if x1 > x0 and y1 > y0:
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fc="#16a34a", alpha=0.6, ec=GREEN, lw=1.5))
        ax.set_title(title, fontsize=11, color=NAVY, fontweight="bold")
    axs[0].text(0.4, 0.15, "น้ำเงิน = กรอบจริง (Ground Truth)   ส้ม = กรอบที่โมเดลทาย   เขียว = ส่วนซ้อนทับ", fontsize=8.6, color=GRAY, transform=axs[0].transData)
    fig.suptitle("IoU = พื้นที่ซ้อนทับ ÷ พื้นที่รวม  (Intersection over Union)", fontsize=12.5, fontweight="bold", color=NAVY, y=1.04)
    save(fig, "fig_iou.png")


# ---------------------------------------------------------------- 5) NMS
def fig_nms():
    img = load_img("plate_004.jpg")
    cx, cy, w, h = load_label("plate_004.jpg")
    H, W = img.shape[:2]
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.0))
    crop = img[int(H * 0.45):int(H * 0.75), int(W * 0.25):int(W * 0.8)]
    ch, cw = crop.shape[:2]
    bx = (cx * W - w * W / 2 - W * 0.25) / cw; by = (cy * H - h * H / 2 - H * 0.45) / ch
    bw = w * W / cw; bh = h * H / ch
    rng = np.random.default_rng(3)
    for k, ax in enumerate(axs):
        ax.imshow(crop); ax.axis("off")
        if k == 0:
            for j, (dx, dy, ds, cf) in enumerate([(0, 0, 1, 0.82), (0.03, 0.04, 1.06, 0.71), (-0.025, 0.03, 0.95, 0.64), (0.05, -0.03, 1.1, 0.55)]):
                x = (bx + dx) * cw; y = (by + dy) * ch
                ax.add_patch(Rectangle((x, y), bw * cw * ds, bh * ch * ds, fill=False, ec=["#16a34a", "#f59e0b", "#f97316", "#ef4444"][j], lw=2.2))
                ax.text(x, y - 6, f"{cf:.2f}", color=["#16a34a", "#f59e0b", "#f97316", "#ef4444"][j], fontsize=9, fontweight="bold")
            ax.set_title("ก่อน NMS: โมเดลทายกรอบซ้อนกันหลายกรอบ", fontsize=10.5, color=NAVY, fontweight="bold")
        else:
            ax.add_patch(Rectangle((bx * cw, by * ch), bw * cw, bh * ch, fill=False, ec="#16a34a", lw=2.6))
            ax.text(bx * cw, by * ch - 6, "0.82", color="#16a34a", fontsize=10, fontweight="bold")
            ax.set_title("หลัง NMS: เหลือกรอบที่มั่นใจที่สุด 1 กรอบ", fontsize=10.5, color=NAVY, fontweight="bold")
    save(fig, "fig_nms.png")


# ---------------------------------------------------------------- 6) k-fold
def fig_kfold():
    fig, ax = plt.subplots(figsize=(10.5, 4.3))
    ax.set_xlim(0, 10.5); ax.set_ylim(0, 4.3); ax.axis("off")
    for f in range(5):
        y = 3.25 - f * 0.62
        ax.text(0.1, y + 0.2, f"Fold {f + 1}", fontsize=9.5, fontweight="bold", color=NAVY, va="center")
        for b in range(5):
            val = (b == f)
            ax.add_patch(Rectangle((1.2 + b * 1.45, y), 1.4, 0.45, fc=AMBER if val else "#dbeafe", ec="white", lw=2))
            ax.text(1.2 + b * 1.45 + 0.7, y + 0.225, "Validation 40" if val else "Train 40", ha="center", va="center", fontsize=8.3,
                    color="#451a03" if val else NAVY)
        ax.add_patch(Rectangle((8.55, y), 1.8, 0.45, fc="#c2410c", ec="white", lw=2))
        ax.text(9.45, y + 0.225, "Test 50 (แยกไว้)", ha="center", va="center", fontsize=8.3, color="white")
    ax.text(5.2, 4.05, "5-Fold Cross-Validation (Train pool 200 รูป) + ชุดทดสอบ 50 รูปที่ไม่ถูกใช้ในทุก fold", ha="center", fontsize=11.5, fontweight="bold", color=NAVY)
    ax.text(5.2, 0.0, "ผลลัพธ์ของแต่ละ fold นำมาเฉลี่ย → ประเมินความเสถียรของโมเดล   |   กลุ่มรถคันเดียวกันไม่ข้าม fold", ha="center", fontsize=9, color=GRAY)
    save(fig, "fig_kfold.png")


# ---------------------------------------------------------------- 7) letterbox
def fig_letterbox():
    img = load_img("plate_004.jpg")
    H, W = img.shape[:2]
    s = 640 / max(H, W)
    nh, nw = int(H * s), int(W * s)
    canvas = np.full((640, 640, 3), 114, np.uint8)
    res = cv2.resize(img, (nw, nh))
    y0 = (640 - nh) // 2
    canvas[y0:y0 + nh, :nw] = res
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.6), gridspec_kw={"width_ratios": [1.33, 0.2, 1]})
    axs[0].imshow(img); axs[0].set_title(f"รูปต้นฉบับ {W}×{H}", fontsize=10.5, color=NAVY, fontweight="bold"); axs[0].axis("off")
    axs[1].axis("off"); axs[1].text(0.5, 0.5, "→\nresize\n+ pad", ha="center", va="center", fontsize=10, color=GRAY)
    axs[2].imshow(canvas); axs[2].set_title("Letterbox 640×640 (เติมขอบสีเทา ไม่บิดสัดส่วน)", fontsize=9.6, color=NAVY, fontweight="bold"); axs[2].axis("off")
    save(fig, "fig_letterbox.png")


# ---------------------------------------------------------------- 8) grid concept
def fig_grid():
    img = load_img("plate_004.jpg")
    H, W = img.shape[:2]
    cx, cy, w, h = load_label("plate_004.jpg")
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    ax.imshow(img); ax.axis("off")
    n = 8
    for i in range(1, n):
        ax.axhline(H * i / n, color="white", lw=0.9, alpha=0.8); ax.axvline(W * i / n, color="white", lw=0.9, alpha=0.8)
    gx, gy = int(cx * n), int(cy * n)
    ax.add_patch(Rectangle((W * gx / n, H * gy / n), W / n, H / n, fc="#f59e0b", alpha=0.35, ec="#f59e0b", lw=2))
    ax.add_patch(Rectangle(((cx - w / 2) * W, (cy - h / 2) * H), w * W, h * H, fill=False, ec="#22c55e", lw=2.8))
    ax.plot([cx * W], [cy * H], "o", color="#ef4444", ms=7)
    ax.set_title("แนวคิด: จุดกึ่งกลางวัตถุตกอยู่ช่องไหน ช่องนั้นรับผิดชอบทำนายกรอบ", fontsize=10.2, color=NAVY, fontweight="bold")
    save(fig, "fig_grid.png")


# ---------------------------------------------------------------- 9) samples w/ boxes
def fig_samples():
    names = ["plate_001.jpg", "plate_012.jpg", "plate_048.jpg", "plate_075.jpg", "plate_130.jpg", "plate_200.jpg"]
    fig, axs = plt.subplots(2, 3, figsize=(11, 6.3))
    for ax, n in zip(axs.flat, names):
        img = load_img(n); H, W = img.shape[:2]
        cx, cy, w, h = load_label(n)
        ax.imshow(img); ax.axis("off")
        ax.add_patch(Rectangle(((cx - w / 2) * W, (cy - h / 2) * H), w * W, h * H, fill=False, ec="#22c55e", lw=2.6))
        ax.set_title(n, fontsize=9, color=GRAY)
    fig.suptitle("ตัวอย่างรูปในชุดข้อมูล พร้อมกรอบป้ายทะเบียนที่ตีไว้ (Ground Truth, สีเขียว)", fontsize=12, fontweight="bold", color=NAVY)
    fig.tight_layout()
    save(fig, "fig_samples.png")


# ---------------------------------------------------------------- 10) crop steps
def fig_crop_steps():
    img = load_img("plate_004.jpg"); H, W = img.shape[:2]
    cx, cy, w, h = load_label("plate_004.jpg")
    x0, y0, x1, y1 = (cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H
    pad = 3
    crop = img[max(0, int(y0) - pad):int(y1) + pad, max(0, int(x0) - pad):int(x1) + pad]
    crop = cv2.resize(crop, (400, 200), interpolation=cv2.INTER_CUBIC)
    fig, axs = plt.subplots(1, 3, figsize=(11.5, 3.5), gridspec_kw={"width_ratios": [1.5, 0.15, 1]})
    axs[0].imshow(img); axs[0].axis("off")
    axs[0].add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="#22c55e", lw=2.6))
    axs[0].set_title("1) YOLO หากรอบป้าย (xyxy)", fontsize=10, color=NAVY, fontweight="bold")
    axs[1].axis("off"); axs[1].text(0.5, 0.5, "→\nคร้อป\n+3px", ha="center", va="center", fontsize=9.5, color=GRAY)
    axs[2].imshow(crop); axs[2].axis("off"); axs[2].set_title("2) ผลลัพธ์ 400×200 px", fontsize=10, color=NAVY, fontweight="bold")
    save(fig, "fig_crop_steps.png")


# ---------------------------------------------------------------- 11) threshold sweep
def fig_threshold():
    rows = list(csv.DictReader(open(ROOT / "outputs/predictions/threshold_sweep.csv")))
    th = [float(r["threshold"]) for r in rows]; rate = [float(r["crop_success_rate"]) * 100 for r in rows]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.plot(th, rate, "-o", color=INDIGO, lw=2.2, ms=5)
    ax.axvspan(0.30, 0.60, color="#dcfce7", alpha=0.7, label="ช่วงที่ให้ผล 98% คงที่")
    ax.set_xlabel("Confidence threshold"); ax.set_ylabel("อัตราครอปสำเร็จ (%)"); ax.set_ylim(55, 102)
    ax.grid(alpha=0.3); ax.legend(loc="lower left")
    ax.set_title("ผลของ confidence threshold ต่ออัตราครอปสำเร็จ (ชุดทดสอบ 50 รูป)", fontsize=11, color=NAVY, fontweight="bold")
    save(fig, "fig_threshold.png")


# ---------------------------------------------------------------- 12) CV bar
def fig_cv():
    rows = list(csv.DictReader(open(ROOT / "outputs/predictions/cv_summary.csv")))
    rows = [r for r in rows if r["fold"] != "average"]
    folds = [f"Fold {r['fold']}" for r in rows]
    P = [float(r["precision"]) for r in rows]; R = [float(r["recall"]) for r in rows]; M = [float(r["map50"]) for r in rows]
    x = np.arange(5); wd = 0.26
    fig, ax = plt.subplots(figsize=(9, 3.9))
    ax.bar(x - wd, P, wd, label="Precision", color="#93c5fd"); ax.bar(x, R, wd, label="Recall", color="#fcd34d"); ax.bar(x + wd, M, wd, label="mAP@0.5", color=INDIGO)
    for i in range(5):
        ax.text(x[i] + wd, M[i] + 0.004, f"{M[i]:.3f}", ha="center", fontsize=8, color=NAVY)
    ax.set_xticks(x); ax.set_xticklabels(folds); ax.set_ylim(0.9, 1.02); ax.grid(axis="y", alpha=0.3); ax.legend(ncol=3, loc="lower center")
    ax.set_title("ผล 5-Fold Cross-Validation แยกตาม fold", fontsize=11, color=NAVY, fontweight="bold")
    save(fig, "fig_cv.png")


# ---------------------------------------------------------------- 13) dataset funnel + license
def fig_dataset():
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.9), gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axs[0]
    rounds = ["รอบ 1\nหมวดป้ายไทย", "รอบ 2\nBangkok/จังหวัด", "รอบ 3\nToyota/Honda", "รอบ 4\nรถคลาสสิก/ยุโรป"]
    dl = [25, 198, 7, 175]; ps = [25, 94, 0, 131]
    x = np.arange(4)
    ax.bar(x - 0.2, dl, 0.4, label="ดาวน์โหลด", color="#cbd5e1"); ax.bar(x + 0.2, ps, 0.4, label="ผ่านเกณฑ์", color=GREEN)
    for i in range(4):
        ax.text(x[i] - 0.2, dl[i] + 3, dl[i], ha="center", fontsize=8.5); ax.text(x[i] + 0.2, ps[i] + 3, ps[i], ha="center", fontsize=8.5, color=GREEN)
    ax.set_xticks(x); ax.set_xticklabels(rounds, fontsize=8.5); ax.legend(); ax.grid(axis="y", alpha=0.3)
    ax.set_title("ผลการเก็บข้อมูลแต่ละรอบ (รวม 405 ไม่ซ้ำ → ผ่าน 250)", fontsize=10.5, color=NAVY, fontweight="bold")
    ax = axs[1]
    labels = ["CC BY-SA 4.0", "CC0", "CC BY 2.0", "CC BY-SA 3.0", "Public domain", "CC BY 3.0"]
    vals = [151, 42, 38, 13, 5, 1]
    ax.pie(vals, labels=[f"{l}\n{v}" for l, v in zip(labels, vals)], startangle=90, counterclock=False, textprops={"fontsize": 8},
           colors=["#6366f1", "#22c55e", "#f59e0b", "#06b6d4", "#94a3b8", "#ec4899"], wedgeprops={"edgecolor": "white"})
    ax.set_title("สัญญาอนุญาตของรูปที่ผ่านเกณฑ์ 250 รูป", fontsize=10.5, color=NAVY, fontweight="bold")
    fig.tight_layout()
    save(fig, "fig_dataset.png")


# ---------------------------------------------------------------- 14) HITL loop
def fig_hitl():
    fig, ax = plt.subplots(figsize=(10.5, 3.9))
    ax.set_xlim(0, 10.5); ax.set_ylim(0, 3.9); ax.axis("off")
    items = [("ผู้ใช้อัปโหลดรูป\nบน Web App", "#dbeafe"), ("โมเดลตรวจจับ\n+ ครอปป้าย", "#e0e7ff"), ("ผู้ใช้ยืนยัน\n'ถูกต้อง' / 'ตำแหน่งผิด'\nหรือวาดกรอบเอง", "#fef3c7"),
             ("เก็บลง outputs/feedback/\n(แยกจาก dataset จริง)", "#f3e8ff"), ("Maintainer ตรวจด้วยตา\nimport_feedback.py", "#fce7f3"), ("นำเข้า data/raw\n→ เทรนรอบถัดไป", "#dcfce7")]
    xs = [0.1, 1.85, 3.6, 5.35, 7.1, 8.85]
    for (t, c), x in zip(items, xs):
        box(ax, x, 1.7, 1.55, 1.3, t, fc=c, fs=8.4)
    for i in range(5):
        arrow(ax, xs[i] + 1.58, 2.35, xs[i + 1] - 0.03, 2.35)
    box(ax, 2.6, 0.2, 5.3, 0.85, "Guardrail: ไม่มีการอัปเดตโมเดลอัตโนมัติ — ทุกข้อมูลต้องผ่านการตรวจของทีมงานก่อน", fc="#0f172a", ec="#0f172a", tc="white", fs=9)
    ax.text(5.25, 3.65, "วงจร Human-in-the-Loop Feedback", ha="center", fontsize=12.5, fontweight="bold", color=NAVY)
    save(fig, "fig_hitl.png")


if __name__ == "__main__":
    fig_project_pipeline(); fig_inference_pipeline(); fig_yolo_arch(); fig_iou(); fig_nms(); fig_kfold(); fig_letterbox()
    fig_grid(); fig_samples(); fig_crop_steps(); fig_threshold(); fig_cv(); fig_dataset(); fig_hitl()
