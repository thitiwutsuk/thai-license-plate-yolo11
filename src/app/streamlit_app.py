"""
เว็บแอปสำหรับระบบตรวจจับและครอปป้ายทะเบียนไทยด้วย YOLO11
อัปโหลดรูปรถ -> ดาวน์โหลดรูปป้ายทะเบียนที่ครอปแล้ว

วิธีรัน:
  streamlit run src/app/streamlit_app.py
"""

import base64
import csv
import io
import re
import sys
import tempfile
import urllib.parse
from pathlib import Path

import cv2
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

# ----- compatibility shim: streamlit-drawable-canvas==0.9.3 vs streamlit==1.51.0 -----
# streamlit-drawable-canvas 0.9.3 import `image_to_url` จาก streamlit.elements.image (ตำแหน่งเก่า)
# แต่ streamlit 1.51 ย้ายฟังก์ชันนี้ไป streamlit.elements.lib.image_utils แล้ว พร้อมเปลี่ยน argument
# ตัวที่สองจาก width (int) เป็น LayoutConfig object — ถ้าไม่แพตช์ไว้ canvas จะ error
# AttributeError ทันทีตอนวาดกรอบ (เลือกใช้วิธีแพตช์แทนการอัปเกรด streamlit เพราะอัปเกรดแล้ว
# CSS ที่ปรับแต่งไว้ทั้งหมดพังตามที่เจอมาก่อน)
import streamlit.elements.image as _st_image_compat  # noqa: E402

if not hasattr(_st_image_compat, "image_to_url"):
    from streamlit.elements.lib.image_utils import image_to_url as _new_image_to_url  # noqa: E402
    from streamlit.elements.lib.layout_utils import LayoutConfig as _LayoutConfig  # noqa: E402

    def _image_to_url_compat(image, width, clamp, channels, output_format, image_id):
        return _new_image_to_url(image, _LayoutConfig(width=width), clamp, channels, output_format, image_id)

    _st_image_compat.image_to_url = _image_to_url_compat

from streamlit_drawable_canvas import st_canvas  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "src"))
from inference.predict_and_crop import predict_and_crop  # noqa: E402
from app.feedback_utils import canvas_rect_to_yolo, save_feedback  # noqa: E402
from crop.crop_utils import crop_plate  # noqa: E402

WEIGHTS_PATH = REPO_ROOT / "models" / "best.pt"
CV_SUMMARY_PATH = REPO_ROOT / "outputs" / "predictions" / "cv_summary.csv"
TEST_SUMMARY_PATH = REPO_ROOT / "outputs" / "predictions" / "test_set_summary.csv"
DATASET_MANIFEST_PATH = REPO_ROOT / "data" / "raw" / "dataset_manifest.csv"

st.set_page_config(page_title="Thai Plate Detector", page_icon="🚘", layout="wide")

if "active_tab" not in st.session_state:
    st.session_state.active_tab = "detect"
if "model_step" not in st.session_state:
    st.session_state.model_step = "1"

DEFAULT_CONF = 0.6
DEFAULT_IOU = 0.45

# ---------- Theme palette — ใช้ธีมสว่าง (ขาว) เพียงธีมเดียวเสมอ ไม่มีโหมดมืดแล้ว ----------
C = {
    "app_bg": "#f8fafc",
    "panel_bg": "#ffffff",
    "panel_bg2": "#eef2f7",
    "border": "#e2e8f0",
    "border2": "#cbd5e1",
    "text_primary": "#0f172a",
    "text_secondary": "#475569",
    "text_muted": "#64748b",
    "text_faint": "#94a3b8",
    "accent": "#0284c7",
    "sidebar_bg": "#f1f5f9",
}

# ---------- ไอคอนเส้นบาง (SVG inline, วาดเองด้วยรูปทรงพื้นฐาน — ไม่พึ่งไลบรารีภายนอก) ----------
ICON_PATHS = {
    "detect": (
        '<path d="M3 7V5a2 2 0 0 1 2-2h2"/><path d="M17 3h2a2 2 0 0 1 2 2v2"/>'
        '<path d="M21 17v2a2 2 0 0 1-2 2h-2"/><path d="M7 21H5a2 2 0 0 1-2-2v-2"/>'
        '<circle cx="12" cy="12" r="3"/>'
    ),
    "settings": (
        '<line x1="4" y1="6" x2="20" y2="6"/><circle cx="9" cy="6" r="2"/>'
        '<line x1="4" y1="12" x2="20" y2="12"/><circle cx="15" cy="12" r="2"/>'
        '<line x1="4" y1="18" x2="20" y2="18"/><circle cx="7" cy="18" r="2"/>'
    ),
    "model": (
        '<rect x="6" y="6" width="12" height="12" rx="2"/><rect x="10" y="10" width="4" height="4"/>'
        '<line x1="2" y1="9" x2="6" y2="9"/><line x1="2" y1="15" x2="6" y2="15"/>'
        '<line x1="18" y1="9" x2="22" y2="9"/><line x1="18" y1="15" x2="22" y2="15"/>'
        '<line x1="9" y1="2" x2="9" y2="6"/><line x1="15" y1="2" x2="15" y2="6"/>'
        '<line x1="9" y1="18" x2="9" y2="22"/><line x1="15" y1="18" x2="15" y2="22"/>'
    ),
    "pipeline": (
        '<circle cx="4" cy="12" r="2"/><circle cx="12" cy="6" r="2"/>'
        '<circle cx="12" cy="18" r="2"/><circle cx="20" cy="12" r="2"/>'
        '<line x1="6" y1="12" x2="10" y2="7"/><line x1="6" y1="12" x2="10" y2="17"/>'
        '<line x1="14" y1="7" x2="18" y2="11"/><line x1="14" y1="17" x2="18" y2="13"/>'
    ),
}

TABS = [
    ("detect", "Detect"),
    ("settings", "Settings"),
    ("model", "Model Info"),
    ("pipeline", "Pipeline"),
]


def css_key(key: str) -> str:
    """แปลง widget key ให้ตรงกับ class ".st-key-<...>" ที่ Streamlit frontend ใส่ให้จริง —
    frontend แทนที่ทุกตัวอักษรที่ไม่ใช่ a-z/A-Z/0-9/_/- ด้วย "-" (เช่น "." ในเลขทศนิยมของ
    cache_key กลายเป็น "-") ต้องแปลงให้ตรงกันก่อนเอาไปต่อเป็น selector เอง ไม่งั้น selector
    จะไม่ตรงกับ class จริงบนหน้าเว็บเลย (เจอบั๊กนี้มาแล้วตอนปุ่ม feedback ไม่ติดสี)"""
    return re.sub(r"[^a-zA-Z0-9_-]", "-", key.strip())


def svg_data_uri(tab_id: str, color: str) -> str:
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" '
        f'stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
        f"{ICON_PATHS[tab_id]}</svg>"
    )
    return "data:image/svg+xml," + urllib.parse.quote(svg)


# ---------- CSS: ดีไซน์โมเดิร์น + กลิ่นอายงานวิจัย/เทคนิค (รองรับ 2 ธีม) ----------
# แถบนำทางหลัก — ปุ่มไอคอน+ข้อความเรียงแนวนอนในแคปซูลเดียว (segmented control) ปุ่ม active
# ลอยเป็นพื้นขาว+เงาตื้นๆ บนแทร็กพื้นเทา ส่วนปุ่มอื่นโปร่งใส ตามดีไซน์อ้างอิงที่ผู้ใช้แนบมา
tab_css = ""
for tab_id, _ in TABS:
    is_active = st.session_state.active_tab == tab_id
    icon_color = C["text_primary"] if is_active else C["text_muted"]
    text_color = C["text_primary"] if is_active else C["text_secondary"]
    bg = "#ffffff" if is_active else "transparent"
    shadow = "0 1px 2px rgba(16,24,40,0.08)" if is_active else "none"
    uri = svg_data_uri(tab_id, icon_color)
    tab_css += f"""
    .st-key-tab_{tab_id} button {{
        background: {bg} !important;
        border: none !important;
        border-radius: 9px !important;
        color: {text_color} !important;
        font-weight: 600 !important;
        font-size: 0.86rem !important;
        box-shadow: {shadow} !important;
        padding: 0.5rem 0.9rem !important;
        width: 100%;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        gap: 0.45rem;
        transition: background 0.15s ease, color 0.15s ease;
    }}
    .st-key-tab_{tab_id} button:hover {{
        background: {'#ffffff' if is_active else 'rgba(15,23,42,0.045)'} !important;
        color: {C['text_primary']} !important;
    }}
    .st-key-tab_{tab_id} button::before {{
        content: "";
        flex: 0 0 auto;
        width: 18px; height: 18px;
        background-image: url("{uri}");
        background-size: contain;
        background-repeat: no-repeat;
        background-position: center;
    }}
    """

# สี accent เฉพาะหน้า Settings — อิงตามไฟล์ตัวอย่าง Settings.dc.html (โทนม่วง/indigo)
SA = "#4F46E5"
SA_SOFT = "#EEF2FF"

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=IBM+Plex+Sans+Thai:wght@400;500;600;700&family=JetBrains+Mono:wght@500;600&display=swap');

    html, body, [class*="css"] {{ font-family: 'Inter', 'IBM Plex Sans Thai', sans-serif; }}

    /* Override ตัวแปรสีหลักของ Streamlit เอง ไม่ให้ native widget (header, ปุ่ม, dropzone, slider)
       ไปตามโหมดมืด/สว่างของเบราว์เซอร์ผู้ใช้ — บังคับให้ตามปุ่มสลับธีมในเว็บแทน */
    :root, .stApp {{
        --background-color: {C['app_bg']} !important;
        --secondary-background-color: {C['panel_bg']} !important;
        --text-color: {C['text_primary']} !important;
        --primary-color: {C['accent']} !important;
    }}

    .stApp {{ background: {C['app_bg']} !important; }}

    header[data-testid="stHeader"] {{
        background: {C['app_bg']} !important;
        border-bottom: 1px solid {C['border']};
    }}
    header[data-testid="stHeader"] span,
    header[data-testid="stHeader"] p,
    header[data-testid="stHeader"] a {{ color: {C['text_primary']} !important; }}
    header[data-testid="stHeader"] svg path {{ stroke: {C['text_primary']} !important; }}

    [data-testid="stFileUploaderDropzone"] {{
        background: {C['panel_bg']} !important;
        border: 1.5px dashed {C['border2']} !important;
    }}
    [data-testid="stFileUploaderFile"] {{
        background: {C['panel_bg']} !important;
        color: {C['text_primary']} !important;
    }}
    [data-testid="stFileUploaderFileData"],
    [data-testid="stFileUploaderFileName"] {{
        color: {C['text_primary']} !important;
    }}
    [data-testid="stFileUploaderFileData"] small {{
        color: {C['text_muted']} !important;
    }}
    [data-testid="stFileUploaderFile"] svg {{
        fill: {C['text_muted']} !important;
    }}

    /* st.caption() ของ Streamlit ลด opacity เหลือ 0.6 โดยปริยาย ทำให้ตัวหนังสือจางมองไม่ชัด
       บนพื้นหลังขาว — บังคับ opacity เต็มแล้วกำหนดสีเองให้ยังคงความแตกต่างจากข้อความหลัก */
    [data-testid="stCaptionContainer"] {{
        opacity: 1 !important;
    }}
    [data-testid="stCaptionContainer"] p {{
        color: {C['text_secondary']} !important;
    }}

    .stButton button, .stDownloadButton button {{
        background: {C['panel_bg']} !important;
        color: {C['text_primary']} !important;
        border: 1px solid {C['border2']} !important;
    }}

    /* แถบนำทางบนสุด — โลโก้ + nav + สถานะโมเดล ในแถบเดียวเรียบๆ (แทนการ์ดหัวเว็บแบบเดิม) */
    .st-key-main_header_row {{
        border-bottom: 1px solid {C['border']};
        padding-bottom: 0.9rem;
        margin-bottom: 1.5rem;
    }}
    .st-key-main_header_row [data-testid="stHorizontalBlock"] > div {{
        display: flex; align-items: center;
    }}
    .main-logo {{ display: flex; align-items: center; gap: 0.7rem; }}
    .main-logo-icon {{
        flex: 0 0 auto; width: 36px; height: 36px; border-radius: 10px;
        background: #1e3a8a; display: flex; align-items: center; justify-content: center;
    }}
    .main-logo-title {{ font-size: 0.95rem; font-weight: 700; color: {C['text_primary']}; line-height: 1.25; }}
    .main-logo-sub {{ font-size: 0.74rem; color: {C['text_muted']}; line-height: 1.25; }}
    .st-key-main_nav_track {{
        background: {C['panel_bg2']}; border-radius: 12px; padding: 4px;
    }}
    .st-key-main_nav_track [data-testid="stHorizontalBlock"] {{ gap: 4px; }}
    .status-pill {{
        display: flex; align-items: center; gap: 0.5rem; padding: 0.4rem 0.75rem;
        border: 1px solid #bbf7d0; background: #f0fdf4; border-radius: 999px;
        font-size: 0.8rem; font-weight: 600; color: #166534; white-space: nowrap; margin-left: auto;
    }}
    .status-dot {{ width: 8px; height: 8px; border-radius: 50%; background: #16a34a; }}

    .badge-row {{ margin-top: 0.9rem; margin-bottom: 1.75rem; display: flex; gap: 0.5rem; flex-wrap: wrap; }}
    .badge {{
        display: inline-block;
        padding: 0.28rem 0.7rem;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 600;
        font-family: 'JetBrains Mono', monospace;
        background: {C['panel_bg2']};
        color: {C['accent']};
        border: 1px solid {C['border2']};
    }}

    .metric-card {{
        background: {C['panel_bg']};
        border: 1px solid {C['border']};
        border-radius: 14px;
        padding: 1rem 1.1rem;
        text-align: left;
        flex: 1 1 150px;
        min-width: 150px;
    }}
    .metric-card-grid {{ display: flex; flex-wrap: wrap; gap: 0.75rem; }}
    .metric-card .label {{
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: {C['text_muted']};
        font-weight: 600;
    }}
    .metric-card .value {{
        font-size: 1.6rem;
        font-weight: 700;
        color: {C['text_primary']};
        font-family: 'JetBrains Mono', monospace;
        margin-top: 0.15rem;
    }}
    .metric-card .sub {{
        font-size: 0.72rem;
        color: {C['text_faint']};
        margin-top: 0.1rem;
    }}
    .metric-card-dark {{ background: #0f172a; border-color: #0f172a; }}
    .metric-card-dark .label {{ color: #d4d4d8; text-transform: none; letter-spacing: normal; }}
    .metric-card-dark .value {{ color: #ffffff; }}
    .metric-card-dark .sub {{ color: #a1a1aa; }}

    /* แท็บ Settings — การ์ดเซคชัน confidence/NMS IoU ทรง SaaS settings page */
    .st-key-settings_conf_section, .st-key-settings_iou_section {{
        background: {C['panel_bg']};
        border: 1px solid {C['border']};
        border-radius: 18px;
        padding: 0 1.5rem 1.5rem 1.5rem;
        margin-bottom: 1.2rem;
    }}
    .settings-section-head {{
        display: flex;
        align-items: flex-start;
        justify-content: space-between;
        gap: 1rem;
        flex-wrap: wrap;
        margin: 0 -1.5rem 1.3rem -1.5rem;
        padding: 1.3rem 1.5rem;
        border-bottom: 1px solid {C['border']};
    }}
    .settings-section-head-left {{ display: flex; align-items: flex-start; gap: 0.8rem; flex: 1 1 auto; min-width: 0; }}
    .settings-icon {{
        flex: 0 0 auto;
        width: 38px;
        height: 38px;
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        background: {SA_SOFT};
        color: #4338CA;
        margin-top: 0.1rem;
    }}
    .settings-card-title {{ font-size: 1rem; font-weight: 700; color: {C['text_primary']}; }}
    .settings-card-desc {{ font-size: 0.8rem; color: {C['text_muted']}; margin-top: 0.2rem; }}
    .settings-section-head-right {{ flex: 0 0 auto; text-align: right; }}
    .settings-badge {{
        display: inline-block;
        font-size: 0.68rem;
        font-weight: 700;
        padding: 0.2rem 0.55rem;
        border-radius: 999px;
        background: #F4F4F5;
        color: #52525B;
        margin-bottom: 0.4rem;
    }}
    .settings-badge.changed {{ background: #FFF7ED; color: #C2410C; }}
    .settings-section-value {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.5rem;
        font-weight: 700;
        color: {C['text_primary']};
        line-height: 1;
    }}

    /* แถบ slider ในหน้า Settings — ย้อมสีแดงเริ่มต้นของ Streamlit ให้เป็นโทน indigo ตามไฟล์ตัวอย่าง
       แล้วขยายปุ่มลาก (thumb) ให้ใหญ่ขึ้นแบบ custom slider ในไฟล์ตัวอย่าง */
    [data-testid="stSlider"] div[data-baseweb="slider"] div[role="slider"] {{
        background-color: {SA} !important;
        border: 3px solid {SA} !important;
        width: 20px !important;
        height: 20px !important;
        box-shadow: 0 1px 4px rgba(0,0,0,0.15) !important;
        position: relative !important;
        z-index: 2 !important;
    }}
    [data-testid="stSlider"] div[style*="height: 0.25rem"] {{
        filter: hue-rotate(239deg) saturate(1.5);
        position: relative !important;
        z-index: 1 !important;
    }}
    [data-testid="stSliderThumbValue"] {{
        color: {SA} !important;
        font-weight: 700 !important;
    }}
    /* เส้นขีดบอกตำแหน่งค่าเริ่มต้นบนแถบ confidence slider (มีเฉพาะ confidence ตามไฟล์ตัวอย่าง ไม่มีใน IoU) —
       ใช้ container key จริง (ไม่ใช่ div จาก markdown) เพื่อให้ ::after วางซ้อนบนแถบ slider จริงได้ */
    .st-key-conf_slider_wrap {{ position: relative; }}
    .st-key-conf_slider_wrap::after {{
        content: "";
        position: absolute;
        left: 50%;
        top: 9px;
        width: 2px;
        height: 20px;
        margin-left: -1px;
        background: #A1A1AA;
        pointer-events: none;
        z-index: 1;
    }}

    /* การ์ดอธิบายข้อดี/ข้อเสีย (เขียว/ส้ม) พร้อมไอคอนลูกศรบอกทิศทาง */
    .settings-tradeoff-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 0.8rem; margin-top: 1.1rem; }}
    .settings-tradeoff-card {{ border-radius: 12px; padding: 0.8rem 0.9rem; font-size: 0.78rem; line-height: 1.5; }}
    .settings-tradeoff-card.good {{ background: #F0FDF4; }}
    .settings-tradeoff-card.good .ttitle {{ color: #14532D; }}
    .settings-tradeoff-card.good {{ color: #3F5A47; }}
    .settings-tradeoff-card.bad {{ background: #FFF7ED; color: #7C2D12; }}
    .settings-tradeoff-card.bad .ttitle {{ color: #7C2D12; }}
    .settings-tradeoff-card .ttitle-icon {{
        width: 20px; height: 20px; border-radius: 6px; flex: 0 0 auto;
        display: flex; align-items: center; justify-content: center;
    }}
    .settings-tradeoff-card.good .ttitle-icon {{ background: #DCFCE7; color: #15803D; }}
    .settings-tradeoff-card.bad .ttitle-icon {{ background: #FFEDD5; color: #C2410C; }}
    .settings-tradeoff-card .ttitle {{ display: flex; align-items: center; gap: 0.35rem; font-weight: 700; margin-bottom: 0.15rem; }}
    .settings-tradeoff-card .ttitle svg {{ flex: 0 0 auto; }}

    /* กล่องค่าต่ำ/ค่าสูง สำหรับ NMS IoU */
    .settings-iou-tips {{ display: grid; grid-template-columns: 1fr 1fr; gap: 0.7rem; margin-top: 1rem; }}
    .settings-iou-tip {{ background: #f8faff; border-radius: 10px; padding: 0.7rem 0.8rem; font-size: 0.76rem; color: {C['text_secondary']}; line-height: 1.5; }}
    .settings-iou-tip b {{ color: {C['text_primary']}; }}

    /* แผนภาพประกอบ NMS (กรอบซ้อนทับ) */
    .settings-iou-figure {{
        border: 1px dashed {C['border2']};
        border-radius: 12px;
        background: #fafafa;
        height: 100%;
        min-height: 150px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 1rem;
    }}
    /* ภาพประกอบ 3 ชั้น: กรอบอ่อน (b1) + กรอบเข้ม (b2) + ส่วนที่ซ้อนทับกันจริงๆ (overlap, ทึบกว่า) — ตำแหน่ง/ขนาด
       overlap คำนวณให้ตรงกับพื้นที่ที่ b1/b2 ทับกันจริงตามไฟล์ตัวอย่างเป๊ะ */
    .settings-iou-diagram {{ position: relative; width: 180px; height: 110px; }}
    .settings-iou-box {{
        position: absolute;
        box-sizing: border-box;
        border-radius: 6px;
        width: 110px;
        height: 64px;
    }}
    .settings-iou-box.b1 {{ left: 10px; top: 14px; border: 2px solid #A5B4FC; background: rgba(165,180,252,0.18); }}
    .settings-iou-box.b2 {{ left: 50px; top: 32px; border: 2px solid {SA}; background: rgba(79,70,229,0.12); }}
    .settings-iou-box.overlap {{ left: 50px; top: 32px; width: 70px; height: 46px; background: rgba(79,70,229,0.28); }}
    .settings-iou-figcaption {{ margin-top: 0.6rem; font-size: 0.72rem; color: {C['text_muted']}; text-align: center; line-height: 1.4; }}

    /* Sidebar — การ์ดค่าที่ใช้อยู่ตอนนี้ / metrics ที่วัดค่าเริ่มต้น */
    .st-key-settings_current_card, .st-key-settings_metrics_card {{
        background: {C['panel_bg']};
        border: 1px solid {C['border']};
        border-radius: 18px;
        padding: 1.3rem 1.5rem;
        margin-bottom: 1.2rem;
    }}
    [data-testid="stMarkdownContainer"] h2.settings-side-title {{
        margin: 0 !important; font-size: 0.95rem !important; font-weight: 700 !important;
        color: {C['text_primary']} !important; line-height: 1.4 !important;
    }}
    .settings-side-desc {{ font-size: 0.76rem; color: {C['text_muted']}; margin: 0.2rem 0 0.9rem 0; padding-bottom: 0.9rem; border-bottom: 1px solid {C['border']}; }}
    .settings-old-value {{ text-decoration: line-through; color: {C['text_faint']}; font-weight: 400; margin-right: 0.4rem; }}
    .settings-new-value {{ color: #C2410C; }}
    .settings-const-label {{ display: inline-flex; align-items: center; gap: 0.3rem; }}
    .settings-const-icon {{ color: {C['text_faint']}; display: inline-flex; }}
    .st-key-reset_settings button {{
        width: 100%;
        margin-top: 1rem;
        background: transparent !important;
        border: 1px solid {C['border']} !important;
        color: {C['text_secondary']} !important;
    }}
    .st-key-reset_settings button:hover {{
        border-color: {SA} !important;
        color: {SA} !important;
    }}

    .settings-metrics-head {{ display: flex; align-items: center; gap: 0.6rem; margin-bottom: 0.5rem; }}
    .settings-metrics-icon {{ flex: 0 0 auto; width: 32px; height: 32px; border-radius: 9px; background: #FFF7ED; color: #C2410C; display: flex; align-items: center; justify-content: center; }}
    [data-testid="stMarkdownContainer"] h2.settings-metrics-title {{
        margin: 0 !important; font-size: 0.9rem !important; font-weight: 700 !important;
        color: {C['text_primary']} !important; line-height: 1.3 !important;
    }}
    .settings-metrics-desc {{ font-size: 0.76rem; color: {C['text_muted']}; line-height: 1.55; margin-bottom: 1rem; }}
    .settings-metrics-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 0.7rem; }}
    .settings-metric-box {{ background: {C['panel_bg2']}; border-radius: 10px; padding: 0.7rem 0.8rem; }}
    .settings-metric-box .mlabel {{ font-size: 0.68rem; color: {C['text_muted']}; }}
    .settings-metric-box .mvalue {{ font-family: 'JetBrains Mono', monospace; font-size: 1.1rem; font-weight: 700; color: {C['text_primary']}; margin-top: 0.1rem; }}

    .result-panel {{
        background: {C['panel_bg2']};
        border-radius: 14px;
        padding: 1.1rem;
        min-height: 360px;
        box-sizing: border-box;
    }}
    .result-panel-center {{ text-align: center; }}
    /* ให้ทั้งสอง panel ชิดขอบบนเท่ากันเสมอ (ไม่ centered เป็นก้อน) แล้วให้ลูกศรเองยืดเต็มความสูง
       ของคอลัมน์ตัวเอง (เพราะ Streamlit ยืด column ให้สูงเท่าแถวโดย default) แล้วค่อย flex-center
       ตัวลูกศรภายในคอลัมน์นั้นอีกที — วิธีนี้ไม่กระทบตำแหน่งเนื้อหาจริงใน panel ทั้งสองฝั่งเลย */
    .result-arrow {{
        height: 100%;
        min-height: 220px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 2.2rem;
        font-weight: 700;
        color: #f8bd38;
        animation: flowCardIn 0.5s ease-out 0.3s both;
    }}
    .result-arrow-muted {{ color: {C['border2']}; animation: none; }}
    @media (prefers-reduced-motion: reduce) {{
        .result-arrow {{ animation: none; }}
    }}
    /* กรอบ layout เปล่า ก่อนมีการอัปโหลดรูป */
    .result-placeholder {{
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        gap: 0.6rem;
        min-height: 220px;
        border: 1.5px dashed {C['border2']};
        border-radius: 12px;
        color: {C['text_muted']};
        margin-top: 0.5rem;
    }}
    .result-placeholder-text {{ font-size: 0.85rem; }}
    .result-panel h4 {{
        color: {C['text_secondary']};
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin: 0 0 0.6rem 0;
    }}

    .conf-pill {{
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 8px;
        font-family: 'JetBrains Mono', monospace;
        font-weight: 600;
        font-size: 0.95rem;
    }}
    .conf-high {{ background: rgba(34,197,94,0.15); color: #16a34a; border: 1px solid rgba(34,197,94,0.35); }}
    .conf-mid  {{ background: rgba(234,179,8,0.15); color: #ca8a04; border: 1px solid rgba(234,179,8,0.35); }}
    .conf-low  {{ background: rgba(239,68,68,0.15); color: #dc2626; border: 1px solid rgba(239,68,68,0.35); }}

    /* บังคับสีตัวอักษรของ widget ของ Streamlit เองให้ตามธีม */
    .stApp [data-testid="stMarkdownContainer"] p,
    .stApp [data-testid="stMarkdownContainer"] li,
    .stApp [data-testid="stMarkdownContainer"] h1,
    .stApp [data-testid="stMarkdownContainer"] h2,
    .stApp [data-testid="stMarkdownContainer"] h3,
    .stApp [data-testid="stMarkdownContainer"] h4,
    .stApp label,
    .stApp [data-testid="stFileUploaderDropzoneInstructions"] span,
    .stApp [data-testid="stExpander"] summary {{ color: {C['text_primary']} !important; }}
    .stApp [data-testid="stFileUploaderDropzoneInstructions"] small {{ color: {C['text_muted']} !important; }}

    [data-testid="stExpander"] {{
        background: {C['panel_bg']} !important;
        border: 1px solid {C['border']} !important;
        border-radius: 10px;
    }}
    [data-testid="stExpander"] summary {{ background: {C['panel_bg']} !important; }}
    [data-testid="stExpander"] summary:hover {{ background: {C['panel_bg2']} !important; }}
    [data-testid="stExpander"] [data-testid="stExpanderDetails"] {{ background: {C['panel_bg']} !important; }}
    [data-testid="stExpander"] svg path {{ fill: {C['text_primary']} !important; }}

    /* แถบแท็บไอคอน */
    {tab_css}

    /* Pipeline flowchart — header */
    .pipeline-eyebrow {{
        font-size: 0.72rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase;
        color: {C['accent']}; margin-bottom: 0.35rem;
    }}
    .pipeline-header-row {{
        display: flex; justify-content: space-between; align-items: flex-end; flex-wrap: wrap; gap: 0.5rem 1.5rem;
    }}
    .pipeline-title {{
        font-size: 2.1rem; font-weight: 800; color: {C['text_primary']}; letter-spacing: -0.02em; margin: 0; line-height: 1.1;
    }}
    .pipeline-meta {{
        font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; color: {C['text_muted']};
        letter-spacing: 0.03em; white-space: nowrap; padding-bottom: 0.3rem;
    }}
    .pipeline-desc {{ color: {C['text_secondary']}; font-size: 0.9rem; margin-top: 0.7rem; max-width: 760px; line-height: 1.65; }}

    /* Model Info — panel กราฟ/ตาราง (แต่ละ container key map ไปสไตล์ .model-panel) */
    .st-key-model_panel_cv, .st-key-model_panel_valtest, .st-key-model_panel_crop,
    .st-key-model_panel_split, .st-key-model_panel_license,
    .st-key-model_panel_split1, .st-key-model_panel_spec {{
        background: {C['panel_bg']};
        border: 1px solid {C['border']};
        border-radius: 16px;
        padding: 1.4rem 1.5rem 1.3rem 1.5rem;
    }}

    /* Model Info — หัวข้อย่อยแบบ step (เหมือน pipeline header แต่เล็กกว่า ใช้ซ้ำได้หลายส่วนในหน้าเดียว) */
    .model-step {{ margin-top: 2.6rem; }}
    .model-step:first-of-type {{ margin-top: 0.3rem; }}
    .model-step-eyebrow {{
        font-size: 0.72rem; font-weight: 800; letter-spacing: 0.08em; text-transform: uppercase;
        color: {C['accent']}; margin-bottom: 0.4rem;
    }}
    .model-step-title {{
        font-size: 1.6rem; font-weight: 800; color: {C['text_primary']}; letter-spacing: -0.01em;
        margin: 0 0 0.45rem 0; line-height: 1.25;
    }}
    .model-step-desc {{ color: {C['text_secondary']}; font-size: 0.86rem; margin-bottom: 1.2rem; max-width: 760px; line-height: 1.6; }}

    /* Model Info — แถบ sub-tab เลือก Step 1-4 แบบ segmented control (เหมือนเว็บ SaaS เช่น Linear/Vercel):
       แทร็กพื้นเทาโค้งมน ปุ่มลอยอยู่ข้างในไม่มีเส้นขอบ ปุ่มที่ active เป็นพื้นขาว+เงาตื้นๆ ลอยเด่นขึ้นมา */
    .model-substep-bar {{ margin: 1.4rem 0 2.2rem 0; }}
    .st-key-model_substep_track {{
        background: {C['panel_bg2']};
        border: 1px solid {C['border']};
        border-radius: 14px;
        padding: 5px;
    }}
    .st-key-model_substep_track [data-testid="stHorizontalBlock"] {{ gap: 4px; }}

    /* Model Info Step 1 — ตารางแบ่ง fold (แต่ละแถวคือ 1 fold, 5 บล็อกคือ 5 ส่วนของ train) */
    .fold-table {{ display: flex; flex-direction: column; gap: 0.5rem; margin-top: 1rem; }}
    .fold-row {{ display: flex; align-items: center; gap: 0.7rem; border-radius: 8px; padding: 0.25rem 0.4rem; border: 1px solid transparent; }}
    .fold-row.fold-row-best {{ background: {C['accent']}0d; border-color: {C['accent']}55; }}
    .fold-row-label {{
        flex: 0 0 4.4rem; font-size: 0.78rem; font-weight: 700; color: {C['text_primary']};
        display: flex; align-items: center; gap: 0.35rem; white-space: nowrap;
    }}
    .fold-blocks {{ flex: 1 1 auto; display: flex; gap: 4px; }}
    .fold-block {{
        flex: 1 1 0; height: 28px; border-radius: 6px; background: {C['panel_bg2']}; border: 1px solid {C['border']};
        display: flex; align-items: center; justify-content: center;
        font-size: 0.62rem; font-weight: 700; color: {C['text_faint']};
    }}
    .fold-block.val {{ background: #f59e0b; color: #fff; border-color: #f59e0b; }}
    .fold-row-badge {{
        flex: 0 0 auto; font-size: 0.6rem; font-weight: 800; text-transform: uppercase; letter-spacing: 0.03em;
        color: {C['accent']}; background: {C['accent']}1a; padding: 0.15rem 0.5rem; border-radius: 999px;
    }}

    /* Model Info Step 1 — ตาราง key-value สเปกโมเดล */
    .spec-row {{
        display: flex; justify-content: space-between; align-items: center;
        padding: 0.6rem 0; border-bottom: 1px solid {C['border']}; font-size: 0.82rem;
    }}
    .spec-row:last-child {{ border-bottom: none; }}
    .spec-key {{ color: {C['text_muted']}; }}
    .spec-val {{ color: {C['text_primary']}; font-weight: 700; font-family: 'JetBrains Mono', monospace; }}

    /* Model Info — การ์ดสถิติ (รองรับแบบพื้นเข้มสำหรับค่าเด่น) */
    .model-metric-row {{ display: flex; flex-wrap: wrap; gap: 0.8rem; margin-bottom: 1.1rem; }}
    .model-metric-card {{
        flex: 1 1 170px; min-width: 150px; background: {C['panel_bg']}; border: 1px solid {C['border']};
        border-radius: 14px; padding: 1rem 1.15rem;
    }}
    .model-metric-card.dark {{ background: #0f172a; border-color: #0f172a; }}
    .model-metric-card .mlabel {{
        font-size: 0.66rem; text-transform: uppercase; letter-spacing: 0.05em; font-weight: 700;
        color: {C['text_muted']};
    }}
    .model-metric-card.dark .mlabel {{ color: #94a3b8; }}
    .model-metric-card .mvalue {{
        font-family: 'JetBrains Mono', monospace; font-size: 1.5rem; font-weight: 800;
        color: {C['text_primary']}; margin-top: 0.2rem;
    }}
    .model-metric-card.dark .mvalue {{ color: #fff; }}
    .model-metric-card .msub {{ font-size: 0.7rem; color: {C['text_faint']}; margin-top: 0.25rem; }}
    .model-metric-card.dark .msub {{ color: #64748b; }}

    /* Model Info — กล่อง panel ขาวพร้อมหัวข้อ ใช้ใส่กราฟ/ตาราง */
    .model-panel-grid {{ display: flex; gap: 1rem; align-items: stretch; flex-wrap: wrap; }}
    .model-panel {{
        flex: 1 1 320px; min-width: 0; background: {C['panel_bg']}; border: 1px solid {C['border']};
        border-radius: 16px; padding: 1.4rem 1.5rem 1.3rem 1.5rem;
    }}
    .model-panel-title {{
        font-size: 0.9rem; font-weight: 700; color: {C['text_primary']}; margin-bottom: 1.1rem;
        display: flex; align-items: center; justify-content: space-between; gap: 0.6rem;
    }}
    .model-panel-title .mp-sub {{
        font-size: 0.72rem; font-weight: 600; color: {C['text_muted']}; font-family: 'JetBrains Mono', monospace;
    }}
    .model-legend {{ display: flex; gap: 1rem; margin-bottom: 1rem; flex-wrap: wrap; }}
    .model-legend-item {{ display: flex; align-items: center; gap: 0.4rem; font-size: 0.76rem; color: {C['text_secondary']}; }}
    .model-legend-dot {{ width: 9px; height: 9px; border-radius: 3px; flex: 0 0 auto; }}

    /* กราฟแท่งเทียบค่า — ใช้ร่วมกันทั้งผล per-fold CV และ Validation เทียบ Test set */
    .bar-chart {{ position: relative; }}
    .bar-chart-gridlines {{ position: absolute; left: 2.5rem; right: 0; top: 0; height: 210px; }}
    .bar-chart-gridline {{ position: absolute; left: 0; right: 0; border-top: 1px dashed {C['border']}; }}
    .bar-chart-gridline .gl-label {{
        position: absolute; left: -2.5rem; top: -0.5rem; width: 2.2rem; text-align: right;
        font-size: 0.64rem; font-family: 'JetBrains Mono', monospace; color: {C['text_faint']};
    }}
    .bar-chart-avgline {{ position: absolute; left: 0; right: 0; border-top: 2px dashed #16a34a; z-index: 2; }}
    .bar-chart-avgline .avg-label {{
        position: absolute; right: 0; top: -1.05rem; font-size: 0.64rem; font-family: 'JetBrains Mono', monospace;
        color: #16a34a; font-weight: 700; background: {C['panel_bg']}; padding-left: 0.3rem;
    }}
    .bar-chart-body {{
        position: relative; display: flex; align-items: flex-end; gap: 1.1rem;
        height: 210px; padding-left: 2.5rem; z-index: 1;
    }}
    .bar-group {{
        flex: 1 1 0; display: flex; flex-direction: column; align-items: center; justify-content: flex-end;
        height: 100%; position: relative; border-radius: 8px 8px 0 0;
    }}
    .bar-group.best-group {{ background: {C['accent']}0d; }}
    .bar-set {{ display: flex; align-items: flex-end; gap: 8px; justify-content: center; width: 100%; }}
    .bar {{ width: 15px; border-radius: 3px 3px 0 0; position: relative; min-height: 2px; }}
    .bar .bar-val {{
        position: absolute; top: -1.2rem; left: 50%; transform: translateX(-50%);
        font-size: 0.6rem; font-family: 'JetBrains Mono', monospace; font-weight: 700; white-space: nowrap;
    }}
    .bar-group-label {{ margin-top: 0.65rem; font-size: 0.78rem; font-weight: 700; color: {C['text_primary']}; }}
    .bar-group-sub {{ margin-top: 0.15rem; font-size: 0.64rem; font-family: 'JetBrains Mono', monospace; color: {C['text_faint']}; }}
    .bar-group-badge {{
        margin-top: 0.35rem; font-size: 0.6rem; font-weight: 800; text-transform: uppercase; letter-spacing: 0.03em;
        color: {C['accent']}; background: {C['accent']}1a; padding: 0.15rem 0.5rem; border-radius: 999px;
    }}

    .model-stat-big {{
        font-family: 'JetBrains Mono', monospace; font-weight: 800; color: {C['text_primary']};
    }}
    .model-stat-sub {{ font-size: 0.76rem; color: {C['text_faint']}; }}

    /* Model Info — ตารางสี่เหลี่ยมเล็กแทนรูปทดสอบแต่ละรูป (crop success) */
    .crop-grid {{ display: grid; grid-template-columns: repeat(10, 1fr); gap: 6px; margin-top: 0.3rem; max-width: 280px; }}
    .crop-cell {{ width: 100%; aspect-ratio: 1; border-radius: 4px; background: {C['accent']}; }}
    .crop-cell.fail {{ background: transparent; border: 2px solid #f59e0b; }}

    /* Model Info — แถบสัดส่วนผ่าน/ไม่ผ่านของชุดข้อมูล */
    .split-bar {{ display: flex; height: 34px; border-radius: 8px; overflow: hidden; }}
    .split-bar-seg.pass {{ background: {C['accent']}; }}
    .split-bar-seg.fail {{ background: #f59e0b; }}
    .split-legend {{ display: flex; justify-content: space-between; margin-top: 0.8rem; }}
    .split-legend-item .sl-label {{ font-size: 0.76rem; color: {C['text_secondary']}; display: flex; align-items: center; gap: 0.4rem; }}
    .split-legend-item .sl-value {{ font-family: 'JetBrains Mono', monospace; font-weight: 700; font-size: 0.95rem; color: {C['text_primary']}; margin-top: 0.25rem; }}
    .split-legend-item.right {{ text-align: right; }}
    .split-legend-item.right .sl-label {{ justify-content: flex-end; }}
    .model-divider {{ border-top: 1px solid {C['border']}; margin: 1.1rem 0; }}
    .criteria-label {{
        font-size: 0.7rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.04em;
        color: {C['text_muted']}; margin-bottom: 0.6rem;
    }}
    .criteria-list {{ display: flex; flex-direction: column; gap: 0.5rem; }}
    .criteria-item {{ display: flex; align-items: center; gap: 0.55rem; font-size: 0.8rem; color: {C['text_secondary']}; }}
    .criteria-check {{
        width: 18px; height: 18px; border-radius: 999px; background: #16a34a1a; color: #16a34a;
        display: flex; align-items: center; justify-content: center; flex: 0 0 auto; font-size: 0.66rem; font-weight: 800;
    }}

    /* Model Info — แถวแท่งแนวนอน (ใช้กับสัดส่วนสัญญาอนุญาต) */
    .hbar-row {{ display: flex; align-items: center; gap: 0.8rem; padding: 0.4rem 0; }}
    .hbar-label {{ flex: 0 0 6.8rem; font-size: 0.78rem; color: {C['text_secondary']}; }}
    .hbar-track {{ flex: 1 1 auto; background: {C['border']}; border-radius: 999px; height: 8px; overflow: hidden; }}
    .hbar-fill {{ background: {C['accent']}; height: 100%; border-radius: 999px; }}
    .hbar-count {{ flex: 0 0 auto; width: 2.4rem; text-align: right; font-family: 'JetBrains Mono', monospace; font-weight: 700; font-size: 0.82rem; color: {C['text_primary']}; }}

    /* micro-animation: การ์ด "ไม่พบป้ายทะเบียน" ค่อยๆ ปรากฏ + ไอคอนเต้นเบาๆ 2 ครั้งเพื่อดึงความสนใจ */
    @keyframes flowCardIn {{
        0% {{ opacity: 0; transform: translateY(-6px); }}
        100% {{ opacity: 1; transform: translateY(0); }}
    }}
    @keyframes flowIconPulse {{
        0%, 100% {{ transform: scale(1); }}
        50% {{ transform: scale(1.15); }}
    }}
    .flow-step.flow-end-bad {{ animation: flowCardIn 0.5s ease-out; }}
    .flow-step.flow-end-bad svg {{ animation: flowIconPulse 1s ease-in-out 0.5s 2; transform-origin: center; }}
    @media (prefers-reduced-motion: reduce) {{
        .flow-step.flow-end-bad, .flow-step.flow-end-bad svg {{ animation: none; }}
    }}

    /* การ์ด "ไม่พบป้ายทะเบียน" จริง ที่แสดงผลในแท็บ Detect ตอนตรวจจับไม่เจอ */
    .result-empty {{
        background: #fef2f2;
        border: 1px solid #fecaca;
        border-radius: 16px;
        padding: 2.2rem 1.5rem;
        text-align: center;
        animation: flowCardIn 0.5s ease-out;
    }}
    .result-empty svg {{ animation: flowIconPulse 1s ease-in-out 0.5s 2; transform-origin: center; }}
    .result-empty .result-empty-title {{ font-weight: 700; color: #7f1d1d; font-size: 1.05rem; margin-top: 0.6rem; }}
    .result-empty .result-empty-desc {{ color: #991b1b; font-size: 0.85rem; margin-top: 0.4rem; }}
    @media (prefers-reduced-motion: reduce) {{
        .result-empty, .result-empty svg {{ animation: none; }}
    }}

    /* micro-animation ฝั่งพบป้าย: รูปครอปค่อยๆ pop-in + กรอบเขียวกระพริบสั้นๆ แสดงความสำเร็จ */
    @keyframes resultPopIn {{
        0% {{ opacity: 0; transform: scale(0.94); }}
        100% {{ opacity: 1; transform: scale(1); }}
    }}
    @keyframes resultGlow {{
        0% {{ box-shadow: 0 0 0 0 rgba(22, 163, 74, 0.45); }}
        70% {{ box-shadow: 0 0 0 10px rgba(22, 163, 74, 0); }}
        100% {{ box-shadow: 0 0 0 0 rgba(22, 163, 74, 0); }}
    }}
    @keyframes pillPopIn {{
        0% {{ opacity: 0; transform: scale(0.8); }}
        60% {{ opacity: 1; transform: scale(1.05); }}
        100% {{ opacity: 1; transform: scale(1); }}
    }}
    .result-image-pop {{
        width: 100%;
        border-radius: 12px;
        display: block;
        animation: resultPopIn 0.45s ease-out, resultGlow 1.1s ease-out 0.45s;
    }}
    .result-image-fade {{
        width: 100%;
        max-height: 380px;
        object-fit: contain;
        border-radius: 12px;
        display: block;
        margin: 0 auto;
        animation: resultPopIn 0.45s ease-out;
    }}
    .conf-pill-animated {{
        display: inline-block;
        animation: pillPopIn 0.4s ease-out 0.35s both;
    }}

    /* ป้าย "license_plate x.xx" + กรอบรูปครอป สีส้มเดียวกับกรอบตรวจจับในรูปต้นฉบับ */
    .crop-callout {{ display: inline-block; max-width: 100%; margin-top: 1rem; }}
    .conf-info-row {{ margin-top: 0.9rem; }}

    /* UI ส่วน human feedback (ยืนยัน/แก้ไขผลลัพธ์) */
    .feedback-prompt {{
        margin-top: 1rem;
        font-size: 0.82rem;
        font-weight: 600;
        color: {C['text_secondary']};
    }}
    .feedback-done {{
        margin-top: 1rem;
        font-size: 0.82rem;
        color: #16a34a;
        font-weight: 600;
    }}

    /* แถบสถานะผลตรวจจับ — แสดงเหนือตาราง 2 panel หลังมีผลลัพธ์แล้ว (แทนลูกศรกลาง + ปุ่มที่เคยกระจายอยู่) */
    .st-key-detect_status_bar {{
        background: {C['panel_bg2']}; border-radius: 14px;
        padding: 1.3rem 1.25rem; margin-bottom: 1.3rem;
        display: flex; align-items: center;
    }}
    .st-key-detect_status_bar > div {{ width: 100%; }}
    .st-key-detect_status_bar [data-testid="stMarkdownContainer"] {{ margin: 0; }}
    .st-key-detect_status_bar [data-testid="stHorizontalBlock"] > div {{ display: flex; align-items: center; }}
    .detect-status-left {{ display: flex; align-items: center; gap: 0.75rem; }}
    .detect-status-icon {{ flex: 0 0 auto; width: 40px; height: 40px; border-radius: 12px; display: flex; align-items: center; justify-content: center; }}
    .detect-status-icon.success {{ background: #dcfce7; }}
    .detect-status-icon.fail {{ background: #fee2e2; }}
    .detect-status-title {{ font-size: 1rem; font-weight: 700; color: {C['text_primary']}; margin: 0; }}
    .detect-status-sub {{ font-size: 0.82rem; color: {C['text_muted']}; margin-top: 0.1rem; }}

    /* กล่องคำอธิบายเล็กใต้รูปครอป (แทนป้าย confidence เดิมที่ย้ายไป sidebar) */
    .crop-info-box {{
        display: flex; align-items: flex-start; gap: 0.6rem; padding: 0.75rem; border-radius: 10px;
        background: #f8faff; font-size: 0.8rem; color: {C['text_secondary']}; margin-top: 0.9rem;
    }}

    /* การ์ด sidebar — รายละเอียดผลตรวจจับ (confidence + progress bar + รายการ)
       ใช้ !important + selector เจาะจงเพราะ Streamlit ใส่ font-size เริ่มต้นให้ h2 เองมาก่อน ชนะ
       selector แบบ class เดี่ยวธรรมดา (บั๊กเดียวกับที่เจอกับหัวข้อ sidebar คำแนะนำถ่ายรูปก่อนหน้านี้) */
    [data-testid="stMarkdownContainer"] h2.detect-detail-title {{
        margin: 0 0 0.9rem 0 !important; font-size: 1rem !important; font-weight: 700 !important;
        color: {C['text_primary']} !important; line-height: 1.4 !important;
    }}
    .detail-row {{ display: flex; align-items: baseline; justify-content: space-between; }}
    .detail-label {{ font-size: 0.82rem; color: {C['text_muted']}; }}
    .detail-conf-value {{ font-family: 'JetBrains Mono', monospace; font-size: 1.55rem; font-weight: 700; color: {C['text_primary']}; }}
    .conf-bar-track {{ position: relative; height: 8px; border-radius: 999px; background: {C['panel_bg2']}; margin-top: 0.5rem; }}
    .conf-bar-fill {{ height: 8px; border-radius: 999px; background: {C['accent']}; }}
    .conf-bar-threshold {{ position: absolute; top: -4px; width: 2px; height: 16px; background: {C['text_muted']}; }}
    .conf-bar-scale {{ position: relative; height: 16px; font-family: 'JetBrains Mono', monospace; font-size: 0.68rem; color: {C['text_muted']}; margin-top: 0.2rem; }}
    .conf-bar-scale span {{ position: absolute; }}
    .detail-list {{ display: flex; flex-direction: column; border-top: 1px solid {C['border']}; margin-top: 0.9rem; }}
    .detail-list-row {{ display: flex; justify-content: space-between; gap: 1rem; padding: 0.6rem 0; border-bottom: 1px solid {C['border']}; font-size: 0.86rem; }}
    .detail-list-row:last-child {{ border-bottom: none; padding-bottom: 0; }}
    .detail-list-row .k {{ color: {C['text_muted']}; }}
    .detail-list-row .v {{ font-family: 'JetBrains Mono', monospace; font-weight: 700; color: {C['text_primary']}; }}

    /* การ์ด sidebar — ผลลัพธ์นี้ถูกต้องไหม / มีป้ายทะเบียนไหม (เหตุผลเดียวกับ detect-detail-title ด้านบน) */
    [data-testid="stMarkdownContainer"] h2.fb-card-title {{
        margin: 0 !important; font-size: 1rem !important; font-weight: 700 !important;
        color: {C['text_primary']} !important; line-height: 1.4 !important;
    }}
    .fb-card-sub {{ margin: 0.1rem 0 0 0; font-size: 0.78rem; color: {C['text_muted']}; }}
    .fb-card-hint {{ margin: 0.8rem 0 0 0; font-size: 0.74rem; color: {C['text_faint']}; }}

    /* breadcrumb นำทาง + การ์ดวาดกรอบป้ายเอง */
    .draw-breadcrumb {{ display: flex; align-items: center; gap: 0.4rem; font-size: 0.8rem; color: {C['text_muted']}; margin-bottom: 0.9rem; flex-wrap: wrap; }}
    .draw-breadcrumb .current {{ color: {C['text_primary']}; font-weight: 600; }}
    .draw-breadcrumb .sep {{ color: {C['text_faint']}; }}
    .st-key-detect_draw_card {{
        background: {C['panel_bg']}; border: 1px solid {C['border']}; border-radius: 20px; padding: 1.5rem;
    }}
    [data-testid="stMarkdownContainer"] h2.draw-card-title {{
        margin: 0 !important; font-size: 1.15rem !important; font-weight: 700 !important; color: {C['text_primary']} !important;
    }}
    .draw-card-desc {{ margin: 0.3rem 0 1rem 0; font-size: 0.85rem; color: {C['text_muted']}; }}
    .st-key-detect_draw_status_bar {{
        display: flex; align-items: center; justify-content: space-between; gap: 1rem; flex-wrap: wrap;
        padding: 0.9rem 1.2rem; background: {C['panel_bg2']}; border: 1px solid {C['border']}; border-radius: 14px;
        margin-top: 1rem; font-size: 0.8rem; color: {C['text_muted']};
    }}
    .draw-status-left {{ display: flex; align-items: center; gap: 0.5rem; }}
    .draw-status-dot {{ width: 6px; height: 6px; border-radius: 50%; background: {C['accent']}; flex: 0 0 auto; }}
    .draw-status-note {{ display: flex; align-items: center; gap: 0.4rem; }}

    /* การ์ด sidebar ฝั่งวาดกรอบ: ตัวอย่างป้าย / เช็กลิสต์ / หลังส่งกรอบ */
    .st-key-detect_draw_preview_card, .st-key-detect_draw_checklist_card, .st-key-detect_draw_steps_card {{
        background: {C['panel_bg']}; border: 1px solid {C['border']}; border-radius: 20px;
        padding: 1.3rem 1.5rem 1.6rem 1.5rem; margin-bottom: 1rem;
    }}
    [data-testid="stMarkdownContainer"] h2.draw-side-title {{
        margin: 0 0 0.8rem 0 !important; font-size: 0.95rem !important; font-weight: 700 !important; color: {C['text_primary']} !important;
    }}
    .draw-preview-img {{ width: 100%; border-radius: 10px; display: block; background: {C['panel_bg2']}; }}
    .draw-preview-hint {{ font-size: 0.76rem; color: {C['text_muted']}; margin-top: 0.6rem; }}
    .draw-preview-empty {{
        height: 100px; border: 1px dashed {C['border2']}; border-radius: 10px; display: flex; align-items: center;
        justify-content: center; color: {C['text_faint']}; font-size: 0.8rem; text-align: center; padding: 0 1rem;
    }}
    .draw-step-row {{ display: flex; gap: 0.7rem; align-items: flex-start; margin-top: 0.7rem; }}
    .draw-step-row:first-of-type {{ margin-top: 0; }}
    .draw-step-num {{
        flex: 0 0 auto; width: 20px; height: 20px; border-radius: 999px; background: {C['accent']}1a; color: {C['accent']};
        display: flex; align-items: center; justify-content: center; font-size: 0.7rem; font-weight: 800;
    }}
    .draw-step-text {{ font-size: 0.82rem; color: {C['text_secondary']}; padding-top: 0.1rem; }}

    /* การ์ด sidebar ตอนไม่พบป้าย — ชวนวาดกรอบเอง */
    .st-key-detect_noplate_action_card {{
        background: {C['panel_bg']}; border: 1px solid {C['border']}; border-radius: 20px; padding: 1.3rem 1.5rem; margin-bottom: 1rem;
    }}
    .noplate-action-head {{ display: flex; align-items: center; gap: 0.7rem; margin-bottom: 0.6rem; }}
    .noplate-action-icon {{
        flex: 0 0 auto; width: 38px; height: 38px; border-radius: 10px; background: #eef2ff; color: #4f46e5;
        display: flex; align-items: center; justify-content: center;
    }}
    [data-testid="stMarkdownContainer"] h2.noplate-action-title {{
        margin: 0 !important; font-size: 0.95rem !important; font-weight: 700 !important; color: {C['text_primary']} !important;
    }}
    .noplate-action-desc {{ font-size: 0.8rem; color: {C['text_muted']}; margin-bottom: 1rem; }}
    .noplate-secondary-link {{
        display: block; text-align: center; margin-top: 0.6rem; font-size: 0.78rem; color: {C['text_muted']};
    }}
    /* ปุ่มพื้นเข้ม "วาดกรอบป้ายเอง" — ต้องไล่ความ specificity ของสีตัวอักษรให้ชนะ
       กฎ global ที่บังคับสี p ภายใน stMarkdownContainer เป็นสีเข้มเสมอ (ไม่งั้นตัวหนังสือจมกับพื้นดำ)
       (ปุ่ม "ส่งกรอบป้าย" สไตล์แยกแบบ dynamic ตรงจุดที่ render เพราะ key มี cache_key ต่อท้าย) */
    .st-key-detect_noplate_draw_btn button {{
        background: #0f172a !important; border-color: #0f172a !important; color: #ffffff !important;
        font-weight: 600 !important; border-radius: 10px !important;
    }}
    .st-key-detect_noplate_draw_btn button [data-testid="stMarkdownContainer"] p {{
        color: #ffffff !important;
    }}

    /* การ์ดเดียวครอบทั้งหมด — "เลือกรูปรถ" + แถบสถานะ + Original/Cropped ให้อยู่ใต้กรอบเดียวกัน
       แทนที่จะแยกเป็นกล่องย่อยหลายกล่อง (เดิม .st-key-detect_upload_card/.detect_status_bar/.result-panel
       มีกรอบของตัวเองหมด ตอนนี้เอากรอบออกจากกล่องย่อย ให้เหลือกรอบเดียวที่การ์ดแม่) */
    .st-key-detect_main_card {{
        background: {C['panel_bg']};
        border: 1px solid {C['border']};
        border-radius: 20px;
        padding: 1.5rem;
    }}
    .detect-section-divider {{ border-top: 1px solid {C['border']}; margin: 1.3rem 0; }}
    .detect-card-head {{
        display: flex; align-items: center; justify-content: space-between; gap: 1rem; flex-wrap: wrap;
        margin-bottom: 1rem;
    }}
    .detect-card-head h2 {{ margin: 0; font-size: 1.05rem; font-weight: 700; color: {C['text_primary']}; }}
    .detect-step-label {{ font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; color: {C['text_muted']}; }}

    /* Dropzone อัปโหลดรูป — โทนม่วง/indigo ให้เป็นจุดสนใจหลักของหน้า (CTA) */
    .st-key-detect_main_card [data-testid="stFileUploaderDropzone"] {{
        background: #f8faff !important;
        border: 1.5px dashed #a5b4fc !important;
        border-radius: 16px !important;
    }}
    .st-key-detect_main_card [data-testid="stFileUploaderDropzone"]:hover {{
        border-color: #6366f1 !important;
        background: #eef2ff !important;
    }}

    /* หัวการ์ดผลลัพธ์ (Original+Detection / Cropped Result) พร้อมป้าย kicker มุมขวา */
    .result-panel-head {{ display: flex; align-items: center; justify-content: space-between; gap: 0.6rem; margin-bottom: 0.6rem; }}
    .result-panel-head h4 {{ margin: 0; }}
    .panel-kicker {{
        flex: 0 0 auto; padding: 0.1rem 0.6rem; border-radius: 999px; background: {C['panel_bg2']};
        font-family: 'JetBrains Mono', monospace; font-size: 0.65rem; color: {C['text_muted']};
    }}

    /* การ์ดคำแนะนำการถ่ายรูป (ก่อนอัปโหลด) / การ์ดผลลัพธ์รวม (รายละเอียด+สถานะดาวน์โหลด+feedback
       หลังอัปโหลดสำเร็จ) — sidebar แนวตั้งด้านขวา รวมเป็นกรอบเดียวกันแทนที่จะแยก 3 กล่องแบบเดิม */
    .st-key-detect_tips_card, .st-key-detect_sidebar_result_card {{
        background: {C['panel_bg']};
        border: 1px solid {C['border']};
        border-radius: 20px;
        padding: 1.5rem 1.5rem 1.6rem 1.5rem;
        box-sizing: border-box;
    }}
    .st-key-detect_sidebar_result_card {{ padding-top: 1.1rem; }}
    /* ปุ่มดาวน์โหลดในการ์ดผลลัพธ์รวม — สีจางลงจากพื้นดำเข้มเดิม (ให้ดูเบาขึ้น ไม่เด่นจัดเกินไป)
       ลด margin ด้านล่างลงด้วยเพื่อให้ส่วน "ผลลัพธ์นี้ถูกต้องไหม?" ถัดไปขยับขึ้นมากระชับขึ้น */
    .st-key-detect_sidebar_result_card [data-testid="stDownloadButton"] {{ margin: 1.3rem 0 0.4rem 0; }}
    .st-key-detect_sidebar_result_card [data-testid="stDownloadButton"] button {{
        background: #475569 !important; border-color: #475569 !important; font-weight: 600 !important;
    }}
    .st-key-detect_sidebar_result_card [data-testid="stDownloadButton"] button:hover {{
        background: #334155 !important; border-color: #334155 !important;
    }}
    .st-key-detect_sidebar_result_card [data-testid="stDownloadButton"] button [data-testid="stMarkdownContainer"] p {{
        color: #ffffff !important;
    }}
    /* ใช้ !important + selector เจาะจงเพราะ Streamlit ใส่ font-size เริ่มต้นให้ h2 เองมาก่อน
       ชนะ selector แบบ class เดี่ยวธรรมดา */
    [data-testid="stMarkdownContainer"] h2.detect-tips-title {{
        margin: 0 !important; font-size: 1.05rem !important; font-weight: 700 !important;
        color: {C['text_primary']} !important; line-height: 1.4 !important;
    }}
    .detect-tips-sub {{ margin: 0.25rem 0 0 0; font-size: 0.78rem; color: {C['text_faint']}; }}
    .tip-row {{ display: flex; gap: 0.85rem; padding: 0.85rem 0; border-top: 1px solid {C['border']}; }}
    .tip-row:first-of-type {{ margin-top: 0.9rem; }}
    .tip-row-icon {{
        flex: 0 0 auto; width: 36px; height: 36px; border-radius: 10px; background: {C['accent']}1a;
        display: flex; align-items: center; justify-content: center;
    }}
    .tip-row-text {{ min-width: 0; }}
    .tip-row-title {{ font-size: 0.86rem; font-weight: 700; color: {C['text_primary']}; margin-bottom: 0.15rem; }}
    .tip-row-desc {{ font-size: 0.78rem; color: {C['text_muted']}; line-height: 1.5; }}
    .crop-callout-badge {{
        display: inline-block;
        background: #f8bd38;
        color: #0b0f17;
        font-weight: 700;
        font-size: 0.78rem;
        padding: 4px 10px;
        border-radius: 6px 6px 0 0;
        font-family: 'JetBrains Mono', monospace;
        animation: flowCardIn 0.4s ease-out;
    }}
    .crop-callout-img.result-image-pop {{
        border: 6px solid #f8bd38 !important;
        border-radius: 0 8px 8px 8px !important;
    }}
    @media (prefers-reduced-motion: reduce) {{
        .result-image-pop, .result-image-fade, .conf-pill-animated, .crop-callout-badge {{ animation: none; }}
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


def load_csv_metrics(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}
    if "metric" in rows[0]:
        return {r["metric"]: r["value"] for r in rows}
    avg_row = next((r for r in rows if r.get("fold") == "average"), None)
    return avg_row or {}


def load_cv_folds(path: Path) -> list[dict]:
    """คืนรายการผลแต่ละ fold (ไม่รวมแถว average) ใช้โชว์กราฟเปรียบเทียบใน Model Info"""
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [r for r in rows if r.get("fold") != "average"]


def load_dataset_stats(path: Path) -> dict:
    """สรุปสถิติ data/raw/dataset_manifest.csv (จำนวนที่ผ่าน/ไม่ผ่าน + สัดส่วนสัญญาอนุญาต) ใช้เล่าที่มาข้อมูลใน Model Info"""
    if not path.exists():
        return {"total": 0, "passed": 0, "rejected": 0, "licenses": []}
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    passed_rows = [r for r in rows if r.get("status") == "passed"]
    license_counts: dict[str, int] = {}
    for r in passed_rows:
        lic = r.get("license") or "ไม่ระบุ"
        license_counts[lic] = license_counts.get(lic, 0) + 1
    licenses_sorted = sorted(license_counts.items(), key=lambda kv: -kv[1])
    return {
        "total": len(rows),
        "passed": len(passed_rows),
        "rejected": len(rows) - len(passed_rows),
        "licenses": licenses_sorted,
    }


def draw_box_overlay(image_bgr, box_xyxy, confidence):
    overlay = image_bgr.copy()
    x0, y0, x1, y1 = [int(round(v)) for v in box_xyxy]
    color = (56, 189, 248)  # สีฟ้า accent (BGR)
    thickness = max(2, overlay.shape[1] // 300)
    cv2.rectangle(overlay, (x0, y0), (x1, y1), color, thickness)
    label = f"license_plate {confidence:.2f}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(overlay, (x0, max(0, y0 - th - 10)), (x0 + tw + 10, y0), color, -1)
    cv2.putText(overlay, label, (x0 + 5, max(15, y0 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (11, 15, 23), 2)
    return overlay


if not WEIGHTS_PATH.exists():
    st.error(f"ไม่พบไฟล์โมเดลที่ {WEIGHTS_PATH} — ต้องเทรนโมเดลให้เสร็จก่อน (ดู PLAN.md Phase 3)")
    st.stop()

cv_metrics = load_csv_metrics(CV_SUMMARY_PATH)
test_metrics = load_csv_metrics(TEST_SUMMARY_PATH)
cv_folds = load_cv_folds(CV_SUMMARY_PATH)
dataset_stats = load_dataset_stats(DATASET_MANIFEST_PATH)

# ---------- แถบนำทางบนสุด: โลโก้ + nav แบบ segmented control + สถานะโมเดล ----------
with st.container(key="main_header_row"):
    logo_col, nav_col, status_col = st.columns([2.2, 5, 2.2])
    with logo_col:
        car_icon = (
            '<svg width="20" height="20" viewBox="-6 -6 36 36" fill="none" stroke="#fff" '
            'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
            '<rect x="2" y="6" width="20" height="12" rx="2"/><path d="M6 10h12M8 14h8"/></svg>'
        )
        st.markdown(
            f'<div class="main-logo"><div class="main-logo-icon">{car_icon}</div>'
            '<div><div class="main-logo-title">Thai LPR</div>'
            '<div class="main-logo-sub">License Plate Detection</div></div></div>',
            unsafe_allow_html=True,
        )
    with nav_col:
        with st.container(key="main_nav_track"):
            tab_cols = st.columns(len(TABS))
            for col, (tab_id, label) in zip(tab_cols, TABS):
                with col:
                    if st.button(label, key=f"tab_{tab_id}", use_container_width=True):
                        st.session_state.active_tab = tab_id
                        st.rerun()
    with status_col:
        st.markdown(
            '<div class="status-pill"><span class="status-dot"></span>โมเดลพร้อมใช้งาน</div>',
            unsafe_allow_html=True,
        )

active = st.session_state.active_tab

# ======================================================================
# TAB: Detect
# ======================================================================
if active == "detect":
    # ---------- Hero: eyebrow + title + desc + tag pills ----------
    st.markdown(
        '<div class="pipeline-eyebrow">YOLO11N OBJECT DETECTION + OPENCV CROPPING</div>'
        '<h1 class="pipeline-title" style="margin-bottom:0.5rem">ตรวจจับและครอปป้ายทะเบียนรถไทย อัตโนมัติ</h1>'
        '<p class="pipeline-desc" style="margin-top:0">อัปโหลดรูปรถ 1 คัน ระบบจะหาตำแหน่งป้ายทะเบียน '
        'แล้วครอปออกมาเป็นรูปขนาด 400×200 พร้อมใช้งานต่อ</p>'
        '<div class="badge-row">'
        '<span class="badge">YOLO11n</span>'
        '<span class="badge">single-class · license_plate</span>'
        '<span class="badge">250 images dataset</span>'
        '<span class="badge">5-fold cross-validation</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    # ---------- Model metrics row ----------
    m1, m2, m3, m4 = st.columns(4)
    metric_defs = [
        (m1, "mAP@0.5 · เฉลี่ย 5-fold CV", cv_metrics.get("map50"), "5-fold cross-validation", True),
        (m2, "Precision · Test", test_metrics.get("precision"), "held-out 50 รูป", False),
        (m3, "Recall · Test", test_metrics.get("recall"), "held-out 50 รูป", False),
        (m4, "Crop success", test_metrics.get("crop_success_rate"), "IoU ≥ 0.5", False),
    ]
    for col, label, value, sub, dark in metric_defs:
        display_val = f"{float(value)*100:.1f}%" if value not in (None, "") else "—"
        cls = "metric-card metric-card-dark" if dark else "metric-card"
        col.markdown(
            f"""<div class="{cls}">
                <div class="label">{label}</div>
                <div class="value">{display_val}</div>
                <div class="sub">{sub}</div>
            </div>""",
            unsafe_allow_html=True,
        )
    st.write("")

    def _tip_icon(name: str) -> str:
        base = (
            f'<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="{C["accent"]}" '
            'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
        )
        paths = {
            "angle": '<path d="M4 7h3l2-2h6l2 2h3a1 1 0 0 1 1 1v10a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1V8a1 1 0 0 1 1-1z"/><circle cx="12" cy="13" r="3.5"/>',
            "light": '<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.2 4.2l1.4 1.4M18.4 18.4l1.4 1.4M2 12h2M20 12h2M4.2 19.8l1.4-1.4M18.4 5.6l1.4-1.4"/>',
            "distance": '<path d="M4 12h16"/><path d="M4 12l4-4M4 12l4 4"/><path d="M20 12l-4-4M20 12l-4 4"/>',
            "plate": '<rect x="3" y="6" width="18" height="12" rx="2"/><path d="M7 11h10M7 15h6"/>',
            "count": '<path d="M5 11l1.5-4.5A2 2 0 0 1 8.4 5h7.2a2 2 0 0 1 1.9 1.5L19 11"/><rect x="3" y="11" width="18" height="6" rx="2"/><circle cx="7.5" cy="17" r="1.5"/><circle cx="16.5" cy="17" r="1.5"/>',
        }
        return base + paths[name] + "</svg>"

    tips = [
        ("angle", "มุมกล้อง", "ถ่ายหน้าตรงหรือมุมหลังตรง ให้ป้ายทะเบียนอยู่ในแนวตรงกับกล้องมากที่สุด หลีกเลี่ยงมุมเฉียง 3/4 หรือมุมข้าง"),
        ("light", "แสงสว่าง", "ถ่ายในที่มีแสงเพียงพอ เช่น ช่วงกลางวัน หลีกเลี่ยงแสงย้อนหรือเงาที่บดบังตัวป้าย"),
        ("distance", "ระยะห่าง", "จัดให้รถเต็มคันพอดีเฟรม ระยะห่างจากรถประมาณ 1.5–3 เมตร"),
        ("plate", "ความชัดเจนของป้าย", "ป้ายทะเบียนต้องเห็นครบทั้งแผ่น ไม่มีสิ่งบดบัง เช่น คราบสกปรก สติกเกอร์ หรือเงา"),
        ("count", "จำนวนรถในเฟรม", "ควรมีรถหลักเพียง 1 คันอยู่กลางภาพ ถ้ามีหลายคันระบบจะเลือกป้ายของคันที่ใหญ่ที่สุดในภาพให้อัตโนมัติ"),
    ]
    retry_tips = [
        ("plate", "ใช้รูปที่ติดป้ายทะเบียนไทย", "ระบบรองรับเฉพาะป้ายทะเบียนไทยมาตรฐาน ไม่รวมป้ายต่างประเทศหรือป้ายโชว์"),
        ("angle", "ถ่ายหน้าตรงหรือหลังตรง", "เลี่ยงมุมเฉียง 3/4 หรือด้านข้าง"),
        ("distance", "เข้าใกล้ขึ้น", "ห่างจากรถประมาณ 1.5–3 เมตร ให้เต็มคันพอดีเฟรม"),
        ("light", "แสงพอ ไม่มีเงาบัง", "เลี่ยงแสงย้อนหรือเงาที่ตกบนป้าย"),
    ]

    def _render_result_feedback(cache_key, fb_key, mode_key, original_bgr, last_upload, info):
        """การ์ด sidebar ฝั่งผลลัพธ์สำเร็จ: ยืนยันถูก/ผิด (เคสเจอป้าย)"""
        if st.session_state.get(fb_key):
            st.markdown(
                '<h2 class="fb-card-title">ขอบคุณสำหรับ feedback 🙏</h2>'
                '<p class="fb-card-sub">บันทึกแล้ว ใช้พัฒนาโมเดลรอบถัดไป</p>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<h2 class="fb-card-title">ผลลัพธ์นี้ถูกต้องไหม?</h2>'
                '<p class="fb-card-sub">คำตอบของคุณช่วยให้โมเดลแม่นยำขึ้นในรอบถัดไป</p>',
                unsafe_allow_html=True,
            )
            ck = css_key(cache_key)
            st.markdown(
                f"""
                <style>
                .st-key-fb_correct_{ck} button {{
                    border: 1px solid #bbf7d0 !important; background: #dcfce7 !important; color: #15803d !important;
                    font-weight: 700 !important; border-radius: 10px !important;
                }}
                .st-key-fb_correct_{ck} button:hover {{
                    background: #bbf7d0 !important; border-color: #86efac !important;
                }}
                .st-key-fb_correct_{ck} button [data-testid="stMarkdownContainer"] p {{
                    color: #15803d !important;
                }}
                .st-key-fb_wrong_{ck} button {{
                    border: 1px solid #fecaca !important; background: #fee2e2 !important; color: #b91c1c !important;
                    font-weight: 700 !important; border-radius: 10px !important;
                }}
                .st-key-fb_wrong_{ck} button:hover {{
                    background: #fecaca !important; border-color: #fca5a5 !important;
                }}
                .st-key-fb_wrong_{ck} button [data-testid="stMarkdownContainer"] p {{
                    color: #b91c1c !important;
                }}
                </style>
                """,
                unsafe_allow_html=True,
            )
            fbc1, fbc2 = st.columns(2)
            with fbc1:
                if st.button("ถูกต้อง", key=f"fb_correct_{cache_key}", use_container_width=True):
                    img_h_fb, img_w_fb = original_bgr.shape[:2]
                    _, img_encoded = cv2.imencode(".jpg", original_bgr)
                    save_feedback(
                        image_bytes=img_encoded.tobytes(),
                        verdict="correct",
                        source_filename=last_upload["name"],
                        img_w=img_w_fb,
                        img_h=img_h_fb,
                        model_box_xyxy=info["box_xyxy"],
                        model_confidence=info["confidence"],
                    )
                    st.session_state[fb_key] = True
                    st.rerun()
            with fbc2:
                if st.button("ตำแหน่งผิด", key=f"fb_wrong_{cache_key}", use_container_width=True):
                    st.session_state[mode_key] = "correct_box"
                    st.rerun()
            st.markdown('<p class="fb-card-hint">ถ้าเลือก "ตำแหน่งผิด" คุณจะวาดกรอบป้ายใหม่ได้เอง</p>', unsafe_allow_html=True)

    def _render_noplate_action(cache_key, fb_key, mode_key, original_bgr, last_upload):
        """การ์ด sidebar ตอนไม่พบป้าย: ชวนวาดกรอบเอง + ทางเลือกยืนยันว่าไม่มีป้ายจริง + tips ลองใหม่"""
        if st.session_state.get(fb_key):
            with st.container(key="detect_noplate_action_card"):
                st.markdown(
                    '<h2 class="noplate-action-title">ขอบคุณสำหรับ feedback 🙏</h2>'
                    '<p class="fb-card-sub">บันทึกแล้ว ใช้พัฒนาโมเดลรอบถัดไป</p>',
                    unsafe_allow_html=True,
                )
            return
        pin_icon = (
            '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#4f46e5" '
            'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
            '<path d="M12 21s-7-5.1-7-11a7 7 0 0 1 14 0c0 5.9-7 11-7 11z"/><circle cx="12" cy="10" r="2.5"/></svg>'
        )
        with st.container(key="detect_noplate_action_card"):
            st.markdown(
                '<div class="noplate-action-head">'
                f'<div class="noplate-action-icon">{pin_icon}</div>'
                '<h2 class="noplate-action-title">ช่วยระบุตำแหน่งป้าย</h2>'
                '</div>'
                '<div class="noplate-action-desc">มีป้ายทะเบียนไทยแต่รูปนี้ไม่เจอใช่ไหม? วาดกรอบตำแหน่งเองได้</div>',
                unsafe_allow_html=True,
            )
            st.markdown('<div class="st-key-detect_noplate_draw_btn">', unsafe_allow_html=True)
            if st.button("✏ วาดกรอบป้ายเอง", key="detect_noplate_draw_btn", use_container_width=True):
                st.session_state[mode_key] = "draw_box"
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
            if st.button("หรือไม่มีป้ายทะเบียนในรูปนี้จริง", key=f"fb_noplate_{cache_key}", use_container_width=True):
                img_h_fb, img_w_fb = original_bgr.shape[:2]
                _, img_encoded = cv2.imencode(".jpg", original_bgr)
                save_feedback(
                    image_bytes=img_encoded.tobytes(),
                    verdict="no_plate",
                    source_filename=last_upload["name"],
                    img_w=img_w_fb,
                    img_h=img_h_fb,
                )
                st.session_state[fb_key] = True
                st.rerun()

        with st.container(key="detect_tips_card"):
            tip_rows_html = "".join(
                '<div class="tip-row">'
                f'<div class="tip-row-icon">{_tip_icon(icon_name)}</div>'
                f'<div class="tip-row-text"><div class="tip-row-title">{title}</div>'
                f'<div class="tip-row-desc">{desc}</div></div>'
                '</div>'
                for icon_name, title, desc in retry_tips
            )
            st.markdown(
                '<h2 class="detect-tips-title">ลองถ่ายใหม่แบบนี้</h2>'
                '<p class="detect-tips-sub">สาเหตุที่พบบ่อยเมื่อระบบหาป้ายไม่เจอ</p>'
                f'{tip_rows_html}',
                unsafe_allow_html=True,
            )

    def _render_draw_sidebar(preview_crop_b64, has_box):
        """การ์ด sidebar ฝั่งวาดกรอบเอง: ตัวอย่างป้ายที่จะครอป (อัปเดตสด) / เช็กลิสต์ / หลังส่งกรอบ"""
        with st.container(key="detect_draw_preview_card"):
            if preview_crop_b64:
                st.markdown(
                    '<h2 class="draw-side-title">ตัวอย่างป้ายที่จะครอป</h2>'
                    f'<img class="draw-preview-img" src="data:image/png;base64,{preview_crop_b64}" alt="preview crop"/>'
                    '<div class="draw-preview-hint">อัปเดตทันทีเมื่อปรับกรอบ ถ้าตัวอักษรถูกตัดขอบ ให้ขยายกรอบออกเล็กน้อย</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<h2 class="draw-side-title">ตัวอย่างป้ายที่จะครอป</h2>'
                    '<div class="draw-preview-empty">วาดกรอบบนรูปด้านซ้ายก่อน ตัวอย่างจะแสดงที่นี่</div>',
                    unsafe_allow_html=True,
                )
        with st.container(key="detect_draw_checklist_card"):
            st.markdown(
                '<h2 class="draw-side-title">วาดยังไงให้ดี</h2>'
                '<div class="criteria-list">'
                '<div class="criteria-item"><div class="criteria-check">✓</div>กรอบชิดขอบป้ายทั้ง 4 ด้าน ไม่กว้างหรือแคบเกิน</div>'
                '<div class="criteria-item"><div class="criteria-check">✓</div>ครอบทั้งแผ่น รวมชื่อจังหวัด</div>'
                '<div class="criteria-item"><div class="criteria-check">✓</div>วาดเฉพาะป้ายทะเบียนไทย 1 ป้ายต่อรูป</div>'
                '</div>',
                unsafe_allow_html=True,
            )
        with st.container(key="detect_draw_steps_card"):
            st.markdown(
                '<h2 class="draw-side-title">หลังส่งกรอบ</h2>'
                '<div class="draw-step-row"><div class="draw-step-num">1</div><div class="draw-step-text">เก็บเป็นข้อมูลใหม่</div></div>'
                '<div class="draw-step-row"><div class="draw-step-num">2</div><div class="draw-step-text">ทีมงานตรวจสอบความถูกต้อง</div></div>'
                '<div class="draw-step-row"><div class="draw-step-num">3</div><div class="draw-step-text">ใช้เทรนโมเดลรอบถัดไป</div></div>',
                unsafe_allow_html=True,
            )

    left_col, right_col = st.columns([2, 1], gap="medium")

    # state เก็บไว้ตัดสินใจว่า right_col จะโชว์อะไร — ต้องรู้ผลจาก left_col ก่อน (ซึ่งรันเสร็จก่อน
    # right_col ด้านล่างนี้) Streamlit วาง element ตาม column ที่ประกาศไว้ ไม่ใช่ตามลำดับโค้ด
    # จึงสลับลำดับการรันได้โดยที่ผลลัพธ์ยังอยู่ถูกคอลัมน์
    last_upload = None
    cropped = message = info = original_bgr = None
    cache_key = fb_key = mode_key = conf_setting = None
    draw_preview_b64 = None
    draw_has_box = False

    with left_col:
        with st.container(key="detect_main_card"):
            upload_icon = (
                '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#4f46e5" '
                'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
                '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><path d="m17 8-5-5-5 5"/><path d="M12 3v12"/></svg>'
            )
            st.markdown(
                '<div class="detect-card-head">'
                '<div style="display:flex; align-items:center; gap:0.7rem">'
                f'<div class="noplate-action-icon">{upload_icon}</div>'
                '<div><h2>เลือกรูปรถเพื่อเริ่มตรวจจับ</h2>'
                f'<div style="font-size:0.78rem; color:{C["text_muted"]}; margin-top:0.1rem">'
                'JPG, JPEG, PNG · สูงสุด 200MB · รถ 1 คันต่อรูป</div></div>'
                '</div></div>',
                unsafe_allow_html=True,
            )
            uploaded_file = st.file_uploader("เลือกรูปรถ", type=["jpg", "jpeg", "png"],
                                              key="plate_uploader", label_visibility="collapsed")
            st.markdown(
                '<div style="display:flex; gap:0.5rem; align-items:flex-start; margin:0.7rem 0 0.2rem 0; '
                f'font-size:0.8rem; color:{C["text_muted"]}">🔒 รูปที่อัปโหลดอาจถูกเก็บไว้เพื่อพัฒนาความแม่นยำของโมเดล '
                'หากคุณช่วยยืนยัน/แก้ไขผลลัพธ์ด้านล่าง</div>',
                unsafe_allow_html=True,
            )

            # เก็บ snapshot ของไฟล์ที่อัปโหลดไว้ใน session_state ทันที แล้วใช้ snapshot นี้ในการแสดงผลต่อ
            # (ไม่ใช้ตัวแปร uploaded_file ที่ได้จาก widget ตรงๆ) เพื่อให้ผลลัพธ์ยังอยู่ครบแม้เกิด rerun จาก
            # วิดเจ็ตอื่นที่ไม่เกี่ยวกับการอัปโหลดรูป เช่น ปุ่มสลับธีม light/dark — แต่ถ้าผู้ใช้กด X เอารูปออก
            # จาก file_uploader จริงๆ (uploaded_file กลายเป็น None) ต้องล้าง snapshot ทิ้งด้วย ไม่งั้นผลลัพธ์
            # เก่าจะค้างแสดงอยู่ต่อแม้ไม่มีไฟล์ในตัวอัปโหลดแล้ว
            if uploaded_file is not None:
                st.session_state["last_upload"] = {
                    "bytes": uploaded_file.getvalue(),
                    "name": uploaded_file.name,
                    "id": uploaded_file.file_id,
                }
            else:
                st.session_state.pop("last_upload", None)

            last_upload = st.session_state.get("last_upload")
            in_draw_mode = False

            if last_upload is not None:
                conf_setting = st.session_state.get("conf_threshold", DEFAULT_CONF)
                iou_setting = st.session_state.get("iou_threshold", DEFAULT_IOU)
                # แคชผลลัพธ์ต่อ (ไฟล์, conf, iou) ไว้ใน session_state — กันไม่ให้รัน YOLO ซ้ำทุกครั้งที่
                # เกิด rerun จากเหตุอื่นที่ไม่เกี่ยวกับรูป (เช่น สลับธีม light/dark) เพื่อให้ output ของรูปเดิม
                # ยังคงแสดงผลเหมือนเดิมทุกประการ ไม่ใช่คำนวณใหม่แล้วอาจได้ผลต่างกันเล็กน้อย
                cache_key = f"detect_result_{last_upload['id']}_{conf_setting}_{iou_setting}"

                if cache_key not in st.session_state:
                    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                        tmp.write(last_upload["bytes"])
                        tmp_path = Path(tmp.name)

                    with st.spinner("กำลังตรวจจับป้ายทะเบียน..."):
                        cropped, message, info = predict_and_crop(
                            tmp_path,
                            WEIGHTS_PATH,
                            conf=conf_setting,
                            iou=iou_setting,
                            device="auto",
                        )

                    original_bgr = cv2.imread(str(tmp_path))
                    tmp_path.unlink(missing_ok=True)
                    st.session_state[cache_key] = {
                        "cropped": cropped,
                        "message": message,
                        "info": info,
                        "original_bgr": original_bgr,
                    }

                cached_result = st.session_state[cache_key]
                cropped = cached_result["cropped"]
                message = cached_result["message"]
                info = cached_result["info"]
                original_bgr = cached_result["original_bgr"]

                fb_key = f"feedback_done_{cache_key}"
                mode_key = f"feedback_mode_{cache_key}"
                feedback_mode = st.session_state.get(mode_key)
                in_draw_mode = feedback_mode in ("correct_box", "draw_box")

                if not in_draw_mode:
                    st.markdown('<div class="detect-section-divider"></div>', unsafe_allow_html=True)
                    # ---------- แถบสถานะผลตรวจจับ ----------
                    # เคส "เจอป้าย" ย้ายข้อความ + ปุ่มดาวน์โหลดไปอยู่ sidebar ใต้การ์ด "รายละเอียดผลตรวจจับ"
                    # แทน (ดู right_col ด้านล่าง) เหลือไว้ตรงนี้เฉพาะเคส "ไม่พบป้ายทะเบียน"
                    if cropped is None:
                        with st.container(key="detect_status_bar"):
                            fail_icon = (
                                '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#dc2626" '
                                'stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">'
                                '<circle cx="12" cy="12" r="10"/><path d="M15 9l-6 6M9 9l6 6"/></svg>'
                            )
                            st.markdown(
                                '<div class="detect-status-left">'
                                f'<div class="detect-status-icon fail">{fail_icon}</div>'
                                '<p class="detect-status-title" style="margin:0">ไม่พบป้ายทะเบียน</p>'
                                '</div>',
                                unsafe_allow_html=True,
                            )

                    # ---------- ตาราง 2 panel: รูปต้นฉบับ+ผลตรวจจับ / ป้ายที่ครอปแล้ว ----------
                    col1, col2 = st.columns(2, gap="medium")

                    with col1:
                        if info["box_xyxy"] is not None:
                            overlay = draw_box_overlay(original_bgr, info["box_xyxy"], info["confidence"])
                            orig_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
                        else:
                            orig_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
                        orig_buf = io.BytesIO()
                        Image.fromarray(orig_rgb).save(orig_buf, format="PNG")
                        orig_b64 = base64.b64encode(orig_buf.getvalue()).decode("utf-8")
                        # รวมเป็น st.markdown ครั้งเดียว — ถ้าเปิด/ปิด <div> คนละ call กัน เบราว์เซอร์จะ
                        # auto-close แท็กที่เปิดค้างไว้ทันทีตอนจบแต่ละ call ทำให้เนื้อหาหลุดออกนอกกล่อง
                        st.markdown(
                            '<div class="result-panel result-panel-center"><div class="result-panel-head"><h4>Original + Detection</h4><span class="panel-kicker">DETECTION</span></div>'
                            f'<img class="result-image-fade" src="data:image/png;base64,{orig_b64}" alt="original with detection"/>'
                            '</div>',
                            unsafe_allow_html=True,
                        )

                    with col2:
                        if cropped is None:
                            x_icon = (
                                '<svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#dc2626" '
                                'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
                                '<circle cx="12" cy="12" r="9"/><path d="M9 9l6 6M15 9l-6 6"/></svg>'
                            )
                            st.markdown(
                                '<div class="result-panel result-panel-center"><div class="result-panel-head"><h4>Cropped Result</h4><span class="panel-kicker">400×200</span></div>'
                                f'<div class="result-empty">{x_icon}'
                                f'<div class="result-empty-title">{message}</div>'
                                '<div class="result-empty-desc">ลองถ่ายรูปใหม่ ให้เห็นป้ายทะเบียนชัดเจน ไม่มีสิ่งบัง</div>'
                                '</div>'
                                '</div>',
                                unsafe_allow_html=True,
                            )
                        else:
                            cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
                            cropped_pil = Image.fromarray(cropped_rgb)
                            png_buf = io.BytesIO()
                            cropped_pil.save(png_buf, format="PNG")
                            img_b64 = base64.b64encode(png_buf.getvalue()).decode("utf-8")
                            info_icon = (
                                '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#4f46e5" '
                                'stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="flex:none;margin-top:2px">'
                                '<circle cx="12" cy="12" r="10"/><path d="M12 16v-4"/><path d="M12 8h.01"/></svg>'
                            )
                            st.markdown(
                                '<div class="result-panel result-panel-center"><div class="result-panel-head"><h4>Cropped Result</h4><span class="panel-kicker">400×200</span></div>'
                                f'<img class="result-image-pop" style="width:100%;border-radius:12px;display:block" '
                                f'src="data:image/png;base64,{img_b64}" alt="cropped plate"/>'
                                f'<div class="crop-info-box">{info_icon}ขยายขอบรอบป้าย 3px แล้วปรับขนาดเป็น 400×200</div>'
                                '</div>',
                                unsafe_allow_html=True,
                            )
            else:
                st.markdown('<div class="detect-section-divider"></div>', unsafe_allow_html=True)
                # ยังไม่มีรูปอัปโหลด — แสดง layout เปล่าไว้ก่อน (กรอบเส้นประ + ไอคอน) กันหน้าดูโล่งเกินไป
                placeholder_icon = (
                    '<svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
                    'stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">'
                    '<rect x="3" y="5" width="18" height="14" rx="2"/>'
                    '<circle cx="8.5" cy="10" r="1.5"/><path d="M21 15l-5-5-9 9"/></svg>'
                )
                ecol1, ecol2 = st.columns(2, gap="medium")
                with ecol1:
                    st.markdown(
                        '<div class="result-panel result-panel-center"><div class="result-panel-head"><h4>Original + Detection</h4><span class="panel-kicker">DETECTION</span></div>'
                        f'<div class="result-placeholder">{placeholder_icon}'
                        '<div class="result-placeholder-text">รอรูปภาพที่อัปโหลด</div>'
                        '</div></div>',
                        unsafe_allow_html=True,
                    )
                with ecol2:
                    st.markdown(
                        '<div class="result-panel result-panel-center"><div class="result-panel-head"><h4>Cropped Result</h4><span class="panel-kicker">400×200</span></div>'
                        f'<div class="result-placeholder">{placeholder_icon}'
                        '<div class="result-placeholder-text">ผลลัพธ์จะแสดงที่นี่</div>'
                        '</div></div>',
                        unsafe_allow_html=True,
                    )

        # ---------- หน้าวาดกรอบป้ายเอง: แยกเป็นการ์ดของตัวเอง นอกกรอบการ์ดหลัก ----------
        if last_upload is not None and in_draw_mode:
            prev_label = "ตำแหน่งผิด" if feedback_mode == "correct_box" else "ไม่พบป้ายทะเบียน"
            st.markdown(
                '<div class="draw-breadcrumb" style="margin-top:1.2rem">Detect<span class="sep">›</span>'
                f'{prev_label}<span class="sep">›</span>'
                '<span class="current">วาดกรอบเอง</span></div>',
                unsafe_allow_html=True,
            )
            with st.container(key="detect_draw_card"):
                st.markdown(
                    '<h2 class="draw-card-title">วาดกรอบป้ายทะเบียน</h2>'
                    '<p class="draw-card-desc">ช่วยบอกระบบว่าป้ายอยู่ตรงไหน กรอบของคุณจะช่วยให้โมเดลรอบถัดไปหาป้ายเจอแม่นขึ้น</p>',
                    unsafe_allow_html=True,
                )
                img_h_canvas, img_w_canvas = original_bgr.shape[:2]
                canvas_w = min(600, img_w_canvas)
                canvas_h = int(canvas_w * img_h_canvas / img_w_canvas)
                bg_image = Image.fromarray(cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)).resize((canvas_w, canvas_h))
                canvas_result = st_canvas(
                    fill_color="rgba(79, 70, 229, 0.2)",
                    stroke_width=3,
                    stroke_color="#4f46e5",
                    background_image=bg_image,
                    update_streamlit=True,
                    height=canvas_h,
                    width=canvas_w,
                    drawing_mode="rect",
                    key=f"canvas_{cache_key}",
                )
                canvas_objects = (canvas_result.json_data or {}).get("objects", []) if canvas_result else []
                draw_has_box = bool(canvas_objects)
                if draw_has_box:
                    try:
                        rect = canvas_objects[-1]
                        yolo_box_preview = canvas_rect_to_yolo(rect, canvas_w, canvas_h, img_w_canvas, img_h_canvas)
                        cx, cy, bw, bh = yolo_box_preview
                        x0 = (cx - bw / 2) * img_w_canvas
                        y0 = (cy - bh / 2) * img_h_canvas
                        x1 = (cx + bw / 2) * img_w_canvas
                        y1 = (cy + bh / 2) * img_h_canvas
                        preview_crop = crop_plate(original_bgr, (x0, y0, x1, y1))
                        preview_rgb = cv2.cvtColor(preview_crop, cv2.COLOR_BGR2RGB)
                        preview_buf = io.BytesIO()
                        Image.fromarray(preview_rgb).save(preview_buf, format="PNG")
                        draw_preview_b64 = base64.b64encode(preview_buf.getvalue()).decode("utf-8")
                    except Exception:
                        draw_preview_b64 = None

                # ปุ่ม "ส่งกรอบป้าย" (พื้นเข้ม) ต้องสูงเท่า "ยกเลิก" (ปุ่มเริ่มต้นของ Streamlit) พอดี —
                # สไตล์ผ่าน key ของปุ่มเองตรงๆ (ไม่ครอบด้วย div ปลอมเหมือนก่อนหน้า เพราะ st.markdown
                # กับ st.button สร้างเป็นคนละ element แยกกัน ครอบกันจริงไม่ได้) แล้วบังคับความสูงเท่ากันทั้งคู่
                ck = css_key(cache_key)
                st.markdown(
                    f"""
                    <style>
                    .st-key-fb_save_box_{ck} button {{
                        background: #0f172a !important; border-color: #0f172a !important;
                        font-weight: 600 !important; border-radius: 10px !important; height: 44px !important;
                    }}
                    .st-key-fb_save_box_{ck} button [data-testid="stMarkdownContainer"] p {{
                        color: #ffffff !important;
                    }}
                    .st-key-fb_cancel_{ck} button {{ height: 44px !important; }}
                    </style>
                    """,
                    unsafe_allow_html=True,
                )
                save_col, cancel_col = st.columns(2)
                with save_col:
                    submit_clicked = st.button("➤ ส่งกรอบป้าย", key=f"fb_save_box_{cache_key}", use_container_width=True)
                    if submit_clicked:
                        if not canvas_objects:
                            st.warning("กรุณาวาดกรอบรอบป้ายทะเบียนก่อนบันทึก")
                        else:
                            rect = canvas_objects[-1]
                            yolo_box = canvas_rect_to_yolo(rect, canvas_w, canvas_h, img_w_canvas, img_h_canvas)
                            _, img_encoded = cv2.imencode(".jpg", original_bgr)
                            verdict = "corrected" if feedback_mode == "correct_box" else "found_missed"
                            save_feedback(
                                image_bytes=img_encoded.tobytes(),
                                verdict=verdict,
                                source_filename=last_upload["name"],
                                img_w=img_w_canvas,
                                img_h=img_h_canvas,
                                model_box_xyxy=info["box_xyxy"],
                                model_confidence=info["confidence"],
                                corrected_yolo_box=yolo_box,
                            )
                            st.session_state[fb_key] = True
                            st.session_state.pop(mode_key, None)
                            st.rerun()
                with cancel_col:
                    if st.button("ยกเลิก", key=f"fb_cancel_{cache_key}", use_container_width=True):
                        st.session_state.pop(mode_key, None)
                        st.rerun()

                lock_icon = (
                    '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
                    'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
                    '<rect x="3" y="11" width="18" height="11" rx="2"/><path d="M7 11V7a5 5 0 0 1 10 0v4"/></svg>'
                )
                box_count = 1 if draw_has_box else 0
                st.markdown(
                    '<div class="st-key-detect_draw_status_bar">'
                    f'<div class="draw-status-left"><span class="draw-status-dot"></span>'
                    f'วาดแล้ว {box_count} กรอบ · 1 ป้ายต่อรูป</div>'
                    f'<div class="draw-status-note">{lock_icon}กรอบที่ส่งจะถูกเก็บแยก รอทีมงานตรวจก่อนนำไปเทรน</div>'
                    '</div>',
                    unsafe_allow_html=True,
                )

    # ---------- ขวา: sidebar ----------
    with right_col:
        if last_upload is None:
            with st.container(key="detect_tips_card"):
                tip_rows_html = "".join(
                    '<div class="tip-row">'
                    f'<div class="tip-row-icon">{_tip_icon(icon_name)}</div>'
                    f'<div class="tip-row-text"><div class="tip-row-title">{title}</div>'
                    f'<div class="tip-row-desc">{desc}</div></div>'
                    '</div>'
                    for icon_name, title, desc in tips
                )
                st.markdown(
                    '<h2 class="detect-tips-title">ถ่ายรูปยังไงให้ได้ผลดีที่สุด</h2>'
                    '<p class="detect-tips-sub">5 ข้อที่ช่วยให้โมเดลหาป้ายเจอแม่นขึ้น</p>'
                    f'{tip_rows_html}',
                    unsafe_allow_html=True,
                )
        else:
            feedback_mode = st.session_state.get(mode_key)
            if feedback_mode in ("correct_box", "draw_box"):
                _render_draw_sidebar(draw_preview_b64, draw_has_box)
            elif cropped is not None:
                # การ์ดผลลัพธ์รวม sidebar: รายละเอียดผลตรวจจับ + สถานะ/ดาวน์โหลด + feedback
                # อยู่ในกรอบเดียวกัน คั่นแต่ละส่วนด้วยเส้นแบ่งแทนการแยกเป็นกล่องขาวหลายกล่อง
                with st.container(key="detect_sidebar_result_card"):
                    conf = info["confidence"]
                    thr = conf_setting
                    conf_pct = max(0.0, min(100.0, conf * 100))
                    thr_pct = max(0.0, min(100.0, thr * 100))
                    st.markdown(
                        '<h2 class="detect-detail-title">รายละเอียดผลตรวจจับ</h2>'
                        '<div class="detail-row"><div class="detail-label">Confidence</div>'
                        f'<div class="detail-conf-value">{conf:.2f}</div></div>'
                        '<div class="conf-bar-track">'
                        f'<div class="conf-bar-fill" style="width:{conf_pct:.1f}%"></div>'
                        f'<div class="conf-bar-threshold" style="left:{thr_pct:.1f}%"></div>'
                        '</div>'
                        '<div class="conf-bar-scale">'
                        '<span style="left:0">0</span>'
                        f'<span style="left:{thr_pct:.1f}%;transform:translateX(-50%)">threshold {thr:.2f}</span>'
                        '<span style="right:0">1.0</span>'
                        '</div>'
                        '<div class="detail-list">'
                        '<div class="detail-list-row"><div class="k">Class</div><div class="v">license_plate</div></div>'
                        f'<div class="detail-list-row"><div class="k">Detections</div><div class="v">{info["num_detections"]}</div></div>'
                        f'<div class="detail-list-row"><div class="k">เวลาประมวลผล</div><div class="v">{info["elapsed_ms"]:.0f} ms</div></div>'
                        '</div>',
                        unsafe_allow_html=True,
                    )

                    st.markdown('<div class="detect-section-divider"></div>', unsafe_allow_html=True)

                    ok_icon = (
                        '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#15803d" '
                        'stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">'
                        '<path d="M20 6 9 17l-5-5"/></svg>'
                    )
                    st.markdown(
                        '<div class="detect-status-left">'
                        f'<div class="detect-status-icon success">{ok_icon}</div>'
                        '<div><p class="detect-status-title">ตรวจพบป้ายทะเบียน 1 ป้าย</p>'
                        '<div class="detect-status-sub">ครอปป้ายเรียบร้อย พร้อมดาวน์โหลด</div></div></div>',
                        unsafe_allow_html=True,
                    )
                    cropped_rgb_dl = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
                    cropped_pil_dl = Image.fromarray(cropped_rgb_dl)
                    dl_buf = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
                    cropped_pil_dl.save(dl_buf.name, format="JPEG")
                    with open(dl_buf.name, "rb") as f:
                        st.download_button(
                            label="⬇ ดาวน์โหลด",
                            data=f.read(),
                            file_name=f"{Path(last_upload['name']).stem}_plate.jpg",
                            mime="image/jpeg",
                            use_container_width=True,
                            key=f"dl_{cache_key}",
                        )
                    Path(dl_buf.name).unlink(missing_ok=True)

                    st.markdown(
                        '<div class="detect-section-divider" style="margin:0.7rem 0"></div>',
                        unsafe_allow_html=True,
                    )

                    _render_result_feedback(cache_key, fb_key, mode_key, original_bgr, last_upload, info)
            else:
                _render_noplate_action(cache_key, fb_key, mode_key, original_bgr, last_upload)
# ======================================================================
# TAB: Settings
# ======================================================================
elif active == "settings":
    # ค่า "ค่าเริ่มต้น" ของ confidence ในหน้านี้อิงตาม settings-page.html ทุกประการ คือ 0.50 (ค่าที่ใช้วัดผลโมเดล
    # จริงบนชุด test) ซึ่งต่างจาก DEFAULT_CONF=0.6 ที่เป็นค่าตั้งต้นจริงตอนเปิดแอป (ยังไม่เคยมาหน้า Settings) —
    # ทำให้ตอนเปิดหน้านี้ครั้งแรก badge จะขึ้น "ปรับจากค่าเริ่มต้น" ทันที ตรงกับสถานะในไฟล์ตัวอย่างเป๊ะ
    CONF_DEFAULT = 0.5

    if st.session_state.pop("settings_reset_pending", False):
        st.session_state["conf_threshold"] = CONF_DEFAULT
        st.session_state["iou_threshold"] = DEFAULT_IOU
    st.session_state.setdefault("conf_threshold", DEFAULT_CONF)
    st.session_state.setdefault("iou_threshold", DEFAULT_IOU)
    conf_now = st.session_state["conf_threshold"]
    iou_now = st.session_state["iou_threshold"]
    conf_changed = abs(conf_now - CONF_DEFAULT) > 1e-6
    iou_changed = abs(iou_now - DEFAULT_IOU) > 1e-6

    # ---------- Hero ----------
    st.markdown(
        '<div class="pipeline-eyebrow">SETTINGS · INFERENCE</div>'
        '<h1 class="pipeline-title" style="margin-bottom:0.5rem">ตั้งค่าการตรวจจับ</h1>'
        '<p class="pipeline-desc" style="margin-top:0">ปรับว่าโมเดลต้องมั่นใจแค่ไหนถึงจะนับว่าเจอป้าย '
        'และจัดการกรอบที่ซ้อนกันอย่างไร ค่าที่ปรับมีผลทันทีกับหน้า Detect</p>',
        unsafe_allow_html=True,
    )

    main_col, side_col = st.columns([2, 1], gap="medium")

    with main_col:
        # ---------- Section: Confidence threshold ----------
        with st.container(key="settings_conf_section"):
            badge_cls = "settings-badge changed" if conf_changed else "settings-badge"
            badge_txt = "ปรับจากค่าเริ่มต้น" if conf_changed else "ค่าเริ่มต้น"
            st.markdown(
                '<div class="settings-section-head">'
                '<div class="settings-section-head-left">'
                '<div class="settings-icon">'
                '<svg width="20" height="20" viewBox="-8 -8 40 40" fill="none" stroke="currentColor" '
                'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
                '<circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/>'
                '</svg></div>'
                '<div><div class="settings-card-title">Confidence threshold</div>'
                '<div class="settings-card-desc">โมเดลต้องมั่นใจอย่างน้อยเท่านี้ ถึงจะนับว่าเจอป้ายทะเบียน</div>'
                '</div></div>'
                '<div class="settings-section-head-right">'
                f'<div class="{badge_cls}">{badge_txt}</div>'
                f'<div class="settings-section-value">{conf_now:.2f}</div>'
                '</div></div>',
                unsafe_allow_html=True,
            )
            with st.container(key="conf_slider_wrap"):
                st.slider("Confidence threshold", 0.0, 1.0, step=0.05,
                          key="conf_threshold", label_visibility="collapsed")
            st.markdown(
                '<div class="conf-bar-scale" style="position:static;display:flex;justify-content:space-between;margin-top:-0.4rem;">'
                '<span style="position:static">0.00</span>'
                f'<span style="position:static">ค่าเริ่มต้น {CONF_DEFAULT:.2f}</span>'
                '<span style="position:static">1.00</span></div>',
                unsafe_allow_html=True,
            )

            st.markdown(
                '<div class="settings-tradeoff-grid">'
                '<div class="settings-tradeoff-card good">'
                '<div class="ttitle"><span class="ttitle-icon">'
                '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
                'stroke-width="3" stroke-linecap="round" stroke-linejoin="round"><path d="M6 9l6 6 6-6"/></svg>'
                '</span>ทายผิดน้อยลง</div>'
                'กรอบที่ไม่ใช่ป้ายจริงจะถูกตัดทิ้งมากขึ้น</div>'
                '<div class="settings-tradeoff-card bad">'
                '<div class="ttitle"><span class="ttitle-icon">'
                '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
                'stroke-width="3" stroke-linecap="round" stroke-linejoin="round"><path d="M6 15l6-6 6 6"/></svg>'
                '</span>พลาดป้ายได้มากขึ้น</div>'
                'ป้ายที่เล็ก เบลอ หรือมุมเฉียง อาจไม่ผ่านเกณฑ์</div>'
                '</div>',
                unsafe_allow_html=True,
            )

        # ---------- Section: NMS IoU threshold ----------
        with st.container(key="settings_iou_section"):
            badge_cls = "settings-badge changed" if iou_changed else "settings-badge"
            badge_txt = "ปรับจากค่าเริ่มต้น" if iou_changed else "ค่าเริ่มต้น"
            st.markdown(
                '<div class="settings-section-head">'
                '<div class="settings-section-head-left">'
                '<div class="settings-icon">'
                '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
                'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
                '<rect x="3" y="3" width="13" height="13" rx="2"/><rect x="8" y="8" width="13" height="13" rx="2"/>'
                '</svg></div>'
                '<div><div class="settings-card-title">NMS IoU threshold</div>'
                '<div class="settings-card-desc">กรอบที่ซ้อนทับกันเกินค่านี้ จะถูกรวมให้เหลือกรอบเดียว</div>'
                '</div></div>'
                '<div class="settings-section-head-right">'
                f'<div class="{badge_cls}">{badge_txt}</div>'
                f'<div class="settings-section-value">{iou_now:.2f}</div>'
                '</div></div>',
                unsafe_allow_html=True,
            )
            iou_left, iou_right = st.columns([3, 2], gap="medium")
            with iou_left:
                st.slider("NMS IoU threshold", 0.0, 1.0, step=0.05,
                          key="iou_threshold", label_visibility="collapsed")
                st.markdown(
                    '<div class="conf-bar-scale" style="position:static;display:flex;justify-content:space-between;margin-top:-0.4rem;">'
                    '<span style="position:static">0.00</span>'
                    f'<span style="position:static">ค่าเริ่มต้น {DEFAULT_IOU:.2f}</span>'
                    '<span style="position:static">1.00</span></div>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    '<div class="settings-iou-tips">'
                    '<div class="settings-iou-tip"><b>ค่าต่ำ</b> — รวมกรอบที่ซ้อนกันเล็กน้อยก็ถูกตัดทิ้ง เหลือกรอบน้อยลง</div>'
                    '<div class="settings-iou-tip"><b>ค่าสูง</b> — ยอมให้กรอบซ้อนกันได้มากขึ้นก่อนจะถูกรวม</div>'
                    '</div>',
                    unsafe_allow_html=True,
                )
            with iou_right:
                st.markdown(
                    '<div class="settings-iou-figure">'
                    '<div class="settings-iou-diagram">'
                    '<div class="settings-iou-box b1"></div>'
                    '<div class="settings-iou-box b2"></div>'
                    '<div class="settings-iou-box overlap"></div>'
                    '</div>'
                    '<div class="settings-iou-figcaption">'
                    f'ส่วนที่ซ้อนกัน <b>&gt; {iou_now:.2f}</b><br/>เหลือกรอบที่มั่นใจกว่า 1 กรอบ</div>'
                    '</div>',
                    unsafe_allow_html=True,
                )

    with side_col:
        # ---------- Sidebar: current values ----------
        with st.container(key="settings_current_card"):
            st.markdown(
                '<h2 class="settings-side-title">ค่าที่ใช้อยู่ตอนนี้</h2>'
                '<div class="settings-side-desc">ใช้กับทุกรูปที่ตรวจในหน้า Detect</div>',
                unsafe_allow_html=True,
            )
            conf_value_html = (
                f'<span class="settings-old-value">{CONF_DEFAULT:.2f}</span>'
                f'<span class="settings-new-value">{conf_now:.2f}</span>'
                if conf_changed else f"{conf_now:.2f}"
            )
            iou_value_html = (
                f'<span class="settings-old-value">{DEFAULT_IOU:.2f}</span>'
                f'<span class="settings-new-value">{iou_now:.2f}</span>'
                if iou_changed else f"{iou_now:.2f}"
            )
            st.markdown(
                '<div class="detail-list" style="margin-top:0;border-top:none">'
                f'<div class="detail-list-row"><div class="k">Confidence</div><div class="v">{conf_value_html}</div></div>'
                f'<div class="detail-list-row"><div class="k">NMS IoU</div><div class="v">{iou_value_html}</div></div>'
                '<div class="detail-list-row"><div class="k settings-const-label">ขยายขอบครอป'
                '<span class="settings-const-icon" aria-label="ค่าคงที่">'
                '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
                'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
                '<rect x="3" y="11" width="18" height="11" rx="2"/><path d="M7 11V7a5 5 0 0 1 10 0v4"/>'
                '</svg></span></div><div class="v">3 px</div></div>'
                '<div class="detail-list-row"><div class="k settings-const-label">ขนาดรูปครอป'
                '<span class="settings-const-icon" aria-label="ค่าคงที่">'
                '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
                'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
                '<rect x="3" y="11" width="18" height="11" rx="2"/><path d="M7 11V7a5 5 0 0 1 10 0v4"/>'
                '</svg></span></div><div class="v">400×200</div></div>'
                '</div>',
                unsafe_allow_html=True,
            )
            if st.button("↺ รีเซ็ตเป็นค่าเริ่มต้น", key="reset_settings"):
                st.session_state["settings_reset_pending"] = True
                st.rerun()

        # ---------- Sidebar: metrics measured at default ----------
        precision_val = test_metrics.get("precision")
        recall_val = test_metrics.get("recall")
        precision_disp = f"{float(precision_val)*100:.1f}%" if precision_val not in (None, "") else "—"
        recall_disp = f"{float(recall_val)*100:.1f}%" if recall_val not in (None, "") else "—"
        with st.container(key="settings_metrics_card"):
            st.markdown(
                '<div class="settings-metrics-head">'
                '<div class="settings-metrics-icon">'
                '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
                'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
                '<circle cx="12" cy="12" r="9"/><path d="M12 8v5"/><path d="M12 16h.01"/>'
                '</svg></div>'
                '<h2 class="settings-metrics-title">ตัวเลขวัดผลอิงค่าเริ่มต้น</h2>'
                '</div>'
                f'<div class="settings-metrics-desc">Precision {precision_disp} และ Recall {recall_disp} ในหน้า Detect '
                f'วัดที่ confidence {CONF_DEFAULT:.2f} · IoU {DEFAULT_IOU:.2f} ถ้าปรับค่า ผลจริงจะต่างจากตัวเลขนี้</div>'
                '<div class="settings-metrics-grid">'
                f'<div class="settings-metric-box"><div class="mlabel">Precision · Test</div><div class="mvalue">{precision_disp}</div></div>'
                f'<div class="settings-metric-box"><div class="mlabel">Recall · Test</div><div class="mvalue">{recall_disp}</div></div>'
                '</div>',
                unsafe_allow_html=True,
            )

# ======================================================================
# TAB: Model Info
# ======================================================================
elif active == "model":
    # ---------- Header ----------
    st.markdown(
        '<div class="pipeline-eyebrow">MODEL CARD · YOLO11n</div>'
        '<div class="pipeline-header-row">'
        '<h1 class="pipeline-title">Model Info</h1>'
        f'<div class="pipeline-meta">{len(cv_folds) or 5}-FOLD CV · {dataset_stats["passed"]} IMAGES · SINGLE-CLASS</div>'
        '</div>'
        '<p class="pipeline-desc">สรุปสถาปัตยกรรม ผลการเทรนรายโฟลด์ ผลทดสอบกับชุดข้อมูลที่ไม่เคยเห็น '
        'และที่มาของชุดข้อมูลทั้งหมดที่ใช้เทรนโมเดลนี้</p>',
        unsafe_allow_html=True,
    )

    # ---------- Sub-tab เลือก Step 1-4 แบบ segmented control (แทนการเลื่อนดูยาวทั้งหน้า) ----------
    STEP_DEFS = [
        ("1", "Architecture"),
        ("2", "Cross-Validation"),
        ("3", "Test Set"),
        ("4", "Dataset"),
    ]
    substep_css = ""
    for step_id, _ in STEP_DEFS:
        is_active = st.session_state.model_step == step_id
        if is_active:
            bg, fg, shadow, weight = C["panel_bg"], C["accent"], "0 1px 3px rgba(15,23,42,0.12), 0 1px 1px rgba(15,23,42,0.06)", 700
        else:
            bg, fg, shadow, weight = "transparent", C["text_muted"], "none", 600
        substep_css += f"""
    .st-key-mstep_{step_id} button {{
        background: {bg} !important; color: {fg} !important; border: none !important;
        box-shadow: {shadow} !important; border-radius: 10px !important;
        font-weight: {weight} !important; font-size: 0.84rem !important;
        padding: 0.55rem 0.4rem !important; width: 100%; transition: background 0.15s ease, color 0.15s ease;
    }}
    .st-key-mstep_{step_id} button:hover {{
        background: {C['panel_bg'] if is_active else 'rgba(15,23,42,0.045)'} !important;
        color: {C['accent'] if is_active else C['text_primary']} !important;
    }}
    """
    st.markdown(f"<style>{substep_css}</style>", unsafe_allow_html=True)
    st.markdown('<div class="model-substep-bar">', unsafe_allow_html=True)
    with st.container(key="model_substep_track"):
        substep_cols = st.columns(len(STEP_DEFS))
        for col, (step_id, label) in zip(substep_cols, STEP_DEFS):
            with col:
                if st.button(f"{step_id}  ·  {label}", key=f"mstep_{step_id}", use_container_width=True):
                    st.session_state.model_step = step_id
                    st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    model_step = st.session_state.model_step

    # ---------- Step content: แต่ละ step เป็นไฟล์ HTML ที่เตรียมไว้ล่วงหน้า (ไม่ generate ด้วย CSS/โค้ดของแอป) ----------
    STEP_PAGES = {
        "1": ("assets/model_info/step1_architecture.html", 870),
        "2": ("assets/model_info/step2_cross_validation.html", 910),
        "3": ("assets/model_info/step3_test_set.html", 870),
        "4": ("assets/model_info/step4_dataset.html", 810),
    }
    html_path, iframe_height = STEP_PAGES[model_step]
    page_html = (APP_DIR / html_path).read_text(encoding="utf-8")
    # หน้า HTML แต่ละ step ย่อขนาด (zoom) ตัวเองให้พอดีความกว้างจริงของ iframe ด้วย JS ในตัว
    # (ดู script ท้าย body ของแต่ละไฟล์) แล้วปรับความสูง iframe ให้พอดีเนื้อหาอัตโนมัติ — ไม่ต้องมี
    # scrollbar ใดๆ อีก — height ที่ส่งมาเป็นแค่ค่าเริ่มต้นก่อน JS จะปรับจริง
    components.html(page_html, height=iframe_height, scrolling=False)

# ======================================================================
# TAB: Pipeline — "จากรูปรถ ถึงป้ายที่ครอปแล้ว" ตามไฟล์ตัวอย่าง pipeline-page.html ทุกประการ
# ======================================================================
elif active == "pipeline":
    # เนื้อหาทั้งหน้าใช้ไฟล์ HTML จริงจากไฟล์ตัวอย่าง pipeline-page_1.html แบบคำต่อคำ (เหมือนวิธีที่ใช้ใน
    # แท็บ Model Info) ฝังผ่าน iframe เพื่อให้ได้ฟอนต์/สี/เลย์เอาต์ตรงกับไฟล์ต้นฉบับทุกประการ โดยตัดส่วน
    # <header> ของไฟล์ออก (ซ้ำกับ header จริงของแอปด้านบนอยู่แล้ว) เหลือแค่ <main>...</main>
    pipeline_html_path = APP_DIR / "assets" / "pipeline" / "pipeline.html"
    pipeline_page_html = pipeline_html_path.read_text(encoding="utf-8")
    components.html(pipeline_page_html, height=900, scrolling=False)
