"""
แพ็กไฟล์ส่งผู้ตรวจ -> submission/submission.zip
เนื้อหาใน zip:
  Present.docx, Present.pdf     รายงาน (จาก docs/)
  thai-license-plate-yolo11/    โปรเจกต์ (โค้ดทั้งหมด + models/best.pt + samples + label/manifest ของ dataset + README)
ไม่รวม: รูปชุดข้อมูลดิบ (data/raw/*.jpg), ผลเทรนทุก fold, โฟลเดอร์ชั่วคราว (ดูเหตุผลใน README ส่วน "เรื่อง Dataset")

ใช้: python make_submission.py
"""
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "submission" / "submission.zip"
TOP = "thai-license-plate-yolo11"
REPORT_FILES = ["Present.docx", "Present.pdf"]

INCLUDE_DIRS = ["src", "samples", "docs", "data/labels"]
INCLUDE_FILES = [
    "models/best.pt", "RUN_COMMANDS.txt", "requirements.txt", "requirements-cli.txt", "README.md", "PLAN.md", ".gitignore",
    "data/classes.txt", "data/raw/CREDITS.md", "data/raw/dataset_manifest.csv",
    "outputs/predictions/cv_summary.csv", "outputs/predictions/test_set_summary.csv", "outputs/predictions/threshold_sweep.csv",
]
EMPTY_DIRS = ["outputs/crops"]
SKIP_PARTS = {"__pycache__", ".DS_Store", ".ipynb_checkpoints"}
SKIP_IN_DOCS = set(REPORT_FILES)  # อยู่ที่ระดับบนสุดของ zip แล้ว ไม่ซ้ำใน docs/


def skip(p: Path) -> bool:
    return (any(part in SKIP_PARTS for part in p.parts) or p.name.startswith("~$")
            or p.suffix == ".pyc" or (p.parent.name == "docs" and p.name in SKIP_IN_DOCS))


def main():
    OUT.parent.mkdir(exist_ok=True)
    files: list[Path] = []
    for d in INCLUDE_DIRS:
        files += [p for p in (ROOT / d).rglob("*") if p.is_file() and not skip(p)]
    for f in INCLUDE_FILES:
        p = ROOT / f
        if p.exists():
            files.append(p)
        else:
            print("[เตือน] ไม่พบไฟล์:", f)
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for name in REPORT_FILES:
            p = ROOT / "docs" / name
            if p.exists():
                z.write(p, name)
            else:
                print("[เตือน] ไม่พบรายงาน:", p)
        z.write(ROOT / "RUN_COMMANDS.txt", "RUN_COMMANDS.txt")  # ไว้ระดับบนสุด ให้ผู้ตรวจเห็นทันที
        for p in sorted(set(files)):
            z.write(p, f"{TOP}/{p.relative_to(ROOT).as_posix()}")
        for d in EMPTY_DIRS:
            z.writestr(f"{TOP}/{d}/.gitkeep", "")
    print(f"สร้างแล้ว: {OUT}  ({OUT.stat().st_size / 1e6:.1f} MB, {len(files)} ไฟล์ในโปรเจกต์)")


if __name__ == "__main__":
    main()
