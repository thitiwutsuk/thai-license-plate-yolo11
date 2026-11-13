"""
แบ่งข้อมูล 250 รูปที่ตีกรอบแล้ว เป็น:
  - ชุดทดสอบ (test) 50 รูป
  - ชุดสอน (train pool) 200 รูป -> แบ่งเป็น 5 fold (fold ละ 40 รูปสำหรับ validation)

กันรถคันเดียวกัน (หลายรูปจากหลายมุม) ไม่ให้หลุดไปอยู่คนละชุด/คนละ fold
โดยจัดกลุ่มจากชื่อไฟล์ต้นฉบับใน dataset_manifest.csv (source_url) ก่อนแบ่ง

ผลลัพธ์: สร้าง symlink ภาพ+label ไว้ที่
  data/test/images/, data/test/labels/
  data/cv/fold_{1..5}/train/images/, train/labels/, val/images/, val/labels/
และเขียน data/cv/fold_{1..5}/data.yaml ให้แต่ละ fold

วิธีใช้:
  python src/data_prep/split_folds.py
"""

import csv
import os
import random
import re
import urllib.parse
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw"
LABELS_DIR = REPO_ROOT / "data" / "labels"
MANIFEST_PATH = RAW_DIR / "dataset_manifest.csv"
TEST_DIR = REPO_ROOT / "data" / "test"
CV_DIR = REPO_ROOT / "data" / "cv"

TEST_SIZE = 50
N_FOLDS = 5
SEED = 42


def extract_group_key(source_url: str) -> str:
    """ดึง 'ชื่อรถ' จาก URL ต้นฉบับ เพื่อจัดกลุ่มรูปที่เป็นรถคันเดียวกัน"""
    name = source_url.split("File:")[-1] if "File:" in source_url else source_url
    name = urllib.parse.unquote(name)
    name = re.sub(r"\.(jpg|jpeg|png|JPG|JPEG|PNG)$", "", name)
    name = re.sub(r"\s*\(\d{6,}\)\s*$", "", name)  # flickr id ท้ายชื่อ
    name = re.sub(r"[\s_]*(front|rear|side|left|right|cropped|\d{1,2})[\s_]*$", "", name, flags=re.IGNORECASE)
    name = re.sub(r"\s*\([^)]*\)\s*$", "", name)
    return name.strip().lower()


def greedy_split(groups: list[tuple[str, list[str]]], target_a: int, seed: int):
    """สุ่มลำดับกลุ่ม แล้วแจกเข้า A จนใกล้ target_a มากที่สุด ที่เหลือเข้า B (ไม่ตัดกลุ่ม)"""
    rng = random.Random(seed)
    groups = groups.copy()
    rng.shuffle(groups)

    set_a, set_b = [], []
    count_a = 0
    for key, files in groups:
        if count_a + len(files) <= target_a:
            set_a.extend(files)
            count_a += len(files)
        else:
            set_b.extend(files)
    return set_a, set_b


def make_symlinks(file_list: list[str], images_dir: Path, labels_dir: Path):
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    for fname in file_list:
        img_src = (RAW_DIR / fname).resolve()
        label_src = (LABELS_DIR / (Path(fname).stem + ".txt")).resolve()
        img_link = images_dir / fname
        label_link = labels_dir / (Path(fname).stem + ".txt")
        if not img_link.exists():
            os.symlink(img_src, img_link)
        if not label_link.exists():
            if label_src.exists():
                os.symlink(label_src, label_link)
            else:
                print(f"[เตือน] ไม่พบ label สำหรับ {fname}")


def write_data_yaml(path: Path, train_dir: Path, val_dir: Path):
    content = (
        f"train: {train_dir.resolve()}\n"
        f"val: {val_dir.resolve()}\n"
        f"nc: 1\n"
        f"names: ['license_plate']\n"
    )
    path.write_text(content, encoding="utf-8")


def main():
    rows = list(csv.DictReader(open(MANIFEST_PATH, encoding="utf-8")))
    passed = [r for r in rows if r["status"] == "passed"]
    assert len(passed) == 250, f"คาดว่าจะมี 250 รูปที่ passed แต่เจอ {len(passed)}"

    groups_map = defaultdict(list)
    for r in passed:
        key = extract_group_key(r["source_url"])
        groups_map[key].append(r["filename"])
    groups = list(groups_map.items())
    print(f"จำนวนรูปทั้งหมด: {len(passed)} | จำนวนกลุ่มรถ: {len(groups)}")

    # แบ่ง test (50) ออกจากที่เหลือ (train pool, ควรได้ ~200)
    test_files, train_pool_files = greedy_split(groups, TEST_SIZE, seed=SEED)
    print(f"ชุดทดสอบ: {len(test_files)} รูป | ชุดสอน (pool): {len(train_pool_files)} รูป")

    # เขียน test
    make_symlinks(test_files, TEST_DIR / "images", TEST_DIR / "labels")

    # แบ่ง train pool เป็น 5 fold โดยยังกันกลุ่มรถไม่ให้ข้าม fold
    train_pool_set = set(train_pool_files)
    pool_groups = [(k, [f for f in v if f in train_pool_set]) for k, v in groups]
    pool_groups = [(k, v) for k, v in pool_groups if v]

    rng = random.Random(SEED)
    pool_groups_shuffled = pool_groups.copy()
    rng.shuffle(pool_groups_shuffled)

    fold_assign = [[] for _ in range(N_FOLDS)]
    fold_counts = [0] * N_FOLDS
    target_per_fold = len(train_pool_files) / N_FOLDS

    for key, files in pool_groups_shuffled:
        # ใส่กลุ่มนี้ลง fold ที่มีรูปน้อยที่สุดตอนนี้ (ทำให้สมดุลใกล้เคียงกัน)
        min_fold = min(range(N_FOLDS), key=lambda i: fold_counts[i])
        fold_assign[min_fold].extend(files)
        fold_counts[min_fold] += len(files)

    print("จำนวนรูปต่อ fold:", fold_counts)

    for k in range(N_FOLDS):
        val_files = fold_assign[k]
        train_files = [f for i in range(N_FOLDS) if i != k for f in fold_assign[i]]

        fold_dir = CV_DIR / f"fold_{k+1}"
        make_symlinks(train_files, fold_dir / "train" / "images", fold_dir / "train" / "labels")
        make_symlinks(val_files, fold_dir / "val" / "images", fold_dir / "val" / "labels")
        write_data_yaml(fold_dir / "data.yaml", fold_dir / "train" / "images", fold_dir / "val" / "images")
        print(f"fold_{k+1}: train {len(train_files)} | val {len(val_files)}")

    print("\nเสร็จแล้ว โครงสร้างข้อมูลอยู่ที่ data/test/ และ data/cv/fold_1..5/")


if __name__ == "__main__":
    main()
