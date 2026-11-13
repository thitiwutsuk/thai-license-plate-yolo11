import ast, json
from pathlib import Path
R = Path("..")
def func(path, name):
    src = (R / path).read_text(encoding="utf-8"); tree = ast.parse(src); lines = src.splitlines()
    for n in ast.walk(tree):
        if isinstance(n, ast.FunctionDef) and n.name == name:
            body = lines[n.lineno - 1:n.end_lineno]
            return n.lineno, "\n".join(body)
    raise KeyError(name)
def span(path, start_anchor, end_anchor):
    lines = (R / path).read_text(encoding="utf-8").splitlines()
    s = next(i for i, l in enumerate(lines) if start_anchor in l)
    e = next(i for i in range(s, len(lines)) if end_anchor in lines[i])
    return s + 1, "\n".join(lines[s:e + 1])
out = []
def add(key, path, ln, code): out.append({"key": key, "path": path, "line": ln, "code": code})
p = "src/data_prep/ingest_images.py"; add("sha", p, *func(p, "sha256_of_file")); add("resize", p, *func(p, "resize_and_save"))
p = "src/data_prep/split_folds.py"; add("group", p, *func(p, "extract_group_key"))
add("fold", p, *span(p, "fold_assign = [[]", 'print("จำนวนรูปต่อ fold:"'))
add("yaml", p, *func(p, "write_data_yaml"))
p = "src/train/train_yolo11.py"; add("train", p, *span(p, "model = YOLO(args.model)", "precision = float"))
p = "src/inference/predict_and_crop.py"; add("predict", p, *span(p, "t0 = time.perf_counter()", "best_conf = float"))
p = "src/crop/crop_utils.py"; add("crop", p, *func(p, "crop_plate"))
p = "src/train/evaluate_test_set.py"; add("iou", p, *func(p, "iou")); 
p = "src/app/feedback_utils.py"; add("canvas", p, *func(p, "canvas_rect_to_yolo"))
json.dump(out, open("snippets.json", "w"), ensure_ascii=False, indent=1)
for o in out: print(o["key"], o["path"], o["line"], len(o["code"].splitlines()))
