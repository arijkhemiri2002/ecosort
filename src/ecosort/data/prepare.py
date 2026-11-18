"""Étape DVC `prepare` : TACO (COCO) -> dataset YOLO à 6 classes + split train/val/test.

Usage : python -m ecosort.data.prepare
"""

from __future__ import annotations

import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml
from PIL import Image, ImageOps
from tqdm import tqdm

from ecosort.config import REPORTS_DIR, get_classes, load_params, load_yaml, resolve
from ecosort.data.download import flat_name

SPLITS = ("train", "val", "test")
ASPECT_TOLERANCE = 0.02  # 2 % d'écart toléré sur le ratio largeur/hauteur


def norm(name: str) -> str:
    """Normalise un nom de catégorie (casse et espaces)."""
    return " ".join(name.lower().split())


def build_category_map(
    categories: list[dict[str, Any]], mapping_cfg: dict[str, Any], classes: list[str]
) -> tuple[dict[int, int], list[str]]:
    """Retourne {id catégorie TACO -> index classe EcoSort} et la liste des avertissements."""
    class_to_idx = {c: i for i, c in enumerate(classes)}
    default_class = mapping_cfg.get("default_class", "other")
    if default_class not in class_to_idx:
        raise ValueError(f"default_class '{default_class}' absente de params.classes")

    name_to_class: dict[str, str] = {}
    for cls, names in mapping_cfg["mapping"].items():
        if cls not in class_to_idx:
            raise ValueError(f"Classe '{cls}' du mapping absente de params.classes")
        for name in names or []:
            name_to_class[norm(name)] = cls

    warnings: list[str] = []
    cat_map: dict[int, int] = {}
    taco_names = set()
    for cat in categories:
        key = norm(cat["name"])
        taco_names.add(key)
        cls = name_to_class.get(key)
        if cls is None:
            warnings.append(f"Catégorie TACO non mappée -> '{default_class}' : {cat['name']}")
            cls = default_class
        cat_map[cat["id"]] = class_to_idx[cls]

    for key in name_to_class:
        if key not in taco_names:
            warnings.append(f"Entrée du mapping absente de TACO : {key}")
    return cat_map, warnings


def load_oriented(path: Path, ann_w: int, ann_h: int) -> Image.Image | None:
    """Ouvre l'image dans l'orientation qui correspond aux annotations (EXIF ou brute).

    Retourne None si aucune orientation ne correspond (image remplacée, corrompue...).
    """
    target = ann_w / ann_h
    with Image.open(path) as img:
        img.load()
        for candidate in (ImageOps.exif_transpose(img), img):
            w, h = candidate.size
            if abs(w / h - target) / target <= ASPECT_TOLERANCE:
                return candidate.convert("RGB")
    return None


def resize_max(img: Image.Image, max_side: int) -> Image.Image:
    w, h = img.size
    scale = max_side / max(w, h)
    if scale < 1:
        img = img.resize((round(w * scale), round(h * scale)), Image.Resampling.LANCZOS)
    return img


def to_yolo_lines(
    anns: list[dict[str, Any]],
    cat_map: dict[int, int],
    ann_w: int,
    ann_h: int,
    out_w: int,
    out_h: int,
    min_box: float,
) -> tuple[list[str], int]:
    """Convertit des bbox COCO [x, y, w, h] (pixels) en lignes YOLO normalisées."""
    lines: list[str] = []
    dropped = 0
    for ann in anns:
        x, y, w, h = ann["bbox"]
        x1, y1 = max(0.0, x), max(0.0, y)
        x2, y2 = min(float(ann_w), x + w), min(float(ann_h), y + h)
        bw, bh = (x2 - x1) / ann_w, (y2 - y1) / ann_h
        if bw * out_w < min_box or bh * out_h < min_box:
            dropped += 1
            continue
        xc, yc = (x1 + x2) / 2 / ann_w, (y1 + y2) / 2 / ann_h
        cls = cat_map[ann["category_id"]]
        lines.append(f"{cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
    return lines, dropped


def split_items(
    names: list[str], val_ratio: float, test_ratio: float, seed: int
) -> dict[str, list[str]]:
    """Split déterministe : même seed + même ensemble de noms = même split."""
    names = sorted(names)
    random.Random(seed).shuffle(names)
    n_test = round(len(names) * test_ratio)
    n_val = round(len(names) * val_ratio)
    return {
        "test": names[:n_test],
        "val": names[n_test : n_test + n_val],
        "train": names[n_test + n_val :],
    }


def main() -> None:
    params = load_params()
    data_cfg = params["data"]
    classes = get_classes()
    raw_dir = resolve(data_cfg["raw_dir"])
    images_dir = raw_dir / "images"
    out_dir = resolve(data_cfg["processed_dir"])

    coco = json.loads((raw_dir / "annotations.json").read_text(encoding="utf-8"))
    mapping_cfg = load_yaml(data_cfg["mapping_file"])
    cat_map, warnings = build_category_map(coco["categories"], mapping_cfg, classes)
    for warning in warnings:
        print(f"[WARN] {warning}")

    anns_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for ann in coco["annotations"]:
        anns_by_image[ann["image_id"]].append(ann)

    by_name = {
        flat_name(img["file_name"]): img
        for img in coco["images"]
        if (images_dir / flat_name(img["file_name"])).exists()
    }
    missing_raw = len(coco["images"]) - len(by_name)
    splits = split_items(
        list(by_name), data_cfg["val_ratio"], data_cfg["test_ratio"], params["seed"]
    )

    if out_dir.exists():
        shutil.rmtree(out_dir)
    for split in SPLITS:
        (out_dir / "images" / split).mkdir(parents=True)
        (out_dir / "labels" / split).mkdir(parents=True)

    stats = {s: {"images": 0, "background": 0, "instances": Counter()} for s in SPLITS}
    skipped: Counter[str] = Counter()
    dropped_total = 0

    for split in SPLITS:
        for name in tqdm(splits[split], desc=f"prepare {split}"):
            info = by_name[name]
            ann_w, ann_h = info["width"], info["height"]
            try:
                img = load_oriented(images_dir / name, ann_w, ann_h)
            except Exception:
                skipped["corrupt"] += 1
                continue
            if img is None:
                skipped["aspect_mismatch"] += 1
                continue

            img = resize_max(img, data_cfg["max_image_side"])
            lines, dropped = to_yolo_lines(
                anns_by_image[info["id"]], cat_map, ann_w, ann_h, *img.size,
                min_box=data_cfg["min_box_size"],
            )
            dropped_total += dropped

            stem = Path(name).stem
            img.save(out_dir / "images" / split / f"{stem}.jpg", quality=95)
            label_text = "\n".join(lines) + ("\n" if lines else "")
            (out_dir / "labels" / split / f"{stem}.txt").write_text(label_text, encoding="utf-8")

            stats[split]["images"] += 1
            stats[split]["background"] += int(not lines)
            for line in lines:
                stats[split]["instances"][classes[int(line.split()[0])]] += 1

    dataset_cfg = {
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(classes),
        "names": dict(enumerate(classes)),
    }
    with (out_dir / "dataset.yaml").open("w", encoding="utf-8") as f:
        f.write("# Généré par ecosort.data.prepare - ne pas éditer à la main\n")
        f.write("# Pas de clé 'path' : Ultralytics prend le dossier de ce fichier comme racine\n")
        yaml.safe_dump(dataset_cfg, f, sort_keys=False, allow_unicode=True)

    report = {
        "images": {s: stats[s]["images"] for s in SPLITS},
        "background_images": {s: stats[s]["background"] for s in SPLITS},
        "instances": {
            s: {c: stats[s]["instances"].get(c, 0) for c in classes} for s in SPLITS
        },
        "missing_raw": missing_raw,
        "skipped": {k: skipped.get(k, 0) for k in ("aspect_mismatch", "corrupt")},
        "dropped_small_boxes": dropped_total,
        "mapping_warnings": len(warnings),
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "data_stats.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\nInstances par classe :")
    print(f"{'classe':<10} " + " ".join(f"{s:>7}" for s in SPLITS))
    for c in classes:
        print(f"{c:<10} " + " ".join(f"{report['instances'][s][c]:>7}" for s in SPLITS))
    print(f"\nImages : {report['images']} | ignorées : {report['skipped']} "
          f"| absentes : {missing_raw} | petites boîtes retirées : {dropped_total}")
    for split in ("val", "test"):
        empty = [c for c in classes if report["instances"][split][c] == 0]
        if empty:
            print(f"[WARN] Classes absentes du split {split} : {empty}")
    print(f"[OK] Dataset YOLO écrit dans {out_dir}")


if __name__ == "__main__":
    main()