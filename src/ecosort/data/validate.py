"""Étape DVC `validate_data` : contrôle qualité du dataset YOLO + aperçus annotés.

Usage : python -m ecosort.data.validate
"""

from __future__ import annotations

import json
import random
import shutil
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from ecosort.config import REPORTS_DIR, get_classes, load_params, resolve

SPLITS = ("train", "val", "test")
PALETTE = [(230, 57, 70), (69, 123, 157), (42, 157, 143),
           (244, 162, 97), (131, 56, 236), (108, 117, 125)]
EPS = 1e-4


def check_label_file(path: Path, nc: int) -> list[str]:
    errors: list[str] = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        parts = line.split()
        if len(parts) != 5:
            errors.append(f"{path.name}:{i} attendu 5 valeurs, reçu {len(parts)}")
            continue
        try:
            cls = int(parts[0])
            xc, yc, w, h = map(float, parts[1:])
        except ValueError:
            errors.append(f"{path.name}:{i} valeur non numérique")
            continue
        if not 0 <= cls < nc:
            errors.append(f"{path.name}:{i} classe {cls} hors [0, {nc - 1}]")
        if w <= 0 or h <= 0:
            errors.append(f"{path.name}:{i} largeur/hauteur nulle")
        if (xc - w / 2 < -EPS or xc + w / 2 > 1 + EPS
                or yc - h / 2 < -EPS or yc + h / 2 > 1 + EPS):
            errors.append(f"{path.name}:{i} boîte hors de l'image")
    return errors


def draw_sample(img_path: Path, label_path: Path, classes: list[str], dest: Path) -> None:
    img = Image.open(img_path).convert("RGB")
    W, H = img.size
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    for line in label_path.read_text(encoding="utf-8").splitlines():
        cls_str, *coords = line.split()
        cls = int(cls_str)
        xc, yc, w, h = map(float, coords)
        x1, y1 = (xc - w / 2) * W, (yc - h / 2) * H
        x2, y2 = (xc + w / 2) * W, (yc + h / 2) * H
        color = PALETTE[cls % len(PALETTE)]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        label = classes[cls]
        left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
        tw, th = right - left, bottom - top
        ty = max(0.0, y1 - th - 6)
        draw.rectangle([x1, ty, x1 + tw + 6, ty + th + 6], fill=color)
        draw.text((x1 + 3, ty + 3 - top), label, fill=(255, 255, 255), font=font)
    img.save(dest, quality=90)


def main(n_samples: int = 8) -> None:
    params = load_params()
    classes = get_classes()
    out_dir = resolve(params["data"]["processed_dir"])
    errors: list[str] = []
    summary: dict[str, int] = {}

    try:
        from ultralytics.data.utils import check_det_dataset

        info = check_det_dataset(str(out_dir / "dataset.yaml"))
        if info["nc"] != len(classes):
            errors.append(f"nc={info['nc']} différent de {len(classes)} classes")
    except Exception as exc:
        errors.append(f"dataset.yaml refusé par Ultralytics : {exc}")

    for split in SPLITS:
        images = sorted((out_dir / "images" / split).glob("*.jpg"))
        labels = sorted((out_dir / "labels" / split).glob("*.txt"))
        summary[f"{split}_images"] = len(images)
        summary[f"{split}_labels"] = len(labels)
        if not images:
            errors.append(f"split '{split}' vide")
        img_stems, lbl_stems = {p.stem for p in images}, {p.stem for p in labels}
        errors += [f"{split}: label manquant pour {s}" for s in sorted(img_stems - lbl_stems)]
        errors += [f"{split}: label orphelin {s}" for s in sorted(lbl_stems - img_stems)]
        for label_path in labels:
            errors += [f"{split}/{e}" for e in check_label_file(label_path, len(classes))]

    samples_dir = REPORTS_DIR / "data_samples"
    if samples_dir.exists():
        shutil.rmtree(samples_dir)
    samples_dir.mkdir(parents=True)
    train_labels = sorted((out_dir / "labels" / "train").glob("*.txt"))
    candidates = [p for p in train_labels if p.stat().st_size > 0]
    rng = random.Random(params["seed"])
    for label_path in rng.sample(candidates, min(n_samples, len(candidates))):
        img_path = out_dir / "images" / "train" / f"{label_path.stem}.jpg"
        draw_sample(img_path, label_path, classes, samples_dir / f"{label_path.stem}.jpg")

    report = {"ok": int(not errors), "n_errors": len(errors), **summary}
    (REPORTS_DIR / "data_validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    for err in errors[:30]:
        print(f"[FAIL] {err}")
    if errors:
        raise SystemExit(f"{len(errors)} erreur(s) de validation.")
    print(f"[OK] Dataset valide : {summary}")
    print(f"[OK] Aperçus annotés dans {samples_dir}")


if __name__ == "__main__":
    main()