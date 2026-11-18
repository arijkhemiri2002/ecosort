"""Tests unitaires de la préparation des données (sans téléchargement)."""

import pytest
from PIL import Image

from ecosort.config import get_classes, load_yaml
from ecosort.data.download import flat_name
from ecosort.data.prepare import (
    build_category_map,
    load_oriented,
    norm,
    split_items,
    to_yolo_lines,
)


def test_flat_name():
    assert flat_name("batch_1/000006.jpg") == "batch_1_000006.jpg"


def test_mapping_file_covers_all_classes_without_duplicates():
    mapping = load_yaml("configs/taco_mapping.yaml")["mapping"]
    assert set(mapping) == set(get_classes())
    names = [norm(n) for group in mapping.values() for n in group]
    assert len(names) == len(set(names)) == 60


def test_build_category_map_sends_unknown_to_default():
    categories = [{"id": 0, "name": "Drink can"}, {"id": 1, "name": "Mystery item"}]
    cfg = {"default_class": "other", "mapping": {"metal": ["Drink  CAN"], "other": []}}
    cat_map, warnings = build_category_map(categories, cfg, ["plastic", "metal", "other"])
    assert cat_map == {0: 1, 1: 2}
    assert any("Mystery item" in w for w in warnings)


def test_to_yolo_lines_normalizes_clips_and_drops_small_boxes():
    anns = [
        {"bbox": [10, 20, 30, 40], "category_id": 0},  # boîte normale
        {"bbox": [90, 90, 50, 50], "category_id": 0},  # déborde -> rognée
        {"bbox": [0, 0, 1, 1], "category_id": 0},      # trop petite -> retirée
    ]
    lines, dropped = to_yolo_lines(anns, {0: 2}, 100, 100, 100, 100, min_box=4)
    assert dropped == 1
    assert lines[0] == "2 0.250000 0.400000 0.300000 0.400000"
    _, xc, _, w, _ = map(float, lines[1].split())
    assert w == pytest.approx(0.1)
    assert xc + w / 2 <= 1.0


def test_split_is_deterministic_and_complete():
    names = [f"img_{i}.jpg" for i in range(100)]
    a = split_items(names, 0.15, 0.10, seed=42)
    b = split_items(list(reversed(names)), 0.15, 0.10, seed=42)
    assert a == b
    assert (len(a["train"]), len(a["val"]), len(a["test"])) == (75, 15, 10)
    assert sorted(a["train"] + a["val"] + a["test"]) == sorted(names)


def test_load_oriented_uses_exif_rotation(tmp_path):
    path = tmp_path / "rotated.jpg"
    exif = Image.Exif()
    exif[0x0112] = 6  # orientation EXIF : rotation de 90°
    Image.new("RGB", (200, 100)).save(path, exif=exif)

    assert load_oriented(path, 100, 200).size == (100, 200)  # annotations en portrait
    assert load_oriented(path, 200, 100).size == (200, 100)  # annotations en paysage
    assert load_oriented(path, 300, 100) is None             # ratio incohérent