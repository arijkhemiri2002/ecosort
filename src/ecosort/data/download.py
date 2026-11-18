"""Étape DVC `download` : annotations TACO + images Flickr (reprise possible).

Usage : python -m ecosort.data.download
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path
from typing import Any

import requests
from PIL import Image
from tqdm import tqdm

from ecosort.config import load_params, resolve

HEADERS = {"User-Agent": "EcoSort/0.1 (projet pedagogique)"}


def flat_name(file_name: str) -> str:
    """'batch_1/000006.jpg' -> 'batch_1_000006.jpg' (noms uniques, sans sous-dossiers)."""
    return file_name.replace("\\", "/").replace("/", "_")


def fetch(url: str, timeout: int = 30, retries: int = 3) -> bytes:
    """GET avec retries ; rejette la page 'photo indisponible' de Flickr."""
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(url, headers=HEADERS, timeout=timeout)
            if response.status_code == 200 and "photo_unavailable" not in response.url:
                return response.content
            last_error = RuntimeError(f"HTTP {response.status_code} ({response.url})")
        except requests.RequestException as exc:
            last_error = exc
        time.sleep(1.5 * attempt)
    raise RuntimeError(f"échec après {retries} tentatives : {last_error}")


def load_annotations(url: str, dest: Path) -> dict[str, Any]:
    if not dest.exists():
        print(f"Téléchargement des annotations : {url}")
        dest.write_bytes(fetch(url, timeout=120))
    return json.loads(dest.read_text(encoding="utf-8"))


def download_image(image: dict[str, Any], images_dir: Path, url_field: str) -> str | None:
    """Télécharge une image. Retourne None si OK (ou déjà présente), sinon l'erreur."""
    dest = images_dir / flat_name(image["file_name"])
    if dest.exists() and dest.stat().st_size > 0:
        return None

    candidates = (image.get(url_field), image.get("flickr_url"))
    urls = list(dict.fromkeys(u for u in candidates if u))  # dédoublonne en gardant l'ordre
    if not urls:
        return "aucune URL disponible"

    errors = []
    for url in urls:
        try:
            content = fetch(url)
            with Image.open(BytesIO(content)) as img:
                img.verify()  # rejette les fichiers corrompus / non-images
            tmp = dest.with_name(dest.name + ".part")
            tmp.write_bytes(content)
            tmp.replace(dest)  # écriture atomique
            return None
        except Exception as exc:
            errors.append(f"{url} -> {exc}")
    return " | ".join(errors)


def main() -> None:
    data_cfg = load_params()["data"]
    raw_dir = resolve(data_cfg["raw_dir"])
    images_dir = raw_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    for leftover in images_dir.glob("*.part"):
        leftover.unlink()

    coco = load_annotations(data_cfg["taco_annotations_url"], raw_dir / "annotations.json")
    images = coco["images"]
    print(
        f"TACO : {len(images)} images, {len(coco['annotations'])} annotations, "
        f"{len(coco['categories'])} catégories"
    )

    failures: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=data_cfg["download_workers"]) as pool:
        futures = {
            pool.submit(download_image, img, images_dir, data_cfg["image_url_field"]): img
            for img in images
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Images"):
            error = future.result()
            if error:
                failures.append({"file_name": futures[future]["file_name"], "error": error})

    ok = len(images) - len(failures)
    report = {"total": len(images), "downloaded": ok, "failed": len(failures), "failures": failures}
    (raw_dir / "download_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"[OK] {ok}/{len(images)} images disponibles ({len(failures)} échecs)")
    if ok == 0:
        raise SystemExit("Aucune image téléchargée : vérifie ta connexion.")
    if len(failures) > 0.2 * len(images):
        print("[WARN] Plus de 20 % d'échecs : relance `dvc repro download` pour reprendre.")


if __name__ == "__main__":
    main()