"""Vérification complète de l'environnement EcoSort (étape 1)."""

from __future__ import annotations

import importlib
import platform
import shutil
import subprocess
import sys

REQUIRED = [
    "torch", "torchvision", "ultralytics", "onnx", "onnxruntime", "mlflow",
    "fastapi", "uvicorn", "streamlit", "pandas", "plotly", "cv2", "yaml", "PIL", "httpx",
]


def check_packages() -> bool:
    ok = True
    print("\n=== Paquets Python ===")
    for name in REQUIRED:
        try:
            mod = importlib.import_module(name)
            version = getattr(mod, "__version__", "?")
            print(f"[OK]   {name:<12} {version}")
        except Exception as exc:  # noqa: BLE001
            ok = False
            print(f"[FAIL] {name:<12} {exc}")
    return ok


def check_gpu() -> str:
    import torch

    print("\n=== GPU ===")
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"[OK]   CUDA {torch.version.cuda} | {name} | {vram:.1f} Go VRAM")
        return "0"
    print("[WARN] CUDA indisponible : l'entraînement se fera sur CPU (ou Kaggle/Colab)")
    return "cpu"


def check_cli(tool: str, args: list[str], required: bool) -> bool:
    path = shutil.which(tool)
    if path is None:
        level = "FAIL" if required else "WARN"
        print(f"[{level}] {tool:<12} introuvable dans le PATH")
        return not required
    try:
        out = subprocess.run([tool, *args], capture_output=True, text=True, timeout=60)
        first_line = (out.stdout or out.stderr).strip().splitlines()[0]
        print(f"[OK]   {tool:<12} {first_line}")
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] {tool:<12} installé mais erreur : {exc}")
    return True


def check_project() -> bool:
    print("\n=== Package ecosort ===")
    try:
        from ecosort import __version__
        from ecosort.config import ROOT_DIR, get_classes

        print(f"[OK]   ecosort {__version__} | racine : {ROOT_DIR}")
        print(f"[OK]   classes : {get_classes()}")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] ecosort : {exc} (as-tu lancé `pip install -e .` ?)")
        return False


def smoke_test_yolo(device: str) -> bool:
    print("\n=== Test d'inférence YOLO ===")
    try:
        from ultralytics import YOLO
        from ultralytics.utils import ASSETS

        from ecosort.config import load_params

        model_name = load_params()["train"]["model"]
        model = YOLO(model_name)  # télécharge les poids pré-entraînés COCO au 1er lancement
        results = model.predict(ASSETS / "bus.jpg", device=device, verbose=False)
        n = len(results[0].boxes)
        labels = sorted({results[0].names[int(c)] for c in results[0].boxes.cls})
        print(f"[OK]   {model_name} sur {device} : {n} objets détectés -> {labels}")
        return n > 0
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] inférence YOLO : {exc}")
        return False


def main() -> int:
    print(f"Python {sys.version.split()[0]} | {platform.system()} {platform.release()}")
    ok = check_packages()
    device = check_gpu()

    print("\n=== Outils CLI ===")
    ok &= check_cli("git", ["--version"], required=True)
    ok &= check_cli("dvc", ["--version"], required=True)
    ok &= check_cli("docker", ["--version"], required=False)  # requis seulement à l'étape 8

    ok &= check_project()
    ok &= smoke_test_yolo(device)

    print("\n" + ("✅ Environnement prêt." if ok else "❌ Des vérifications ont échoué."))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())