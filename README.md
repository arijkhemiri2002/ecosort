# ♻️ EcoSort

Détection multi-déchets (YOLO26) avec un pipeline MLOps complet :
DVC · MLflow · ONNX · FastAPI · Streamlit · Docker · GitHub Actions.

## Installation (Windows / Anaconda)

```powershell
conda create -n ecosort python=3.11 -y
conda activate ecosort
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements/dev.txt
pip install -e .
python scripts/check_env.py
```

## Classes
`plastic`, `paper`, `glass`, `metal`, `cigarette`, `other`