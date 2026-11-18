"""Chargement centralisé de la configuration (params.yaml) et résolution des chemins."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

# src/ecosort/config.py -> parents[2] = racine du projet (aussi valable dans Docker : /app)
ROOT_DIR: Path = Path(os.getenv("ECOSORT_ROOT", Path(__file__).resolve().parents[2]))
PARAMS_PATH: Path = ROOT_DIR / "params.yaml"
CONFIGS_DIR: Path = ROOT_DIR / "configs"
MODELS_DIR: Path = ROOT_DIR / "models"
REPORTS_DIR: Path = ROOT_DIR / "reports"


@lru_cache(maxsize=1)
def load_params(path: str | Path = PARAMS_PATH) -> dict[str, Any]:
    """Charge params.yaml une seule fois (mis en cache)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Fichier de paramètres introuvable : {path}")
    with path.open(encoding="utf-8") as f:
        params = yaml.safe_load(f)
    if not isinstance(params, dict):
        raise ValueError(f"params.yaml invalide : {path}")
    return params


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Charge un fichier YAML quelconque (mapping, conseils de tri...)."""
    with resolve(path).open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve(path_str: str | Path) -> Path:
    """Convertit un chemin relatif (depuis params.yaml) en chemin absolu depuis la racine."""
    p = Path(path_str)
    return p if p.is_absolute() else ROOT_DIR / p


def get_classes() -> list[str]:
    """Liste ordonnée des classes (index = id YOLO)."""
    return list(load_params()["classes"])