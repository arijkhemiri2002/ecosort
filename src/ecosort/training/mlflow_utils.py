"""Utilitaires MLflow partagés par l'entraînement, l'évaluation et l'export."""

from __future__ import annotations

import os
import re
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import mlflow
import requests
import yaml

from ecosort.config import REPORTS_DIR, ROOT_DIR


def sanitize(name: str) -> str:
    """MLflow n'accepte que [a-zA-Z0-9_-. /:] dans les noms de métriques."""
    return re.sub(r"[^\w\-./: ]", "", name)


def setup_mlflow(cfg: dict[str, Any]) -> str:
    """Configure l'URI et l'expérience MLflow. Les variables d'environnement sont prioritaires."""
    uri = os.getenv("MLFLOW_TRACKING_URI", cfg["tracking_uri"])
    if uri.startswith("http"):
        try:
            requests.get(uri.rstrip("/") + "/health", timeout=5)
        except requests.RequestException as exc:
            raise SystemExit(
                f"Serveur MLflow injoignable : {uri}\n"
                "Démarre-le dans un autre terminal : .\\scripts\\start_mlflow.ps1"
            ) from exc
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", cfg["experiment_name"]))
    return uri


def git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT_DIR, capture_output=True, text=True, check=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def dvc_out_md5(stage: str, out_path: str, lock_path: Path | None = None) -> str:
    """Hash DVC d'une sortie de stage (lu dans dvc.lock) pour relier run et données."""
    lock_path = lock_path or ROOT_DIR / "dvc.lock"
    if not lock_path.exists():
        return "unknown"
    lock = yaml.safe_load(lock_path.read_text(encoding="utf-8")) or {}
    for out in lock.get("stages", {}).get(stage, {}).get("outs", []):
        if out.get("path") == out_path:
            return str(out.get("md5", "unknown"))
    return "unknown"


def make_epoch_logger() -> Callable[[Any], None]:
    """Callback Ultralytics `on_fit_epoch_end` : log des métriques par époque dans MLflow."""
    state = {"last_epoch": -1}

    def on_fit_epoch_end(trainer: Any) -> None:
        # Ultralytics rappelle ce callback après la validation finale du best.pt :
        # on ignore ce second appel pour ne pas dupliquer la dernière époque.
        if trainer.epoch == state["last_epoch"]:
            return
        state["last_epoch"] = trainer.epoch
        metrics = {
            **trainer.label_loss_items(trainer.tloss, prefix="train"),
            **trainer.metrics,
            **trainer.lr,
        }
        clean = {sanitize(k): float(v) for k, v in metrics.items() if v is not None}
        mlflow.log_metrics(clean, step=trainer.epoch)

    return on_fit_epoch_end


def log_run_artifacts(save_dir: Path, data_yaml: Path) -> None:
    """Courbes, matrice de confusion, CSV des résultats et configuration."""
    for pattern in ("*.png", "*.jpg", "*.csv", "args.yaml"):
        for file in sorted(save_dir.glob(pattern)):
            mlflow.log_artifact(str(file), artifact_path="training")
    mlflow.log_artifact(str(data_yaml), artifact_path="config")
    for extra in (ROOT_DIR / "params.yaml", REPORTS_DIR / "data_stats.json"):
        if extra.exists():
            mlflow.log_artifact(str(extra), artifact_path="config")