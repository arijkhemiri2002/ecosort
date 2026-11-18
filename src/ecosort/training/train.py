"""Étape DVC `train` : entraînement YOLO avec suivi MLflow.

Usage :
  python -m ecosort.training.train                                   # complet (params.yaml)
  python -m ecosort.training.train --epochs 2 --fraction 0.1 --run-name smoke-test
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import mlflow
from ultralytics import YOLO, settings

from ecosort.config import MODELS_DIR, REPORTS_DIR, ROOT_DIR, get_classes, load_params, resolve
from ecosort.training.mlflow_utils import (
    dvc_out_md5,
    git_commit,
    log_run_artifacts,
    make_epoch_logger,
    setup_mlflow,
)

TRAIN_KEYS = (
    "epochs", "imgsz", "batch", "patience", "device", "workers",
    "optimizer", "lr0", "cos_lr", "close_mosaic", "amp",
)
FINAL_METRICS = {
    "metrics/precision(B)": "precision",
    "metrics/recall(B)": "recall",
    "metrics/mAP50(B)": "mAP50",
    "metrics/mAP50-95(B)": "mAP50_95",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entraînement YOLO EcoSort")
    parser.add_argument("--epochs", type=int, help="surcharge train.epochs")
    parser.add_argument("--batch", type=int, help="surcharge train.batch")
    parser.add_argument("--device", type=str, help="surcharge train.device ('0' ou 'cpu')")
    parser.add_argument("--imgsz", type=int, help="surcharge train.imgsz")
    parser.add_argument("--fraction", type=float, default=1.0, help="part du train set (0-1]")
    parser.add_argument("--run-name", type=str, help="nom du run MLflow")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    params = load_params()
    cfg = dict(params["train"])
    for key in ("epochs", "batch", "device", "imgsz"):
        value = getattr(args, key)
        if value is not None:
            cfg[key] = value

    data_yaml = resolve(params["data"]["processed_dir"]) / "dataset.yaml"
    if not data_yaml.exists():
        raise SystemExit(f"{data_yaml} introuvable : lance d'abord `dvc repro validate_data`.")

    settings.update({"mlflow": False})  # on gère MLflow nous-mêmes (run gardé ouvert)
    tracking_uri = setup_mlflow(params["mlflow"])
    is_smoke = args.fraction < 1.0
    run_name = args.run_name or (
        f"{Path(cfg['model']).stem}-e{cfg['epochs']}-{time.strftime('%Y%m%d-%H%M')}"
    )

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags({
            "git_commit": git_commit(),
            "data_md5": dvc_out_md5("prepare", "data/processed"),
            "smoke_test": str(is_smoke).lower(),
            "stage": "train",
        })
        mlflow.log_params({k: cfg[k] for k in TRAIN_KEYS})
        mlflow.log_params({
            "model": cfg["model"],
            "fraction": args.fraction,
            "seed": params["seed"],
            "classes": ",".join(get_classes()),
        })
        stats_path = REPORTS_DIR / "data_stats.json"
        if stats_path.exists():
            stats = json.loads(stats_path.read_text(encoding="utf-8"))
            mlflow.log_params({f"data_{s}_images": n for s, n in stats["images"].items()})

        model = YOLO(cfg["model"])
        model.add_callback("on_fit_epoch_end", make_epoch_logger())
        model.train(
            data=str(data_yaml),
            epochs=cfg["epochs"],
            imgsz=cfg["imgsz"],
            batch=cfg["batch"],
            patience=cfg["patience"],
            device=cfg["device"],
            workers=cfg["workers"],
            optimizer=cfg["optimizer"],
            lr0=cfg["lr0"],
            cos_lr=cfg["cos_lr"],
            close_mosaic=cfg["close_mosaic"],
            amp=cfg["amp"],
            fraction=args.fraction,
            seed=params["seed"],
            deterministic=True,
            project=str(ROOT_DIR / "runs"),
            name="train",
            exist_ok=True,
            plots=True,
        )

        trainer = model.trainer
        save_dir, best = Path(trainer.save_dir), Path(trainer.best)
        if not best.exists():
            raise SystemExit(f"Poids introuvables : {best}")

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best, MODELS_DIR / "best.pt")

        final = {
            name: round(float(trainer.metrics[key]), 5)
            for key, name in FINAL_METRICS.items()
            if key in trainer.metrics
        }
        final["epochs_trained"] = trainer.epoch + 1
        mlflow.log_metrics({f"best_{k}": float(v) for k, v in final.items()})

        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        metrics_path = REPORTS_DIR / "train_metrics.json"
        metrics_path.write_text(json.dumps(final, indent=2), encoding="utf-8")
        log_run_artifacts(save_dir, data_yaml)
        mlflow.log_artifact(str(MODELS_DIR / "best.pt"), artifact_path="weights")

        run_info = {
            "run_id": run.info.run_id,
            "experiment_id": run.info.experiment_id,
            "run_name": run_name,
            "tracking_uri": tracking_uri,
            "smoke_test": is_smoke,
        }
        (MODELS_DIR / "run_info.json").write_text(json.dumps(run_info, indent=2), encoding="utf-8")

    print(f"\n[OK] Entraînement terminé | run MLflow : {run_name} ({run.info.run_id})")
    print(f"[OK] Métriques val (best.pt) : {final}")
    print(f"[OK] Poids : {MODELS_DIR / 'best.pt'}")


if __name__ == "__main__":
    main()