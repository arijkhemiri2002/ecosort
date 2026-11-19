"""Étape DVC `evaluate` : seuil de confiance (val) + métriques finales (test) + MLflow.

Usage : python -m ecosort.training.evaluate [--device 0]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from typing import Any

import mlflow
import numpy as np
import requests
from ultralytics import YOLO, settings

from ecosort.config import MODELS_DIR, REPORTS_DIR, ROOT_DIR, get_classes, load_params, resolve
from ecosort.training.mlflow_utils import git_commit

EVAL_DIR = REPORTS_DIR / "eval"
METRIC_KEYS = ("precision", "recall", "mAP50", "mAP50_95")


def best_threshold(box: Any) -> tuple[float, float]:
    """Seuil de confiance qui maximise le F1 moyen (toutes classes) -> (seuil, F1)."""
    f1 = np.asarray(box.f1_curve, dtype=float)  # (nc, 1000)
    px = np.asarray(box.px, dtype=float)        # (1000,) seuils de 0 à 1
    if f1.size == 0:
        return 0.25, 0.0
    mean_f1 = f1.mean(axis=0)
    i = int(mean_f1.argmax())
    return float(px[i]), float(mean_f1[i])


def operating_point(box: Any, conf: float) -> dict[str, float]:
    """Précision, rappel et F1 moyens au seuil `conf` (ce que verra l'utilisateur)."""
    px = np.asarray(box.px, dtype=float)
    i = int(np.abs(px - conf).argmin())
    return {
        "precision": float(np.asarray(box.p_curve, dtype=float).mean(axis=0)[i]),
        "recall": float(np.asarray(box.r_curve, dtype=float).mean(axis=0)[i]),
        "f1": float(np.asarray(box.f1_curve, dtype=float).mean(axis=0)[i]),
    }


def per_class(box: Any, classes: list[str]) -> dict[str, dict[str, float]]:
    """Métriques par classe ; une classe absente du split reste à 0."""
    out = {c: dict.fromkeys(METRIC_KEYS, 0.0) for c in classes}
    for i, cls_idx in enumerate(box.ap_class_index):
        p, r, ap50, ap = box.class_result(i)
        out[classes[int(cls_idx)]] = {
            "precision": round(float(p), 4),
            "recall": round(float(r), 4),
            "mAP50": round(float(ap50), 4),
            "mAP50_95": round(float(ap), 4),
        }
    return out


def run_val(model: YOLO, data_yaml: str, split: str, cfg: dict, imgsz: int, name: str) -> Any:
    return model.val(
        data=data_yaml,
        split=split,
        imgsz=imgsz,
        batch=cfg["batch"],
        conf=cfg["conf"],
        iou=cfg["iou"],
        device=cfg["device"],
        plots=True,
        project=str(ROOT_DIR / "runs"),
        name=name,
        exist_ok=True,
        verbose=False,
    )


def log_to_mlflow(report: dict[str, Any], params: dict[str, Any]) -> None:
    """Complète le run d'entraînement (run_id de run_info.json) avec les métriques de test."""
    info_path = MODELS_DIR / "run_info.json"
    info = json.loads(info_path.read_text(encoding="utf-8")) if info_path.exists() else {}
    uri = (
        os.getenv("MLFLOW_TRACKING_URI")
        or info.get("tracking_uri")
        or params["mlflow"]["tracking_uri"]
    )
    if uri.startswith("http"):
        try:
            requests.get(uri.rstrip("/") + "/health", timeout=5)
        except requests.RequestException:
            print(f"[WARN] MLflow injoignable ({uri}) : métriques non loggées (fichiers OK).")
            return
    mlflow.set_tracking_uri(uri)

    metrics = {f"test_{k}": float(report[k]) for k in METRIC_KEYS}
    metrics["best_conf"] = float(report["best_conf"])
    metrics.update({f"test_op_{k}": float(v) for k, v in report["operating_point"].items()})
    for cls, m in report["per_class"].items():
        metrics[f"test_mAP50_{cls}"] = float(m["mAP50"])

    try:
        if "run_id" not in info:
            raise KeyError("run_id absent de models/run_info.json")
        with mlflow.start_run(run_id=info["run_id"]):
            mlflow.log_metrics(metrics)
            mlflow.set_tag("evaluated_at_commit", git_commit())
            mlflow.log_artifacts(str(EVAL_DIR), artifact_path="evaluation")
            mlflow.log_artifact(str(REPORTS_DIR / "eval_metrics.json"), artifact_path="evaluation")
        print(f"[OK] Métriques ajoutées au run MLflow {info['run_id']} ({uri})")
    except Exception as exc:
        print(f"[WARN] Run d'origine inaccessible ({exc}) : métriques non loggées dans MLflow.")
        print("       Vérifie MLFLOW_TRACKING_USERNAME / MLFLOW_TRACKING_PASSWORD (DagsHub).")


def main() -> None:
    parser = argparse.ArgumentParser(description="Évaluation EcoSort")
    parser.add_argument("--device", type=str, help="surcharge evaluate.device")
    args = parser.parse_args()

    params = load_params()
    cfg = dict(params["evaluate"])
    if args.device:
        cfg["device"] = args.device
    imgsz = params["train"]["imgsz"]
    classes = get_classes()

    weights = MODELS_DIR / "best.pt"
    if not weights.exists():
        raise SystemExit(f"{weights} introuvable : lance `dvc pull -r origin train`.")
    data_yaml = str(resolve(params["data"]["processed_dir"]) / "dataset.yaml")

    settings.update({"mlflow": False})
    model = YOLO(str(weights))

    print(f"[1/2] Choix du seuil sur le split '{cfg['threshold_split']}'...")
    val_metrics = run_val(model, data_yaml, cfg["threshold_split"], cfg, imgsz, "eval_val")
    conf_star, f1_star = best_threshold(val_metrics.box)

    print(f"[2/2] Mesure finale sur le split '{cfg['split']}'...")
    test_metrics = run_val(model, data_yaml, cfg["split"], cfg, imgsz, "eval_test")
    box = test_metrics.box

    report = {
        "split": cfg["split"],
        "precision": round(float(box.mp), 4),
        "recall": round(float(box.mr), 4),
        "mAP50": round(float(box.map50), 4),
        "mAP50_95": round(float(box.map), 4),
        "best_conf": round(conf_star, 3),
        "best_conf_val_f1": round(f1_star, 4),
        "operating_point": {k: round(v, 4) for k, v in operating_point(box, conf_star).items()},
        "per_class": per_class(box, classes),
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "eval_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    if EVAL_DIR.exists():
        shutil.rmtree(EVAL_DIR)
    EVAL_DIR.mkdir(parents=True)
    for png in sorted((ROOT_DIR / "runs" / "eval_test").glob("*.png")):
        shutil.copy2(png, EVAL_DIR / png.name)
    val_f1 = ROOT_DIR / "runs" / "eval_val" / "BoxF1_curve.png"
    if val_f1.exists():
        shutil.copy2(val_f1, EVAL_DIR / "val_BoxF1_curve.png")

    log_to_mlflow(report, params)

    print(f"\nTest ({cfg['split']}) : P={report['precision']} R={report['recall']} "
          f"mAP50={report['mAP50']} mAP50-95={report['mAP50_95']}")
    op = report["operating_point"]
    print(f"Seuil optimal (val) : {report['best_conf']} -> sur test : "
          f"P={op['precision']} R={op['recall']} F1={op['f1']}")
    print(f"\n{'classe':<10} {'P':>7} {'R':>7} {'mAP50':>7} {'mAP50-95':>9}")
    for cls, m in report["per_class"].items():
        print(f"{cls:<10} {m['precision']:>7} {m['recall']:>7} {m['mAP50']:>7} {m['mAP50_95']:>9}")
    print(f"\n[OK] Rapport : {REPORTS_DIR / 'eval_metrics.json'} | figures : {EVAL_DIR}")


if __name__ == "__main__":
    main()