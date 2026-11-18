"""Tests unitaires des utilitaires MLflow (sans GPU ni serveur)."""

from types import SimpleNamespace

from ecosort.training import mlflow_utils
from ecosort.training.mlflow_utils import dvc_out_md5, make_epoch_logger, sanitize


def test_sanitize_removes_parentheses():
    assert sanitize("metrics/mAP50-95(B)") == "metrics/mAP50-95B"
    assert sanitize("lr/pg0") == "lr/pg0"


def test_dvc_out_md5_reads_lock(tmp_path):
    lock = tmp_path / "dvc.lock"
    lock.write_text(
        "schema: '2.0'\nstages:\n  prepare:\n    outs:\n"
        "    - path: data/processed\n      md5: abc123.dir\n",
        encoding="utf-8",
    )
    assert dvc_out_md5("prepare", "data/processed", lock) == "abc123.dir"
    assert dvc_out_md5("prepare", "autre", lock) == "unknown"
    assert dvc_out_md5("prepare", "data/processed", tmp_path / "absent.lock") == "unknown"


def test_epoch_logger_logs_once_per_epoch(monkeypatch):
    calls = []
    monkeypatch.setattr(
        mlflow_utils.mlflow, "log_metrics", lambda m, step: calls.append((step, m))
    )
    trainer = SimpleNamespace(
        epoch=0,
        tloss=None,
        metrics={"metrics/mAP50(B)": 0.5},
        lr={"lr/pg0": 0.01},
        label_loss_items=lambda loss, prefix: {f"{prefix}/box_loss": 1.2},
    )
    logger = make_epoch_logger()
    logger(trainer)
    logger(trainer)  # même époque (validation finale) : ignorée
    trainer.epoch = 1
    logger(trainer)

    assert [step for step, _ in calls] == [0, 1]
    assert calls[0][1] == {"train/box_loss": 1.2, "metrics/mAP50B": 0.5, "lr/pg0": 0.01}