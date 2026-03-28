from __future__ import annotations

from pathlib import Path
import shutil
import uuid

import pandas as pd
import yaml
from fastapi.testclient import TestClient

from churnxgb.api.app import create_app
from churnxgb.modeling.model_utils import save_model_artifacts


class DummyModel:
    def predict_proba(self, X: pd.DataFrame):
        probs = X["rev_sum_90d"].astype(float) / 100.0
        probs = probs.clip(lower=0.0, upper=1.0)
        return pd.concat([1.0 - probs, probs], axis=1).to_numpy()


ROOT = Path(__file__).resolve().parents[1]


def _make_repo_root() -> Path:
    path = ROOT / "tests_artifacts" / f"api_app_{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_config(repo_root: Path) -> None:
    config_dir = repo_root / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    with open(config_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(
            {
                "eval": {"budgets": [0.05, 0.10]},
                "mlflow": {"tracking_uri": "sqlite:///mlflow.db"},
            },
            f,
            sort_keys=False,
        )


def _feature_cols() -> list[str]:
    return [
        "rev_sum_30d",
        "rev_sum_90d",
        "rev_sum_180d",
        "freq_30d",
        "freq_90d",
        "rev_std_90d",
        "return_count_90d",
        "aov_90d",
        "gap_days_prev",
    ]


def test_api_health_and_predict_flow() -> None:
    repo_root = _make_repo_root()
    _write_config(repo_root)
    save_model_artifacts(repo_root, DummyModel(), _feature_cols(), model_name="churn_xgb_v1")

    try:
        with TestClient(create_app(repo_root)) as client:
            health = client.get("/health")
            assert health.status_code == 200
            assert health.json()["status"] == "ok"

            payload = {
                "rows": [
                    {
                        "CustomerID": 123,
                        "invoice_month": "2010-01",
                        "T": "2010-01-31T00:00:00",
                        "rev_sum_30d": 50.0,
                        "rev_sum_90d": 80.0,
                        "rev_sum_180d": 120.0,
                        "freq_30d": 2.0,
                        "freq_90d": 4.0,
                        "rev_std_90d": 5.0,
                        "return_count_90d": 0.0,
                        "aov_90d": 20.0,
                        "gap_days_prev": 3.0,
                    }
                ]
            }
            response = client.post("/predict", json=payload)

            assert response.status_code == 200
            body = response.json()
            assert list(body[0].keys()) == [
                "CustomerID",
                "invoice_month",
                "T",
                "churn_prob",
                "value_pos",
                "policy_ml",
            ]
            assert body[0]["CustomerID"] == 123
            assert body[0]["invoice_month"] == "2010-01"
    finally:
        shutil.rmtree(repo_root, ignore_errors=True)


def test_api_rejects_training_only_columns() -> None:
    repo_root = _make_repo_root()
    _write_config(repo_root)
    save_model_artifacts(repo_root, DummyModel(), _feature_cols(), model_name="churn_xgb_v1")

    try:
        with TestClient(create_app(repo_root)) as client:
            response = client.post(
                "/predict",
                json={
                    "rows": [
                        {
                            "rev_sum_30d": 50.0,
                            "rev_sum_90d": 80.0,
                            "rev_sum_180d": 120.0,
                            "freq_30d": 2.0,
                            "freq_90d": 4.0,
                            "rev_std_90d": 5.0,
                            "return_count_90d": 0.0,
                            "aov_90d": 20.0,
                            "gap_days_prev": 3.0,
                            "churn_90d": 1,
                        }
                    ]
                },
            )

            assert response.status_code == 422
            assert "Training-only columns" in response.json()["detail"]
    finally:
        shutil.rmtree(repo_root, ignore_errors=True)
