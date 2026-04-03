from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


def load_repo_config(repo_root: Path) -> dict:
    cfg_path = repo_root / "config" / "config.yaml"
    if not cfg_path.exists():
        return {}
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_runtime_root(repo_root: Path, cfg: dict | None = None) -> Path:
    use_cfg = cfg if cfg is not None else load_repo_config(repo_root)
    runtime_rel = (use_cfg.get("runtime", {}) or {}).get("root")
    runtime_dir = runtime_rel or ".runtime"
    return (repo_root / runtime_dir).resolve()


@dataclass(frozen=True)
class ArtifactPaths:
    repo_root: Path
    runtime_root: Path

    @classmethod
    def for_repo(cls, repo_root: Path, cfg: dict | None = None) -> "ArtifactPaths":
        return cls(
            repo_root=repo_root,
            runtime_root=resolve_runtime_root(repo_root, cfg),
        )

    @property
    def data_dir(self) -> Path:
        return self.runtime_root / "data"

    @property
    def interim_dir(self) -> Path:
        return self.data_dir / "interim"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def models_dir(self) -> Path:
        return self.runtime_root / "models"

    @property
    def registry_dir(self) -> Path:
        return self.models_dir / "registry"

    @property
    def promoted_dir(self) -> Path:
        return self.models_dir / "promoted"

    @property
    def reports_dir(self) -> Path:
        return self.runtime_root / "reports"

    @property
    def evaluation_dir(self) -> Path:
        return self.reports_dir / "evaluation"

    @property
    def figures_dir(self) -> Path:
        return self.reports_dir / "figures"

    @property
    def monitoring_dir(self) -> Path:
        return self.reports_dir / "monitoring"

    @property
    def outputs_dir(self) -> Path:
        return self.runtime_root / "outputs"

    @property
    def predictions_dir(self) -> Path:
        return self.outputs_dir / "predictions"

    @property
    def targets_dir(self) -> Path:
        return self.outputs_dir / "targets"

    def feature_table_path(self) -> Path:
        return self.processed_dir / "customer_month_features.parquet"

    def customer_month_path(self) -> Path:
        return self.processed_dir / "customer_month.parquet"

    def customer_month_labeled_path(self) -> Path:
        return self.processed_dir / "customer_month_labeled.parquet"

    def transactions_clean_path(self) -> Path:
        return self.interim_dir / "transactions_clean.parquet"

    def invoice_df_path(self) -> Path:
        return self.interim_dir / "invoice_df.parquet"

    def customer_events_path(self) -> Path:
        return self.interim_dir / "customer_events.parquet"

    def model_registry_dir(self, model_name: str) -> Path:
        return self.registry_dir / model_name

    def promotion_record_path(self) -> Path:
        return self.promoted_dir / "production.json"

    def model_comparison_path(self) -> Path:
        return self.reports_dir / "model_comparison.csv"

    def training_manifest_path(self) -> Path:
        return self.reports_dir / "training_manifest.json"

    def predictions_path(self, split_name: str = "all") -> Path:
        return self.predictions_dir / f"predictions_{split_name}.parquet"

    def inference_predictions_path(self) -> Path:
        return self.predictions_dir / "predictions_inference.parquet"

    def target_list_path(self, split_name: str, budget_pct: int) -> Path:
        return self.targets_dir / f"targets_{split_name}_k{budget_pct:02d}.parquet"
