from __future__ import annotations

from pathlib import Path
import yaml

from churnxgb.artifacts import ArtifactPaths
from churnxgb.data.load import load_raw_csv
from churnxgb.data.clean import clean_transactions
from churnxgb.data.invoices import build_invoice_df
from churnxgb.data.validation import (
    require_columns,
    validate_date_order,
    validate_null_thresholds,
    validate_unique_keys,
)
from churnxgb.features.events import build_customer_events
from churnxgb.features.assemble import build_customer_month
from churnxgb.labeling.churn_90d import label_churn_90d
from churnxgb.features.rolling import build_rolling_features
from churnxgb.features.recency import add_recency_features
from churnxgb.features.value import add_customer_value_90d
from churnxgb.utils.io import atomic_write_parquet


def main() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    cfg_path = repo_root / "config" / "config.yaml"

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    artifacts = ArtifactPaths.for_repo(repo_root, cfg)

    raw_csv = repo_root / cfg["data"]["raw_csv"]
    horizon_days = int(cfg["label"]["horizon_days"])

    # Load raw -> clean -> invoices
    df_raw = load_raw_csv(raw_csv)
    require_columns(
        df_raw,
        ["Invoice", "InvoiceDate", "Quantity", "Price", "Customer ID"],
        "raw_transactions",
    )
    df_clean = clean_transactions(df_raw)
    validate_null_thresholds(
        df_clean,
        {"InvoiceDate": 0.0, "CustomerID": 0.0, "Quantity": 0.0, "Price": 0.0},
        "transactions_clean",
    )
    validate_date_order(df_clean, "CustomerID", "InvoiceDate", "transactions_clean")
    invoice_df = build_invoice_df(df_clean)
    validate_unique_keys(invoice_df, ["Invoice", "CustomerID"], "invoice_df")
    validate_date_order(invoice_df, "CustomerID", "InvoiceDate", "invoice_df")

    # Event table
    event_df = build_customer_events(invoice_df)
    validate_unique_keys(event_df, ["CustomerID", "InvoiceDate"], "event_df")
    validate_date_order(event_df, "CustomerID", "InvoiceDate", "event_df")

    # Customer-month + label
    customer_month = build_customer_month(invoice_df)
    validate_unique_keys(customer_month, ["CustomerID", "invoice_month"], "customer_month")
    customer_month_labeled = label_churn_90d(
        customer_month, event_df, horizon_days=horizon_days
    )

    # Rolling historical features (at event grain) and merge onto T
    feat_events = build_rolling_features(event_df)

    feature_table = customer_month_labeled.merge(
        feat_events.rename(columns={"InvoiceDate": "T"}),
        on=["CustomerID", "T"],
        how="left",
        validate="one_to_one",
    )

    # Recency + customer value in next 90 days
    feature_table = add_recency_features(feature_table, event_df)
    feature_table = add_customer_value_90d(
        feature_table, event_df, horizon_days=horizon_days
    )
    validate_unique_keys(feature_table, ["CustomerID", "invoice_month"], "feature_table")
    validate_null_thresholds(
        feature_table,
        {"CustomerID": 0.0, "invoice_month": 0.0, "T": 0.0, "churn_90d": 0.0},
        "feature_table",
    )

    # Basic fills (align with notebook expectations)
    num_cols = [
        "rev_sum_30d",
        "rev_sum_90d",
        "rev_sum_180d",
        "freq_30d",
        "freq_90d",
        "rev_std_90d",
        "return_count_90d",
        "aov_90d",
        "gap_days_prev",
        "customer_value_90d",
    ]
    for c in num_cols:
        if c in feature_table.columns:
            feature_table[c] = feature_table[c].fillna(0.0)

    # Write artifacts
    interim_dir = artifacts.interim_dir
    processed_dir = artifacts.processed_dir
    interim_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    atomic_write_parquet(df_clean, artifacts.transactions_clean_path(), index=False)
    atomic_write_parquet(invoice_df, artifacts.invoice_df_path(), index=False)
    atomic_write_parquet(event_df, artifacts.customer_events_path(), index=False)

    atomic_write_parquet(customer_month, artifacts.customer_month_path(), index=False)
    atomic_write_parquet(
        customer_month_labeled,
        artifacts.customer_month_labeled_path(),
        index=False,
    )
    atomic_write_parquet(
        feature_table,
        artifacts.feature_table_path(),
        index=False,
    )

    # Print sanity checks
    print("=== FEATURE TABLE SHAPE ===")
    print("feature_table:", feature_table.shape)
    print(
        "dup keys (CustomerID, invoice_month):",
        feature_table.duplicated(["CustomerID", "invoice_month"]).sum(),
    )

    print("\n=== FEATURE NA CHECK (selected) ===")
    for c in [
        "rev_sum_90d",
        "freq_90d",
        "rev_std_90d",
        "gap_days_prev",
        "customer_value_90d",
    ]:
        if c in feature_table.columns:
            print(c, "na:", int(feature_table[c].isna().sum()))

    print("\n=== SAMPLE ROW ===")
    print(
        feature_table.head(5)[
            [
                "CustomerID",
                "invoice_month",
                "T",
                "churn_90d",
                "rev_sum_90d",
                "freq_90d",
                "rev_std_90d",
                "return_count_90d",
                "aov_90d",
                "gap_days_prev",
                "customer_value_90d",
            ]
        ]
    )


if __name__ == "__main__":
    main()
