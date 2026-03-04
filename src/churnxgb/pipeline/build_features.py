from __future__ import annotations

from pathlib import Path
import yaml

from churnxgb.data.load import load_raw_csv
from churnxgb.data.clean import clean_transactions
from churnxgb.data.invoices import build_invoice_df
from churnxgb.features.events import build_customer_events
from churnxgb.features.assemble import build_customer_month
from churnxgb.labeling.churn_90d import label_churn_90d
from churnxgb.features.rolling import build_rolling_features
from churnxgb.features.recency import add_recency_features
from churnxgb.features.value import add_customer_value_90d


def main() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    cfg_path = repo_root / "config" / "config.yaml"

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    raw_csv = repo_root / cfg["data"]["raw_csv"]
    horizon_days = int(cfg["label"]["horizon_days"])

    # Load raw -> clean -> invoices
    df_raw = load_raw_csv(raw_csv)
    df_clean = clean_transactions(df_raw)
    invoice_df = build_invoice_df(df_clean)

    # Event table
    event_df = build_customer_events(invoice_df)

    # Customer-month + label
    customer_month = build_customer_month(invoice_df)
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
    interim_dir = repo_root / "data" / "interim"
    processed_dir = repo_root / "data" / "processed"
    interim_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    df_clean.to_parquet(interim_dir / "transactions_clean.parquet", index=False)
    invoice_df.to_parquet(interim_dir / "invoice_df.parquet", index=False)
    event_df.to_parquet(interim_dir / "customer_events.parquet", index=False)

    customer_month.to_parquet(processed_dir / "customer_month.parquet", index=False)
    customer_month_labeled.to_parquet(
        processed_dir / "customer_month_labeled.parquet", index=False
    )
    feature_table.to_parquet(
        processed_dir / "customer_month_features.parquet", index=False
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
