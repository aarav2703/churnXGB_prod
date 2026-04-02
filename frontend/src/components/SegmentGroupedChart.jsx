import { useMemo, useState } from "react";
import ChartCard from "./ChartCard.jsx";
import { EmptyState } from "./StatusState.jsx";
import { formatCompact, formatNumber } from "../utils/format.js";

export default function SegmentGroupedChart({
  title,
  rows,
  metricKey,
  metricLabel,
  selectedSegment,
  onSelectSegment,
}) {
  const [normalized, setNormalized] = useState(false);
  const [sortKey, setSortKey] = useState("net_benefit_at_k");

  const grouped = useMemo(() => {
    const values = rows.reduce((acc, row) => {
      const segment = row.segment_value;
      if (!acc[segment]) acc[segment] = {};
      acc[segment][row.model] = row;
      return acc;
    }, {});
    const items = Object.entries(values).map(([segment, modelRows]) => {
      const sample = Object.values(modelRows)[0];
      const result = {
        segment,
        rows: modelRows,
        score: Number(sample?.[sortKey] ?? 0),
      };
      return result;
    });
    items.sort((a, b) => b.score - a.score);
    return items;
  }, [rows, sortKey]);

  if (!rows?.length) {
    return (
      <ChartCard
        title={title}
        actions={
          <button type="button" className="mini-toggle" onClick={() => setNormalized((value) => !value)}>
            {normalized ? "Show absolute" : "Show normalized"}
          </button>
        }
      >
        <EmptyState title="No segment data" message="The API did not return segment metrics for this view." />
      </ChartCard>
    );
  }

  const allValues = rows.map((row) => Number(row[metricKey] ?? 0));
  const maxValue = Math.max(...allValues, 1e-6);
  const models = [...new Set(rows.map((row) => row.model))];
  const colors = {
    lightgbm: "#1d3557",
    logistic_regression: "#2a9d8f",
    xgboost: "#bc6c25",
  };

  return (
    <ChartCard
      title={title}
      actions={
        <div className="panel-actions-inline">
          <select value={sortKey} onChange={(event) => setSortKey(event.target.value)}>
            <option value="net_benefit_at_k">Sort by net benefit</option>
            <option value="lift_at_k">Sort by lift</option>
          </select>
          <button type="button" className="mini-toggle" onClick={() => setNormalized((value) => !value)}>
            {normalized ? "Show absolute" : "Show normalized"}
          </button>
        </div>
      }
    >
      <div className="segment-chart">
        {grouped.map((group) => (
          <button
            type="button"
            key={group.segment}
            className={selectedSegment === group.segment ? "segment-card active" : "segment-card"}
            onClick={() => onSelectSegment?.(group.segment)}
          >
            <div className="segment-card-title">{group.segment}</div>
            <div className="segment-group-bars">
              {models.map((model) => {
                const row = group.rows[model];
                const value = Number(row?.[metricKey] ?? 0);
                const displayValue = normalized ? value / maxValue : value;
                return (
                  <div key={model} className="segment-bar-row" title={`${model}: ${formatNumber(value, 3)}`}>
                    <span className="segment-bar-label">{model}</span>
                    <div className="segment-bar-track">
                      <div
                        className="segment-bar-fill"
                        style={{
                          width: `${Math.abs(displayValue / Math.max(normalized ? 1 : maxValue, 1e-6)) * 100}%`,
                          background: colors[model] ?? "#5b7083",
                        }}
                      />
                    </div>
                    <span className="segment-bar-value">
                      {normalized ? formatNumber(displayValue, 2) : formatCompact(value)}
                    </span>
                  </div>
                );
              })}
            </div>
            <div className="segment-card-meta">{metricLabel}</div>
          </button>
        ))}
      </div>
    </ChartCard>
  );
}
