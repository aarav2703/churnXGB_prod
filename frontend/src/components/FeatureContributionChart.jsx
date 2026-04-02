import ChartCard from "./ChartCard.jsx";
import { EmptyState } from "./StatusState.jsx";
import { formatNumber } from "../utils/format.js";

export default function FeatureContributionChart({ title, rows, tone = "positive" }) {
  if (!rows?.length) {
    return (
      <ChartCard title={title}>
        <EmptyState title="No contributions returned" message="This customer explanation did not include feature contribution details." />
      </ChartCard>
    );
  }

  const maxValue = Math.max(...rows.map((row) => Math.abs(Number(row.contribution ?? 0))), 1e-6);
  return (
    <ChartCard title={title}>
      <div className="bar-chart-list">
        {rows.map((row) => (
          <div key={`${title}-${row.feature}`} className="bar-chart-row">
            <div className="bar-chart-header">
              <span>{row.feature}</span>
              <span>{formatNumber(row.contribution, 3)}</span>
            </div>
            <div className="bar-chart-track">
              <div
                className={`bar-chart-fill ${tone}`}
                style={{ width: `${(Math.abs(Number(row.contribution ?? 0)) / maxValue) * 100}%` }}
              />
            </div>
            {row.featureValue !== undefined ? (
              <div className="contribution-caption">Feature value: {formatNumber(row.featureValue, 2)}</div>
            ) : null}
          </div>
        ))}
      </div>
    </ChartCard>
  );
}
