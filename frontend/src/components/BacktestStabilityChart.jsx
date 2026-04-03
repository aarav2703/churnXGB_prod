import ChartCard from "./ChartCard.jsx";
import { EmptyState } from "./StatusState.jsx";
import { formatCompact, formatNumber } from "../utils/format.js";

function toNumber(value) {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : 0;
}

function makeScale(domainMin, domainMax, rangeMin, rangeMax) {
  const span = Math.max(domainMax - domainMin, 1e-6);
  return (value) => rangeMin + ((value - domainMin) / span) * (rangeMax - rangeMin);
}

export default function BacktestStabilityChart({ rows, modelName }) {
  if (!rows?.length) {
    return (
      <ChartCard title="Backtest Stability" eyebrow="Trust signal view">
        <EmptyState
          title="Backtest view unavailable"
          message="Saved backtest folds were not available for the current run."
        />
      </ChartCard>
    );
  }

  const modelRows = rows.filter((row) => row.model === modelName);
  if (!modelRows.length) {
    return (
      <ChartCard title="Backtest Stability" eyebrow="Trust signal view">
        <EmptyState
          title="No champion backtest rows"
          message="The promoted model does not have saved backtest folds for this budget."
        />
      </ChartCard>
    );
  }

  const width = 760;
  const height = 320;
  const padding = { top: 28, right: 24, bottom: 56, left: 70 };
  const netBenefitValues = modelRows.map((row) => toNumber(row.net_benefit_at_k));
  const minNet = Math.min(...netBenefitValues, 0);
  const maxNet = Math.max(...netBenefitValues, 1);
  const avgNet = netBenefitValues.reduce((sum, value) => sum + value, 0) / Math.max(netBenefitValues.length, 1);
  const deviation = Math.sqrt(
    netBenefitValues.reduce((sum, value) => sum + (value - avgNet) ** 2, 0) / Math.max(netBenefitValues.length, 1),
  );
  const x = makeScale(0, Math.max(modelRows.length - 1, 1), padding.left, width - padding.right);
  const y = makeScale(minNet, maxNet, height - padding.bottom, padding.top);
  const path = modelRows
    .map((row, index) => `${index === 0 ? "M" : "L"} ${x(index)} ${y(toNumber(row.net_benefit_at_k))}`)
    .join(" ");
  const unstableCount = modelRows.filter((row) => Math.abs(toNumber(row.net_benefit_at_k) - avgNet) > deviation).length;
  const takeaway = `Takeaway: ${unstableCount === 0 ? "Business value is stable across saved folds." : `Business value is broadly stable, but ${unstableCount} fold${unstableCount > 1 ? "s" : ""} sit outside the stability band.`}`;

  return (
    <ChartCard title="Backtest Stability" eyebrow="Trust signal view" className="span-12">
      <div className="chart-takeaway">{takeaway}</div>
      <div className="frontier-chart-shell">
        <svg viewBox={`0 0 ${width} ${height}`} className="line-chart interactive frontier-chart">
          <line x1={padding.left} x2={width - padding.right} y1={height - padding.bottom} y2={height - padding.bottom} className="axis-line" />
          <line x1={padding.left} x2={padding.left} y1={padding.top} y2={height - padding.bottom} className="axis-line" />

          <rect
            x={padding.left}
            y={y(avgNet + deviation)}
            width={width - padding.left - padding.right}
            height={Math.max(y(avgNet - deviation) - y(avgNet + deviation), 4)}
            className="stability-band"
          />
          <line x1={padding.left} x2={width - padding.right} y1={y(avgNet)} y2={y(avgNet)} className="threshold-line warn" />
          <text x={width - padding.right} y={y(avgNet) - 8} className="threshold-label">
            stability threshold
          </text>
          <path d={path} fill="none" stroke="#1c5d99" strokeWidth="4" />

          {modelRows.map((row, index) => {
            const unstable = Math.abs(toNumber(row.net_benefit_at_k) - avgNet) > deviation;
            return (
              <g key={row.fold}>
                {unstable ? (
                  <rect
                    x={Math.max(x(index) - 24, padding.left)}
                    y={padding.top}
                    width="48"
                    height={height - padding.top - padding.bottom}
                    className="unstable-column"
                  />
                ) : null}
                <circle
                  cx={x(index)}
                  cy={y(toNumber(row.net_benefit_at_k))}
                  r={unstable ? 7 : 5}
                  fill={unstable ? "#c05621" : "#1c5d99"}
                  stroke="#ffffff"
                  strokeWidth="2"
                />
                <text x={x(index)} y={height - 18} className="axis-label centered">
                  {row.fold}
                </text>
              </g>
            );
          })}
        </svg>

        <div className="frontier-summary frontier-summary-dense">
          <div className="frontier-summary-card">
            <div className="decision-label">Average net benefit</div>
            <div className="decision-value">{formatCompact(avgNet)}</div>
            <div className="decision-subvalue">{modelName} across saved folds</div>
          </div>
          <div className="frontier-summary-card">
            <div className="decision-label">Stability band</div>
            <div className="decision-value">{formatCompact(deviation)}</div>
            <div className="decision-subvalue">Higher values mean the recommendation is less consistent over time.</div>
          </div>
          <div className="frontier-summary-card">
            <div className="decision-label">Unstable periods</div>
            <div className="decision-value">{unstableCount}</div>
            <div className="decision-subvalue">Highlighted in amber on the chart.</div>
          </div>
        </div>
      </div>
    </ChartCard>
  );
}
