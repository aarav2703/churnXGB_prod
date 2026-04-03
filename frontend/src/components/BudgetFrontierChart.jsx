import { useMemo, useState } from "react";
import ChartCard from "./ChartCard.jsx";
import { EmptyState } from "./StatusState.jsx";
import { formatCompact, formatPct } from "../utils/format.js";

function toNumber(value) {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : 0;
}

function makeScale(domainMin, domainMax, rangeMin, rangeMax) {
  const span = Math.max(domainMax - domainMin, 1e-6);
  return (value) => rangeMin + ((value - domainMin) / span) * (rangeMax - rangeMin);
}

function starPoints(cx, cy, outer, inner) {
  const values = [];
  for (let index = 0; index < 10; index += 1) {
    const angle = (-Math.PI / 2) + (index * Math.PI) / 5;
    const radius = index % 2 === 0 ? outer : inner;
    values.push(`${cx + Math.cos(angle) * radius},${cy + Math.sin(angle) * radius}`);
  }
  return values.join(" ");
}

const metricModes = {
  net_benefit: {
    label: "Net benefit",
    valueKey: "net_benefit_at_k",
    baselineKey: "net_benefit_at_k",
    selectedLabel: "Net benefit at selected budget",
  },
  value_at_risk: {
    label: "Value at risk",
    valueKey: "value_at_risk",
    baselineKey: "value_at_risk",
    selectedLabel: "Value at selected budget",
  },
  delta_vs_baseline: {
    label: "Delta vs baseline",
    valueKey: "delta_vs_baseline",
    baselineKey: null,
    selectedLabel: "Delta vs baseline",
  },
};

export default function BudgetFrontierChart({
  selectedRows,
  baselineRows,
  selectedBudget,
  onSelectBudget,
  selectedPolicy,
}) {
  const [metricMode, setMetricMode] = useState("net_benefit");
  const [hoveredBudget, setHoveredBudget] = useState(null);

  if (!selectedRows?.length) {
    return (
      <ChartCard title="Budget Frontier" eyebrow="Flagship decision chart">
        <EmptyState
          title="Budget frontier unavailable"
          message="Run training and scoring so the saved frontier can show how policy value changes with budget."
        />
      </ChartCard>
    );
  }

  const width = 760;
  const height = 360;
  const padding = { top: 30, right: 26, bottom: 56, left: 76 };
  const selectedSorted = [...selectedRows].sort((left, right) => toNumber(left.budget_k) - toNumber(right.budget_k));
  const baselineSorted = [...baselineRows].sort((left, right) => toNumber(left.budget_k) - toNumber(right.budget_k));
  const baselineByBudget = baselineSorted.reduce(
    (acc, row) => ({ ...acc, [toNumber(row.budget_k).toFixed(4)]: row }),
    {},
  );

  const preparedRows = selectedSorted.map((row) => {
    const budgetKey = toNumber(row.budget_k).toFixed(4);
    const baseline = baselineByBudget[budgetKey];
    const delta = baseline ? toNumber(row.net_benefit_at_k) - toNumber(baseline.net_benefit_at_k) : 0;
    return {
      ...row,
      delta_vs_baseline: delta,
      pct_improvement: baseline && toNumber(baseline.net_benefit_at_k) !== 0
        ? delta / Math.abs(toNumber(baseline.net_benefit_at_k))
        : null,
    };
  });

  const activeMode = metricModes[metricMode];
  const values = [
    ...preparedRows.map((row) => toNumber(row[activeMode.valueKey])),
    ...(activeMode.baselineKey ? baselineSorted.map((row) => toNumber(row[activeMode.baselineKey])) : []),
  ];
  const x = makeScale(0, Math.max(...preparedRows.map((row) => toNumber(row.budget_k)), 0.2), padding.left, width - padding.right);
  const y = makeScale(Math.min(...values, 0), Math.max(...values, 1), height - padding.bottom, padding.top);

  const selectedPoint = preparedRows.find((row) => Math.abs(toNumber(row.budget_k) - selectedBudget) < 1e-6) ?? preparedRows[0];
  const hoveredPoint = preparedRows.find((row) => Math.abs(toNumber(row.budget_k) - hoveredBudget) < 1e-6) ?? selectedPoint;
  const bestPoint = preparedRows.reduce((best, row) => (
    toNumber(row.net_benefit_at_k) > toNumber(best.net_benefit_at_k) ? row : best
  ), preparedRows[0]);
  const baselinePoint = baselineByBudget[toNumber(hoveredPoint.budget_k).toFixed(4)];

  const selectedPath = preparedRows
    .map((row, index) => `${index === 0 ? "M" : "L"} ${x(toNumber(row.budget_k))} ${y(toNumber(row[activeMode.valueKey]))}`)
    .join(" ");
  const baselinePath = activeMode.baselineKey
    ? baselineSorted
      .map((row, index) => `${index === 0 ? "M" : "L"} ${x(toNumber(row.budget_k))} ${y(toNumber(row[activeMode.baselineKey]))}`)
      .join(" ")
    : "";

  const takeaway = `Takeaway: ${selectedPolicy.replaceAll("_", " ")} delivers ${formatCompact(selectedPoint.net_benefit_at_k)} at ${formatPct(selectedPoint.budget_k, 0)} budget, ${hoveredPoint.delta_vs_baseline >= 0 ? "adding" : "losing"} ${formatCompact(Math.abs(hoveredPoint.delta_vs_baseline))} vs baseline.`;

  return (
    <ChartCard
      title="Budget Frontier"
      eyebrow="The core decision engine"
      className="span-12"
      actions={
        <div className="panel-actions-inline">
          {Object.entries(metricModes).map(([key, item]) => (
            <button
              key={key}
              type="button"
              className={metricMode === key ? "mini-toggle active" : "mini-toggle"}
              onClick={() => setMetricMode(key)}
            >
              {item.label}
            </button>
          ))}
        </div>
      }
    >
      <div className="chart-takeaway">{takeaway}</div>
      <div className="frontier-chart-shell">
        <svg viewBox={`0 0 ${width} ${height}`} className="line-chart interactive frontier-chart">
          <line x1={padding.left} x2={width - padding.right} y1={height - padding.bottom} y2={height - padding.bottom} className="axis-line" />
          <line x1={padding.left} x2={padding.left} y1={padding.top} y2={height - padding.bottom} className="axis-line" />

          {[0.05, 0.1, 0.2].filter((tick) => tick <= Math.max(...preparedRows.map((row) => toNumber(row.budget_k)), 0.2)).map((tick) => (
            <g key={tick}>
              <line x1={x(tick)} x2={x(tick)} y1={padding.top} y2={height - padding.bottom} className="chart-gridline" />
              <text x={x(tick)} y={height - 18} className="axis-label centered">
                {formatPct(tick, 0)}
              </text>
            </g>
          ))}

          {[0, 0.33, 0.66, 1].map((ratio) => {
            const value = Math.min(...values, 0) + (Math.max(...values, 1) - Math.min(...values, 0)) * ratio;
            return (
              <g key={ratio}>
                <line x1={padding.left} x2={width - padding.right} y1={y(value)} y2={y(value)} className="chart-gridline" />
                <text x={16} y={y(value) + 4} className="axis-label">
                  {formatCompact(value)}
                </text>
              </g>
            );
          })}

          {activeMode.baselineKey ? (
            <path d={baselinePath} fill="none" stroke="#7b8794" strokeWidth="3" strokeDasharray="10 7" />
          ) : null}
          <path d={selectedPath} fill="none" stroke="#0f766e" strokeWidth="4" />

          {preparedRows.map((row) => {
            const budgetValue = toNumber(row.budget_k);
            const isSelected = Math.abs(budgetValue - toNumber(selectedPoint.budget_k)) < 1e-6;
            return (
              <circle
                key={`selected-${budgetValue}`}
                cx={x(budgetValue)}
                cy={y(toNumber(row[activeMode.valueKey]))}
                r={isSelected ? 9 : 5}
                fill={isSelected ? "#0f766e" : "#5ab6a8"}
                stroke={isSelected ? "#082f49" : "#ffffff"}
                strokeWidth={isSelected ? "3" : "1.5"}
                onMouseEnter={() => setHoveredBudget(budgetValue)}
                onMouseLeave={() => setHoveredBudget(null)}
                onClick={() => onSelectBudget?.(budgetValue)}
              />
            );
          })}

          {activeMode.baselineKey ? baselineSorted.map((row) => {
            const budgetValue = toNumber(row.budget_k);
            return (
              <rect
                key={`baseline-${budgetValue}`}
                x={x(budgetValue) - 4}
                y={y(toNumber(row[activeMode.baselineKey])) - 4}
                width="8"
                height="8"
                fill="#7b8794"
                stroke="#ffffff"
                strokeWidth="1.5"
                onMouseEnter={() => setHoveredBudget(budgetValue)}
                onMouseLeave={() => setHoveredBudget(null)}
                onClick={() => onSelectBudget?.(budgetValue)}
              />
            );
          }) : null}

          <polygon
            points={starPoints(x(toNumber(bestPoint.budget_k)), y(toNumber(bestPoint[activeMode.valueKey])), 10, 5)}
            fill="#f59e0b"
            stroke="#7c2d12"
            strokeWidth="1.5"
          />

          <line x1={x(toNumber(selectedPoint.budget_k))} x2={x(toNumber(selectedPoint.budget_k))} y1={padding.top} y2={height - padding.bottom} className="selection-line" />

          <text x={Math.min(x(toNumber(selectedPoint.budget_k)) + 14, width - 190)} y={padding.top + 16} className="annotation-label">
            Selected: {formatPct(selectedPoint.budget_k, 0)} {"->"} {formatCompact(selectedPoint[activeMode.valueKey])}
          </text>
          <text x={Math.min(x(toNumber(bestPoint.budget_k)) + 14, width - 190)} y={padding.top + 36} className="annotation-label emphasis">
            Best: {formatPct(bestPoint.budget_k, 0)} {"->"} {formatCompact(bestPoint.net_benefit_at_k)}
          </text>
          {baselinePoint ? (
            <text x={Math.min(x(toNumber(hoveredPoint.budget_k)) + 14, width - 190)} y={padding.top + 56} className="annotation-label">
              {hoveredPoint.delta_vs_baseline >= 0 ? "+" : "-"}
              {formatCompact(Math.abs(hoveredPoint.delta_vs_baseline))} vs baseline
            </text>
          ) : null}
        </svg>

        <div className="frontier-summary frontier-summary-dense">
          <div className="frontier-summary-card">
            <div className="decision-label">{activeMode.selectedLabel}</div>
            <div className="decision-value">{formatCompact(selectedPoint[activeMode.valueKey])}</div>
            <div className="decision-subvalue">{formatPct(selectedPoint.budget_k, 0)} budget</div>
          </div>
          <div className="frontier-summary-card">
            <div className="decision-label">Delta vs baseline</div>
            <div className="decision-value">{formatCompact(hoveredPoint.delta_vs_baseline)}</div>
            <div className="decision-subvalue">
              {hoveredPoint.pct_improvement === null ? "No baseline comparison" : `${formatPct(hoveredPoint.pct_improvement, 1)} improvement`}
            </div>
          </div>
          <div className="frontier-summary-card">
            <div className="decision-label">Hover budget</div>
            <div className="decision-value">{formatPct(hoveredPoint.budget_k, 0)}</div>
            <div className="decision-subvalue">Click to update the whole dashboard.</div>
          </div>
        </div>

        <div className="legend-row">
          <div className="legend-item">
            <span className="legend-swatch" style={{ background: "#0f766e" }} />
            <span>{selectedPolicy} policy</span>
          </div>
          {activeMode.baselineKey ? (
            <div className="legend-item">
              <span className="legend-swatch baseline-swatch" />
              <span>ML baseline</span>
            </div>
          ) : null}
          <div className="legend-item">
            <span className="legend-star">★</span>
            <span>Best budget point</span>
          </div>
        </div>
      </div>
    </ChartCard>
  );
}
