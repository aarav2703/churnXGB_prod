import { useMemo, useState } from "react";
import ChartCard from "./ChartCard.jsx";
import { EmptyState } from "./StatusState.jsx";
import { formatCompact, formatPct } from "../utils/format.js";

function toNumber(value) {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : 0;
}

export default function AssumptionSensitivityView({
  frontierRows,
  selectedBudget,
  onBudgetChange,
  targetRows,
}) {
  const [interventionCost, setInterventionCost] = useState(25);
  const [successRate, setSuccessRate] = useState(0.2);

  const selectedRow = frontierRows.find((row) => Math.abs(toNumber(row.budget_k) - selectedBudget) < 1e-6);

  const scenario = useMemo(() => {
    if (!targetRows?.length) return null;
    const adjustedRetainedValue = targetRows.reduce((sum, row) => {
      const baseSuccessRate = Math.max(toNumber(row.assumed_success_rate_customer), 1e-6);
      const scale = successRate / baseSuccessRate;
      return sum + toNumber(row.expected_retained_value) * scale;
    }, 0);
    const totalCost = targetRows.length * interventionCost;
    const netBenefit = adjustedRetainedValue - totalCost;
    return {
      targetedCustomers: targetRows.length,
      adjustedRetainedValue,
      totalCost,
      netBenefit,
    };
  }, [interventionCost, successRate, targetRows]);

  if (!frontierRows?.length) {
    return (
      <ChartCard title="Assumption Sensitivity" eyebrow="Stress-test the recommendation">
        <EmptyState
          title="No scenario data"
          message="This view needs saved frontier and target artifacts before it can estimate sensitivity."
        />
      </ChartCard>
    );
  }

  return (
    <ChartCard title="Assumption Sensitivity" eyebrow="Assumption-driven scenario explorer" className="span-12">
      <div className="sensitivity-layout">
        <div className="sensitivity-controls">
          <label className="slider-field">
            <span>Budget</span>
            <input
              type="range"
              min="5"
              max="20"
              step="5"
              value={Math.round(selectedBudget * 100)}
              onChange={(event) => onBudgetChange?.(Number(event.target.value) / 100)}
            />
            <strong>{formatPct(selectedBudget, 0)}</strong>
          </label>
          <label className="slider-field">
            <span>Intervention cost per customer</span>
            <input
              type="range"
              min="5"
              max="100"
              step="5"
              value={interventionCost}
              onChange={(event) => setInterventionCost(Number(event.target.value))}
            />
            <strong>{formatCompact(interventionCost)}</strong>
          </label>
          <label className="slider-field">
            <span>Success rate</span>
            <input
              type="range"
              min="0.05"
              max="0.6"
              step="0.01"
              value={successRate}
              onChange={(event) => setSuccessRate(Number(event.target.value))}
            />
            <strong>{formatPct(successRate, 0)}</strong>
          </label>
        </div>
        <div className="sensitivity-summary">
          <div className="decision-item">
            <div className="decision-label">Scenario net benefit</div>
            <div className="decision-value">{formatCompact(scenario?.netBenefit)}</div>
            <div className="decision-subvalue">
              Estimated from the currently loaded target list and saved expected retained value fields.
            </div>
          </div>
          <div className="decision-item">
            <div className="decision-label">Selected frontier point</div>
            <div className="decision-value">{formatCompact(selectedRow?.net_benefit_at_k)}</div>
            <div className="decision-subvalue">
              Current saved frontier net benefit at {formatPct(selectedBudget, 0)} budget.
            </div>
          </div>
          <div className="decision-item">
            <div className="decision-label">Recommendation shift</div>
            <div className="decision-value">
              {scenario && selectedRow && scenario.netBenefit >= toNumber(selectedRow.net_benefit_at_k) * 0.9
                ? "Recommendation holds"
                : "Review assumptions"}
            </div>
            <div className="decision-subvalue">
              This view is assumption-driven and not a causal uplift estimate.
            </div>
          </div>
        </div>
      </div>
    </ChartCard>
  );
}
