import ChartCard from "./ChartCard.jsx";
import { formatCompact, formatPct } from "../utils/format.js";

function riskLevel(simulationRow, frontierPoint) {
  const overlap = Number(simulationRow?.selection_overlap_at_k ?? 0);
  const delta = Number(simulationRow?.comparison_minus_baseline ?? 0);
  const lift = Number(frontierPoint?.lift_at_k ?? 0);
  if (delta > 0 && overlap >= 0.9 && lift >= 0.7) return "Low";
  if (delta >= 0 && lift >= 0.5) return "Medium";
  return "High";
}

function recommendation(simulationRow, frontierPoint) {
  const delta = Number(simulationRow?.comparison_minus_baseline ?? 0);
  const netBenefit = Number(frontierPoint?.net_benefit_at_k ?? 0);
  if (netBenefit <= 0) return "Hold budget steady and review campaign economics";
  if (delta > 0) return "Use the cost-aware targeting policy at the selected budget";
  return "Use the current budget, but monitor overlap and incremental gains closely";
}

export default function DecisionPanel({ budgetPct, simulationRow, frontierPoint, targetedCountOverride = null }) {
  const baselineNetBenefit = Number(simulationRow?.baseline_net_benefit_at_k ?? 0);
  const comparisonNetBenefit = Number(simulationRow?.comparison_net_benefit_at_k ?? 0);
  const netBenefit =
    simulationRow?.comparison_net_benefit_at_k != null
      ? comparisonNetBenefit
      : Number(frontierPoint?.net_benefit_at_k ?? 0);
  const deltaAbsolute = simulationRow?.comparison_minus_baseline;
  const deltaPct =
    baselineNetBenefit !== 0 && simulationRow?.comparison_net_benefit_at_k != null
      ? (comparisonNetBenefit - baselineNetBenefit) / Math.abs(baselineNetBenefit)
      : null;
  const targetedCount = targetedCountOverride ?? frontierPoint?.targeted_count ?? "n/a";

  return (
    <ChartCard title="Decision Card" eyebrow="What should we do?">
      <div className="decision-panel">
        <div className="decision-recommendation">
          <div className="decision-label">Recommendation</div>
          <div className="decision-headline">{recommendation(simulationRow, frontierPoint)}</div>
        </div>
        <div className="decision-grid">
          <div className="decision-item">
            <div className="decision-label">Expected Impact</div>
            <div className="decision-value">Net benefit: {formatCompact(netBenefit)}</div>
            <div className="decision-subvalue">Customers targeted: {targetedCount}</div>
            <div className="decision-subvalue">
              Delta vs baseline:{" "}
              {deltaAbsolute == null
                ? deltaPct === null
                  ? "n/a"
                  : formatPct(deltaPct)
                : `${formatCompact(deltaAbsolute)}${deltaPct === null ? "" : ` (${formatPct(deltaPct)})`}`}
            </div>
          </div>
          <div className="decision-item">
            <div className="decision-label">Tradeoffs</div>
            <div className="decision-value">Selection overlap: {formatPct(simulationRow?.selection_overlap_at_k)}</div>
            <div className="decision-subvalue">Risk level: {riskLevel(simulationRow, frontierPoint)}</div>
            <div className="decision-subvalue">Budget: {budgetPct}%</div>
          </div>
          <div className="decision-item">
            <div className="decision-label">Notes</div>
            <div className="decision-subvalue">
              The economics view is assumption-driven. Net benefit and overlap come from saved scored outputs, not observed treatment effects.
            </div>
          </div>
        </div>
      </div>
    </ChartCard>
  );
}
