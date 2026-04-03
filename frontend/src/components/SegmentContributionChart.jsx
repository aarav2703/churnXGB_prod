import { useMemo, useState } from "react";
import ChartCard from "./ChartCard.jsx";
import { EmptyState } from "./StatusState.jsx";
import { formatCompact, formatNumber, formatPct } from "../utils/format.js";

function toNumber(value) {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : 0;
}

export default function SegmentContributionChart({
  rows,
  selectedSegment,
  onSelectSegment,
  segmentType,
}) {
  const [viewMode, setViewMode] = useState("contribution");

  const chosenRows = useMemo(
    () => rows.filter((row) => row.policy === "policy_net_benefit"),
    [rows],
  );
  const baselineBySegment = useMemo(
    () => rows.filter((row) => row.policy === "policy_ml").reduce((acc, row) => ({ ...acc, [row.segment_value]: row }), {}),
    [rows],
  );

  if (!chosenRows.length) {
    return (
      <ChartCard
        title="Segment Contribution"
        eyebrow="Where value comes from"
        actions={
          <div className="panel-actions-inline">
            <button type="button" className="mini-toggle active">Contribution</button>
            <button type="button" className="mini-toggle">Policy shift</button>
          </div>
        }
      >
        <EmptyState
          title="Segment view unavailable"
          message="Segment metrics are not available for the current model, budget, and segment family yet."
        />
      </ChartCard>
    );
  }

  const enriched = chosenRows
    .map((row) => {
      const baseline = baselineBySegment[row.segment_value];
      const netBenefit = toNumber(row.net_benefit_at_k);
      const deltaVsBaseline = baseline ? netBenefit - toNumber(baseline.net_benefit_at_k) : 0;
      return {
        ...row,
        netBenefit,
        deltaVsBaseline,
        contributionShare: 0,
      };
    })
    .sort((left, right) => right.netBenefit - left.netBenefit);

  const positiveTotal = enriched.reduce((sum, row) => sum + Math.max(row.netBenefit, 0), 0);
  const rowsWithShare = enriched.map((row) => ({
    ...row,
    contributionShare: positiveTotal > 0 ? Math.max(row.netBenefit, 0) / positiveTotal : 0,
  }));

  const insightSource = rowsWithShare[0];
  const negativeSegment = rowsWithShare.find((row) => row.netBenefit < 0);
  const maxAbs = Math.max(
    ...rowsWithShare.map((row) => Math.max(Math.abs(row.netBenefit), Math.abs(row.deltaVsBaseline), Math.abs(toNumber(baselineBySegment[row.segment_value]?.net_benefit_at_k)))),
    1e-6,
  );

  const takeaway = insightSource
    ? `${insightSource.segment_value} drives about ${formatPct(insightSource.contributionShare, 0)} of positive net benefit${negativeSegment ? `, while ${negativeSegment.segment_value} is negative ROI and should be deprioritized.` : "."}`
    : "Segment value is concentrated in a small subset of the portfolio.";

  return (
    <ChartCard
      title="Segment Contribution"
      eyebrow={`Ranked ${segmentType.replace("segment_", "").replaceAll("_", " ")}`}
      className="span-12"
      actions={
        <div className="panel-actions-inline">
          <button
            type="button"
            className={viewMode === "contribution" ? "mini-toggle active" : "mini-toggle"}
            onClick={() => setViewMode("contribution")}
          >
            Contribution
          </button>
          <button
            type="button"
            className={viewMode === "policy_shift" ? "mini-toggle active" : "mini-toggle"}
            onClick={() => setViewMode("policy_shift")}
          >
            Policy shift
          </button>
        </div>
      }
    >
      <div className="chart-takeaway">Takeaway: {takeaway}</div>
      <div className="segment-contribution-list">
        {rowsWithShare.map((row) => {
          const baseline = baselineBySegment[row.segment_value];
          const active = selectedSegment === row.segment_value;
          const isNegative = row.netBenefit < 0;
          const shareWidth = `${Math.max((Math.abs(row.netBenefit) / maxAbs) * 100, 4)}%`;
          const baselineWidth = `${Math.max((Math.abs(toNumber(baseline?.net_benefit_at_k)) / maxAbs) * 100, 4)}%`;

          return (
            <button
              key={`${row.segment_value}-${viewMode}`}
              type="button"
              className={active ? "segment-contribution-row active" : "segment-contribution-row"}
              onClick={() => onSelectSegment?.(row.segment_value)}
            >
              <div>
                <div className="segment-contribution-title">{row.segment_value}</div>
                <div className="segment-contribution-meta">
                  {viewMode === "contribution"
                    ? `${formatPct(row.contributionShare, 0)} of positive value | lift ${formatNumber(row.lift_at_k, 2)}`
                    : `Baseline ${formatCompact(baseline?.net_benefit_at_k)} -> policy ${formatCompact(row.netBenefit)}`}
                </div>
              </div>

              {viewMode === "contribution" ? (
                <div className="segment-contribution-bar">
                  <div
                    className={isNegative ? "segment-contribution-fill negative" : "segment-contribution-fill"}
                    style={{ width: shareWidth }}
                  />
                </div>
              ) : (
                <div className="segment-shift-track">
                  <div className="segment-shift-lane baseline" style={{ width: baselineWidth }} />
                  <div className={isNegative ? "segment-shift-lane policy negative" : "segment-shift-lane policy"} style={{ width: shareWidth }} />
                </div>
              )}

              <div className="segment-contribution-values">
                <div>{formatCompact(row.netBenefit)}</div>
                <div className={row.deltaVsBaseline >= 0 ? "segment-improve" : "segment-warning"}>
                  {row.deltaVsBaseline >= 0 ? "+" : "-"}
                  {formatCompact(Math.abs(row.deltaVsBaseline))} vs baseline
                </div>
              </div>
            </button>
          );
        })}
      </div>
    </ChartCard>
  );
}
