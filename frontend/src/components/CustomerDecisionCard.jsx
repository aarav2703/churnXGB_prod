import ChartCard from "./ChartCard.jsx";
import { EmptyState } from "./StatusState.jsx";
import { formatCompact, formatPct } from "../utils/format.js";

export default function CustomerDecisionCard({
  customer,
  explanation,
  rank,
  totalRows,
}) {
  if (!customer) {
    return (
      <ChartCard title="Customer Decision Card" eyebrow="Why this customer was targeted">
        <EmptyState
          title="No customer selected"
          message="Pick a customer-month row from the explorer to inspect the risk, value, and feature drivers."
        />
      </ChartCard>
    );
  }

  const percentile = totalRows && rank ? 1 - (rank - 1) / totalRows : null;
  const positiveDrivers = explanation?.top_positive_contributors?.slice(0, 3) ?? [];

  return (
    <ChartCard title="Customer Decision Card" eyebrow="Risk, value, and action context" className="span-12">
      <div className="customer-decision-grid">
        <div className="customer-hero">
          <div className="decision-label">Selected customer</div>
          <div className="decision-headline">
            {customer.CustomerID} | {customer.invoice_month}
          </div>
          <div className="decision-note">
            This card combines churn risk, value proxy, targeting rank, and the top feature signals behind the decision.
          </div>
        </div>
        <div className="customer-metric">
          <div className="decision-label">Churn probability</div>
          <div className="decision-value">{formatPct(customer.churn_prob)}</div>
        </div>
        <div className="customer-metric">
          <div className="decision-label">Value proxy</div>
          <div className="decision-value">{formatCompact(customer.value_pos)}</div>
        </div>
        <div className="customer-metric">
          <div className="decision-label">Policy net benefit</div>
          <div className="decision-value">{formatCompact(customer.policy_net_benefit)}</div>
        </div>
        <div className="customer-metric">
          <div className="decision-label">Rank percentile</div>
          <div className="decision-value">{percentile === null ? "n/a" : formatPct(percentile, 0)}</div>
        </div>
      </div>
      <div className="customer-driver-strip">
        {positiveDrivers.length ? (
          positiveDrivers.map((driver) => (
            <div key={driver.feature} className="driver-chip">
              <span>{driver.feature}</span>
              <strong>{formatCompact(driver.contribution_logit ?? driver.shap_value)}</strong>
            </div>
          ))
        ) : (
          <div className="decision-note">Feature drivers will appear here when the explanation payload includes them.</div>
        )}
      </div>
    </ChartCard>
  );
}
