import { useState } from "react";
import ChartCard from "./ChartCard.jsx";

const fallbackSections = {
  what_this_shows: "This explanation reflects the current selection: budget, policy, segment, customer, and chart state.",
  why_it_matters: "Use the guided actions to get a grounded readout of the decision, not a generic chat response.",
  what_to_do: "Choose one action to explain the current chart, customer, or risk view.",
  caution: "Economic outputs are assumption-driven and do not estimate causal uplift.",
};

const labels = {
  what_this_shows: "What this shows",
  why_it_matters: "Why it matters",
  what_to_do: "What to do",
  caution: "Caution",
};

export default function ExplanationPanel({
  title,
  subtitle,
  actions,
  explanation,
  loading = false,
  activeAction = "",
  onRunAction,
}) {
  const [expanded, setExpanded] = useState(false);
  const sections = explanation?.sections ?? fallbackSections;
  const summary = sections.why_it_matters ?? fallbackSections.why_it_matters;
  const actionHint = activeAction ? actions.find((action) => action.id === activeAction)?.label : null;

  return (
    <ChartCard title={title} eyebrow={subtitle} className="explanation-panel">
      <div className="explanation-actions">
        {actions.map((action) => (
          <button
            key={action.id}
            type="button"
            className={activeAction === action.id ? "explanation-chip active" : "explanation-chip"}
            onClick={() => onRunAction(action)}
            disabled={loading}
          >
            {action.label}
          </button>
        ))}
      </div>
      <div className="explanation-compact">
        <div className="explanation-summary">
          {loading && activeAction ? "Updating explanation..." : summary}
        </div>
        <div className="explanation-caution">{sections.caution}</div>
        <div className="explanation-toolbar">
          <span>{actionHint ? `Action: ${actionHint}` : "Pick an action to explain the current view."}</span>
          <button
            type="button"
            className="mini-toggle"
            onClick={() => setExpanded((value) => !value)}
          >
            {expanded ? "Collapse" : "Expand"}
          </button>
        </div>
      </div>
      <div className={expanded ? "explanation-body expanded" : "explanation-body"}>
        {Object.entries(labels).map(([key, label]) => (
          <div key={key} className="explanation-section">
            <div className="explanation-label">{label}</div>
            <div className="explanation-text">{loading && activeAction ? "Updating explanation..." : sections[key]}</div>
          </div>
        ))}
      </div>
    </ChartCard>
  );
}
