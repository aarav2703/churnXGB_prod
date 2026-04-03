import { useEffect, useMemo, useState } from "react";
import { apiGet, apiPost } from "./api.js";
import ChartCard from "./components/ChartCard.jsx";
import MetricCard from "./components/MetricCard.jsx";
import FeatureContributionChart from "./components/FeatureContributionChart.jsx";
import DataTable from "./components/DataTable.jsx";
import BudgetFrontierChart from "./components/BudgetFrontierChart.jsx";
import SegmentContributionChart from "./components/SegmentContributionChart.jsx";
import CustomerDecisionCard from "./components/CustomerDecisionCard.jsx";
import AssumptionSensitivityView from "./components/AssumptionSensitivityView.jsx";
import BacktestStabilityChart from "./components/BacktestStabilityChart.jsx";
import ExplanationPanel from "./components/ExplanationPanel.jsx";
import { EmptyState, ErrorState, LoadingGrid } from "./components/StatusState.jsx";
import { formatCompact, formatNumber, formatPct } from "./utils/format.js";
import { addLocalSegments } from "./utils/segments.js";

const pages = [
  ["overview", "Overview", "Default recommendation for the current retention strategy."],
  ["policy", "Policy Explorer", "Budget tradeoffs for the selected targeting policy."],
  ["segments", "Segment Insights", "Where the policy creates value and where it does not."],
  ["customers", "Customer Explorer", "Why a specific customer was targeted."],
  ["monitoring", "Monitoring & Trust", "How much to trust the recommendation over time."],
];
const caveats = [
  "All economics are assumption-driven and come from saved offline artifacts.",
  "The current repo does not estimate causal uplift.",
];
const policyLabels = {
  policy_net_benefit: "Cost-aware policy",
  policy_ml: "ML baseline",
  policy_recency: "Recency heuristic",
  policy_rfm: "RFM heuristic",
  policy_random: "Random baseline",
};

const toNum = (v) => (Number.isFinite(Number(v)) ? Number(v) : 0);
const atBudget = (rows, k) => rows.find((row) => Math.abs(toNum(row.budget_k) - k) < 1e-6) ?? null;

export default function App() {
  const [baseUrl, setBaseUrl] = useState("http://127.0.0.1:8000");
  const [page, setPage] = useState("overview");
  const [budgetPct, setBudgetPct] = useState(10);
  const [selectedPolicy, setSelectedPolicy] = useState("policy_net_benefit");
  const [segmentType, setSegmentType] = useState("segment_value_band");
  const [selectedSegment, setSelectedSegment] = useState("");
  const [selectedPrediction, setSelectedPrediction] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [refresh, setRefresh] = useState(0);
  const [health, setHealth] = useState(null);
  const [summary, setSummary] = useState(null);
  const [comparison, setComparison] = useState([]);
  const [policyMetrics, setPolicyMetrics] = useState([]);
  const [frontier, setFrontier] = useState([]);
  const [segments, setSegments] = useState([]);
  const [backtest, setBacktest] = useState([]);
  const [drift, setDrift] = useState(null);
  const [driftHistory, setDriftHistory] = useState([]);
  const [decisionDrift, setDecisionDrift] = useState([]);
  const [targets, setTargets] = useState([]);
  const [predictions, setPredictions] = useState([]);
  const [policySimulation, setPolicySimulation] = useState(null);
  const [customerExplanation, setCustomerExplanation] = useState(null);
  const [customerError, setCustomerError] = useState("");
  const [explanation, setExplanation] = useState(null);
  const [explanationLoading, setExplanationLoading] = useState(false);
  const [activeAction, setActiveAction] = useState("");

  const apiBase = useMemo(() => baseUrl.replace(/\/$/, ""), [baseUrl]);
  const budget = budgetPct / 100;
  const bestModel = summary?.manifest?.best_model ?? health?.model_name ?? "lightgbm";

  async function run(task) {
    setLoading(true);
    setError("");
    try {
      await task();
    } catch (err) {
      setError(err.message || "Request failed");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    setExplanation(null);
    setActiveAction("");
  }, [page, budgetPct, selectedPolicy, selectedSegment, selectedPrediction]);

  useEffect(() => {
    run(async () => {
      const results = await Promise.allSettled([
        apiGet(apiBase, "/health"),
        apiGet(apiBase, "/model-summary"),
        apiGet(apiBase, "/model-comparison"),
        apiGet(apiBase, "/policy-metrics"),
        apiGet(apiBase, "/frontier"),
        apiGet(apiBase, `/segments?segment_type=${encodeURIComponent(segmentType)}`),
        apiGet(apiBase, `/backtest?budget_pct=${budgetPct}`),
        apiGet(apiBase, "/drift/latest"),
        apiGet(apiBase, "/drift/history?limit=24"),
        apiGet(apiBase, `/drift/decision?budget_pct=${budgetPct}`),
        apiGet(apiBase, `/targets/${budgetPct}?limit=200`),
        apiGet(apiBase, "/predictions?limit=250&sort_by=policy_net_benefit"),
        apiPost(apiBase, "/simulate-policy", { budgets: [budget] }),
      ]);
      const [a, b, c, d, e, f, g, h, i, j, k, l, m] = results;
      if (a.status === "fulfilled") setHealth(a.value);
      if (b.status === "fulfilled") setSummary(b.value);
      if (c.status === "fulfilled") setComparison(c.value.rows ?? []);
      if (d.status === "fulfilled") setPolicyMetrics(d.value.rows ?? []);
      if (e.status === "fulfilled") setFrontier(e.value.rows ?? []);
      if (f.status === "fulfilled") setSegments(f.value.rows ?? []);
      if (g.status === "fulfilled") setBacktest(g.value.rows ?? []);
      if (h.status === "fulfilled") setDrift(h.value);
      if (i.status === "fulfilled") setDriftHistory(i.value.rows ?? []);
      if (j.status === "fulfilled") setDecisionDrift(j.value.rows ?? []);
      if (k.status === "fulfilled") setTargets(k.value.rows ?? []);
      if (l.status === "fulfilled") setPredictions(l.value.rows ?? []);
      if (m.status === "fulfilled") setPolicySimulation(m.value);
      if (results.every((item) => item.status === "rejected")) {
        throw new Error(results[0]?.reason?.message || "Could not load dashboard data.");
      }
    });
  }, [apiBase, refresh, budgetPct, budget, segmentType]);

  const selectedPolicyRows = useMemo(() => policyMetrics.filter((row) => row.policy === selectedPolicy), [policyMetrics, selectedPolicy]);
  const baselineRows = useMemo(() => policyMetrics.filter((row) => row.policy === "policy_ml"), [policyMetrics]);
  const selectedPolicyRow = useMemo(() => atBudget(selectedPolicyRows, budget), [selectedPolicyRows, budget]);
  const baselineRow = useMemo(() => atBudget(baselineRows, budget), [baselineRows, budget]);
  const championSegments = useMemo(
    () => segments.filter((row) => row.model === bestModel && row.segment_type === segmentType && toNum(row.budget_k) === budget),
    [segments, bestModel, segmentType, budget],
  );
  useEffect(() => {
    if (!selectedSegment && championSegments.length) setSelectedSegment(championSegments[0].segment_value);
  }, [championSegments, selectedSegment]);

  const annotatedPredictions = useMemo(() => addLocalSegments(predictions, segmentType), [predictions, segmentType]);
  const filteredPredictions = useMemo(() => {
    const targetKey = `target_k${String(budgetPct).padStart(2, "0")}`;
    return annotatedPredictions.filter((row) => {
      const inBudget = row[targetKey] === 1 || row[targetKey] === "1";
      return inBudget && (!selectedSegment || row.local_segment_value === selectedSegment);
    });
  }, [annotatedPredictions, budgetPct, selectedSegment]);

  useEffect(() => {
    if (!filteredPredictions.length) return setSelectedPrediction("");
    const exists = filteredPredictions.some((row) => `${row.CustomerID}||${row.invoice_month}` === selectedPrediction);
    if (!exists) {
      const row = filteredPredictions[0];
      setSelectedPrediction(`${row.CustomerID}||${row.invoice_month}`);
    }
  }, [filteredPredictions, selectedPrediction]);

  const selectedCustomer = useMemo(
    () => filteredPredictions.find((row) => `${row.CustomerID}||${row.invoice_month}` === selectedPrediction) ?? null,
    [filteredPredictions, selectedPrediction],
  );
  const selectedCustomerRank = useMemo(
    () => filteredPredictions.findIndex((row) => `${row.CustomerID}||${row.invoice_month}` === selectedPrediction) + 1 || null,
    [filteredPredictions, selectedPrediction],
  );

  useEffect(() => {
    if (page !== "customers" || !selectedCustomer) return setCustomerExplanation(null);
    run(async () => {
      try {
        setCustomerError("");
        setCustomerExplanation(
          await apiGet(
            apiBase,
            `/customers/explain?customer_id=${encodeURIComponent(selectedCustomer.CustomerID)}&invoice_month=${encodeURIComponent(selectedCustomer.invoice_month)}&top_n=5`,
          ),
        );
      } catch (err) {
        setCustomerExplanation(null);
        setCustomerError(err.message || "Could not load customer explanation.");
      }
    });
  }, [page, apiBase, selectedCustomer]);

  const positiveRows = (customerExplanation?.top_positive_contributors ?? []).map((row) => ({
    feature: row.feature,
    contribution: row.contribution_logit ?? row.shap_value ?? 0,
    featureValue: row.feature_value,
  }));
  const negativeRows = (customerExplanation?.top_negative_contributors ?? []).map((row) => ({
    feature: row.feature,
    contribution: row.contribution_logit ?? row.shap_value ?? 0,
    featureValue: row.feature_value,
  }));
  const policySimulationRow = policySimulation?.results?.[0] ?? null;
  const latestPsi = toNum(driftHistory?.[0]?.top_psi ?? drift?.alerts?.top_psi);
  const previousPsi = toNum(driftHistory?.[1]?.top_psi ?? latestPsi);
  const comparisonSorted = useMemo(() => [...comparison].sort((l, r) => toNum(r.test_net_benefit_at_k) - toNum(l.test_net_benefit_at_k)), [comparison]);
  const runnerUp = comparisonSorted.find((row) => row.model !== bestModel);
  const unstableCount = useMemo(() => {
    const modelRows = backtest.filter((row) => row.model === bestModel);
    if (!modelRows.length) return 0;
    const values = modelRows.map((row) => toNum(row.net_benefit_at_k));
    const avg = values.reduce((sum, value) => sum + value, 0) / values.length;
    const deviation = Math.sqrt(values.reduce((sum, value) => sum + (value - avg) ** 2, 0) / values.length);
    return modelRows.filter((row) => Math.abs(toNum(row.net_benefit_at_k) - avg) > deviation).length;
  }, [backtest, bestModel]);

  const pageMeta = pages.find((item) => item[0] === page);
  const pageTakeaways = {
    overview: selectedPolicyRow && baselineRow
      ? `${policyLabels[selectedPolicy] ?? selectedPolicy} is currently the better operating policy, with ${formatCompact(toNum(selectedPolicyRow.net_benefit_at_k) - toNum(baselineRow.net_benefit_at_k))} more net benefit than the baseline at ${budgetPct}% budget.`
      : "Use this page to confirm the default policy, budget, and key caveat before exploring details.",
    policy: selectedPolicyRow && baselineRow
      ? `${policyLabels[selectedPolicy] ?? selectedPolicy} delivers ${formatCompact(selectedPolicyRow.net_benefit_at_k)} at ${budgetPct}% budget, ${toNum(selectedPolicyRow.net_benefit_at_k) >= toNum(baselineRow.net_benefit_at_k) ? "beating" : "trailing"} the baseline by ${formatCompact(Math.abs(toNum(selectedPolicyRow.net_benefit_at_k) - toNum(baselineRow.net_benefit_at_k)))}.`
      : "Use this page to compare budget tradeoffs before changing campaign volume.",
    segments: championSegments.length
      ? `${championSegments.filter((row) => row.policy === "policy_net_benefit").sort((left, right) => toNum(right.net_benefit_at_k) - toNum(left.net_benefit_at_k))[0]?.segment_value ?? "Top segment"} is carrying most of the value, while negative-ROI segments should be deprioritized.`
      : "Use this page to see which customer segments actually create value for the policy.",
    customers: selectedCustomer
      ? `Customer ${selectedCustomer.CustomerID} is shown in the context of current budget, segment, and policy ranking.`
      : "Use this page to inspect the exact customer-level rationale behind a targeting decision.",
    monitoring: unstableCount > 0
      ? `Recommendation is broadly stable, but ${unstableCount} saved period${unstableCount > 1 ? "s" : ""} show business-value volatility.`
      : "Recommendation looks stable across the saved backtest periods, but economics remain assumption-driven.",
  };
  const llmActions = {
    overview: [["summarize_recommendation", "Summarize recommendation"], ["summarize_risk", "Summarize risk"]],
    policy: [["explain_chart", "Explain this chart"], ["explain_budget_tradeoff", "Why this budget?"], ["explain_policy", "Compare policies"]],
    segments: [["explain_segment", "Summarize this segment"], ["explain_chart", "Explain this chart"]],
    customers: [["explain_customer", "Why this customer?"]],
    monitoring: [["summarize_risk", "Summarize risk"], ["explain_chart", "Explain this chart"]],
  }[page];

  const basePayload = () => ({
    page: pageMeta?.[1] ?? page,
    selected_budget: budgetPct,
    selected_policy: selectedPolicy,
    selected_model: bestModel,
    selected_segment: selectedSegment ? { segment_type: segmentType, segment_value: selectedSegment } : null,
    selected_customer: selectedCustomer ? { customer_id: selectedCustomer.CustomerID, invoice_month: selectedCustomer.invoice_month } : null,
    chart_data: null,
    key_metrics: {},
    baseline_metrics: {},
    caveats: [...caveats],
    assumption_flags: ["No causal uplift is estimated in the current repo."],
  });

  async function runExplanation(action) {
    const payload = basePayload();
    let path = "";
    if (action === "summarize_recommendation") {
      path = "/llm/summarize/recommendation";
      payload.key_metrics = { net_benefit_at_k: selectedPolicyRow?.net_benefit_at_k, value_at_risk: selectedPolicyRow?.value_at_risk };
      payload.baseline_metrics = { baseline_net_benefit_at_k: baselineRow?.net_benefit_at_k };
    } else if (action === "summarize_risk") {
      path = "/llm/summarize/risk";
      payload.key_metrics = { top_psi: latestPsi, backtest_rows: backtest.length };
      payload.caveats.push("Use drift and backtest artifacts before relying on the recommendation.");
    } else if (action === "explain_chart") {
      path = "/llm/explain/chart";
      payload.chart_type = page === "policy" ? "budget_frontier" : page === "monitoring" ? "backtest_stability" : "segment_value_composition";
      payload.chart_data = page === "policy"
        ? selectedPolicyRows.map((row) => ({ budget_k: row.budget_k, net_benefit_at_k: row.net_benefit_at_k, value_at_risk: row.value_at_risk }))
        : page === "monitoring"
          ? backtest.map((row) => ({ fold: row.fold, net_benefit_at_k: row.net_benefit_at_k, roc_auc: row.roc_auc }))
          : championSegments.map((row) => ({ segment_value: row.segment_value, net_benefit_at_k: row.net_benefit_at_k, lift_at_k: row.lift_at_k }));
      payload.selected_point = page === "policy" ? selectedPolicyRow : page === "segments" ? championSegments.find((row) => row.segment_value === selectedSegment) : backtest[0] ?? null;
    } else if (action === "explain_budget_tradeoff") {
      path = "/llm/explain/budget-tradeoff";
      payload.chart_data = selectedPolicyRows.map((row) => ({ budget_k: row.budget_k, net_benefit_at_k: row.net_benefit_at_k, value_at_risk: row.value_at_risk }));
      payload.key_metrics = { selected_budget: budgetPct, selected_net_benefit: selectedPolicyRow?.net_benefit_at_k };
      payload.baseline_metrics = { baseline_net_benefit_at_k: baselineRow?.net_benefit_at_k };
    } else if (action === "explain_policy") {
      path = "/llm/explain/policy";
      payload.baseline_policy = "policy_ml";
      payload.comparison_policy = selectedPolicy;
      payload.key_metrics = { comparison_net_benefit_at_k: selectedPolicyRow?.net_benefit_at_k, comparison_value_at_risk: selectedPolicyRow?.value_at_risk };
      payload.baseline_metrics = { baseline_net_benefit_at_k: baselineRow?.net_benefit_at_k, baseline_value_at_risk: baselineRow?.value_at_risk };
    } else if (action === "explain_segment") {
      const row = championSegments.find((item) => item.segment_value === selectedSegment) ?? championSegments[0];
      path = "/llm/explain/segment";
      payload.segment_type = segmentType;
      payload.segment_value = row?.segment_value ?? selectedSegment;
      payload.split = "test";
      payload.chart_data = championSegments.map((item) => ({ segment_value: item.segment_value, net_benefit_at_k: item.net_benefit_at_k, lift_at_k: item.lift_at_k }));
      payload.key_metrics = { segment_net_benefit_at_k: row?.net_benefit_at_k, segment_lift_at_k: row?.lift_at_k };
    } else if (action === "explain_customer" && selectedCustomer) {
      path = "/llm/explain/customer";
      payload.customer_id = String(selectedCustomer.CustomerID);
      payload.invoice_month = String(selectedCustomer.invoice_month);
      payload.top_n = 5;
      payload.key_metrics = { churn_prob: selectedCustomer.churn_prob, policy_net_benefit: selectedCustomer.policy_net_benefit, value_pos: selectedCustomer.value_pos };
      payload.baseline_metrics = { rank_position: selectedCustomerRank };
    } else {
      return;
    }
    setExplanationLoading(true);
    setActiveAction(action);
    try {
      setExplanation(await apiPost(apiBase, path, payload));
    } catch (err) {
      setExplanation({ sections: { what_this_shows: "The explanation request failed.", why_it_matters: "The dashboard still reflects saved artifacts, but the explanation service could not respond.", what_to_do: "Retry after confirming the API is running.", caution: err.message || "LLM request failed." } });
    } finally {
      setExplanationLoading(false);
    }
  }

  if (error && !summary && !loading) return <div className="app-shell"><ErrorState message={error} onRetry={() => setRefresh((x) => x + 1)} /></div>;

  return (
    <div className="app-shell app-shell-redesign">
      <header className="app-header app-header-redesign">
        <div>
          <div className="eyebrow">Retention Decision Intelligence</div>
          <h1>Leakage-safe churn targeting, built for action.</h1>
          <p className="subtle">A decision interface over calibrated churn scoring, budget-aware policy selection, saved monitoring artifacts, and grounded LLM explanations.</p>
        </div>
        <div className="header-actions">
          <label className="api-input"><span>API Base URL</span><input value={baseUrl} onChange={(e) => setBaseUrl(e.target.value)} /></label>
          <label className="api-input compact"><span>Budget</span><select value={budgetPct} onChange={(e) => setBudgetPct(Number(e.target.value))}><option value={5}>5%</option><option value={10}>10%</option><option value={20}>20%</option></select></label>
          <label className="api-input compact"><span>Policy</span><select value={selectedPolicy} onChange={(e) => setSelectedPolicy(e.target.value)}>{[...new Set(policyMetrics.map((row) => row.policy))].map((policy) => <option key={policy} value={policy}>{policyLabels[policy] ?? policy}</option>)}</select></label>
          <button type="button" className="retry-button subtle-button" onClick={() => setRefresh((x) => x + 1)}>Refresh</button>
        </div>
      </header>

      <section className="hero-strip hero-strip-redesign">
        <div className="hero-copy">
          <div className="hero-title">What should we do now?</div>
          <div className="hero-text">{selectedPolicyRow && baselineRow ? `${policyLabels[selectedPolicy] ?? selectedPolicy} adds ${formatCompact(toNum(selectedPolicyRow.net_benefit_at_k) - toNum(baselineRow.net_benefit_at_k))} net benefit vs baseline at ${budgetPct}% budget.` : "Review the promoted model, selected budget, and caveats before changing campaign volume."}</div>
        </div>
        <div className="hero-kpis"><div className="hero-chip">API {health?.status === "ok" ? "live" : health?.status ?? "n/a"}</div><div className="hero-chip">Champion {bestModel}</div><div className="hero-chip">Budget {formatPct(budget, 0)}</div></div>
      </section>

      <nav className="tab-row">{pages.map(([id, label]) => <button key={id} type="button" className={id === page ? "tab active" : "tab"} onClick={() => setPage(id)}>{label}</button>)}</nav>
      {error ? <div className="banner error">{error}<button type="button" className="retry-button inline" onClick={() => setRefresh((x) => x + 1)}>Retry</button></div> : null}
      {loading && !summary ? <LoadingGrid lines={4} /> : null}

      <section className="page-intro">
        <div className="page-headline-group">
          <div><div className="panel-eyebrow">{pageMeta?.[1]}</div><h2>{pageMeta?.[2]}</h2></div>
          <div className="page-takeaway"><strong>Takeaway:</strong> {pageTakeaways[page]}</div>
        </div>
      </section>

      <div className="workspace-grid">
        <main className="workspace-main">
          {page === "overview" && <div className="page-grid">
            <div className="metrics-grid span-12">
              <MetricCard label="Recommended policy" value={policyLabels[selectedPolicy] ?? selectedPolicy} tone="primary" />
              <MetricCard label="Selected budget" value={formatPct(budget, 0)} tone="primary" />
              <MetricCard label="Net benefit vs baseline" value={selectedPolicyRow && baselineRow ? `${toNum(selectedPolicyRow.net_benefit_at_k) >= toNum(baselineRow.net_benefit_at_k) ? "+" : "-"}${formatCompact(Math.abs(toNum(selectedPolicyRow.net_benefit_at_k) - toNum(baselineRow.net_benefit_at_k)))}` : "n/a"} tone={toNum(selectedPolicyRow?.net_benefit_at_k) >= toNum(baselineRow?.net_benefit_at_k) ? "gain" : "loss"} />
              <MetricCard label="Primary caveat" value="Assumption-driven (no causal uplift)" tone="primary" />
            </div>
            <ChartCard title="Recommendation Summary" eyebrow="Default operating point" className="span-12"><div className="overview-summary-grid"><div className="decision-recommendation"><div className="decision-label">Recommendation</div><div className="decision-headline">{toNum(selectedPolicyRow?.net_benefit_at_k) >= toNum(baselineRow?.net_benefit_at_k) ? `Use ${policyLabels[selectedPolicy] ?? selectedPolicy} at ${budgetPct}% budget` : "Hold the budget and reassess whether the policy earns its complexity"}</div><div className="decision-note">This recommendation is driven by saved offline policy metrics, not observed intervention effects.</div></div><div className="decision-item"><div className="decision-label">Expected retained value</div><div className="decision-value">{formatCompact(selectedPolicyRow?.net_benefit_at_k)}</div><div className="decision-subvalue">Value at risk: {formatCompact(selectedPolicyRow?.value_at_risk)}</div></div><div className="decision-item"><div className="decision-label">Key risk</div><div className="decision-value">Assumption-driven</div><div className="decision-subvalue">No causal uplift is estimated in the current repo.</div></div></div></ChartCard>
            <ChartCard title="Why the winner won" eyebrow="Compact champion analysis" className="span-12">{comparisonSorted.length ? <div className="winner-grid"><div className="decision-item"><div className="decision-label">Champion model</div><div className="decision-value">{bestModel}</div><div className="decision-subvalue">Net benefit {formatCompact(comparisonSorted[0]?.test_net_benefit_at_k)} | Brier {formatNumber(comparisonSorted[0]?.test_brier_score, 3)}</div></div><div className="decision-item"><div className="decision-label">Runner-up</div><div className="decision-value">{runnerUp?.model ?? "n/a"}</div><div className="decision-subvalue">Gap {runnerUp ? formatCompact(toNum(comparisonSorted[0]?.test_net_benefit_at_k) - toNum(runnerUp?.test_net_benefit_at_k)) : "n/a"} in test net benefit.</div></div><div className="decision-item"><div className="decision-label">What to do</div><div className="decision-subvalue">Keep the promoted model if calibration, drift, and budget economics remain stable; otherwise revisit the policy explorer and monitoring pages.</div></div></div> : <EmptyState title="No champion analysis" message="Model comparison artifacts were not available." />}</ChartCard>
          </div>}

          {page === "policy" && <div className="page-grid">
            <BudgetFrontierChart selectedRows={selectedPolicyRows} baselineRows={baselineRows} selectedBudget={budget} onSelectBudget={(next) => setBudgetPct(Math.round(next * 100))} selectedPolicy={selectedPolicy} />
            <div className="metrics-grid span-12">
              <MetricCard label="Selected policy net benefit" value={formatCompact(selectedPolicyRow?.net_benefit_at_k)} tone="gain" />
              <MetricCard label="Baseline net benefit" value={formatCompact(baselineRow?.net_benefit_at_k)} tone="primary" />
              <MetricCard label="Selection overlap" value={formatPct(policySimulation?.results?.[0]?.selection_overlap_at_k)} tone="primary" />
              <MetricCard label="Net benefit vs baseline" value={policySimulationRow?.comparison_minus_baseline == null ? "n/a" : `${toNum(policySimulationRow?.comparison_minus_baseline) >= 0 ? "+" : "-"}${formatCompact(Math.abs(toNum(policySimulationRow?.comparison_minus_baseline)))}`} tone={toNum(policySimulationRow?.comparison_minus_baseline) >= 0 ? "gain" : "loss"} />
            </div>
            <ChartCard title="Policy Comparison" eyebrow="How the selected policy differs from the ML baseline" className="span-12"><div className="winner-grid"><div className="decision-item"><div className="decision-label">Selected policy</div><div className="decision-value">{policyLabels[selectedPolicy] ?? selectedPolicy}</div><div className="decision-subvalue">Net benefit {formatCompact(selectedPolicyRow?.net_benefit_at_k)} | VaR {formatCompact(selectedPolicyRow?.value_at_risk)}</div></div><div className="decision-item"><div className="decision-label">Baseline</div><div className="decision-value">ML ranking</div><div className="decision-subvalue">Net benefit {formatCompact(baselineRow?.net_benefit_at_k)} | VaR {formatCompact(baselineRow?.value_at_risk)}</div></div><div className="decision-item"><div className="decision-label">What to do</div><div className="decision-subvalue">{toNum(policySimulationRow?.comparison_minus_baseline) >= 0 ? "Keep the cost-aware policy at this budget unless drift or assumptions deteriorate." : "The baseline remains competitive. Tighten assumptions before increasing campaign volume."}</div></div></div></ChartCard>
            <AssumptionSensitivityView frontierRows={selectedPolicyRows} selectedBudget={budget} onBudgetChange={(next) => setBudgetPct(Math.round(next * 100))} targetRows={targets} />
          </div>}

          {page === "segments" && <div className="page-grid">
            <ChartCard title="Segment Controls" eyebrow="Segment family" className="span-12" actions={<label className="api-input compact"><span>Segment family</span><select value={segmentType} onChange={(e) => setSegmentType(e.target.value)}><option value="segment_value_band">Value bands</option><option value="segment_recency_bucket">Recency buckets</option><option value="segment_frequency_bucket">Frequency buckets</option></select></label>}><div className="decision-note">Click a segment to keep that context active across this page and Customer Explorer.</div></ChartCard>
            <SegmentContributionChart rows={championSegments} selectedSegment={selectedSegment} onSelectSegment={setSelectedSegment} segmentType={segmentType} />
            <div className="metrics-grid span-12">
              <MetricCard label="Selected segment" value={selectedSegment || "n/a"} tone="primary" />
              <MetricCard label="Segment net benefit" value={formatCompact(championSegments.find((row) => row.segment_value === selectedSegment)?.net_benefit_at_k)} tone={toNum(championSegments.find((row) => row.segment_value === selectedSegment)?.net_benefit_at_k) >= 0 ? "gain" : "loss"} />
              <MetricCard label="Segment lift" value={formatNumber(championSegments.find((row) => row.segment_value === selectedSegment)?.lift_at_k, 2)} tone="primary" />
              <MetricCard label="Negative ROI segments" value={championSegments.filter((row) => toNum(row.net_benefit_at_k) < 0).length} tone={championSegments.some((row) => toNum(row.net_benefit_at_k) < 0) ? "loss" : "gain"} />
            </div>
          </div>}

          {page === "customers" && <div className="page-grid">
            <ChartCard title="Customer Controls" eyebrow="Cross-filtered by budget and segment" className="span-12">{annotatedPredictions.length ? <><div className="controls-grid"><label className="api-input"><span>Segment filter</span><select value={selectedSegment} onChange={(e) => setSelectedSegment(e.target.value)}><option value="">All segments</option>{[...new Set(annotatedPredictions.map((row) => row.local_segment_value))].map((segment) => <option key={segment} value={segment}>{segment}</option>)}</select></label><label className="api-input"><span>Customer-month</span><select value={selectedPrediction} onChange={(e) => setSelectedPrediction(e.target.value)}>{filteredPredictions.map((row) => { const value = `${row.CustomerID}||${row.invoice_month}`; return <option key={value} value={value}>{row.CustomerID} | {row.invoice_month}</option>; })}</select></label></div>{customerError ? <div className="customer-inline-note">{customerError}</div> : null}</> : <EmptyState title="No scored predictions loaded" message="Run the scoring pipeline and start the API to explore saved customer decisions." />}</ChartCard>
            <CustomerDecisionCard customer={selectedCustomer} explanation={customerExplanation} rank={selectedCustomerRank} totalRows={filteredPredictions.length} />
            <FeatureContributionChart title="Positive feature drivers" rows={positiveRows} tone="blue" />
            <FeatureContributionChart title="Negative feature drivers" rows={negativeRows} tone="amber" />
            <DataTable title="Targeted customers in current slice" rows={filteredPredictions.slice(0, 15)} searchable={false} defaultLimit={8} highlightMetric="policy_net_benefit" negativeMetric="policy_net_benefit" />
          </div>}

          {page === "monitoring" && <div className="page-grid">
            <div className="metrics-grid span-12">
              <MetricCard label="Drift status" value={drift?.alerts?.overall_status === "ok" ? "stable" : drift?.alerts?.overall_status ?? "n/a"} tone={drift?.alerts?.overall_status === "ok" ? "gain" : drift?.alerts?.overall_status === "warn" ? "primary" : "loss"} />
              <MetricCard label="Latest PSI" value={formatNumber(latestPsi, 3)} tone={latestPsi >= 0.2 ? "loss" : latestPsi >= 0.1 ? "primary" : "gain"} />
              <MetricCard label="Key risk" value={unstableCount > 0 ? `${unstableCount} unstable period${unstableCount > 1 ? "s" : ""}` : "Assumption-driven economics"} tone={unstableCount > 0 ? "loss" : "primary"} />
            </div>
            <ChartCard title="Trust Summary" eyebrow="Monitoring takeaway" className="span-12"><div className="trust-banner">{unstableCount > 0 ? `Model is stable overall, but ${unstableCount} saved period${unstableCount > 1 ? "s" : ""} show meaningful volatility in business value.` : "Model is stable across the saved backtest periods, but economics are still assumption-driven."}</div></ChartCard>
            <BacktestStabilityChart rows={backtest} modelName={bestModel} />
            <ChartCard title="What to watch" eyebrow="Main trust risks" className="span-12"><div className="winner-grid"><div className="decision-item"><div className="decision-label">Drift</div><div className="decision-value">{drift?.alerts?.overall_status === "ok" ? "stable" : drift?.alerts?.overall_status ?? "n/a"}</div><div className="decision-subvalue">Top PSI {formatNumber(latestPsi, 3)} and trend {latestPsi > previousPsi ? "up" : latestPsi < previousPsi ? "down" : "flat"}.</div></div><div className="decision-item"><div className="decision-label">Temporal stability</div><div className="decision-subvalue">Highlighted folds sit outside the stability band and deserve closer review.</div></div><div className="decision-item"><div className="decision-label">Assumption sensitivity</div><div className="decision-subvalue">Campaign economics still depend on configured intervention cost and success-rate assumptions.</div></div></div></ChartCard>
            <AssumptionSensitivityView frontierRows={selectedPolicyRows} selectedBudget={budget} onBudgetChange={(next) => setBudgetPct(Math.round(next * 100))} targetRows={targets} />
          </div>}
        </main>

        <aside className="workspace-side">
          <ExplanationPanel title="Grounded Explanation" subtitle="Context-aware explanation layer" actions={llmActions.map(([id, label]) => ({ id, label }))} explanation={explanation} loading={explanationLoading} activeAction={activeAction} onRunAction={(action) => runExplanation(action.id)} />
        </aside>
      </div>
    </div>
  );
}
