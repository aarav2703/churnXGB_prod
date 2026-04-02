import { useEffect, useMemo, useState } from "react";
import { apiGet, apiPost } from "./api.js";
import MetricCard from "./components/MetricCard.jsx";
import ChartCard from "./components/ChartCard.jsx";
import DecisionPanel from "./components/DecisionPanel.jsx";
import InteractiveLineChart from "./components/InteractiveLineChart.jsx";
import SegmentGroupedChart from "./components/SegmentGroupedChart.jsx";
import FeatureContributionChart from "./components/FeatureContributionChart.jsx";
import DataTable from "./components/DataTable.jsx";
import { EmptyState, ErrorState, LoadingGrid } from "./components/StatusState.jsx";
import { formatCompact, formatNumber, formatPct } from "./utils/format.js";
import { addLocalSegments } from "./utils/segments.js";

const tabs = [
  "Executive Summary",
  "Targeting Strategy",
  "Model Performance",
  "Segment Analysis",
  "Drift Monitoring",
  "Customer Explorer",
  "Ask / Explain",
];

function extractMessage(result) {
  return result?.reason?.message || result?.reason || "Request failed";
}

function recommendationText(frontierPoint, simulationRow) {
  const netBenefit = Number(frontierPoint?.net_benefit_at_k ?? simulationRow?.comparison_net_benefit_at_k ?? 0);
  const delta = Number(simulationRow?.comparison_minus_baseline ?? 0);
  if (netBenefit <= 0) return "Targeting at this budget is weak economically. Review campaign cost assumptions first.";
  if (delta > 0) return "The cost-aware policy is adding value over the baseline at this budget.";
  return "The selected budget is viable, but the baseline ranking remains competitive.";
}

export default function App() {
  const [baseUrl, setBaseUrl] = useState("http://127.0.0.1:8000");
  const [activeTab, setActiveTab] = useState("Executive Summary");
  const [budgetPct, setBudgetPct] = useState(10);
  const [selectedBudgetX, setSelectedBudgetX] = useState(0.1);
  const [segmentType, setSegmentType] = useState("segment_value_band");
  const [selectedSegment, setSelectedSegment] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [refreshToken, setRefreshToken] = useState(0);

  const [health, setHealth] = useState(null);
  const [summary, setSummary] = useState(null);
  const [comparison, setComparison] = useState([]);
  const [policyMetrics, setPolicyMetrics] = useState([]);
  const [frontier, setFrontier] = useState([]);
  const [segments, setSegments] = useState([]);
  const [backtest, setBacktest] = useState([]);
  const [drift, setDrift] = useState(null);
  const [decisionDrift, setDecisionDrift] = useState([]);
  const [driftHistory, setDriftHistory] = useState([]);
  const [targets, setTargets] = useState([]);
  const [targetSummary, setTargetSummary] = useState({ totalRows: null, returnedRows: 0 });
  const [predictions, setPredictions] = useState([]);
  const [selectedPrediction, setSelectedPrediction] = useState("");
  const [customerExplanation, setCustomerExplanation] = useState(null);
  const [llmDecision, setLlmDecision] = useState(null);
  const [customerError, setCustomerError] = useState("");
  const [policySimulation, setPolicySimulation] = useState(null);
  const [chatQuery, setChatQuery] = useState("What should we do at 10% budget?");
  const [chatResult, setChatResult] = useState(null);

  const normalizedBaseUrl = useMemo(() => baseUrl.replace(/\/$/, ""), [baseUrl]);
  const selectedBudgetFraction = Number(selectedBudgetX ?? budgetPct / 100);
  const summaryRow = summary?.comparison_row ?? null;
  const policySimulationRow = policySimulation?.results?.[0] ?? null;

  async function runLoad(task) {
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

  const retryLoad = () => setRefreshToken((value) => value + 1);

  useEffect(() => {
    setError("");
  }, [activeTab]);

  useEffect(() => {
    runLoad(async () => {
      const results = await Promise.allSettled([
        apiGet(normalizedBaseUrl, "/health"),
        apiGet(normalizedBaseUrl, "/model-summary"),
        apiGet(normalizedBaseUrl, "/model-comparison"),
        apiGet(normalizedBaseUrl, "/frontier"),
        apiGet(normalizedBaseUrl, "/policy-metrics"),
      ]);
      const [healthRes, summaryRes, comparisonRes, frontierRes, policyRes] = results;

      if (healthRes.status === "fulfilled") setHealth(healthRes.value);
      if (summaryRes.status === "fulfilled") setSummary(summaryRes.value);
      if (comparisonRes.status === "fulfilled") setComparison(comparisonRes.value.rows ?? []);
      if (frontierRes.status === "fulfilled") setFrontier(frontierRes.value.rows ?? []);
      if (policyRes.status === "fulfilled") setPolicyMetrics(policyRes.value.rows ?? []);

      const nSuccess = results.filter((result) => result.status === "fulfilled").length;
      if (nSuccess === 0) {
        throw new Error(extractMessage(results.find((result) => result.status === "rejected")));
      }
    });
  }, [normalizedBaseUrl, refreshToken]);

  useEffect(() => {
    runLoad(async () => {
      const results = await Promise.allSettled([
        apiGet(normalizedBaseUrl, `/drift/decision?budget_pct=${budgetPct}`),
        apiPost(normalizedBaseUrl, "/simulate-policy", { budgets: [budgetPct / 100] }),
      ]);
      const [decisionRes, policyRes] = results;
      if (decisionRes.status === "fulfilled") setDecisionDrift(decisionRes.value.rows ?? []);
      if (policyRes.status === "fulfilled") setPolicySimulation(policyRes.value);
      if (results.every((result) => result.status === "rejected")) {
        throw new Error(extractMessage(results[0]));
      }
    });
  }, [normalizedBaseUrl, budgetPct, refreshToken]);

  useEffect(() => {
    if (!["Model Performance", "Segment Analysis"].includes(activeTab)) return;
    runLoad(async () => {
      const [segmentsRes, backtestRes] = await Promise.all([
        apiGet(normalizedBaseUrl, `/segments?segment_type=${encodeURIComponent(segmentType)}`),
        apiGet(normalizedBaseUrl, `/backtest?budget_pct=${budgetPct}`),
      ]);
      const rows = segmentsRes.rows ?? [];
      setSegments(rows);
      if (!selectedSegment && rows.length) setSelectedSegment(rows[0].segment_value);
      setBacktest(backtestRes.rows ?? []);
    });
  }, [activeTab, normalizedBaseUrl, segmentType, budgetPct, refreshToken]);

  useEffect(() => {
    if (activeTab !== "Drift Monitoring") return;
    runLoad(async () => {
      const results = await Promise.allSettled([
        apiGet(normalizedBaseUrl, "/drift/latest"),
        apiGet(normalizedBaseUrl, "/drift/history?limit=24"),
        apiGet(normalizedBaseUrl, `/drift/decision?budget_pct=${budgetPct}`),
      ]);
      const [latest, history, decision] = results;
      if (latest.status === "fulfilled") setDrift(latest.value);
      if (history.status === "fulfilled") setDriftHistory(history.value.rows ?? []);
      if (decision.status === "fulfilled") setDecisionDrift(decision.value.rows ?? []);
      if (results.every((result) => result.status === "rejected")) {
        throw new Error(extractMessage(results[0]));
      }
    });
  }, [activeTab, normalizedBaseUrl, budgetPct, refreshToken]);

  useEffect(() => {
    if (activeTab !== "Targeting Strategy") return;
    runLoad(async () => {
      const targetRes = await apiGet(normalizedBaseUrl, `/targets/${budgetPct}?limit=100`);
      setTargets(targetRes.rows ?? []);
      setTargetSummary({
        totalRows: targetRes.total_rows ?? null,
        returnedRows: targetRes.returned_rows ?? 0,
      });
    });
  }, [activeTab, normalizedBaseUrl, budgetPct, refreshToken]);

  useEffect(() => {
    if (activeTab !== "Customer Explorer") return;
    setCustomerError("");
    setCustomerExplanation(null);
    setLlmDecision(null);
    runLoad(async () => {
      const results = await Promise.allSettled([
        apiGet(normalizedBaseUrl, "/predictions?limit=250&sort_by=policy_net_benefit"),
        apiGet(normalizedBaseUrl, `/targets/${budgetPct}?limit=100`),
      ]);
      const [predictionRes, targetRes] = results;

      let sourceRows = [];
      if (predictionRes.status === "fulfilled") {
        sourceRows = predictionRes.value.rows ?? [];
      } else if (targetRes.status === "fulfilled") {
        sourceRows = targetRes.value.rows ?? [];
        setCustomerError(
          "Saved prediction rows could not be loaded, so the explorer is using the current target list instead.",
        );
      } else {
        setPredictions([]);
        setCustomerError(
          "Customer rows could not be loaded from the API. Check that the backend is still running, then retry.",
        );
        return;
      }

      setPredictions(sourceRows);
      setError("");
      if (sourceRows.length > 0) {
        const fallbackKey = `${sourceRows[0].CustomerID}||${sourceRows[0].invoice_month}`;
        setSelectedPrediction((current) => {
          const exists = sourceRows.some(
            (row) => `${row.CustomerID}||${row.invoice_month}` === current,
          );
          return exists ? current : fallbackKey;
        });
      }
    });
  }, [activeTab, normalizedBaseUrl, refreshToken, budgetPct]);

  const selectedFrontierPoint = frontier.find(
    (row) => Number(row.budget_k) === Number(selectedBudgetFraction),
  ) ?? frontier.find((row) => Number(row.budget_k) === Number(budgetPct / 100));

  const annotatedPredictions = useMemo(
    () => addLocalSegments(predictions, segmentType),
    [predictions, segmentType],
  );

  const selectedSegmentPredictions = useMemo(() => {
    let rows = annotatedPredictions;
    const targetColumn = `target_k${String(budgetPct).padStart(2, "0")}`;
    rows = rows.filter((row) => row[targetColumn] === 1);
    if (selectedSegment) {
      rows = rows.filter((row) => row.local_segment_value === selectedSegment);
    }
    return rows;
  }, [annotatedPredictions, selectedSegment, budgetPct]);

  useEffect(() => {
    if (activeTab !== "Customer Explorer") return;
    if (!selectedPrediction || !selectedSegmentPredictions.length) return;
    const exists = selectedSegmentPredictions.some(
      (row) => `${row.CustomerID}||${row.invoice_month}` === selectedPrediction,
    );
    if (!exists) {
      const next = selectedSegmentPredictions[0];
      setSelectedPrediction(`${next.CustomerID}||${next.invoice_month}`);
    }
  }, [activeTab, selectedPrediction, selectedSegmentPredictions]);

  useEffect(() => {
    if (activeTab !== "Customer Explorer" || !selectedPrediction || !selectedSegmentPredictions.length) {
      if (activeTab === "Customer Explorer" && !selectedSegmentPredictions.length) {
        setCustomerExplanation(null);
        setLlmDecision(null);
      }
      return;
    }
    const [customerId, invoiceMonth] = selectedPrediction.split("||");
    runLoad(async () => {
      const results = await Promise.allSettled([
        apiGet(
          normalizedBaseUrl,
          `/customers/explain?customer_id=${encodeURIComponent(customerId)}&invoice_month=${encodeURIComponent(invoiceMonth)}&top_n=5`,
        ),
        apiPost(normalizedBaseUrl, "/llm/explain_customer", {
          customer_id: customerId,
          invoice_month: invoiceMonth,
          top_n: 5,
        }),
      ]);
      const [explanationRes, llmRes] = results;
      if (explanationRes.status === "fulfilled") {
        setCustomerExplanation(explanationRes.value);
        setCustomerError("");
      }
      if (llmRes.status === "fulfilled") {
        setLlmDecision(llmRes.value);
      } else if (explanationRes.status === "fulfilled") {
        setLlmDecision({
          answer: "Grounded decision narrative is temporarily unavailable, but the prediction and feature contributions are still shown below.",
        });
      }
      if (explanationRes.status === "rejected") {
        setCustomerError(
          "The selected customer could not be explained from saved predictions. Try another row or rerun scoring.",
        );
      }
    });
  }, [activeTab, normalizedBaseUrl, selectedPrediction, refreshToken, selectedSegmentPredictions.length]);

  const selectedCustomerRow = useMemo(() => {
    const sourceRows = selectedSegmentPredictions.length ? selectedSegmentPredictions : [];
    const index = sourceRows.findIndex(
      (row) => `${row.CustomerID}||${row.invoice_month}` === selectedPrediction,
    );
    return {
      rows: sourceRows,
      rank: index >= 0 ? index + 1 : null,
    };
  }, [annotatedPredictions, selectedPrediction, selectedSegmentPredictions]);

  async function submitChatQuery() {
    await runLoad(async () => {
      const result = await apiPost(normalizedBaseUrl, "/llm/query", {
        query: chatQuery,
        include_raw_data: true,
      });
      setChatResult(result);
    });
  }

  const contributionPositiveRows = (customerExplanation?.top_positive_contributors ?? []).map((row) => ({
    feature: row.feature,
    contribution: row.contribution_logit ?? row.shap_value ?? 0,
    featureValue: row.feature_value,
  }));
  const contributionNegativeRows = (customerExplanation?.top_negative_contributors ?? []).map((row) => ({
    feature: row.feature,
    contribution: row.contribution_logit ?? row.shap_value ?? 0,
    featureValue: row.feature_value,
  }));

  const latestPsi = Number(driftHistory?.[0]?.top_psi ?? drift?.alerts?.top_psi ?? 0);
  const previousPsi = Number(driftHistory?.[1]?.top_psi ?? latestPsi);
  const trendDirection = latestPsi > previousPsi ? "Up" : latestPsi < previousPsi ? "Down" : "Flat";

  if (error && !health && !loading) {
    return (
      <div className="app-shell">
        <ErrorState message={error} onRetry={retryLoad} />
      </div>
    );
  }

  return (
    <div className="app-shell">
      <header className="app-header">
        <div>
          <div className="eyebrow">Decision Intelligence Workspace</div>
          <h1>ChurnXGB Frontend</h1>
          <p className="subtle">
            An interactive analytics layer over the existing FastAPI backend, with shared budget
            and segment state, stronger explainability, and decision-focused reporting.
          </p>
        </div>
        <div className="header-actions">
          <label className="api-input">
            <span>API Base URL</span>
            <input value={baseUrl} onChange={(event) => setBaseUrl(event.target.value)} />
          </label>
          <label className="api-input compact">
            <span>Budget K</span>
            <select
              value={budgetPct}
              onChange={(event) => {
                const next = Number(event.target.value);
                setBudgetPct(next);
                setSelectedBudgetX(next / 100);
              }}
            >
              <option value={5}>5%</option>
              <option value={10}>10%</option>
              <option value={20}>20%</option>
            </select>
          </label>
          <button type="button" className="retry-button subtle-button" onClick={retryLoad}>
            Retry data
          </button>
        </div>
      </header>

      <section className="hero-strip">
        <div className="hero-copy">
          <div className="hero-title">Interactive decision analytics for retention targeting</div>
          <div className="hero-text">
            Hover, click, and filter across the budget frontier, segments, drift trends, and
            customer list. The whole dashboard stays tied to the selected budget and segment.
          </div>
        </div>
        <div className="hero-kpis">
          <div className="hero-chip">API {health?.status ?? "n/a"}</div>
          <div className="hero-chip">Budget {formatPct(selectedBudgetFraction, 0)}</div>
          <div className="hero-chip">Model {summary?.manifest?.best_model ?? "n/a"}</div>
        </div>
      </section>

      <nav className="tab-row">
        {tabs.map((tab) => (
          <button
            key={tab}
            type="button"
            className={tab === activeTab ? "tab active" : "tab"}
            onClick={() => setActiveTab(tab)}
          >
            {tab}
          </button>
        ))}
      </nav>

      {error ? (
        <div className="banner error">
          {error}
          <button type="button" className="retry-button inline" onClick={retryLoad}>
            Retry
          </button>
        </div>
      ) : null}

      {loading && !summary ? <LoadingGrid lines={4} /> : null}

      {activeTab === "Executive Summary" && (
        <div className="page-grid">
          <div className="metrics-grid span-12">
            <MetricCard label="Promoted Model" value={health?.model_name ?? "n/a"} tone="primary" />
            <MetricCard label="Selection Policy" value={summary?.manifest?.selection_policy ?? "n/a"} tone="primary" />
            <MetricCard label="Test VaR" value={formatCompact(summaryRow?.test_value_at_risk)} tone="primary" />
            <MetricCard label="Test Net Benefit" value={formatCompact(summaryRow?.test_net_benefit_at_k)} tone="gain" />
            <MetricCard label="Brier Score" value={formatNumber(summaryRow?.test_brier_score, 3)} tone="primary" />
          </div>
          <InteractiveLineChart
            title="Budget Frontier"
            eyebrow="Hover, click, and lock points"
            rows={frontier}
            xKey="budget_k"
            selectedX={selectedBudgetFraction}
            onSelectX={(value) => {
              setSelectedBudgetX(value);
              setBudgetPct(Math.round(Number(value) * 100));
            }}
            baselineKey="net_benefit_at_k"
            showOptimal
            series={[
              { key: "value_at_risk", label: "VaR@K", color: "#1c5d99" },
              { key: "net_benefit_at_k", label: "Net Benefit@K", color: "#2a9d8f" },
              { key: "lift_at_k", label: "Lift@K", color: "#bc4749" },
            ]}
          />
          <DecisionPanel
            budgetPct={budgetPct}
            simulationRow={policySimulationRow}
            frontierPoint={selectedFrontierPoint}
          />
          <ChartCard title="Recommendation Notes" eyebrow="Short readout" className="span-12">
            <div className="decision-note">{recommendationText(selectedFrontierPoint, policySimulationRow)}</div>
          </ChartCard>
          <DataTable
            title="Model Comparison"
            rows={comparison}
            highlightMetric="test_net_benefit_at_k"
            negativeMetric="test_net_benefit_at_k"
          />
        </div>
      )}

      {activeTab === "Targeting Strategy" && (
        <div className="page-grid">
          <div className="metrics-grid span-12">
            <MetricCard label="Selection Overlap" value={formatPct(policySimulationRow?.selection_overlap_at_k)} tone="primary" />
            <MetricCard label="Net Benefit Delta" value={formatCompact(policySimulationRow?.comparison_minus_baseline)} tone={Number(policySimulationRow?.comparison_minus_baseline ?? 0) >= 0 ? "gain" : "loss"} />
            <MetricCard label="Customers Targeted" value={selectedFrontierPoint?.targeted_count ?? "n/a"} tone="primary" />
            <MetricCard label="Risk Level" value={Number(policySimulationRow?.comparison_minus_baseline ?? 0) >= 0 ? "Medium" : "High"} tone="primary" />
          </div>
          <InteractiveLineChart
            title="Budget vs VaR vs Net Benefit"
            eyebrow="Click a point to update the whole dashboard"
            rows={frontier}
            xKey="budget_k"
            selectedX={selectedBudgetFraction}
            onSelectX={(value) => {
              setSelectedBudgetX(value);
              setBudgetPct(Math.round(Number(value) * 100));
            }}
            baselineKey="net_benefit_at_k"
            showOptimal
            series={[
              { key: "value_at_risk", label: "VaR@K", color: "#1d3557" },
              { key: "net_benefit_at_k", label: "Net Benefit@K", color: "#2a9d8f" },
            ]}
          />
          <DecisionPanel
            budgetPct={budgetPct}
            simulationRow={policySimulationRow}
            frontierPoint={selectedFrontierPoint}
            targetedCountOverride={targetSummary.totalRows}
          />
          <DataTable
            title="Top Target Customers"
            rows={targets}
            highlightMetric="policy_net_benefit"
            negativeMetric="policy_net_benefit"
          />
        </div>
      )}

      {activeTab === "Model Performance" && (
        <div className="page-grid">
          <InteractiveLineChart
            title="Backtest Stability Across Folds"
            eyebrow="Hover to inspect fold-level values"
            rows={backtest}
            xKey="fold"
            selectedX={backtest?.[0]?.fold}
            series={[
              { key: "value_at_risk", label: "VaR@K", color: "#1d3557" },
              { key: "net_benefit_at_k", label: "Net Benefit@K", color: "#2a9d8f" },
              { key: "roc_auc", label: "ROC-AUC", color: "#bc4749" },
            ]}
          />
          <DataTable
            title="Backtest Detail"
            rows={backtest}
            highlightMetric="net_benefit_at_k"
            negativeMetric="net_benefit_at_k"
          />
        </div>
      )}

      {activeTab === "Segment Analysis" && (
        <div className="page-grid">
          <ChartCard
            title="Segment Controls"
            className="span-12"
            actions={
              <label className="api-input compact">
                <span>Segment family</span>
                <select
                  value={segmentType}
                  onChange={(event) => {
                    setSegmentType(event.target.value);
                    setSelectedSegment("");
                  }}
                >
                  <option value="segment_value_band">Value bands</option>
                  <option value="segment_recency_bucket">Recency buckets</option>
                  <option value="segment_frequency_bucket">Frequency buckets</option>
                </select>
              </label>
            }
          >
            <div className="subtle">Click a segment card to filter the customer list in the Customer Explorer.</div>
          </ChartCard>
          <SegmentGroupedChart
            title="Segment Net Benefit Comparison"
            rows={segments}
            metricKey="net_benefit_at_k"
            metricLabel="Net Benefit@K"
            selectedSegment={selectedSegment}
            onSelectSegment={setSelectedSegment}
          />
          <SegmentGroupedChart
            title="Segment Lift Comparison"
            rows={segments}
            metricKey="lift_at_k"
            metricLabel="Lift@K"
            selectedSegment={selectedSegment}
            onSelectSegment={setSelectedSegment}
          />
          <DataTable
            title="Segment Detail"
            rows={segments}
            highlightMetric="net_benefit_at_k"
            negativeMetric="net_benefit_at_k"
          />
        </div>
      )}

      {activeTab === "Drift Monitoring" && (
        <div className="page-grid">
          <div className="metrics-grid span-12">
            <MetricCard label="Drift Status" value={drift?.alerts?.overall_status ?? "n/a"} tone={drift?.alerts?.overall_status === "ok" ? "gain" : drift?.alerts?.overall_status === "warn" ? "primary" : "loss"} />
            <MetricCard label="Latest PSI" value={formatNumber(latestPsi, 3)} tone={latestPsi >= 0.2 ? "loss" : latestPsi >= 0.1 ? "primary" : "gain"} />
            <MetricCard label="Trend Direction" value={trendDirection} tone="primary" />
            <MetricCard label="Rows Scored" value={drift?.n_rows_current ?? "n/a"} tone="primary" />
          </div>
          <InteractiveLineChart
            title="Decision Drift Over Time"
            eyebrow="Top-K decision quality by month"
            rows={decisionDrift}
            xKey="invoice_month"
            selectedX={decisionDrift?.[decisionDrift.length - 1]?.invoice_month}
            series={[
              { key: "avg_churn_prob_top_k", label: "Avg Churn Prob Top-K", color: "#1d3557" },
              { key: "avg_value_pos_top_k", label: "Avg Value Top-K", color: "#2a9d8f" },
              { key: "var_at_k", label: "VaR@K", color: "#bc4749" },
            ]}
          />
          <InteractiveLineChart
            title="Drift History"
            eyebrow="Thresholds shown for PSI"
            rows={[...driftHistory].reverse()}
            xKey="generated_at_utc"
            selectedX={driftHistory?.[0]?.generated_at_utc}
            thresholdLines={[
              { value: 0.1, label: "PSI warn = 0.1", tone: "warn" },
              { value: 0.2, label: "PSI critical = 0.2", tone: "critical" },
            ]}
            series={[
              { key: "top_psi", label: "Top PSI", color: "#bc4749" },
              { key: "score_mean", label: "Mean Score", color: "#1d3557" },
            ]}
          />
          <DataTable title="Feature Drift Detail" rows={Object.entries(drift?.features ?? {}).map(([feature, info]) => ({ feature, ...info }))} />
        </div>
      )}

      {activeTab === "Customer Explorer" && (
        <div className="page-grid">
          <ChartCard title="Customer Controls" className="span-12" eyebrow="Cross-filtered by budget and segment">
            {annotatedPredictions.length ? (
              <>
                <div className="controls-grid">
                  <label className="api-input">
                    <span>Segment filter</span>
                    <select value={selectedSegment} onChange={(event) => setSelectedSegment(event.target.value)}>
                      <option value="">All segments</option>
                      {[...new Set(annotatedPredictions.map((row) => row.local_segment_value))].map((segment) => (
                        <option key={segment} value={segment}>
                          {segment}
                        </option>
                      ))}
                    </select>
                  </label>
                  <label className="api-input">
                    <span>Customer-month row</span>
                    <select value={selectedPrediction} onChange={(event) => setSelectedPrediction(event.target.value)}>
                      {selectedSegmentPredictions.map((row) => {
                        const value = `${row.CustomerID}||${row.invoice_month}`;
                        return (
                          <option key={value} value={value}>
                            {row.CustomerID} | {row.invoice_month}
                          </option>
                        );
                      })}
                    </select>
                  </label>
                </div>
                {customerError ? <div className="customer-inline-note">{customerError}</div> : null}
                {!selectedSegmentPredictions.length ? (
                  <div className="customer-inline-note">
                    No customers matched the current budget and segment filter. Try a different segment or budget.
                  </div>
                ) : null}
              </>
            ) : (
              <EmptyState title="No scored predictions loaded" message="Run the scoring pipeline and start the API to explore customer rows." />
            )}
          </ChartCard>
          <div className="metrics-grid span-12">
            <MetricCard label="Rank Position" value={selectedCustomerRow.rank ?? "n/a"} tone="primary" />
            <MetricCard label="Churn Probability" value={formatPct(customerExplanation?.prediction?.churn_prob)} tone="primary" />
            <MetricCard label="Policy Score" value={formatCompact(customerExplanation?.prediction?.policy_ml)} tone="primary" />
            <MetricCard label="Policy Net Benefit" value={formatCompact(customerExplanation?.prediction?.policy_net_benefit)} tone={Number(customerExplanation?.prediction?.policy_net_benefit ?? 0) >= 0 ? "gain" : "loss"} />
          </div>
          <DecisionPanel
            budgetPct={budgetPct}
            simulationRow={policySimulationRow}
            frontierPoint={selectedFrontierPoint}
            targetedCountOverride={selectedSegmentPredictions.length || null}
          />
          <ChartCard title="Why this customer was selected" eyebrow="Model-driven explanation" className="span-12">
            <div className="decision-note">
              This customer was selected because their churn score and economic value place them near the top of the
              current policy ranking. The contribution chart below shows which features pushed the risk score up or down.
            </div>
            <div className="chat-answer compact">{llmDecision?.answer ?? "Select a customer to generate a grounded explanation."}</div>
          </ChartCard>
          <FeatureContributionChart title="Positive Contributors" rows={contributionPositiveRows} tone="blue" />
          <FeatureContributionChart title="Negative Contributors" rows={contributionNegativeRows} tone="amber" />
          <DataTable
            title="Customer Prediction Detail"
            rows={customerExplanation?.prediction ? [customerExplanation.prediction] : []}
            negativeMetric="policy_net_benefit"
          />
          <DataTable
            title="Filtered Customer List"
            rows={selectedSegmentPredictions}
            highlightMetric="policy_net_benefit"
            negativeMetric="policy_net_benefit"
          />
        </div>
      )}

      {activeTab === "Ask / Explain" && (
        <div className="page-grid">
          <ChartCard title="Ask the backend" className="span-5">
            <p className="subtle panel-copy">
              Ask for a business summary, drift readout, targeting recommendation, or a project walkthrough. This uses grounded backend tools.
            </p>
            <textarea rows={5} value={chatQuery} onChange={(event) => setChatQuery(event.target.value)} />
            <button type="button" className="retry-button" onClick={submitChatQuery}>
              Run Query
            </button>
          </ChartCard>
          <ChartCard title="Decision Narrative" className="span-7">
            <div className="chat-answer">
              {chatResult?.answer ?? "Ask a question like 'What should we do at 10% budget?', 'What changed in the data recently?', or 'How does this project work?'."}
            </div>
          </ChartCard>
          <DataTable title="Tool Calls" rows={chatResult?.tools_used ?? []} />
          <ChartCard title="Grounding Data" className="span-12">
            <pre>{JSON.stringify(chatResult?.raw_data ?? null, null, 2)}</pre>
          </ChartCard>
        </div>
      )}
    </div>
  );
}
