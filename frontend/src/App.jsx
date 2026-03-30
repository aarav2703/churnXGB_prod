import { useEffect, useMemo, useState } from "react";
import { apiGet, apiPost } from "./api.js";

const tabs = [
  "Overview",
  "Targeting Simulator",
  "Customer Explanation",
  "Drift Monitoring",
  "Experiment Simulation",
  "Chat / Ask",
];

function MetricCard({ label, value, note }) {
  return (
    <div className="metric-card">
      <div className="metric-label">{label}</div>
      <div className="metric-value">{value}</div>
      {note ? <div className="metric-note">{note}</div> : null}
    </div>
  );
}

function DataTable({ rows }) {
  if (!rows || rows.length === 0) {
    return <div className="empty-state">No rows available.</div>;
  }
  const columns = Object.keys(rows[0]);
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            {columns.map((column) => (
              <th key={column}>{column}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, idx) => (
            <tr key={idx}>
              {columns.map((column) => (
                <td key={column}>{renderCell(row[column])}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function formatNumber(value, digits = 1) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "n/a";
  }
  return Number(value).toLocaleString(undefined, {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}

function formatPct(value, digits = 1) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "n/a";
  }
  return `${(Number(value) * 100).toFixed(digits)}%`;
}

function formatCompact(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "n/a";
  }
  return Number(value).toLocaleString(undefined, {
    notation: "compact",
    maximumFractionDigits: 1,
  });
}

function SegmentedBar({ segments }) {
  const total = segments.reduce((sum, segment) => sum + Math.max(segment.value, 0), 0);
  if (total <= 0) {
    return <div className="empty-state">No visual breakdown available.</div>;
  }
  return (
    <div className="segmented-bar">
      {segments.map((segment) => (
        <div
          key={segment.label}
          className="segmented-bar-piece"
          style={{
            width: `${(Math.max(segment.value, 0) / total) * 100}%`,
            background: segment.color,
          }}
          title={`${segment.label}: ${segment.value}`}
        />
      ))}
    </div>
  );
}

function ContributionBars({ title, rows, tone }) {
  if (!rows || rows.length === 0) {
    return (
      <div className="panel">
        <h2>{title}</h2>
        <div className="empty-state">No contributor rows available.</div>
      </div>
    );
  }

  const maxAbs = Math.max(
    ...rows.map((row) => Math.abs(Number(row.contribution_logit ?? 0))),
    1e-6,
  );

  return (
    <div className="panel">
      <h2>{title}</h2>
      <div className="contribution-list">
        {rows.map((row) => {
          const width = (Math.abs(Number(row.contribution_logit ?? 0)) / maxAbs) * 100;
          return (
            <div key={`${title}-${row.feature}`} className="contribution-row">
              <div className="contribution-meta">
                <div className="contribution-feature">{row.feature}</div>
                <div className="contribution-value">
                  contribution {formatNumber(row.contribution_logit, 3)} | value{" "}
                  {formatNumber(row.feature_value, 2)}
                </div>
              </div>
              <div className="contribution-track">
                <div
                  className={`contribution-fill ${tone}`}
                  style={{ width: `${width}%` }}
                />
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function TrendBars({ rows, valueKey, labelKey }) {
  if (!rows || rows.length === 0) {
    return <div className="empty-state">No trend history available.</div>;
  }
  const values = rows.map((row) => Number(row[valueKey] ?? 0));
  const maxValue = Math.max(...values, 1e-6);
  return (
    <div className="trend-bars">
      {rows.map((row, idx) => {
        const value = Number(row[valueKey] ?? 0);
        return (
          <div key={`${row[labelKey] ?? idx}`} className="trend-bar-col">
            <div className="trend-bar-wrap">
              <div
                className="trend-bar-fill"
                style={{ height: `${(value / maxValue) * 100}%` }}
              />
            </div>
            <div className="trend-bar-label">{String(row[labelKey] ?? idx).slice(5, 10)}</div>
          </div>
        );
      })}
    </div>
  );
}

function HorizontalMetricChart({ title, rows, labelKey, valueKey, formatter = formatNumber }) {
  if (!rows || rows.length === 0) {
    return (
      <div className="panel">
        <h2>{title}</h2>
        <div className="empty-state">No chart data available.</div>
      </div>
    );
  }

  const maxValue = Math.max(...rows.map((row) => Number(row[valueKey] ?? 0)), 1e-6);
  return (
    <div className="panel">
      <h2>{title}</h2>
      <div className="bar-chart-list">
        {rows.map((row) => {
          const value = Number(row[valueKey] ?? 0);
          const width = (value / maxValue) * 100;
          return (
            <div key={`${title}-${row[labelKey]}`} className="bar-chart-row">
              <div className="bar-chart-header">
                <span>{row[labelKey]}</span>
                <span>{formatter(value)}</span>
              </div>
              <div className="bar-chart-track">
                <div className="bar-chart-fill" style={{ width: `${width}%` }} />
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function renderCell(value) {
  if (value === null || value === undefined) return "";
  if (typeof value === "number") return Number(value).toFixed(3);
  if (Array.isArray(value)) return value.join(", ");
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
}

export default function App() {
  const [baseUrl, setBaseUrl] = useState("http://127.0.0.1:8000");
  const [activeTab, setActiveTab] = useState("Overview");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [health, setHealth] = useState(null);
  const [modelSummary, setModelSummary] = useState(null);
  const [policyMetrics, setPolicyMetrics] = useState(null);
  const [modelComparison, setModelComparison] = useState(null);
  const [featureImportance, setFeatureImportance] = useState(null);
  const [drift, setDrift] = useState(null);
  const [driftHistory, setDriftHistory] = useState(null);
  const [targetsBudget, setTargetsBudget] = useState(10);
  const [targets, setTargets] = useState(null);
  const [policySimulation, setPolicySimulation] = useState(null);
  const [experimentSimulation, setExperimentSimulation] = useState(null);
  const [predictions, setPredictions] = useState(null);
  const [selectedPrediction, setSelectedPrediction] = useState("");
  const [customerExplanation, setCustomerExplanation] = useState(null);
  const [chatQuery, setChatQuery] = useState("Who should I target at 10% budget?");
  const [chatResult, setChatResult] = useState(null);
  const [customerLoadError, setCustomerLoadError] = useState("");

  const normalizedBaseUrl = useMemo(() => baseUrl.replace(/\/$/, ""), [baseUrl]);
  const policyResult = policySimulation?.results?.[0] ?? null;
  const experimentResult = experimentSimulation?.results?.[0] ?? null;
  const comparisonRow = modelSummary?.comparison_row ?? null;
  const driftHistoryRows = driftHistory?.rows ?? [];
  const explanationMethod = customerExplanation?.explanation_method ?? null;

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

  useEffect(() => {
    runLoad(async () => {
      const [healthRes, summaryRes] = await Promise.all([
        apiGet(normalizedBaseUrl, "/health"),
        apiGet(normalizedBaseUrl, "/model-summary"),
      ]);
      setHealth(healthRes);
      setModelSummary(summaryRes);
    });
  }, [normalizedBaseUrl]);

  useEffect(() => {
    if (activeTab !== "Overview") return;
    runLoad(async () => {
      const [policy, comparison, importance] = await Promise.all([
        apiGet(normalizedBaseUrl, "/policy-metrics"),
        apiGet(normalizedBaseUrl, "/model-comparison"),
        apiGet(normalizedBaseUrl, "/feature-importance?limit=8"),
      ]);
      setPolicyMetrics(policy);
      setModelComparison(comparison);
      setFeatureImportance(importance);
    });
  }, [activeTab, normalizedBaseUrl]);

  useEffect(() => {
    if (activeTab !== "Targeting Simulator") return;
    runLoad(async () => {
      const [targetRows, simulation] = await Promise.all([
        apiGet(normalizedBaseUrl, `/targets/${targetsBudget}?limit=50`),
        apiPost(normalizedBaseUrl, "/simulate-policy", {
          budgets: [targetsBudget / 100],
        }),
      ]);
      setTargets(targetRows);
      setPolicySimulation(simulation);
    });
  }, [activeTab, normalizedBaseUrl, targetsBudget]);

  useEffect(() => {
    if (activeTab !== "Customer Explanation") return;
    runLoad(async () => {
      setCustomerLoadError("");
      setCustomerExplanation(null);
      try {
        const predictionRes = await apiGet(
          normalizedBaseUrl,
          "/predictions?limit=100&sort_by=policy_net_benefit",
        );
        setPredictions(predictionRes);
        const first = predictionRes.rows?.[0];
        if (first) {
          const key = `${first.CustomerID}||${first.invoice_month}`;
          setSelectedPrediction(key);
        } else {
          setSelectedPrediction("");
          setCustomerLoadError(
            "No scored customer rows were returned. Run the offline pipeline first so the explanation page has saved predictions to inspect.",
          );
        }
      } catch (err) {
        setPredictions(null);
        setSelectedPrediction("");
        setCustomerLoadError(
          err.message ||
            "The frontend could not load saved predictions. Make sure the API is running and the offline pipeline has been scored.",
        );
      }
    });
  }, [activeTab, normalizedBaseUrl]);

  useEffect(() => {
    if (activeTab !== "Customer Explanation") return;
    if (!selectedPrediction) return;
    const [customerId, invoiceMonth] = selectedPrediction.split("||");
    runLoad(async () => {
      try {
        const explanation = await apiGet(
          normalizedBaseUrl,
          `/customers/explain?customer_id=${encodeURIComponent(customerId)}&invoice_month=${encodeURIComponent(invoiceMonth)}&top_n=5`,
        );
        setCustomerExplanation(explanation);
      } catch (err) {
        setCustomerExplanation(null);
        setCustomerLoadError(
          err.message ||
            "The explanation request failed. Check that the API is running and the selected customer exists in saved predictions.",
        );
      }
    });
  }, [activeTab, normalizedBaseUrl, selectedPrediction]);

  useEffect(() => {
    if (activeTab !== "Drift Monitoring") return;
    runLoad(async () => {
      const [driftRes, driftHistoryRes] = await Promise.all([
        apiGet(normalizedBaseUrl, "/drift/latest"),
        apiGet(normalizedBaseUrl, "/drift/history?limit=50"),
      ]);
      setDrift(driftRes);
      setDriftHistory(driftHistoryRes);
    });
  }, [activeTab, normalizedBaseUrl]);

  useEffect(() => {
    if (activeTab !== "Experiment Simulation") return;
    runLoad(async () => {
      const simulation = await apiPost(normalizedBaseUrl, "/simulate-experiment", {
        budgets: [targetsBudget / 100],
      });
      setExperimentSimulation(simulation);
    });
  }, [activeTab, normalizedBaseUrl, targetsBudget]);

  const selectedPredictionRows = customerExplanation
    ? [customerExplanation.prediction]
    : [];

  async function submitChatQuery() {
    await runLoad(async () => {
      const result = await apiPost(normalizedBaseUrl, "/llm/query", {
        query: chatQuery,
        include_raw_data: true,
      });
      setChatResult(result);
    });
  }

  return (
    <div className="app-shell">
      <header className="app-header">
        <div>
          <div className="eyebrow">Decision Support UI</div>
          <h1>ChurnXGB React Frontend</h1>
          <p className="subtle">
            React UI backed by FastAPI endpoints. Backend remains the source of
            truth for scoring, simulation, and explanations.
          </p>
        </div>
        <label className="api-input">
          <span>API Base URL</span>
          <input value={baseUrl} onChange={(e) => setBaseUrl(e.target.value)} />
        </label>
      </header>

      <section className="hero-strip">
        <div className="hero-copy">
          <div className="hero-title">Decision system, not just score tables</div>
          <div className="hero-text">
            Explore targeting, explanations, drift, and simulations from the backend as
            a single decision workflow.
          </div>
        </div>
        <button
          type="button"
          className="hero-cta"
          onClick={() => setActiveTab("Chat / Ask")}
        >
          Open Chat / Ask
        </button>
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

      {loading ? <div className="banner info">Loading...</div> : null}
      {error ? <div className="banner error">{error}</div> : null}

      {activeTab === "Overview" && (
        <section className="page-grid">
          <div className="metrics-grid">
            <MetricCard label="API Status" value={health?.status ?? "n/a"} />
            <MetricCard label="Promoted Model" value={health?.model_name ?? "n/a"} />
            <MetricCard
              label="Best Model"
              value={modelSummary?.manifest?.best_model ?? "n/a"}
            />
            <MetricCard
              label="Chosen Budget"
              value={modelSummary?.manifest?.chosen_budget ?? "n/a"}
            />
          </div>
          <div className="panel span-7">
            <h2>Decision Snapshot</h2>
            <div className="snapshot-grid">
              <div className="snapshot-card">
                <div className="snapshot-label">Selection Policy</div>
                <div className="snapshot-value">
                  {modelSummary?.manifest?.selection_policy ?? "n/a"}
                </div>
              </div>
              <div className="snapshot-card">
                <div className="snapshot-label">Test VaR</div>
                <div className="snapshot-value">
                  {formatCompact(comparisonRow?.test_value_at_risk)}
                </div>
              </div>
              <div className="snapshot-card">
                <div className="snapshot-label">Test Net Benefit</div>
                <div className="snapshot-value">
                  {formatCompact(comparisonRow?.test_net_benefit_at_k)}
                </div>
              </div>
              <div className="snapshot-card">
                <div className="snapshot-label">ROC-AUC</div>
                <div className="snapshot-value">{formatNumber(comparisonRow?.test_roc_auc, 3)}</div>
              </div>
            </div>
          </div>
          <div className="panel span-5">
            <h2>Policy Strength</h2>
            <div className="meter-stack">
              <div>
                <div className="meter-label">Value at risk captured</div>
                <SegmentedBar
                  segments={[
                    {
                      label: "covered",
                      value: Number(comparisonRow?.test_var_covered_frac ?? 0),
                      color: "linear-gradient(90deg, #113c68, #2d6aa3)",
                    },
                    {
                      label: "remaining",
                      value: 1 - Number(comparisonRow?.test_var_covered_frac ?? 0),
                      color: "#dfe8ef",
                    },
                  ]}
                />
                <div className="meter-caption">
                  {formatPct(comparisonRow?.test_var_covered_frac)}
                </div>
              </div>
              <div>
                <div className="meter-label">Precision at chosen budget</div>
                <SegmentedBar
                  segments={[
                    {
                      label: "precision",
                      value: Number(comparisonRow?.test_precision_at_k ?? 0),
                      color: "linear-gradient(90deg, #a85d2d, #d28b49)",
                    },
                    {
                      label: "other",
                      value: 1 - Number(comparisonRow?.test_precision_at_k ?? 0),
                      color: "#ece8e1",
                    },
                  ]}
                />
                <div className="meter-caption">
                  {formatPct(comparisonRow?.test_precision_at_k)}
                </div>
              </div>
            </div>
          </div>
          <div className="panel">
            <h2>Model Summary</h2>
            <pre>{JSON.stringify(modelSummary, null, 2)}</pre>
          </div>
          <div className="panel">
            <h2>Policy Metrics</h2>
            <DataTable rows={policyMetrics?.rows ?? []} />
          </div>
          <HorizontalMetricChart
            title="Model Comparison By Test Net Benefit"
            rows={modelComparison?.rows ?? []}
            labelKey="model"
            valueKey="test_net_benefit_at_k"
            formatter={(value) => formatCompact(value)}
          />
          <HorizontalMetricChart
            title="Global Feature Importance"
            rows={featureImportance?.rows ?? []}
            labelKey="feature"
            valueKey="importance"
            formatter={(value) => formatNumber(value, 3)}
          />
        </section>
      )}

      {activeTab === "Targeting Simulator" && (
        <section className="page-grid">
          <div className="panel controls">
            <h2>Budget</h2>
            <select
              value={targetsBudget}
              onChange={(e) => setTargetsBudget(Number(e.target.value))}
            >
              <option value={5}>5%</option>
              <option value={10}>10%</option>
              <option value={20}>20%</option>
            </select>
          </div>
          <div className="metrics-grid">
            <MetricCard
              label="Ranking Changed"
              value={policySimulation?.results?.[0]?.ranking_changed ? "Yes" : "No"}
            />
            <MetricCard
              label="% Rank Changed"
              value={
                policySimulation?.results?.[0]?.pct_customers_rank_changed !== undefined
                  ? `${(policySimulation.results[0].pct_customers_rank_changed * 100).toFixed(1)}%`
                  : "n/a"
              }
            />
            <MetricCard
              label="Only In VaR Top-K"
              value={policySimulation?.results?.[0]?.n_selected_only_baseline ?? "n/a"}
            />
            <MetricCard
              label="Only In Net-Benefit Top-K"
              value={policySimulation?.results?.[0]?.n_selected_only_comparison ?? "n/a"}
            />
          </div>
          <div className="panel span-6">
            <h2>Selection Overlap</h2>
            <p className="subtle panel-copy">
              This compares how much the baseline ranking and net-benefit ranking overlap
              at the selected budget.
            </p>
            <SegmentedBar
              segments={[
                {
                  label: "overlap",
                  value: Number(policyResult?.selection_overlap_at_k ?? 0),
                  color: "linear-gradient(90deg, #153e75, #2c7bc6)",
                },
                {
                  label: "different",
                  value: 1 - Number(policyResult?.selection_overlap_at_k ?? 0),
                  color: "linear-gradient(90deg, #e7b18c, #d2743a)",
                },
              ]}
            />
            <div className="meter-caption">
              overlap {formatPct(policyResult?.selection_overlap_at_k)}
            </div>
          </div>
          <div className="panel span-6">
            <h2>Economic Difference</h2>
            <div className="big-number">
              {formatCompact(policyResult?.comparison_minus_baseline)}
            </div>
            <div className="subtle">
              Comparison policy minus baseline at the selected budget.
            </div>
          </div>
          <div className="panel">
            <h2>Policy Simulation</h2>
            <p className="subtle">
              Assumption-driven threshold analysis with flat configured intervention
              cost and success rate. This is not causal inference.
            </p>
            <pre>{JSON.stringify(policySimulation, null, 2)}</pre>
          </div>
          <div className="panel">
            <h2>Target List</h2>
            <DataTable rows={targets?.rows ?? []} />
          </div>
        </section>
      )}

      {activeTab === "Customer Explanation" && (
        <section className="page-grid">
          <div className="panel controls span-4">
            <h2>Customer Row</h2>
            <p className="subtle panel-copy">
              This page explains a saved scored customer row from the backend. If this is
              empty, the API is unreachable or the offline scoring outputs are missing.
            </p>
            <select
              value={selectedPrediction}
              onChange={(e) => setSelectedPrediction(e.target.value)}
              disabled={!predictions?.rows?.length}
            >
              {!predictions?.rows?.length ? (
                <option value="">No customer rows loaded</option>
              ) : null}
              {(predictions?.rows ?? []).map((row) => {
                const key = `${row.CustomerID}||${row.invoice_month}`;
                return (
                  <option key={key} value={key}>
                    {row.CustomerID} - {row.invoice_month}
                  </option>
                );
              })}
            </select>
            {customerLoadError ? <div className="inline-error">{customerLoadError}</div> : null}
          </div>
          <div className="panel span-8">
            <h2>Prediction</h2>
            <p className="subtle">
              {explanationMethod === "logistic_pipeline_logit_contributions"
                ? "For the logistic-regression explanation path, contributions are shown relative to the standardized baseline, which corresponds to the training-data mean after scaling."
                : "This explanation comes from the backend explanation path for the selected scored customer."}
            </p>
            {customerExplanation ? (
              <div className="snapshot-grid">
                <div className="snapshot-card">
                  <div className="snapshot-label">Customer</div>
                  <div className="snapshot-value">
                    {customerExplanation.identifiers?.CustomerID ?? "n/a"}
                  </div>
                </div>
                <div className="snapshot-card">
                  <div className="snapshot-label">Month</div>
                  <div className="snapshot-value">
                    {customerExplanation.identifiers?.invoice_month ?? "n/a"}
                  </div>
                </div>
                <div className="snapshot-card">
                  <div className="snapshot-label">Churn Probability</div>
                  <div className="snapshot-value">
                    {formatPct(customerExplanation.prediction?.churn_prob)}
                  </div>
                </div>
                <div className="snapshot-card">
                  <div className="snapshot-label">Policy Net Benefit</div>
                  <div className="snapshot-value">
                    {formatNumber(customerExplanation.prediction?.policy_net_benefit, 2)}
                  </div>
                </div>
              </div>
            ) : (
              <div className="empty-state">
                Pick a customer row after the backend loads saved predictions.
              </div>
            )}
          </div>
          <ContributionBars
            title="Top Positive Contributors"
            rows={customerExplanation?.top_positive_contributors ?? []}
            tone="positive"
          />
          <ContributionBars
            title="Top Negative Contributors"
            rows={customerExplanation?.top_negative_contributors ?? []}
            tone="negative"
          />
        </section>
      )}

      {activeTab === "Drift Monitoring" && (
        <section className="page-grid">
          <div className="metrics-grid">
            <MetricCard label="Features OK" value={drift?.summary?.n_ok ?? "n/a"} />
            <MetricCard label="Warnings" value={drift?.summary?.n_warn ?? "n/a"} />
            <MetricCard label="Alerts" value={drift?.summary?.n_alert ?? "n/a"} />
            <MetricCard label="Rows Scored" value={drift?.n_rows_current ?? "n/a"} />
          </div>
          <div className="panel">
            <h2>Monitoring Status</h2>
            <div className="snapshot-grid">
              <div className="snapshot-card">
                <div className="snapshot-label">Overall Status</div>
                <div className="snapshot-value">
                  {drift?.alerts?.overall_status ?? "n/a"}
                </div>
              </div>
              <div className="snapshot-card">
                <div className="snapshot-label">Top PSI</div>
                <div className="snapshot-value">
                  {formatNumber(drift?.alerts?.top_psi ?? drift?.summary?.top_psi, 3)}
                </div>
              </div>
              <div className="snapshot-card">
                <div className="snapshot-label">Generated</div>
                <div className="snapshot-value small">
                  {drift?.generated_at_utc ?? "n/a"}
                </div>
              </div>
            </div>
          </div>
          <div className="panel span-7">
            <h2>Feature Drift</h2>
            <DataTable
              rows={Object.entries(drift?.features ?? {}).map(([feature, info]) => ({
                feature,
                ...info,
              }))}
            />
          </div>
          <div className="panel span-5">
            <h2>Score Distribution</h2>
            <pre>{JSON.stringify(drift?.score_current_stats ?? {}, null, 2)}</pre>
          </div>
          <div className="panel">
            <h2>Drift History</h2>
            <TrendBars
              rows={[...driftHistoryRows].reverse()}
              valueKey="top_psi"
              labelKey="generated_at_utc"
            />
            <DataTable rows={driftHistoryRows} />
          </div>
        </section>
      )}

      {activeTab === "Experiment Simulation" && (
        <section className="page-grid">
          <div className="panel controls">
            <h2>Budget</h2>
            <select
              value={targetsBudget}
              onChange={(e) => setTargetsBudget(Number(e.target.value))}
            >
              <option value={5}>5%</option>
              <option value={10}>10%</option>
              <option value={20}>20%</option>
            </select>
          </div>
          <div className="metrics-grid">
            <MetricCard
              label="Targeted Customers"
              value={experimentSimulation?.results?.[0]?.targeted_customers ?? "n/a"}
            />
            <MetricCard
              label="Treatment Customers"
              value={experimentSimulation?.results?.[0]?.treatment_customers ?? "n/a"}
            />
            <MetricCard
              label="Control Customers"
              value={experimentSimulation?.results?.[0]?.control_customers ?? "n/a"}
            />
            <MetricCard
              label="Incremental Retained Value"
              value={
                experimentSimulation?.results?.[0]?.incremental_retained_value !== undefined
                  ? experimentSimulation.results[0].incremental_retained_value.toFixed(2)
                  : "n/a"
              }
            />
          </div>
          <div className="panel span-6">
            <h2>Treatment Mix</h2>
            <SegmentedBar
              segments={[
                {
                  label: "treatment",
                  value: Number(experimentResult?.treatment_customers ?? 0),
                  color: "linear-gradient(90deg, #12476f, #2e84c9)",
                },
                {
                  label: "control",
                  value: Number(experimentResult?.control_customers ?? 0),
                  color: "linear-gradient(90deg, #d99c52, #b86825)",
                },
              ]}
            />
          </div>
          <div className="panel span-6">
            <h2>Average Incremental Value</h2>
            <div className="big-number">
              {formatNumber(
                experimentResult?.average_incremental_retained_value_per_treated_customer,
                2,
              )}
            </div>
            <div className="subtle">Per treated customer in the simulated experiment.</div>
          </div>
          <div className="panel">
            <h2>Experiment Simulation</h2>
            <p className="subtle">
              Deterministic business-case simulation only. The backend does not
              estimate causal uplift or experimental confidence intervals from this
              repo.
            </p>
            <pre>{JSON.stringify(experimentSimulation, null, 2)}</pre>
          </div>
        </section>
      )}

      {activeTab === "Chat / Ask" && (
        <section className="page-grid">
          <div className="panel controls span-5">
            <h2>Ask The System</h2>
            <p className="subtle">
              This chat layer routes to backend tools first, then summarizes tool
              output. It does not generate scores directly.
            </p>
            <textarea
              rows={5}
              value={chatQuery}
              onChange={(e) => setChatQuery(e.target.value)}
            />
            <button type="button" className="tab active" onClick={submitChatQuery}>
              Run Query
            </button>
          </div>
          <div className="panel span-7">
            <h2>Answer</h2>
            <div className="chat-answer">
              {chatResult?.answer ?? "Ask a question like 'Who should I target at 10% budget?' or 'What changed in the data recently?'."}
            </div>
          </div>
          <div className="panel">
            <h2>Tools Used</h2>
            <DataTable rows={chatResult?.tools_used ?? []} />
          </div>
          <div className="panel">
            <h2>Raw Tool Data</h2>
            <pre>{JSON.stringify(chatResult?.raw_data ?? null, null, 2)}</pre>
          </div>
        </section>
      )}
    </div>
  );
}
