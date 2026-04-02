import { useMemo, useState } from "react";
import ChartCard from "./ChartCard.jsx";
import { EmptyState } from "./StatusState.jsx";
import { formatCompact, formatNumber, formatPct } from "../utils/format.js";

function normalizeSeries(rows, series) {
  return rows.map((row) => {
    const next = { ...row };
    series.forEach((item) => {
      const maxValue = Math.max(...rows.map((candidate) => Number(candidate[item.key] ?? 0)), 1e-6);
      next[`${item.key}__normalized`] = Number(row[item.key] ?? 0) / maxValue;
    });
    return next;
  });
}

export default function InteractiveLineChart({
  title,
  eyebrow,
  rows,
  xKey,
  series,
  selectedX,
  onSelectX,
  showOptimal = false,
  baselineKey = null,
  thresholdLines = [],
  actions = null,
}) {
  const [normalized, setNormalized] = useState(false);
  const [hoveredIndex, setHoveredIndex] = useState(null);
  const [lockedIndex, setLockedIndex] = useState(null);

  const processedRows = useMemo(
    () => (normalized ? normalizeSeries(rows, series) : rows),
    [normalized, rows, series],
  );

  if (!rows?.length) {
    return (
      <ChartCard
        title={title}
        eyebrow={eyebrow}
        actions={actions}
      >
        <EmptyState title="No chart data" message="Run the pipeline and API first so the chart has values to display." />
      </ChartCard>
    );
  }

  const width = 720;
  const height = 280;
  const padding = 40;
  const activeIndex = lockedIndex ?? hoveredIndex;
  const metricKeys = series.map((item) => (normalized ? `${item.key}__normalized` : item.key));
  const allValues = metricKeys.flatMap((key) => processedRows.map((row) => Number(row[key] ?? 0)));
  const maxValue = Math.max(...allValues, 1e-6);
  const minValue = Math.min(...allValues, 0);

  const pointsForKey = (key) =>
    processedRows.map((row, index) => {
      const x = padding + (index * (width - padding * 2)) / Math.max(processedRows.length - 1, 1);
      const value = Number(row[key] ?? 0);
      const y =
        height -
        padding -
        ((value - minValue) / Math.max(maxValue - minValue, 1e-6)) * (height - padding * 2);
      return { row, index, x, y, value };
    });

  const paths = series.map((item) => {
    const key = normalized ? `${item.key}__normalized` : item.key;
    const path = pointsForKey(key)
      .map((point) => `${point.index === 0 ? "M" : "L"} ${point.x} ${point.y}`)
      .join(" ");
    return { ...item, key, path, points: pointsForKey(key) };
  });

  const selectedIndex = processedRows.findIndex((row) => String(row[xKey]) === String(selectedX));
  const optimalIndex =
    showOptimal && !normalized
      ? processedRows.reduce((bestIndex, row, index, source) => (
          Number(row.net_benefit_at_k ?? -Infinity) > Number(source[bestIndex]?.net_benefit_at_k ?? -Infinity)
            ? index
            : bestIndex
        ), 0)
      : null;
  const displayIndex = activeIndex ?? (selectedIndex >= 0 ? selectedIndex : null);
  const displayRow = displayIndex !== null ? processedRows[displayIndex] : null;
  const baselineValue =
    displayRow && baselineKey ? Number(processedRows[0]?.[baselineKey] ?? 0) : null;
  const deltaFromBaseline =
    displayRow && baselineKey ? Number(displayRow[baselineKey] ?? 0) - baselineValue : null;

  return (
    <ChartCard
      title={title}
      eyebrow={eyebrow}
      actions={
        <div className="panel-actions-inline">
          {actions}
          <button
            type="button"
            className="mini-toggle"
            onClick={() => setNormalized((value) => !value)}
          >
            {normalized ? "Show absolute" : "Show normalized"}
          </button>
        </div>
      }
    >
      <div className="chart-shell">
        <svg viewBox={`0 0 ${width} ${height}`} className="line-chart interactive">
          <line x1={padding} y1={height - padding} x2={width - padding} y2={height - padding} className="axis-line" />
          <line x1={padding} y1={padding} x2={padding} y2={height - padding} className="axis-line" />
          {thresholdLines.map((threshold) => {
            const value = normalized ? threshold.value / Math.max(maxValue, 1e-6) : threshold.value;
            const y =
              height -
              padding -
              ((value - minValue) / Math.max(maxValue - minValue, 1e-6)) * (height - padding * 2);
            return (
              <g key={threshold.label}>
                <line x1={padding} y1={y} x2={width - padding} y2={y} className={`threshold-line ${threshold.tone ?? ""}`} />
                <text x={width - padding} y={y - 6} className="threshold-label">
                  {threshold.label}
                </text>
              </g>
            );
          })}
          {selectedIndex >= 0 ? (
            <line
              x1={paths[0].points[selectedIndex].x}
              y1={padding}
              x2={paths[0].points[selectedIndex].x}
              y2={height - padding}
              className="selection-line"
            />
          ) : null}
          {paths.map((path) => (
            <path key={path.label} d={path.path} fill="none" stroke={path.color} strokeWidth="3" opacity={displayIndex !== null && activeIndex !== null ? 0.45 : 1} />
          ))}
          {paths.map((path) =>
            path.points.map((point) => (
              <circle
                key={`${path.label}-${point.index}`}
                cx={point.x}
                cy={point.y}
                r={displayIndex === point.index ? 6 : 4}
                fill={path.color}
                className="chart-point"
                onMouseEnter={() => setHoveredIndex(point.index)}
                onMouseLeave={() => setHoveredIndex(null)}
                onClick={() => {
                  setLockedIndex((value) => (value === point.index ? null : point.index));
                  if (onSelectX) onSelectX(point.row[xKey]);
                }}
              />
            )),
          )}
          {optimalIndex !== null ? (
            <g>
              <circle
                cx={paths[0].points[optimalIndex].x}
                cy={paths[1]?.points?.[optimalIndex]?.y ?? paths[0].points[optimalIndex].y}
                r={8}
                className="optimal-point"
              />
              <text
                x={Math.min(paths[0].points[optimalIndex].x + 10, width - 180)}
                y={padding + 12}
                className="optimal-label"
              >
                {`Best K = ${formatPct(rows[optimalIndex][xKey], 0)} -> +${formatCompact(rows[optimalIndex].net_benefit_at_k)}`}
              </text>
            </g>
          ) : null}
        </svg>
        <div className="line-chart-labels">
          {processedRows.map((row, index) => (
            <button
              key={`${row[xKey]}-${index}`}
              type="button"
              className={String(row[xKey]) === String(selectedX) ? "chart-label active" : "chart-label"}
              onClick={() => onSelectX?.(row[xKey])}
            >
              {String(row[xKey])}
            </button>
          ))}
        </div>
        <div className="legend-row">
          {series.map((item) => (
            <div key={item.key} className="legend-item">
              <span className="legend-swatch" style={{ background: item.color }} />
              <span>{item.label}</span>
            </div>
          ))}
        </div>
        {displayRow ? (
          <div className="chart-tooltip-panel">
            <div className="tooltip-title">{String(displayRow[xKey])}</div>
            {series.map((item) => {
              const key = normalized ? `${item.key}__normalized` : item.key;
              const value = displayRow[key];
              return (
                <div key={item.key} className="tooltip-row">
                  <span>{item.label}</span>
                  <span>{normalized ? formatPct(value, 0) : formatNumber(value, 3)}</span>
                </div>
              );
            })}
            {baselineKey && !normalized ? (
              <div className="tooltip-row emphasis">
                <span>Delta vs baseline</span>
                <span>{deltaFromBaseline === null ? "n/a" : formatCompact(deltaFromBaseline)}</span>
              </div>
            ) : null}
          </div>
        ) : null}
      </div>
    </ChartCard>
  );
}
