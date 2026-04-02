import { useMemo, useState } from "react";
import ChartCard from "./ChartCard.jsx";
import { EmptyState } from "./StatusState.jsx";
import { toDisplay } from "../utils/format.js";

export default function DataTable({
  title,
  rows,
  searchable = true,
  defaultLimit = 10,
  highlightMetric = null,
  negativeMetric = null,
}) {
  const [query, setQuery] = useState("");
  const [sortColumn, setSortColumn] = useState(null);
  const [sortDirection, setSortDirection] = useState("desc");
  const [expanded, setExpanded] = useState(false);

  const columns = rows?.length ? Object.keys(rows[0]) : [];
  const filteredRows = useMemo(() => {
    const filtered = (rows ?? []).filter((row) =>
      !query
        ? true
        : Object.values(row).some((value) =>
            String(value ?? "").toLowerCase().includes(query.toLowerCase()),
          ),
    );
    if (!sortColumn) return filtered;
    return [...filtered].sort((left, right) => {
      const a = left[sortColumn];
      const b = right[sortColumn];
      if (typeof a === "number" && typeof b === "number") {
        return sortDirection === "asc" ? a - b : b - a;
      }
      return sortDirection === "asc"
        ? String(a ?? "").localeCompare(String(b ?? ""))
        : String(b ?? "").localeCompare(String(a ?? ""));
    });
  }, [query, rows, sortColumn, sortDirection]);

  const visibleRows = expanded ? filteredRows : filteredRows.slice(0, defaultLimit);

  if (!rows?.length) {
    return (
      <ChartCard title={title}>
        <EmptyState title="No rows returned" message="This table will populate when the API returns data for the selected view." />
      </ChartCard>
    );
  }

  return (
    <ChartCard
      title={title}
      actions={
        <div className="panel-actions-inline">
          {searchable ? (
            <input
              className="table-search"
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder="Search rows"
            />
          ) : null}
          <button type="button" className="mini-toggle" onClick={() => setExpanded((value) => !value)}>
            {expanded ? "Show top 10" : "Expand"}
          </button>
        </div>
      }
    >
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              {columns.map((column) => (
                <th
                  key={column}
                  className="sortable"
                  onClick={() => {
                    if (sortColumn === column) {
                      setSortDirection((value) => (value === "asc" ? "desc" : "asc"));
                    } else {
                      setSortColumn(column);
                      setSortDirection("desc");
                    }
                  }}
                >
                  {column}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {visibleRows.map((row, index) => {
              const isTop = highlightMetric && index === 0;
              const isNegative = negativeMetric && Number(row[negativeMetric] ?? 0) < 0;
              return (
                <tr key={index} className={isTop ? "row-highlight" : isNegative ? "row-negative" : ""}>
                  {columns.map((column) => (
                    <td key={column}>{toDisplay(row[column])}</td>
                  ))}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      {!expanded && filteredRows.length > defaultLimit ? (
        <div className="table-footer-note">Showing top {defaultLimit} of {filteredRows.length} rows.</div>
      ) : null}
    </ChartCard>
  );
}
