export function deriveSegmentValue(row, segmentType) {
  const revenue = Number(row.rev_sum_90d ?? 0);
  const recency = Number(row.gap_days_prev ?? 0);
  const frequency = Number(row.freq_90d ?? 0);

  if (segmentType === "segment_value_band") {
    if (revenue <= 50) return "low_value";
    if (revenue <= 200) return "mid_value";
    return "high_value";
  }
  if (segmentType === "segment_recency_bucket") {
    if (recency <= 30) return "recent";
    if (recency <= 90) return "warming";
    return "stale";
  }
  if (frequency <= 2) return "low_frequency";
  if (frequency <= 5) return "mid_frequency";
  return "high_frequency";
}

export function addLocalSegments(rows, segmentType) {
  return rows.map((row) => ({
    ...row,
    local_segment_value: deriveSegmentValue(row, segmentType),
  }));
}
