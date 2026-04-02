export function LoadingGrid({ lines = 3 }) {
  return (
    <div className="skeleton-grid">
      {Array.from({ length: lines }).map((_, index) => (
        <div key={index} className="skeleton-block" />
      ))}
    </div>
  );
}

export function EmptyState({ title, message }) {
  return (
    <div className="empty-state-card">
      <div className="empty-title">{title}</div>
      <div className="empty-message">{message}</div>
    </div>
  );
}

export function ErrorState({ message, onRetry }) {
  return (
    <div className="error-state-card">
      <div className="empty-title">Could not load dashboard data</div>
      <div className="empty-message">{message}</div>
      {onRetry ? (
        <button type="button" className="retry-button" onClick={onRetry}>
          Retry
        </button>
      ) : null}
    </div>
  );
}
