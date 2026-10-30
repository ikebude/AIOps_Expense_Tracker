export function Skeleton({ rows = 3, label = 'Loading' }) {
  return (
    <div className="skeleton" aria-busy="true" aria-label={label} role="status">
      {Array.from({ length: rows }, (_, i) => (
        <div key={i} className="skeleton-row" />
      ))}
    </div>
  )
}

export function EmptyState({ children = 'No expenses yet — add your first one' }) {
  return <p className="empty">{children}</p>
}

export function ErrorState({ message, onRetry }) {
  return (
    <div className="error" role="alert">
      <p>{message}</p>
      {onRetry && (
        <button type="button" onClick={onRetry}>
          Retry
        </button>
      )}
    </div>
  )
}
