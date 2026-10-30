import { formatNGN } from '../lib/format'

export default function SummaryCards({ summary, month, onMonth }) {
  const top = summary.by_category.slice(0, 3)
  const change = summary.change_pct
  return (
    <section aria-labelledby="summary-h" className="card">
      <h2 id="summary-h">Monthly summary</h2>
      {summary.available_months.length > 0 && (
        <label className="field">
          <span>Month</span>
          <select value={month ?? summary.month ?? ''} onChange={(e) => onMonth(e.target.value)}>
            {summary.available_months.map((m) => (
              <option key={m} value={m}>
                {m}
              </option>
            ))}
          </select>
        </label>
      )}
      <div className="stats">
        <div>
          <p className="stat-label">Total spent</p>
          <p className="stat-value">{formatNGN(summary.total_ngn)}</p>
        </div>
        <div>
          <p className="stat-label">Expenses</p>
          <p className="stat-value">{summary.count}</p>
        </div>
        <div>
          <p className="stat-label">Vs previous month</p>
          <p className="stat-value">
            {change === null || change === undefined
              ? 'n/a'
              : `${change > 0 ? '▲' : change < 0 ? '▼' : '='} ${Math.abs(change).toFixed(1)}%`}
          </p>
        </div>
      </div>
      {top.length > 0 && (
        <>
          <h3>Top categories</h3>
          <ol className="top-cats">
            {top.map((c) => (
              <li key={c.category}>
                <span>{c.category}</span> <strong>{formatNGN(c.total_ngn)}</strong>
              </li>
            ))}
          </ol>
        </>
      )}
    </section>
  )
}
