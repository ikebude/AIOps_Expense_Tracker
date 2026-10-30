import { formatNGN } from '../lib/format'

const W = 600
const H = 220
const PAD = 36

export default function ForecastChart({ forecast, demo }) {
  const hist = Object.entries(forecast.historical_monthly_total_ngn)
  const fc = Object.entries(forecast.forecast_monthly_total_ngn)
  const all = [...hist, ...fc]
  const max = Math.max(...all.map(([, v]) => v), 1)
  const x = (i) => PAD + (i * (W - 2 * PAD)) / Math.max(all.length - 1, 1)
  const y = (v) => H - PAD - (v / max) * (H - 2 * PAD)
  const pts = (arr, off) => arr.map(([, v], i) => `${x(i + off)},${y(v)}`).join(' ')
  const joinFc = hist.length ? [hist[hist.length - 1], ...fc] : fc
  const wape = (forecast.holdout_wape * 100).toFixed(1)

  return (
    <section className="card" aria-labelledby="fc-h">
      <h2 id="fc-h">Spend forecast</h2>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-labelledby="fc-t fc-d" className="chart">
        <title id="fc-t">Monthly spend and forecast</title>
        <desc id="fc-d">Solid line is historical monthly spend, dashed line is the forecast.</desc>
        <polyline fill="none" stroke="var(--accent)" strokeWidth="2.5" points={pts(hist, 0)} />
        <polyline
          fill="none"
          stroke="var(--accent-2)"
          strokeWidth="2.5"
          strokeDasharray="6 5"
          points={joinFc.map(([, v], i) => `${x(i + hist.length - (hist.length ? 1 : 0))},${y(v)}`).join(' ')}
        />
        {all.map(([m], i) => (
          <text key={m} x={x(i)} y={H - 8} fontSize="10" textAnchor="middle" fill="var(--muted)">
            {m.slice(2)}
          </text>
        ))}
      </svg>
      <table className="sr-only">
        <caption>Monthly totals</caption>
        <thead>
          <tr><th scope="col">Month</th><th scope="col">Total</th><th scope="col">Type</th></tr>
        </thead>
        <tbody>
          {hist.map(([m, v]) => (<tr key={m}><td>{m}</td><td>{formatNGN(v)}</td><td>Actual</td></tr>))}
          {fc.map(([m, v]) => (<tr key={m}><td>{m}</td><td>{formatNGN(v)}</td><td>Forecast</td></tr>))}
        </tbody>
      </table>
      <p className="hint">
        Holdout WAPE {wape}%{demo ? ' (backtest on synthetic data)' : ''}. Solid line: actual. Dashed line: forecast.
      </p>
    </section>
  )
}
