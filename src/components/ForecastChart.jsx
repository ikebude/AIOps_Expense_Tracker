import { formatNGN } from '../lib/format'

// Small viewBox so 12-unit text stays legible when scaled to a 320px phone; CSS caps the desktop size.
const W = 400
const H = 190
const PAD_T = 14
const PAD_B = 40 // month + year labels
const PAD_R = 20
const FONT = 12
const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
const LEFT = 56 // room for y-axis labels
const compactNGN = (n) =>
  new Intl.NumberFormat('en-NG', { style: 'currency', currency: 'NGN', notation: 'compact', maximumFractionDigits: 2 }).format(n)

export default function ForecastChart({ forecast, demo }) {
  const hist = Object.entries(forecast.historical_monthly_total_ngn)
  const fc = Object.entries(forecast.forecast_monthly_total_ngn)
  const all = [...hist, ...fc]
  const max = Math.max(...all.map(([, v]) => v), 1)
  const x = (i) => LEFT + 14 + (i * (W - LEFT - 14 - PAD_R)) / Math.max(all.length - 1, 1)
  const y = (v) => H - PAD_B - (v / max) * (H - PAD_B - PAD_T)
  const pts = (arr, off) => arr.map(([, v], i) => `${x(i + off)},${y(v)}`).join(' ')
  const joinFc = hist.length ? [hist[hist.length - 1], ...fc] : fc
  const wape = (forecast.holdout_wape * 100).toFixed(1)

  return (
    <section className="card" aria-labelledby="fc-h">
      <h2 id="fc-h">Spend forecast</h2>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-labelledby="fc-t fc-d" className="chart">
        <title id="fc-t">Monthly spend and forecast</title>
        <desc id="fc-d">Solid line is historical monthly spend, dashed line is the forecast.</desc>
        {[0, max].map((v) => (
          <g key={v}>
            <line x1={LEFT} x2={W - PAD_R} y1={y(v)} y2={y(v)} stroke="var(--border)" />
            <text x={LEFT - 6} y={y(v) + 4} fontSize={FONT} textAnchor="end" fill="var(--muted)">
              {compactNGN(v)}
            </text>
          </g>
        ))}
        <polyline fill="none" stroke="var(--accent)" strokeWidth="2.5" points={pts(hist, 0)} />
        <polyline
          fill="none"
          stroke="var(--accent-2)"
          strokeWidth="2.5"
          strokeDasharray="6 5"
          points={joinFc.map(([, v], i) => `${x(i + hist.length - (hist.length ? 1 : 0))},${y(v)}`).join(' ')}
        />
        {all.map(([m], i) => (
          <text key={m} x={x(i)} y={H - PAD_B + 18} fontSize={FONT} textAnchor="middle" fill="var(--muted)">
            {MONTHS[Number(m.slice(5, 7)) - 1]}
            {(i === 0 || m.endsWith('-01')) && (
              <tspan x={x(i)} dy={FONT + 3}>
                {m.slice(0, 4)}
              </tspan>
            )}
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
