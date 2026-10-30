import { formatNGN } from '../lib/format'

function Anomaly({ a }) {
  if (!a?.flag) return null
  return (
    <details className="anomaly">
      <summary>
        <span className="badge">Unusual</span>
      </summary>
      <span>{a.reason}</span>
    </details>
  )
}

const isAuto = (e) => e.category_source !== 'user' && e.category_source !== 'seed'

export default function ExpenseList({ items, total, onLoadMore }) {
  return (
    <section className="card" aria-labelledby="list-h">
      <h2 id="list-h">Expenses</h2>
      <ul className="cards">
        {items.map((e) => (
          <li key={e.id}>
            <div className="row-top">
              <span>{e.date}</span>
              <strong>{formatNGN(e.amount_ngn)}</strong>
            </div>
            <div>{e.merchant || e.narration}</div>
            <div className="meta">
              {e.category}
              {isAuto(e) && <span className="tag">auto</span>}
            </div>
            <Anomaly a={e.anomaly} />
          </li>
        ))}
      </ul>
      <table className="wide">
        <caption>Expenses, newest first</caption>
        <thead>
          <tr>
            <th scope="col">Date</th>
            <th scope="col">Merchant / narration</th>
            <th scope="col">Category</th>
            <th scope="col" className="num">Amount</th>
          </tr>
        </thead>
        <tbody>
          {items.map((e) => (
            <tr key={e.id}>
              <td>{e.date}</td>
              <td>
                {e.merchant || e.narration}
                <Anomaly a={e.anomaly} />
              </td>
              <td>
                {e.category}
                {isAuto(e) && <span className="tag">auto</span>}
              </td>
              <td className="num">{formatNGN(e.amount_ngn)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {items.length < total && (
        <button type="button" onClick={onLoadMore}>
          Load more
        </button>
      )}
    </section>
  )
}
