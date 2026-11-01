import { useCallback, useEffect, useState } from 'react'
import './App.css'
import {
  ApiError,
  getCategories,
  getForecast,
  getHealth,
  getSummary,
  listExpenses,
  setApiKey,
} from './lib/api'
import SummaryCards from './components/SummaryCards'
import ExpenseForm from './components/ExpenseForm'
import ReceiptParser from './components/ReceiptParser'
import ExpenseList from './components/ExpenseList'
import ForecastChart from './components/ForecastChart'
import ThemeToggle from './components/ThemeToggle'
import { EmptyState, ErrorState, Skeleton } from './components/StatusStates'

const PAGE = 50

export default function App() {
  const [status, setStatus] = useState('loading')
  const [error, setError] = useState('')
  const [health, setHealth] = useState(null)
  const [categories, setCategories] = useState([])
  const [month, setMonth] = useState(null)
  const [summary, setSummary] = useState(null)
  const [list, setList] = useState({ items: [], total: 0 })
  const [forecast, setForecast] = useState({ data: null, error: '' })
  const [prefill, setPrefill] = useState(null)

  const loadForecast = useCallback(async () => {
    try {
      setForecast({ data: await getForecast(3), error: '' })
    } catch (e) {
      setForecast({
        data: null,
        error: e.status === 422 ? 'Need at least 3 months of history' : e.message,
      })
    }
  }, [])

  const load = useCallback(
    async (m) => {
      try {
        const [h, c, s, l] = await Promise.all([
          getHealth(),
          getCategories(),
          getSummary(m ?? undefined),
          listExpenses({ month: m ?? undefined, limit: PAGE }),
        ])
        setHealth(h)
        setCategories(c.categories)
        setSummary(s)
        setList(l)
        setStatus('ready')
        loadForecast()
      } catch (e) {
        setError(e instanceof ApiError ? e.message : 'Something went wrong')
        setStatus('error')
      }
    },
    [loadForecast],
  )

  useEffect(() => {
    // Deferred so the initial fetch is not a synchronous setState inside the effect body.
    const t = setTimeout(() => load(null), 0)
    return () => clearTimeout(t)
  }, [load])

  const refreshSummary = async (m) => {
    try {
      setSummary(await getSummary(m ?? undefined))
    } catch {
      /* summary refresh is best-effort; list already updated */
    }
  }

  const onCreated = (row) => {
    setList((l) => ({ items: [row, ...l.items], total: l.total + 1 }))
    refreshSummary(month)
    loadForecast()
  }

  const onMonth = async (m) => {
    setMonth(m)
    try {
      const [s, l] = await Promise.all([getSummary(m), listExpenses({ month: m, limit: PAGE })])
      setSummary(s)
      setList(l)
    } catch (e) {
      setError(e.message)
      setStatus('error')
    }
  }

  const loadMore = async () => {
    try {
      const more = await listExpenses({ month: month ?? undefined, limit: PAGE, offset: list.items.length })
      setList((l) => ({ items: [...l.items, ...more.items], total: more.total }))
    } catch (e) {
      setError(e.message)
    }
  }

  return (
    <div className="app">
      <header className="top">
        <h1>AIOps Expense Tracker</h1>
        {health?.demo_data && <span className="pill">Demo data (synthetic)</span>}
        <label className="key">
          <span>API key (for adding expenses)</span>
          <input type="password" autoComplete="off" onChange={(e) => setApiKey(e.target.value)} />
        </label>
        <ThemeToggle />
      </header>

      {status === 'loading' && <Skeleton rows={4} label="Loading expenses" />}
      {status === 'error' && <ErrorState message={error} onRetry={() => {
            setStatus('loading')
            load(month)
          }} />}
      {status === 'ready' && (
        <main className="grid">
          <div className="col">
            <SummaryCards summary={summary} month={month} onMonth={onMonth} />
            <ExpenseForm categories={categories} prefill={prefill} onCreated={onCreated} />
            <ReceiptParser onParsed={setPrefill} />
          </div>
          <div className="col">
            {/* Forecast first: below a 50-row list it was ~6,500 px down on a phone. */}
            {forecast.data ? (
              <ForecastChart forecast={forecast.data} demo={Boolean(health?.demo_data)} />
            ) : (
              forecast.error && (
                <section className="card">
                  <h2>Spend forecast</h2>
                  <p className="hint">{forecast.error}</p>
                </section>
              )
            )}
            {list.items.length === 0 ? (
              <section className="card">
                <EmptyState />
              </section>
            ) : (
              <ExpenseList items={list.items} total={list.total} onLoadMore={loadMore} />
            )}
          </div>
        </main>
      )}
    </div>
  )
}
