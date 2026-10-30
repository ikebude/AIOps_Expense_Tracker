import { describe, test, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import App from './App'
import { setApiKey } from './lib/api'

const CATS = ['Feeding & Groceries', 'Miscellaneous']
const row = (o = {}) => ({
  id: 'E1', date: '2026-02-10', merchant: 'SHOPRITE', narration: 'POS SHOPRITE', channel: 'POS', bank: 'GTBank',
  amount_ngn: 12000, category: 'Feeding & Groceries', category_source: 'user', category_confidence: null,
  anomaly: { flag: false, z_mad: 0, reason: 'Within normal category band' }, ...o,
})
const summary = (o = {}) => ({
  month: '2026-02', available_months: ['2026-02', '2026-01'], total_ngn: 12000, count: 1,
  by_category: [{ category: 'Feeding & Groceries', total_ngn: 12000, count: 1 }],
  previous_month: '2026-01', previous_total_ngn: 10000, change_pct: 20, ...o,
})
const json = (data, status = 200) => ({ ok: status < 400, status, json: async () => data })

function mockApi(over = {}) {
  const routes = {
    'GET /health': () => json({ status: 'ok', demo_data: false, expenses_count: 1 }),
    'GET /v1/categories': () => json({ categories: CATS }),
    'GET /v1/summary': () => json(summary()),
    'GET /v1/expenses': () => json({ items: [row()], total: 1 }),
    'GET /v1/forecast': () => json({ detail: 'Need at least 3 months of history' }, 422),
    ...over,
  }
  const f = vi.fn(async (url, opts = {}) => {
    const path = new URL(url).pathname
    const h = routes[`${opts.method ?? 'GET'} ${path}`]
    if (!h) throw new Error(`unmocked ${opts.method ?? 'GET'} ${path}`)
    return h(opts)
  })
  vi.stubGlobal('fetch', f)
  return f
}

beforeEach(() => {
  document.documentElement.removeAttribute('data-theme')
  try { localStorage.clear() } catch { /* ignore */ }
  vi.stubGlobal('matchMedia', (q) => ({ matches: false, media: q, addEventListener() {}, removeEventListener() {} }))
})
afterEach(() => {
  vi.unstubAllGlobals()
  setApiKey('')
})

describe('App', () => {
  test('shows skeleton while loading then rows', async () => {
    mockApi()
    render(<App />)
    expect(screen.getByRole('status', { busy: true })).toHaveAttribute('aria-busy', 'true')
    expect((await screen.findAllByText('SHOPRITE')).length).toBeGreaterThan(0)
  })

  test('empty list shows empty state', async () => {
    mockApi({ 'GET /v1/expenses': () => json({ items: [], total: 0 }), 'GET /v1/summary': () => json(summary({ month: null, available_months: [], total_ngn: 0, count: 0, by_category: [], change_pct: null })) })
    render(<App />)
    expect(await screen.findByText(/No expenses yet/)).toBeInTheDocument()
  })

  test('500 shows error and Retry refetches', async () => {
    let fail = true
    mockApi({ 'GET /health': () => (fail ? json({ detail: 'boom' }, 500) : json({ status: 'ok', demo_data: false })) })
    render(<App />)
    expect(await screen.findByRole('alert')).toHaveTextContent('boom')
    fail = false
    await userEvent.click(screen.getByRole('button', { name: 'Retry' }))
    expect((await screen.findAllByText('SHOPRITE')).length).toBeGreaterThan(0)
  })

  test('network failure shows cannot reach message', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => { throw new TypeError('x') }))
    render(<App />)
    expect(await screen.findByRole('alert')).toHaveTextContent(/Cannot reach the API/)
  })

  test('adding an expense sends numeric amount and shows new row', async () => {
    const f = mockApi({
      'POST /v1/expenses': (opts) => json(row({ id: 'E2', merchant: 'NEW CO', amount_ngn: JSON.parse(opts.body).amount_ngn }), 201),
    })
    render(<App />)
    await screen.findAllByText('SHOPRITE')
    await userEvent.type(screen.getByLabelText('Narration'), 'Lunch')
    await userEvent.type(screen.getByLabelText('Amount (₦)'), '12,000')
    await userEvent.click(screen.getByRole('button', { name: 'Add expense' }))
    await screen.findByText('Expense added')
    const post = f.mock.calls.find(([, o]) => o?.method === 'POST')
    expect(typeof JSON.parse(post[1].body).amount_ngn).toBe('number')
    expect((await screen.findAllByText('₦12,000.00')).length).toBeGreaterThan(1)
  })

  test('invalid amount shows field error with aria-invalid', async () => {
    mockApi()
    render(<App />)
    await screen.findAllByText('SHOPRITE')
    await userEvent.type(screen.getByLabelText('Narration'), 'Lunch')
    await userEvent.type(screen.getByLabelText('Amount (₦)'), 'abc')
    await userEvent.click(screen.getByRole('button', { name: 'Add expense' }))
    expect(screen.getByLabelText(/Amount \(₦\)/)).toHaveAttribute('aria-invalid', 'true')
    expect(screen.getByText(/greater than zero/)).toBeInTheDocument()
  })

  test('API error on submit is shown', async () => {
    mockApi({ 'POST /v1/expenses': () => json({ detail: 'Invalid or missing API key' }, 401) })
    render(<App />)
    await screen.findAllByText('SHOPRITE')
    await userEvent.type(screen.getByLabelText('Narration'), 'Lunch')
    await userEvent.type(screen.getByLabelText('Amount (₦)'), '50')
    await userEvent.click(screen.getByRole('button', { name: 'Add expense' }))
    expect(await screen.findByText('Invalid or missing API key')).toBeInTheDocument()
  })

  test('typed API key is sent as X-API-Key', async () => {
    const f = mockApi({ 'POST /v1/expenses': () => json(row({ id: 'E3' }), 201) })
    render(<App />)
    await screen.findAllByText('SHOPRITE')
    await userEvent.type(screen.getByLabelText(/API key/), 'k123')
    await userEvent.type(screen.getByLabelText('Narration'), 'Lunch')
    await userEvent.type(screen.getByLabelText('Amount (₦)'), '50')
    await userEvent.click(screen.getByRole('button', { name: 'Add expense' }))
    await screen.findByText('Expense added')
    const post = f.mock.calls.find(([, o]) => o?.method === 'POST')
    expect(post[1].headers['X-API-Key']).toBe('k123')
  })

  test('receipt parse prefills amount and merchant', async () => {
    mockApi({ 'POST /v1/receipt/parse': () => json({ merchant: 'NNPC', date: '2026-02-10', total_ngn: 12000, suggested_category: 'Transport & Fuel', category_confidence: 0.8 }) })
    render(<App />)
    await screen.findAllByText('SHOPRITE')
    await userEvent.type(screen.getByLabelText('Paste receipt text'), 'TOTAL 12000')
    await userEvent.click(screen.getByRole('button', { name: 'Parse receipt text' }))
    await waitFor(() => expect(screen.getByLabelText('Amount (₦)')).toHaveValue('12000'))
    expect(screen.getByLabelText('Merchant')).toHaveValue('NNPC')
    expect(screen.getByText(/Suggested: Transport & Fuel/)).toBeInTheDocument()
  })

  test('receipt parser says it reads text, not images', async () => {
    mockApi()
    render(<App />)
    await screen.findAllByText('SHOPRITE')
    expect(screen.getByText(/text only\. It does not read images|only\. It does not read images/)).toBeInTheDocument()
  })

  test('anomaly row shows Unusual and the reason', async () => {
    mockApi({ 'GET /v1/expenses': () => json({ items: [row({ anomaly: { flag: true, z_mad: 5, reason: '₦900,000 is 9.0x category median' } })], total: 1 }) })
    render(<App />)
    expect((await screen.findAllByText('Unusual')).length).toBeGreaterThan(0)
    expect(screen.getAllByText(/9\.0x category median/).length).toBeGreaterThan(0)
  })

  test('auto-categorised rows are tagged', async () => {
    mockApi({ 'GET /v1/expenses': () => json({ items: [row({ category_source: 'ml_tfidf_logreg' })], total: 1 }) })
    render(<App />)
    expect((await screen.findAllByText('auto')).length).toBeGreaterThan(0)
  })

  test('theme toggle flips data-theme and persists', async () => {
    mockApi()
    render(<App />)
    await screen.findAllByText('SHOPRITE')
    await userEvent.click(screen.getByRole('button', { name: /Switch to dark theme/ }))
    expect(document.documentElement.dataset.theme).toBe('dark')
    expect(localStorage.getItem('theme')).toBe('dark')
    await userEvent.click(screen.getByRole('button', { name: /Switch to light theme/ }))
    expect(document.documentElement.dataset.theme).toBe('light')
  })

  test('demo pill shows only when API says demo data', async () => {
    mockApi({ 'GET /health': () => json({ status: 'ok', demo_data: true }) })
    render(<App />)
    expect(await screen.findByText('Demo data (synthetic)')).toBeInTheDocument()
  })

  test('summary shows total, change arrow and top category in naira', async () => {
    mockApi()
    const { container } = render(<App />)
    await screen.findAllByText('SHOPRITE')
    const sum = screen.getByRole('region', { name: 'Monthly summary' })
    expect(within(sum).getByText('▲ 20.0%')).toBeInTheDocument()
    expect(within(sum).getAllByText('₦12,000.00').length).toBeGreaterThan(0)
    expect(container.textContent).not.toMatch(/\$/)
  })

  test('forecast 422 shows need 3 months message', async () => {
    mockApi()
    render(<App />)
    expect(await screen.findByText('Need at least 3 months of history')).toBeInTheDocument()
  })

  test('forecast chart renders accessible svg and wape', async () => {
    mockApi({ 'GET /v1/forecast': () => json({ holdout_wape: 0.081, historical_monthly_total_ngn: { '2026-01': 100, '2026-02': 120 }, forecast_monthly_total_ngn: { '2026-03': 130 } }) })
    render(<App />)
    expect(await screen.findByRole('img', { name: /Monthly spend and forecast/ })).toBeInTheDocument()
    expect(screen.getByText(/Holdout WAPE 8\.1%/)).toBeInTheDocument()
  })

  test('changing month refetches summary and list', async () => {
    const f = mockApi()
    render(<App />)
    await screen.findAllByText('SHOPRITE')
    await userEvent.selectOptions(screen.getByLabelText('Month'), '2026-01')
    await waitFor(() => expect(f.mock.calls.some(([u]) => String(u).includes('month=2026-01'))).toBe(true))
  })

  test('load more appends next page', async () => {
    const f = mockApi({
      'GET /v1/expenses': (o) => json({ items: [row({ id: 'E' + Math.random(), merchant: 'M' + Math.random() })], total: 2 }, 200, o),
    })
    render(<App />)
    await screen.findAllByRole('button', { name: 'Load more' })
    await userEvent.click(screen.getAllByRole('button', { name: 'Load more' })[0])
    await waitFor(() => expect(f.mock.calls.some(([u]) => String(u).includes('offset=1'))).toBe(true))
  })
})
