export const API_BASE = (import.meta.env.VITE_API_BASE ?? 'http://localhost:8000').replace(/\/$/, '')

export class ApiError extends Error {
  constructor(status, message) {
    super(message)
    this.name = 'ApiError'
    this.status = status
  }
}

// Optional key typed by the user at runtime. Kept in memory only; never bundled.
let apiKey = ''
export const setApiKey = (k) => {
  apiKey = k || ''
}

function detailMessage(detail, fallback) {
  if (typeof detail === 'string') return detail
  if (Array.isArray(detail) && detail[0]?.msg) return detail[0].msg
  return fallback
}

async function request(path, { method = 'GET', body, signal } = {}) {
  const ctrl = new AbortController()
  const timer = setTimeout(() => ctrl.abort(), 10_000)
  if (signal) signal.addEventListener('abort', () => ctrl.abort())
  const headers = { 'Content-Type': 'application/json' }
  if (apiKey) headers['X-API-Key'] = apiKey
  let res
  try {
    res = await fetch(`${API_BASE}${path}`, {
      method,
      headers,
      body: body === undefined ? undefined : JSON.stringify(body),
      signal: ctrl.signal,
    })
  } catch {
    throw new ApiError(0, `Cannot reach the API at ${API_BASE}`)
  } finally {
    clearTimeout(timer)
  }
  let data = null
  try {
    data = await res.json()
  } catch {
    data = null
  }
  if (!res.ok) {
    throw new ApiError(res.status, detailMessage(data?.detail, `Request failed (${res.status})`))
  }
  return data
}

const qs = (params) => {
  const u = new URLSearchParams()
  for (const [k, v] of Object.entries(params)) if (v !== undefined && v !== null && v !== '') u.set(k, v)
  const s = u.toString()
  return s ? `?${s}` : ''
}

export const getHealth = () => request('/health')
export const getCategories = () => request('/v1/categories')
export const listExpenses = ({ month, limit = 50, offset = 0 } = {}) =>
  request(`/v1/expenses${qs({ month, limit, offset })}`)
export const createExpense = (payload) => request('/v1/expenses', { method: 'POST', body: payload })
export const categorise = (payload) => request('/v1/categorise', { method: 'POST', body: payload })
export const parseReceipt = (rawText) => request('/v1/receipt/parse', { method: 'POST', body: { raw_text: rawText } })
export const getSummary = (month) => request(`/v1/summary${qs({ month })}`)
export const getForecast = (horizon = 3) => request(`/v1/forecast${qs({ horizon_months: horizon })}`)
