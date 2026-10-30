import { describe, test, expect, vi, afterEach } from 'vitest'
import { API_BASE, ApiError, getSummary, listExpenses, createExpense, setApiKey } from './api'

const ok = (data) => ({ ok: true, status: 200, json: async () => data })
const bad = (status, data) => ({ ok: false, status, json: async () => data })

afterEach(() => {
  vi.unstubAllGlobals()
  setApiKey('')
})

describe('api', () => {
  test('uses API_BASE and builds query strings', async () => {
    const f = vi.fn(async () => ok({}))
    vi.stubGlobal('fetch', f)
    await listExpenses({ month: '2026-02', limit: 10, offset: 20 })
    expect(f.mock.calls[0][0]).toBe(`${API_BASE}/v1/expenses?month=2026-02&limit=10&offset=20`)
  })
  test('422 array detail maps to first msg', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => bad(422, { detail: [{ msg: 'bad amount' }] })))
    await expect(getSummary()).rejects.toMatchObject({ status: 422, message: 'bad amount' })
  })
  test('string detail is used', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => bad(401, { detail: 'Invalid or missing API key' })))
    await expect(getSummary()).rejects.toThrow('Invalid or missing API key')
  })
  test('network failure becomes ApiError status 0', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => { throw new TypeError('fail') }))
    const err = await getSummary().catch((e) => e)
    expect(err).toBeInstanceOf(ApiError)
    expect(err.status).toBe(0)
  })
  test('sends X-API-Key only when set by the user', async () => {
    const f = vi.fn(async () => ok({}))
    vi.stubGlobal('fetch', f)
    await createExpense({ a: 1 })
    expect(f.mock.calls[0][1].headers['X-API-Key']).toBeUndefined()
    setApiKey('k')
    await createExpense({ a: 1 })
    expect(f.mock.calls[1][1].headers['X-API-Key']).toBe('k')
  })
})
