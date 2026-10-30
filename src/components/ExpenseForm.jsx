import { useEffect, useRef, useState } from 'react'
import { categorise, createExpense } from '../lib/api'
import { parseAmount } from '../lib/format'

const today = () => new Date().toISOString().slice(0, 10)

export default function ExpenseForm({ categories, prefill, onCreated }) {
  const [form, setForm] = useState({ date: today(), merchant: '', narration: '', amount: '', category: '' })
  const [errors, setErrors] = useState({})
  const [suggestion, setSuggestion] = useState(null)
  const [message, setMessage] = useState('')
  const [busy, setBusy] = useState(false)
  const timer = useRef(null)

  useEffect(() => {
    if (!prefill) return
    setForm((f) => ({
      ...f,
      date: prefill.date || f.date,
      merchant: prefill.merchant || f.merchant,
      narration: f.narration || (prefill.merchant ? `Receipt: ${prefill.merchant}` : f.narration),
      amount: prefill.total_ngn ? String(prefill.total_ngn) : f.amount,
    }))
    if (prefill.suggested_category) {
      setSuggestion({ category: prefill.suggested_category, confidence: prefill.category_confidence, source: 'receipt' })
    }
  }, [prefill])

  useEffect(() => () => clearTimeout(timer.current), [])

  const set = (k) => (e) => setForm((f) => ({ ...f, [k]: e.target.value }))

  const onNarrationBlur = () => {
    clearTimeout(timer.current)
    if (form.narration.trim().length < 3) return
    timer.current = setTimeout(async () => {
      try {
        const r = await categorise({ merchant: form.merchant, narration: form.narration })
        setSuggestion({ category: r.category, confidence: r.confidence, source: r.source })
      } catch {
        setSuggestion(null)
      }
    }, 400)
  }

  const submit = async (e) => {
    e.preventDefault()
    const errs = {}
    const amount = parseAmount(form.amount)
    if (!form.narration.trim()) errs.narration = 'Narration is required'
    if (amount === null) errs.amount = 'Enter an amount greater than zero'
    if (!form.date) errs.date = 'Date is required'
    setErrors(errs)
    if (Object.keys(errs).length) return
    setBusy(true)
    setMessage('')
    try {
      const created = await createExpense({
        date: form.date,
        merchant: form.merchant,
        narration: form.narration.trim(),
        amount_ngn: amount,
        category: form.category || null,
      })
      setForm({ date: form.date, merchant: '', narration: '', amount: '', category: '' })
      setSuggestion(null)
      setMessage('Expense added')
      onCreated(created)
    } catch (err) {
      setMessage('')
      setErrors({ form: err.message })
    } finally {
      setBusy(false)
    }
  }

  const err = (k) => ({
    'aria-invalid': errors[k] ? 'true' : undefined,
    'aria-describedby': errors[k] ? `${k}-err` : undefined,
  })

  return (
    <form className="card" onSubmit={submit} noValidate aria-labelledby="form-h">
      <h2 id="form-h">Add expense</h2>
      <label className="field">
        <span>Date</span>
        <input type="date" value={form.date} onChange={set('date')} {...err('date')} />
        {errors.date && <small id="date-err" className="field-err">{errors.date}</small>}
      </label>
      <label className="field">
        <span>Merchant</span>
        <input value={form.merchant} onChange={set('merchant')} maxLength={120} />
      </label>
      <label className="field">
        <span>Narration</span>
        <input value={form.narration} onChange={set('narration')} onBlur={onNarrationBlur} maxLength={500} required {...err('narration')} />
        {errors.narration && <small id="narration-err" className="field-err">{errors.narration}</small>}
      </label>
      <label className="field">
        <span>Amount (₦)</span>
        <input inputMode="decimal" value={form.amount} onChange={set('amount')} required {...err('amount')} />
        {errors.amount && <small id="amount-err" className="field-err">{errors.amount}</small>}
      </label>
      <label className="field">
        <span>Category</span>
        <select value={form.category} onChange={set('category')}>
          <option value="">Auto-categorise</option>
          {categories.map((c) => (
            <option key={c} value={c}>
              {c}
            </option>
          ))}
        </select>
      </label>
      {suggestion && (
        <p className="suggestion">
          Suggested: {suggestion.category}
          {typeof suggestion.confidence === 'number' && ` (${Math.round(suggestion.confidence * 100)}% confidence, ${suggestion.source})`}{' '}
          <button type="button" onClick={() => setForm((f) => ({ ...f, category: suggestion.category }))}>
            Use suggestion
          </button>
        </p>
      )}
      {errors.form && <p className="field-err" role="alert">{errors.form}</p>}
      <button type="submit" className="primary" disabled={busy}>
        {busy ? 'Saving…' : 'Add expense'}
      </button>
      <p aria-live="polite" className="ok">{message}</p>
    </form>
  )
}
