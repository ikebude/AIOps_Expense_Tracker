import { useState } from 'react'
import { parseReceipt } from '../lib/api'

export default function ReceiptParser({ onParsed }) {
  const [text, setText] = useState('')
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)

  const run = async () => {
    if (!text.trim()) return
    setBusy(true)
    setError('')
    try {
      onParsed(await parseReceipt(text))
    } catch (e) {
      setError(e.message)
    } finally {
      setBusy(false)
    }
  }

  return (
    <section className="card" aria-labelledby="receipt-h">
      <h2 id="receipt-h">Receipt text</h2>
      <p className="hint">Parses pasted text only. It does not read images or photos.</p>
      <label className="field">
        <span>Paste receipt text</span>
        <textarea rows={4} value={text} onChange={(e) => setText(e.target.value)} maxLength={10000} />
      </label>
      {error && <p className="field-err" role="alert">{error}</p>}
      <button type="button" onClick={run} disabled={busy}>
        Parse receipt text
      </button>
    </section>
  )
}
