export const formatNGN = (n) =>
  new Intl.NumberFormat('en-NG', { style: 'currency', currency: 'NGN' }).format(n)

// Parse a user-typed amount ("₦1,200.50") to a Number rounded to 2dp, or null if invalid.
export function parseAmount(input) {
  if (input === null || input === undefined) return null
  const cleaned = String(input).replace(/₦|NGN/gi, '').replace(/[,\s]/g, '')
  if (cleaned === '' || !/^\d*\.?\d*$/.test(cleaned)) return null
  const n = Number(cleaned)
  if (!Number.isFinite(n) || n <= 0 || n > 1e10) return null
  return Math.round(n * 100) / 100
}

// Sum in integer kobo to avoid float drift (and string concatenation).
export const sumAmounts = (items) =>
  items.reduce((s, x) => s + Math.round(x.amount_ngn * 100), 0) / 100
