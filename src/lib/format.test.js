import { describe, test, expect } from 'vitest'
import { formatNGN, parseAmount, sumAmounts } from './format'

describe('format', () => {
  test('formatNGN uses naira', () => {
    expect(formatNGN(1234.5)).toBe('₦1,234.50')
  })
  test('parseAmount accepts symbols and commas', () => {
    expect(parseAmount('₦1,200.50')).toBe(1200.5)
    expect(parseAmount('NGN 5 000')).toBe(5000)
  })
  test('parseAmount rejects bad input', () => {
    for (const bad of ['-5', '', 'abc', '0', null, '1e3', '99999999999']) {
      expect(parseAmount(bad)).toBeNull()
    }
  })
  test('sumAmounts adds numerically (no string concatenation)', () => {
    expect(sumAmounts([{ amount_ngn: 27 }, { amount_ngn: 31 }])).toBe(58)
  })
  test('sumAmounts has no float drift', () => {
    expect(sumAmounts([{ amount_ngn: 0.1 }, { amount_ngn: 0.2 }])).toBe(0.3)
  })
})
