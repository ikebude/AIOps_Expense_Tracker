import { useState } from 'react'

function readStored() {
  try {
    return localStorage.getItem('theme')
  } catch {
    return null
  }
}

export default function ThemeToggle() {
  const [theme, setTheme] = useState(() => {
    const t = readStored()
    if (t === 'light' || t === 'dark') document.documentElement.dataset.theme = t
    return t
  })
  const prefersDark =
    typeof window !== 'undefined' && window.matchMedia?.('(prefers-color-scheme: dark)').matches
  const effective = theme ?? (prefersDark ? 'dark' : 'light')

  const toggle = () => {
    const next = effective === 'dark' ? 'light' : 'dark'
    document.documentElement.dataset.theme = next
    setTheme(next)
    try {
      localStorage.setItem('theme', next)
    } catch {
      /* storage unavailable: theme still applies for this session */
    }
  }

  return (
    <button type="button" className="theme-toggle" onClick={toggle} aria-label={`Switch to ${effective === 'dark' ? 'light' : 'dark'} theme`}>
      {effective === 'dark' ? 'Light theme' : 'Dark theme'}
    </button>
  )
}
