import '@quen9669/lignes/polices.css'
import '@quen9669/lignes/tokens.css'
import '@quen9669/lignes/composants.css'
import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { App } from './App'
import './style.css'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
