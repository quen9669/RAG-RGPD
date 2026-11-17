import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// En développement, l'API Python (`uv run rag-contrat`) répond sur :8000.
const api = 'http://127.0.0.1:8000'

export default defineConfig({
  plugins: [react()],
  server: { proxy: { '/api': api, '/health': api, '/vitrine.json': api } },
})
