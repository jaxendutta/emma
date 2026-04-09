import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { readFileSync, existsSync } from 'fs'
import { resolve } from 'path'

export default defineConfig({
  plugins: [
    react(),
    {
      name: 'serve-readme',
      configureServer(server) {
        server.middlewares.use('/README.md', (req, res) => {
          const file = resolve(__dirname, '../README.md')
          if (existsSync(file)) {
            res.setHeader('Content-Type', 'text/markdown')
            res.end(readFileSync(file))
          } else {
            res.statusCode = 404
            res.end('Not found')
          }
        })
      }
    }
  ],
  server: {
    fs: { allow: ['..'] },
    proxy: {
      '/chat':       { target: 'http://localhost:8000', changeOrigin: true },
      '/health':     { target: 'http://localhost:8000', changeOrigin: true },
      '/conditions': { target: 'http://localhost:8000', changeOrigin: true },
      '/webhook':    { target: 'http://localhost:8000', changeOrigin: true },
      '/query':      { target: 'http://localhost:8000', changeOrigin: true },
    }
  }
})
