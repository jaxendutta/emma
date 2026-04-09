import { useState, useEffect, useRef } from 'react'
import { marked } from 'marked'
import mermaid from 'mermaid'

mermaid.initialize({
  startOnLoad: false,
  theme: 'base',
  themeVariables: {
    fontFamily: "'Source Sans 3', system-ui, sans-serif",
    fontSize: '13px',
    primaryColor: '#d8f3dc',
    primaryTextColor: '#2a2420',
    primaryBorderColor: '#b7e4c7',
    lineColor: '#8a7d74',
    secondaryColor: '#f5f0eb',
    tertiaryColor: '#faf8f5',
  },
})

export default function TabDocs() {
  const [html, setHtml]       = useState('')
  const [loading, setLoading] = useState(true)
  const [error, setError]     = useState(null)
  const contentRef            = useRef(null)
  const loadedRef             = useRef(false)

  useEffect(() => {
    if (loadedRef.current) return
    loadedRef.current = true

    const renderer = new marked.Renderer()
    renderer.code = ({ text, lang }) => {
      if (lang === 'mermaid') return `<div class="mermaid">${text}</div>`
      return `<pre><code>${text}</code></pre>`
    }

    fetch('/README.md')
      .then(res => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`)
        return res.text()
      })
      .then(text => {
        setHtml(marked.parse(text, { renderer }))
        setLoading(false)
      })
      .catch(err => {
        setError('Could not load README.md. ' + err.message)
        setLoading(false)
      })
  }, [])

  // After HTML renders, set up tables + mermaid
  useEffect(() => {
    if (!html || !contentRef.current) return

    const el = contentRef.current

    // Wrap tables in scroll + fade containers
    el.querySelectorAll('table').forEach(table => {
      if (table.parentElement.classList.contains('table-scroll')) return
      const scroll = document.createElement('div')
      scroll.className = 'table-scroll'
      const outer = document.createElement('div')
      outer.className = 'table-fade-wrap'
      table.parentNode.insertBefore(outer, table)
      scroll.appendChild(table)
      outer.appendChild(scroll)
    })

    el.querySelectorAll('.table-fade-wrap').forEach(outer => {
      const scroll = outer.querySelector('.table-scroll')
      const update = () => {
        const atEnd    = scroll.scrollLeft + scroll.clientWidth >= scroll.scrollWidth - 4
        const atStart  = scroll.scrollLeft <= 4
        const hasOver  = scroll.scrollWidth > scroll.clientWidth
        outer.classList.toggle('fade-right', hasOver && !atEnd)
        outer.classList.toggle('fade-left',  hasOver && !atStart)
      }
      update()
      scroll.addEventListener('scroll', update, { passive: true })
      window.addEventListener('resize', update, { passive: true })
    })

    // Run mermaid then attach zoom/pan toolbars
    mermaid.run({ nodes: el.querySelectorAll('.mermaid') }).then(() => {
      el.querySelectorAll('.mermaid').forEach(diagram => {
        if (diagram.parentElement.classList.contains('mermaid-zoomable')) return

        const wrapper = document.createElement('div')
        wrapper.className = 'mermaid-zoomable'
        diagram.parentNode.insertBefore(wrapper, diagram)
        wrapper.appendChild(diagram)

        const toolbar = document.createElement('div')
        toolbar.className = 'mermaid-toolbar'
        toolbar.innerHTML = `
          <button title="Zoom in">+</button>
          <button title="Zoom out">&ndash;</button>
          <button title="Reset zoom">&#x27F3;</button>
          <button title="Open in new tab">&#x26F6;</button>
        `
        wrapper.insertBefore(toolbar, diagram)

        let scale = 1, panX = 0, panY = 0
        let isPanning = false, startX = 0, startY = 0
        const MIN = 0.5, MAX = 3
        const svg = diagram.querySelector('svg')
        if (!svg) return
        svg.style.transformOrigin = '0 0'

        const updateTransform = () => {
          svg.style.transform = `translate(${panX}px,${panY}px) scale(${scale})`
        }
        const reset = () => { scale = 1; panX = 0; panY = 0; updateTransform() }

        toolbar.children[0].onclick = () => { scale = Math.min(MAX, scale * 1.2); updateTransform() }
        toolbar.children[1].onclick = () => { scale = Math.max(MIN, scale / 1.2); updateTransform() }
        toolbar.children[2].onclick = reset
        toolbar.children[3].onclick = () => {
          const data = new XMLSerializer().serializeToString(svg)
          const blob = new Blob([data], { type: 'image/svg+xml' })
          window.open(URL.createObjectURL(blob), '_blank')
        }

        svg.addEventListener('mousedown', e => {
          if (scale === 1) return
          isPanning = true; startX = e.clientX - panX; startY = e.clientY - panY
          wrapper.style.cursor = 'grabbing'
        })
        window.addEventListener('mousemove', e => {
          if (!isPanning) return
          panX = e.clientX - startX; panY = e.clientY - startY; updateTransform()
        })
        window.addEventListener('mouseup', () => {
          isPanning = false; wrapper.style.cursor = 'grab'
        })
        svg.addEventListener('touchstart', e => {
          if (scale === 1) return
          isPanning = true
          const t = e.touches[0]; startX = t.clientX - panX; startY = t.clientY - panY
        })
        window.addEventListener('touchmove', e => {
          if (!isPanning) return
          const t = e.touches[0]; panX = t.clientX - startX; panY = t.clientY - startY; updateTransform()
        })
        window.addEventListener('touchend', () => { isPanning = false })
      })
    }).catch(() => {/* mermaid errors are non-fatal */})
  }, [html])

  return (
    <div className="docs">
      {loading && (
        <div style={{ padding: '2rem 0', color: 'var(--muted)', fontSize: '0.9rem' }}>
          Loading documentation…
        </div>
      )}
      {error && (
        <div style={{ padding: '2rem 0', color: 'var(--muted)', fontSize: '0.9rem' }}>
          {error}
          <br /><br />
          <small>Tip: make sure the Vite dev server is running from <code>client/</code> so the README.md plugin can serve the file.</small>
        </div>
      )}
      {html && (
        <div
          id="readme-content"
          ref={contentRef}
          dangerouslySetInnerHTML={{ __html: html }}
        />
      )}
    </div>
  )
}
