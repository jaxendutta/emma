import { useState } from 'react'

export default function Nav({ tab, setTab }) {
  const [drawerOpen, setDrawerOpen] = useState(false)

  const handleTab = (name) => {
    setTab(name)
    setDrawerOpen(false)
  }

  const tabs = [
    { id: 'home',  label: 'Agent' },
    { id: 'docs',  label: 'Documentation' },
    { id: 'graph', label: 'Ontology' },
  ]

  return (
    <>
      <nav className={drawerOpen ? 'nav-drawer-open' : ''}>
        <div className="nav-brand">
          <span>EMMA</span>
          <span className="nav-brand-sub">
            Emergency Medicine<br />Mentoring Agent
          </span>
        </div>

        <div className="nav-tabs">
          {tabs.map(t => (
            <button
              key={t.id}
              className={`nav-tab${tab === t.id ? ' active' : ''}`}
              onClick={() => handleTab(t.id)}
            >
              {t.label}
            </button>
          ))}
        </div>

        <button
          className={`nav-hamburger${drawerOpen ? ' open' : ''}`}
          aria-label="Menu"
          onClick={(e) => { e.stopPropagation(); setDrawerOpen(o => !o) }}
        >
          <span /><span /><span />
        </button>

        <span className="nav-meta">DTI 5125 &mdash; Group 23</span>
      </nav>

      <div
        className={`nav-overlay${drawerOpen ? ' open' : ''}`}
        onClick={() => setDrawerOpen(false)}
      />

      <div className={`nav-drawer${drawerOpen ? ' open' : ''}`}>
        {tabs.map(t => (
          <button
            key={t.id}
            className={`nav-drawer-tab${tab === t.id ? ' active' : ''}`}
            onClick={() => handleTab(t.id)}
          >
            {t.label}
          </button>
        ))}
      </div>
    </>
  )
}
