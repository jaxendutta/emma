import { useState, useEffect } from 'react'
import Nav from './components/Nav.jsx'
import TabHome from './components/TabHome.jsx'
import TabDocs from './components/TabDocs.jsx'
import TabGraph from './components/TabGraph.jsx'
import Chat from './components/Chat.jsx'

export default function App() {
  const [tab, setTab] = useState(() => {
    const params = new URLSearchParams(window.location.search)
    const t = params.get('tab')
    return (t === 'docs' || t === 'graph') ? t : 'home'
  })

  useEffect(() => {
    document.body.className = tab === 'graph' ? 'tab-graph' : ''
    window.scrollTo(0, 0)
    const url = new URL(window.location)
    url.searchParams.set('tab', tab)
    window.history.pushState({}, '', url)
  }, [tab])

  return (
    <>
      <Nav tab={tab} setTab={setTab} />
      {tab === 'home'  && <TabHome />}
      {tab === 'docs'  && <TabDocs />}
      {tab === 'graph' && <TabGraph />}
      {tab !== 'graph' && <Chat />}
    </>
  )
}
