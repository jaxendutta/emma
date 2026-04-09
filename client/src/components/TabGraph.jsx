import { useEffect, useRef } from 'react'
import { initGraph } from '../../scripts/graph.js'

export default function TabGraph() {
  const initializedRef = useRef(false)

  useEffect(() => {
    if (initializedRef.current) return
    initializedRef.current = true
    initGraph('graph-container')
  }, [])

  return (
    <div className="tab-graph-panel">
      <div id="graph-container" />
    </div>
  )
}
