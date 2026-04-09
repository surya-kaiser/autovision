import React, { useEffect, useRef } from 'react'
import { Terminal } from 'lucide-react'

export default function TrainingProgress({ logs = [], status }) {
  const endRef = useRef(null)

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [logs])

  const statusColor = {
    running: 'text-yellow-400',
    completed: 'text-green-400',
    failed: 'text-red-400',
    queued: 'text-blue-400',
  }[status] || 'text-gray-400'

  return (
    <div className="bg-dark-900 border border-dark-700 rounded-xl overflow-hidden">
      <div className="flex items-center gap-2 px-4 py-2 border-b border-dark-700 bg-dark-800">
        <Terminal className="w-4 h-4 text-gray-500" />
        <span className="text-xs font-mono text-gray-400">Training Log</span>
        {status && (
          <span className={`ml-auto text-xs font-semibold ${statusColor}`}>
            {status.toUpperCase()}
          </span>
        )}
      </div>
      <div className="h-64 overflow-y-auto p-4 font-mono text-xs text-green-300 space-y-1">
        {logs.length === 0 && (
          <span className="text-gray-600">No logs yet...</span>
        )}
        {logs.map((line, i) => (
          <div key={i} className="leading-relaxed">
            <span className="text-gray-600 mr-2">[{String(i + 1).padStart(3, '0')}]</span>
            {line}
          </div>
        ))}
        <div ref={endRef} />
      </div>
    </div>
  )
}
