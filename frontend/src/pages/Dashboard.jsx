import React, { useEffect, useState } from 'react'
import { Database, Cpu, Award, Activity, Server } from 'lucide-react'
import { getSystemInfo, getLLMStatus } from '../api/client'

function StatCard({ icon: Icon, label, value, sub, color = 'text-accent-400' }) {
  return (
    <div className="bg-dark-800 border border-dark-700 rounded-xl p-5 flex items-start gap-4">
      <div className={`p-2.5 rounded-lg bg-dark-700 ${color}`}>
        <Icon className="w-5 h-5" />
      </div>
      <div>
        <p className="text-xs text-gray-500 mb-0.5">{label}</p>
        <p className="text-2xl font-bold text-white">{value}</p>
        {sub && <p className="text-xs text-gray-500 mt-0.5">{sub}</p>}
      </div>
    </div>
  )
}

export default function Dashboard({ sessionId }) {
  const [sys, setSys] = useState(null)
  const [llm, setLlm] = useState(null)

  useEffect(() => {
    getSystemInfo().then(r => setSys(r.data?.data)).catch(() => {})
    getLLMStatus().then(r => setLlm(r.data?.data)).catch(() => {})
  }, [])

  return (
    <div className="p-6 max-w-5xl mx-auto">
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-white">Dashboard</h1>
        <p className="text-gray-500 text-sm mt-1">AutoVision MLOps Platform — automated ML training &amp; inference</p>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
        <StatCard icon={Database} label="Session" value={sessionId ? '1' : '—'} sub={sessionId ? sessionId.slice(0, 8) + '…' : 'Upload a dataset'} />
        <StatCard icon={Cpu} label="Device" value={sys?.gpu_available ? 'GPU' : 'CPU'} sub={sys?.gpu_name || sys?.platform || '—'} color="text-green-400" />
        <StatCard icon={Activity} label="CPU Usage" value={sys ? `${sys.cpu_percent}%` : '—'} sub={sys ? `${sys.memory_used_gb}/${sys.memory_total_gb} GB RAM` : 'loading…'} color="text-yellow-400" />
        <StatCard icon={Server} label="LLM Engine" value={llm?.ollama_online ? 'Online' : 'Offline'} sub={llm ? `${llm.model} @ ${llm.base_url}` : '—'} color={llm?.ollama_online ? 'text-green-400' : 'text-orange-400'} />
      </div>

      <div className="bg-dark-800 border border-dark-700 rounded-xl p-6">
        <h2 className="text-sm font-semibold text-gray-300 mb-3">Getting Started</h2>
        <ol className="space-y-2 text-sm text-gray-400">
          <li><span className="text-accent-400 font-medium">1.</span> Go to <strong className="text-white">Dataset</strong> → upload your CSV or image ZIP</li>
          <li><span className="text-accent-400 font-medium">2.</span> Click <strong className="text-white">Auto Preprocess</strong> to clean and prepare your data</li>
          <li><span className="text-accent-400 font-medium">3.</span> Go to <strong className="text-white">Training</strong> → run a Pilot, then Full Training</li>
          <li><span className="text-accent-400 font-medium">4.</span> Check <strong className="text-white">Results</strong> to compare models and export</li>
          <li><span className="text-accent-400 font-medium">5.</span> Use <strong className="text-white">Inference</strong> to make predictions on new data</li>
        </ol>
      </div>
    </div>
  )
}
