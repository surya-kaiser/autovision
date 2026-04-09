import React, { useState, useEffect } from 'react'
import api from '../api/client'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Cell
} from 'recharts'
import { Download, RefreshCw, ChevronDown, ChevronRight, Terminal, BarChart2 } from 'lucide-react'

function MetricBadge({ label, value }) {
  if (value == null) return null
  return (
    <div className="text-center">
      <div className="text-lg font-bold text-accent-400">
        {typeof value === 'number' ? value.toFixed(4) : value}
      </div>
      <div className="text-xs text-gray-500">{label}</div>
    </div>
  )
}

function TrainingRunCard({ run, sessionId }) {
  const [expanded, setExpanded] = useState(false)
  const [log, setLog] = useState('')
  const [loadingLog, setLoadingLog] = useState(false)

  async function loadLog() {
    if (log) return
    setLoadingLog(true)
    try {
      const r = await api.get(`/training/log/${sessionId}/${run.model_type}`)
      setLog(r.data.data.log || 'No log available')
    } catch {
      setLog('Could not load log')
    } finally {
      setLoadingLog(false)
    }
  }

  function toggle() {
    setExpanded(e => !e)
    if (!expanded) loadLog()
  }

  return (
    <div className="bg-dark-700 rounded-xl border border-dark-600 overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3">
        <div className="flex items-center gap-3">
          <button onClick={toggle} className="text-gray-400 hover:text-white">
            {expanded ? <ChevronDown className="w-4 h-4" /> : <ChevronRight className="w-4 h-4" />}
          </button>
          <div>
            <span className="text-sm font-semibold text-white">{run.model_type}</span>
            {run.pilot && (
              <span className="ml-2 px-1.5 py-0.5 text-xs bg-yellow-500/20 text-yellow-400 rounded">
                pilot
              </span>
            )}
            <span className="ml-2 text-xs text-gray-500">{run.task_type}</span>
          </div>
        </div>

        <div className="flex items-center gap-4">
          <span className="text-xs text-gray-500">{run.training_time_s?.toFixed(1)}s</span>
          <span className="text-xs text-gray-500">{run.trained_at ? new Date(run.trained_at).toLocaleTimeString() : ''}</span>
          {run.checkpoint_path && (
            <a
              href={`/api/v1/inference/export/${sessionId}?model_type=${run.model_type}&format=pkl`}
              className="flex items-center gap-1 px-2 py-1 bg-dark-600 hover:bg-dark-500 border border-dark-500 text-white rounded text-xs"
            >
              <Download className="w-3 h-3" /> Export
            </a>
          )}
        </div>
      </div>

      {/* Metrics row */}
      <div className="grid grid-cols-4 gap-3 px-4 pb-3 border-t border-dark-600/50 pt-3">
        <MetricBadge label="Accuracy" value={run.metrics?.accuracy} />
        <MetricBadge label="F1 Score" value={run.metrics?.f1} />
        <MetricBadge label="mAP50" value={run.metrics?.map50} />
        <MetricBadge label="RMSE" value={run.metrics?.rmse} />
      </div>

      {/* Expanded log */}
      {expanded && (
        <div className="border-t border-dark-600">
          <div className="flex items-center gap-2 px-4 py-2 bg-dark-800">
            <Terminal className="w-3.5 h-3.5 text-gray-500" />
            <span className="text-xs text-gray-500 font-mono">train.log</span>
          </div>
          <pre className="px-4 py-3 text-xs font-mono text-green-300 max-h-64 overflow-y-auto bg-dark-900 whitespace-pre-wrap">
            {loadingLog ? 'Loading…' : log || 'No log available'}
          </pre>
        </div>
      )}
    </div>
  )
}

export default function ResultsPage({ sessionId }) {
  const [history, setHistory] = useState(null)
  const [allSessions, setAllSessions] = useState([])
  const [viewSession, setViewSession] = useState(sessionId)
  const [loading, setLoading] = useState(false)

  async function loadHistory(sid) {
    if (!sid) return
    setLoading(true)
    try {
      const r = await api.get(`/training/history/${sid}`)
      setHistory(r.data.data)
    } catch (e) {
      setHistory(null)
    } finally {
      setLoading(false)
    }
  }

  async function loadAllSessions() {
    try {
      const r = await api.get('/dataset/sessions')
      setAllSessions(r.data.data || [])
    } catch {}
  }

  useEffect(() => {
    loadAllSessions()
  }, [])

  useEffect(() => {
    if (viewSession) loadHistory(viewSession)
  }, [viewSession])

  useEffect(() => {
    if (sessionId) setViewSession(sessionId)
  }, [sessionId])

  const ds = history?.dataset
  const runs = history?.training_runs || []

  return (
    <div className="p-6 max-w-5xl mx-auto space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold text-white">Results</h1>
        <button
          onClick={() => { loadAllSessions(); if (viewSession) loadHistory(viewSession) }}
          disabled={loading}
          className="flex items-center gap-2 px-3 py-1.5 bg-dark-700 hover:bg-dark-600 border border-dark-600 text-white rounded-lg text-xs font-medium"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin' : ''}`} />
          Refresh
        </button>
      </div>

      {/* Session selector */}
      {allSessions.length > 0 && (
        <div className="bg-dark-800 border border-dark-700 rounded-xl p-4">
          <h2 className="text-xs font-semibold text-gray-400 mb-3">All Sessions</h2>
          <div className="space-y-1">
            {allSessions.map((s) => (
              <button
                key={s.session_id}
                onClick={() => setViewSession(s.session_id)}
                className={`w-full text-left px-3 py-2 rounded-lg text-xs transition-colors flex items-center justify-between
                  ${viewSession === s.session_id ? 'bg-accent-500/20 border border-accent-500/40 text-white' : 'bg-dark-700 hover:bg-dark-600 text-gray-300'}`}
              >
                <div>
                  <span className="font-medium">
                    {s.dataset && s.dataset !== 'unknown' ? s.dataset : `Dataset ${s.session_id.slice(0, 6)}`}
                  </span>
                  {s.task_type && s.task_type !== 'unknown' && (
                    <span className="ml-2 text-gray-500">{s.task_type}</span>
                  )}
                </div>
                <span className="text-gray-500">{s.num_training_runs} run(s)</span>
              </button>
            ))}
          </div>
        </div>
      )}

      {!viewSession && (
        <div className="text-gray-500 text-sm">Upload a dataset and train a model first.</div>
      )}

      {/* Dataset info */}
      {ds && (
        <div className="bg-dark-800 border border-dark-700 rounded-xl p-5">
          <h2 className="text-sm font-semibold text-gray-300 mb-3">Dataset</h2>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 text-xs">
            <div><span className="text-gray-500">Name: </span><span className="text-white font-medium">{ds.name}</span></div>
            <div><span className="text-gray-500">Format: </span><span className="text-white font-medium">{ds.format}</span></div>
            <div><span className="text-gray-500">Task: </span><span className="text-white font-medium">{ds.task_type}</span></div>
            <div><span className="text-gray-500">Samples: </span><span className="text-white font-medium">{ds.num_samples?.toLocaleString()}</span></div>
          </div>
          {ds.classes?.length > 0 && (
            <div className="mt-3 flex flex-wrap gap-1.5">
              {ds.classes.map(c => (
                <span key={c} className="px-2 py-0.5 bg-accent-500/20 text-accent-400 rounded text-xs">{c}</span>
              ))}
            </div>
          )}
          {ds.preprocess_report && (
            <div className="mt-3 grid grid-cols-3 gap-2 text-center text-xs">
              {[
                ['Train', ds.preprocess_report.train_size],
                ['Val', ds.preprocess_report.val_size],
                ['Test', ds.preprocess_report.test_size],
              ].map(([label, val]) => (
                <div key={label} className="bg-dark-700 rounded-lg p-2">
                  <div className="text-lg font-bold text-accent-400">{val ?? '—'}</div>
                  <div className="text-gray-500">{label}</div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Training runs */}
      {runs.length === 0 && viewSession && !loading && (
        <div className="text-gray-500 text-sm">No training runs yet. Go to Training page.</div>
      )}

      {runs.length > 0 && (
        <div className="space-y-5">
          <h2 className="text-sm font-semibold text-gray-300">{runs.length} Training Run(s)</h2>

          {/* ── Metrics comparison chart ──────────────────────────────── */}
          {runs.length >= 1 && (() => {
            const METRIC_KEYS = ['accuracy', 'f1', 'r2', 'map50', 'rmse']
            const presentKeys = METRIC_KEYS.filter(k =>
              runs.some(r => r.metrics?.[k] != null)
            )
            if (presentKeys.length === 0) return null

            const chartData = runs
              .filter(r => !r.pilot)
              .map(r => ({
                name: r.model_type,
                ...Object.fromEntries(
                  presentKeys.map(k => [k, r.metrics?.[k] != null ? +r.metrics[k].toFixed(4) : null])
                ),
              }))

            if (chartData.length === 0) return null
            const COLORS = ['#818cf8', '#34d399', '#fb923c', '#f472b6', '#38bdf8']

            return (
              <div className="bg-dark-800 border border-dark-700 rounded-xl p-5">
                <div className="flex items-center gap-2 mb-4">
                  <BarChart2 className="w-4 h-4 text-accent-400" />
                  <h3 className="text-sm font-semibold text-gray-300">Model Comparison</h3>
                </div>
                <ResponsiveContainer width="100%" height={220}>
                  <BarChart data={chartData} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#2e3346" />
                    <XAxis dataKey="name" stroke="#4b5563" tick={{ fill: '#9ca3af', fontSize: 11 }} />
                    <YAxis stroke="#4b5563" tick={{ fill: '#9ca3af', fontSize: 11 }} domain={[0, 1]} />
                    <Tooltip
                      contentStyle={{ background: '#1a1d27', border: '1px solid #2e3346', borderRadius: 8 }}
                      labelStyle={{ color: '#e5e7eb' }}
                    />
                    <Legend wrapperStyle={{ paddingTop: 10, fontSize: 12 }} />
                    {presentKeys.map((k, i) => (
                      <Bar key={k} dataKey={k} fill={COLORS[i % COLORS.length]} radius={[3, 3, 0, 0]} maxBarSize={40} />
                    ))}
                  </BarChart>
                </ResponsiveContainer>
              </div>
            )
          })()}

          {runs.map((run, i) => (
            <TrainingRunCard key={i} run={run} sessionId={viewSession} />
          ))}
        </div>
      )}
    </div>
  )
}
