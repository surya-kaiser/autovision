import React, { useState, useRef, useEffect } from 'react'
import ModelSelector from '../components/ModelSelector'
import MetricsChart from '../components/MetricsChart'
import { startTraining, startPilot, stopTraining, createTrainingWS, createPilotWS } from '../api/client'
import { useToast } from '../components/Toast'
import { useNavigate } from 'react-router-dom'
import api from '../api/client'
import { Play, Zap, Loader2, Terminal, CheckCircle, XCircle, Clock, Square, Database } from 'lucide-react'

// ── Progress bar ──────────────────────────────────────────────────────────────
function TrainingProgressBar({ pct, status, elapsedS }) {
  const color = status === 'failed' ? 'bg-red-500' : status === 'completed' ? 'bg-green-500' : 'bg-accent-500'
  return (
    <div className="space-y-1.5">
      <div className="flex justify-between text-xs text-gray-400">
        <span>
          {status === 'completed' ? '✓ Done' : status === 'failed' ? '✗ Failed / Stopped' : 'Training…'}
        </span>
        <span className="flex items-center gap-1">
          <Clock className="w-3 h-3" />
          {elapsedS}s
          {pct > 0 && pct < 100 && <span className="ml-2 font-bold text-accent-400">{pct}%</span>}
        </span>
      </div>
      <div className="w-full bg-dark-700 rounded-full h-3 overflow-hidden">
        <div
          className={`h-3 rounded-full transition-all duration-500 ${color} ${status === 'running' ? 'animate-pulse' : ''}`}
          style={{ width: `${pct || (status === 'running' ? 5 : 0)}%` }}
        />
      </div>
    </div>
  )
}

// ── Metric card ───────────────────────────────────────────────────────────────
function MetricCard({ label, value, prev }) {
  if (value == null) return null
  const fmt = typeof value === 'number' ? value.toFixed(4) : value
  const delta = prev != null && typeof value === 'number' ? value - prev : null
  return (
    <div className="bg-dark-700 rounded-xl p-3 text-center">
      <div className="text-xs text-gray-500 mb-0.5">{label}</div>
      <div className="text-xl font-bold text-accent-400">{fmt}</div>
      {delta != null && (
        <div className={`text-xs mt-0.5 ${delta >= 0 ? 'text-green-400' : 'text-red-400'}`}>
          {delta >= 0 ? '▲' : '▼'} {Math.abs(delta).toFixed(4)}
        </div>
      )}
    </div>
  )
}

// ── Log panel ─────────────────────────────────────────────────────────────────
function LogPanel({ logs, status }) {
  const endRef = useRef(null)
  useEffect(() => { endRef.current?.scrollIntoView({ behavior: 'smooth' }) }, [logs])

  return (
    <div className="bg-dark-900 border border-dark-700 rounded-xl overflow-hidden">
      <div className="flex items-center gap-2 px-4 py-2 border-b border-dark-700 bg-dark-800">
        <Terminal className="w-3.5 h-3.5 text-gray-500" />
        <span className="text-xs font-mono text-gray-400">Live Log</span>
        <span className="ml-auto flex items-center gap-1.5 text-xs">
          {status === 'running'   && <><span className="w-1.5 h-1.5 rounded-full bg-yellow-400 animate-pulse" />running</>}
          {status === 'completed' && <><span className="w-1.5 h-1.5 rounded-full bg-green-400" />completed</>}
          {status === 'failed'    && <><span className="w-1.5 h-1.5 rounded-full bg-red-400" />stopped</>}
        </span>
      </div>
      <div className="h-56 overflow-y-auto p-4 font-mono text-xs text-green-300 space-y-0.5">
        {logs.length === 0 && <span className="text-gray-600">Waiting for training to start…</span>}
        {logs.map((line, i) => (
          <div key={i} className={`leading-relaxed ${
            line.startsWith('ERROR') ? 'text-red-400' :
            line.startsWith('=== Auto-fix') ? 'text-yellow-400 font-semibold' :
            line.startsWith('===') ? 'text-accent-400 font-semibold' : ''
          }`}>
            <span className="text-gray-600 mr-2 select-none">{String(i + 1).padStart(3, '0')}</span>
            {line}
          </div>
        ))}
        <div ref={endRef} />
      </div>
    </div>
  )
}

export default function TrainingPage({ sessionId, setSessionId, pendingRecommendation, clearRecommendation }) {
  const toast = useToast()
  const navigate = useNavigate()

  const [modelType, setModelType] = useState('random_forest')
  const [taskType, setTaskType] = useState('classification')
  const [epochs, setEpochs] = useState(10)
  const [batchSize, setBatchSize] = useState(32)
  const [learningRate, setLearningRate] = useState(0.001)

  const [logs, setLogs] = useState([])
  const [status, setStatus] = useState(null)
  const [metrics, setMetrics] = useState({})
  const [prevMetrics, setPrevMetrics] = useState({})
  const [metricsHistory, setMetricsHistory] = useState([])
  const [pct, setPct] = useState(0)
  const [elapsedS, setElapsedS] = useState(0)
  const [running, setRunning] = useState(false)
  const [mode, setMode] = useState(null)
  const [error, setError] = useState(null)

  // Dataset picker
  const [savedSessions, setSavedSessions] = useState([])
  const [activeSession, setActiveSession] = useState(sessionId)
  const [showPicker, setShowPicker] = useState(false)

  const wsRef = useRef(null)
  const timerRef = useRef(null)
  const startTimeRef = useRef(null)

  // Apply pending recommendation from DatasetPage
  useEffect(() => {
    if (pendingRecommendation) {
      try {
        const { model_type, task_type } = pendingRecommendation
        if (model_type) setModelType(model_type)
        if (task_type) setTaskType(task_type)
      } catch {}
      // WS already started by auto-start endpoint — just connect
      if (activeSession) connectWS('training')
      if (clearRecommendation) clearRecommendation()
    }
  }, [])

  useEffect(() => {
    setActiveSession(sessionId)
  }, [sessionId])

  useEffect(() => () => {
    wsRef.current?.close()
    clearInterval(timerRef.current)
  }, [])

  async function loadSavedSessions() {
    try {
      const r = await api.get('/dataset/sessions')
      setSavedSessions(r.data.data || [])
    } catch {}
  }

  function openPicker() {
    loadSavedSessions()
    setShowPicker(true)
  }

  function pickSession(sid) {
    setActiveSession(sid)
    if (setSessionId) setSessionId(sid)
    setShowPicker(false)
    toast('Dataset session selected', 'info')
  }

  function startTimer() {
    startTimeRef.current = Date.now()
    timerRef.current = setInterval(() => {
      setElapsedS(Math.round((Date.now() - startTimeRef.current) / 1000))
    }, 1000)
  }

  function stopTimer() { clearInterval(timerRef.current) }

  function buildConfig(pilot = false) {
    return {
      session_id: activeSession,
      model_type: modelType,
      task_type: taskType,
      epochs,
      batch_size: batchSize,
      learning_rate: learningRate,
      early_stopping: true,
      patience: 5,
      pilot,
      hyperparams: { n_estimators: 100, max_depth: 6, learning_rate: learningRate },
    }
  }

  function connectWS(type) {
    if (!activeSession) return
    const ws = type === 'pilot' ? createPilotWS(activeSession) : createTrainingWS(activeSession)
    wsRef.current = ws

    ws.onmessage = (e) => {
      const data = JSON.parse(e.data)
      if (data.heartbeat) return

      if (data.log) {
        const line = data.log
        setLogs(prev => [...prev, line])

        const accMatch  = line.match(/accuracy[:\s]+([0-9.]+)/i)
        const f1Match   = line.match(/f1[:\s]+([0-9.]+)/i)
        const lossMatch = line.match(/loss[:\s]+([0-9.]+)/i)
        const rmseMatch = line.match(/rmse[:\s]+([0-9.]+)/i)
        const r2Match   = line.match(/r2[:\s]+([0-9.]+)/i)
        const mapMatch  = line.match(/map50[:\s=]+([0-9.]+)/i)

        const parsed = {}
        if (accMatch)  parsed.accuracy = parseFloat(accMatch[1])
        if (f1Match)   parsed.f1       = parseFloat(f1Match[1])
        if (lossMatch) parsed.loss     = parseFloat(lossMatch[1])
        if (rmseMatch) parsed.rmse     = parseFloat(rmseMatch[1])
        if (r2Match)   parsed.r2       = parseFloat(r2Match[1])
        if (mapMatch)  parsed.map50    = parseFloat(mapMatch[1])

        if (Object.keys(parsed).length > 0) {
          setPrevMetrics(m => ({ ...m }))
          setMetrics(m => ({ ...m, ...parsed }))
          setMetricsHistory(prev => {
            const next = [...prev, { epoch: prev.length + 1, ...parsed }]
            const totalEpochs = type === 'pilot' ? 3 : epochs
            setPct(Math.min(Math.round((next.length / totalEpochs) * 100), 95))
            return next
          })
        }
      }

      if (data.progress != null && data.progress > 0) {
        setPct(p => Math.max(p, data.progress))
      }

      if (data.status) {
        setStatus(data.status)
        if (data.status === 'completed') {
          setPct(100)
          stopTimer()
          setRunning(false)
          ws.close()
          toast('Training completed successfully!', 'success')
        }
        if (data.status === 'failed') {
          stopTimer()
          setRunning(false)
          ws.close()
          toast('Training stopped or failed', 'error')
        }
      }

      if (data.metrics && Object.keys(data.metrics).length > 0) {
        setPrevMetrics(m => ({ ...m }))
        setMetrics(m => ({ ...m, ...data.metrics }))
      }
    }

    ws.onerror = () => {
      // Only surface this if we haven't already received a clean status update
      setStatus(s => {
        if (!s || s === 'running') {
          setLogs(prev => [...prev, 'ERROR: WebSocket error — check backend is running'])
        }
        return s || 'failed'
      })
      stopTimer()
      setRunning(false)
    }
    ws.onclose = () => {
      stopTimer()
      setRunning(false)
    }
  }

  async function handlePilot() {
    if (!activeSession) { setError('Select a dataset session first'); return }
    reset()
    setMode('pilot')
    try {
      await startPilot(buildConfig(true))
      startTimer()
      connectWS('pilot')
      toast('Pilot run started (3 epochs)', 'info')
    } catch (e) {
      setError(e.response?.data?.detail || e.message)
      setRunning(false)
    }
  }

  async function handleFullTrain() {
    if (!activeSession) { setError('Select a dataset session first'); return }
    reset()
    setMode('full')
    try {
      await startTraining(buildConfig(false))
      startTimer()
      connectWS('training')
      toast('Full training started', 'info')
    } catch (e) {
      setError(e.response?.data?.detail || e.message)
      setRunning(false)
    }
  }

  async function handleStop() {
    if (!activeSession) return
    try {
      await stopTraining(activeSession)
      wsRef.current?.close()
      stopTimer()
      setRunning(false)
      setStatus('failed')
      toast('Training stopped', 'info')
    } catch (e) {
      toast('Could not stop training: ' + (e.response?.data?.detail || e.message), 'error')
    }
  }

  function reset() {
    wsRef.current?.close()
    clearInterval(timerRef.current)
    setLogs([])
    setMetrics({})
    setPrevMetrics({})
    setMetricsHistory([])
    setPct(0)
    setElapsedS(0)
    setStatus('running')
    setRunning(true)
    setError(null)
  }

  const metricKeys = metricsHistory.length > 0
    ? Object.keys(metricsHistory[0]).filter(k => k !== 'epoch')
    : []

  return (
    <div className="p-6 max-w-4xl mx-auto space-y-6">
      <h1 className="text-2xl font-bold text-white">Training</h1>

      {/* ── Dataset session selector ─────────────────────────────────── */}
      <div className="bg-dark-800 border border-dark-700 rounded-xl p-4 flex items-center justify-between gap-4">
        <div className="flex items-center gap-3 min-w-0">
          <Database className="w-4 h-4 text-accent-400 shrink-0" />
          <div className="min-w-0">
            <p className="text-xs text-gray-500">Active Dataset Session</p>
            <p className="text-sm text-white font-mono truncate">
              {activeSession ? activeSession.slice(0, 16) + '…' : <span className="text-gray-600">None selected</span>}
            </p>
          </div>
        </div>
        <button
          onClick={openPicker}
          className="flex items-center gap-2 px-3 py-1.5 bg-dark-700 hover:bg-dark-600 border border-dark-600 text-white rounded-lg text-xs font-medium transition-colors shrink-0"
        >
          <Database className="w-3.5 h-3.5 text-accent-400" />
          Change Dataset
        </button>
      </div>

      {/* ── Session picker modal ─────────────────────────────────────── */}
      {showPicker && (
        <div className="bg-dark-800 border border-accent-500/40 rounded-xl p-4 space-y-2">
          <div className="flex items-center justify-between mb-1">
            <h2 className="text-sm font-semibold text-white">Select Dataset</h2>
            <button onClick={() => setShowPicker(false)} className="text-gray-500 hover:text-gray-300 text-xs">✕</button>
          </div>
          {savedSessions.length === 0 ? (
            <p className="text-xs text-gray-500">No saved sessions yet. Upload a dataset first.</p>
          ) : (
            <div className="space-y-1.5 max-h-52 overflow-y-auto">
              {savedSessions.map(s => (
                <button
                  key={s.session_id}
                  onClick={() => pickSession(s.session_id)}
                  className={`w-full text-left px-3 py-2.5 rounded-lg text-xs transition-colors flex items-center justify-between
                    ${activeSession === s.session_id ? 'bg-accent-500/20 border border-accent-500/40 text-white' : 'bg-dark-700 hover:bg-dark-600 text-gray-300'}`}
                >
                  <div className="flex items-center gap-2">
                    <span className="font-medium">
                      {s.dataset && s.dataset !== 'unknown' ? s.dataset : `Dataset ${s.session_id.slice(0, 6)}`}
                    </span>
                    {s.task_type && s.task_type !== 'unknown' && (
                      <span className="text-gray-500">{s.task_type}</span>
                    )}
                  </div>
                  <span className="text-gray-500">{s.num_training_runs} run(s)</span>
                </button>
              ))}
            </div>
          )}
        </div>
      )}

      {/* ── Config ──────────────────────────────────────────────────── */}
      <div className="bg-dark-800 border border-dark-700 rounded-xl p-5 grid grid-cols-1 sm:grid-cols-2 gap-4">
        <div>
          <label className="block text-xs font-medium text-gray-400 mb-1">Task Type</label>
          <select value={taskType} onChange={e => setTaskType(e.target.value)}
            className="w-full bg-dark-700 border border-dark-600 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-accent-500">
            <option value="classification">Classification</option>
            <option value="regression">Regression</option>
            <option value="object_detection">Object Detection</option>
          </select>
        </div>
        <ModelSelector taskType={taskType} value={modelType} onChange={setModelType} />
        <div>
          <label className="block text-xs font-medium text-gray-400 mb-1">Epochs</label>
          <input type="number" min={1} max={500} value={epochs}
            onChange={e => setEpochs(+e.target.value)}
            className="w-full bg-dark-700 border border-dark-600 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-accent-500" />
        </div>
        <div>
          <label className="block text-xs font-medium text-gray-400 mb-1">Batch Size</label>
          <input type="number" min={1} max={512} value={batchSize}
            onChange={e => setBatchSize(+e.target.value)}
            className="w-full bg-dark-700 border border-dark-600 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-accent-500" />
        </div>
        <div>
          <label className="block text-xs font-medium text-gray-400 mb-1">Learning Rate</label>
          <input type="number" step={0.0001} min={0.0001} max={1} value={learningRate}
            onChange={e => setLearningRate(+e.target.value)}
            className="w-full bg-dark-700 border border-dark-600 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-accent-500" />
        </div>
      </div>

      {/* ── Buttons ──────────────────────────────────────────────────── */}
      <div className="flex gap-3 flex-wrap items-center">
        <button onClick={handlePilot} disabled={running}
          className="flex items-center gap-2 px-4 py-2 bg-dark-700 hover:bg-dark-600 border border-dark-600 disabled:opacity-50 text-white rounded-lg text-sm font-medium transition-colors">
          {running && mode === 'pilot' ? <Loader2 className="w-4 h-4 animate-spin" /> : <Zap className="w-4 h-4 text-yellow-400" />}
          Pilot Run (3 epochs)
        </button>
        <button onClick={handleFullTrain} disabled={running}
          className="flex items-center gap-2 px-4 py-2 bg-accent-500 hover:bg-accent-400 disabled:opacity-50 text-white rounded-lg text-sm font-medium transition-colors">
          {running && mode === 'full' ? <Loader2 className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4" />}
          Full Training
        </button>
        {running && (
          <button onClick={handleStop}
            className="flex items-center gap-2 px-4 py-2 bg-red-600 hover:bg-red-500 text-white rounded-lg text-sm font-medium transition-colors">
            <Square className="w-4 h-4" />
            Stop Training
          </button>
        )}
        {status === 'completed' && (
          <button onClick={() => navigate('/results')}
            className="flex items-center gap-2 px-4 py-2 bg-green-700 hover:bg-green-600 text-white rounded-lg text-sm font-medium transition-colors ml-auto">
            <CheckCircle className="w-4 h-4" />
            View Results
          </button>
        )}
      </div>

      {error && (
        <div className="bg-red-900/30 border border-red-700 rounded-xl p-4 text-sm text-red-300 flex justify-between">
          {error}
          <button onClick={() => setError(null)} className="text-red-400 text-xs">✕</button>
        </div>
      )}

      {/* ── Live progress ─────────────────────────────────────────────── */}
      {status && (
        <div className="bg-dark-800 border border-dark-700 rounded-xl p-5 space-y-5">
          <TrainingProgressBar pct={pct} status={status} elapsedS={elapsedS} />

          {Object.keys(metrics).length > 0 && (
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
              {Object.entries(metrics).map(([k, v]) => (
                <MetricCard key={k} label={k} value={v} prev={prevMetrics[k]} />
              ))}
            </div>
          )}

          {metricsHistory.length > 1 && metricKeys.length > 0 && (
            <div>
              <p className="text-xs font-medium text-gray-400 mb-2">Metric History</p>
              <MetricsChart data={metricsHistory} keys={metricKeys} />
            </div>
          )}
        </div>
      )}

      {(logs.length > 0 || running) && (
        <LogPanel logs={logs} status={status} />
      )}
    </div>
  )
}
