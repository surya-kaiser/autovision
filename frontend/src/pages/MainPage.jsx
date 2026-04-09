/**
 * MainPage — single-page AutoVision UI.
 *
 * Three sequential sections on one page:
 *   1. Upload Dataset   → detects task type, preprocesses
 *   2. Start Training   → autonomous pipeline, live WebSocket logs
 *   3. View Results     → task-appropriate metrics
 */
import React, { useState, useEffect, useRef, useCallback } from 'react'
import {
  uploadDataset,
  preprocessDataset,
  autoTrain,
  createTrainingWS,
  getLLMStatus,
  listSessions,
  getSessionSummary,
  getTrainingLog,
  deleteDataset,
  listModels,
  predict,
} from '../api/client'

// ── Task badge colours ─────────────────────────────────────────────────────────
const TASK_META = {
  segmentation: {
    label: 'Image Segmentation',
    color: 'bg-purple-900 text-purple-300 border-purple-700',
    metrics: ['iou', 'dice', 'pixel_accuracy'],
    metricLabels: { iou: 'IoU', dice: 'Dice', pixel_accuracy: 'Pixel Acc' },
  },
  classification: {
    label: 'Image / Tabular Classification',
    color: 'bg-blue-900 text-blue-300 border-blue-700',
    metrics: ['accuracy', 'f1'],
    metricLabels: { accuracy: 'Accuracy', f1: 'F1 Score' },
  },
  object_detection: {
    label: 'Object Detection',
    color: 'bg-orange-900 text-orange-300 border-orange-700',
    metrics: ['map50', 'map50_95'],
    metricLabels: { map50: 'mAP50', map50_95: 'mAP50-95' },
  },
  regression: {
    label: 'Regression',
    color: 'bg-green-900 text-green-300 border-green-700',
    metrics: ['rmse', 'r2'],
    metricLabels: { rmse: 'RMSE', r2: 'R²' },
  },
}

// ── Helpers ────────────────────────────────────────────────────────────────────

function TaskBadge({ taskType }) {
  const meta = TASK_META[taskType] || {
    label: taskType || 'Unknown',
    color: 'bg-gray-800 text-gray-300 border-gray-600',
  }
  return (
    <span className={`inline-block px-3 py-1 rounded-full border text-sm font-semibold ${meta.color}`}>
      {meta.label}
    </span>
  )
}

function SectionHeader({ step, title, done, locked }) {
  return (
    <div className="flex items-center gap-3 mb-4">
      <div className={`
        w-8 h-8 rounded-full flex items-center justify-center text-sm font-bold flex-shrink-0
        ${done ? 'bg-green-600 text-white' : locked ? 'bg-gray-700 text-gray-500' : 'bg-blue-600 text-white'}
      `}>
        {done ? '✓' : step}
      </div>
      <h2 className={`text-lg font-semibold ${locked ? 'text-gray-500' : 'text-white'}`}>
        {title}
      </h2>
    </div>
  )
}

function MetricCard({ label, value }) {
  const formatted =
    value === null || value === undefined
      ? '—'
      : typeof value === 'number'
      ? value.toFixed(4)
      : value
  return (
    <div className="bg-dark-800 border border-dark-700 rounded-lg p-4 text-center">
      <div className="text-2xl font-bold text-white mb-1">{formatted}</div>
      <div className="text-sm text-gray-400">{label}</div>
    </div>
  )
}

// ── Main component ─────────────────────────────────────────────────────────────

const TASK_OPTIONS = [
  { value: 'classification', label: 'Image Classification' },
  { value: 'segmentation',   label: 'Image Segmentation' },
  { value: 'object_detection', label: 'Object Detection' },
  { value: 'regression',     label: 'Tabular Regression' },
]

export default function MainPage() {
  // Section 1 state
  const [file, setFile] = useState(null)
  const [sessionId, setSessionId] = useState(null)
  const [datasetInfo, setDatasetInfo] = useState(null)   // from upload response
  const [selectedTask, setSelectedTask] = useState(null) // user-chosen or auto-detected
  const [uploadError, setUploadError] = useState(null)
  const [uploading, setUploading] = useState(false)
  const [preprocessing, setPreprocessing] = useState(false)
  const [preprocessDone, setPreprocessDone] = useState(false)

  // Section 2 state
  const [training, setTraining] = useState(false)
  const [trainingDone, setTrainingDone] = useState(false)
  const [trainingStatus, setTrainingStatus] = useState(null) // running|completed|failed
  const [logs, setLogs] = useState([])
  const [liveMetrics, setLiveMetrics] = useState({})
  const wsRef = useRef(null)

  // Section 3 state
  const [finalMetrics, setFinalMetrics] = useState(null)
  const [bestModel, setBestModel] = useState(null)

  // UI tab
  const [activeTab, setActiveTab] = useState('train') // 'train' | 'predict' | 'history'

  // System
  const [llmOnline, setLlmOnline] = useState(false)

  useEffect(() => {
    getLLMStatus()
      .then(r => setLlmOnline(r.data?.data?.ollama_online ?? false))
      .catch(() => {})
  }, [])

  // ── Section 1: Upload + Preprocess ─────────────────────────────────────────

  const handleFileChange = (e) => {
    setFile(e.target.files[0] || null)
    setUploadError(null)
    setDatasetInfo(null)
    setSelectedTask(null)
    setPreprocessDone(false)
    setTrainingDone(false)
    setFinalMetrics(null)
    setLogs([])
  }

  const handleUpload = async () => {
    if (!file) return
    setUploading(true)
    setUploadError(null)
    try {
      const res = await uploadDataset(file, null)
      const info = res.data?.data
      if (!info) throw new Error('Upload response missing data')
      setDatasetInfo(info)
      setSessionId(info.session_id)
      // Pre-select the auto-detected task; user can override below
      setSelectedTask(info.task_type || 'classification')
    } catch (err) {
      const msg = err.response?.data?.detail || err.response?.data?.message || err.message
      setUploadError(`Upload failed: ${msg}`)
    } finally {
      setUploading(false)
    }
  }

  const handlePreprocess = async () => {
    if (!sessionId) return
    setPreprocessing(true)
    setUploadError(null)
    try {
      const res = await preprocessDataset({ session_id: sessionId, task_type_hint: selectedTask })
      const info = res.data?.data?.info
      // Enrich datasetInfo with num_samples and num_classes from the preprocess response
      if (info) {
        setDatasetInfo(prev => ({
          ...prev,
          num_samples: info.num_samples ?? prev?.num_samples,
          num_classes: info.num_classes ?? prev?.num_classes,
          class_names: info.class_names ?? prev?.class_names,
        }))
      }
      setPreprocessDone(true)
    } catch (err) {
      const msg = err.response?.data?.detail || err.response?.data?.message || err.message
      setUploadError(`Preprocessing failed: ${msg}`)
    } finally {
      setPreprocessing(false)
    }
  }

  // ── Section 2: Autonomous Training ──────────────────────────────────────────

  const handleStartTraining = useCallback(async () => {
    if (!sessionId || !preprocessDone) return
    setTraining(true)
    setTrainingDone(false)
    setTrainingStatus('running')
    setLogs([])
    setFinalMetrics(null)
    setBestModel(null)

    // Open WebSocket for live logs
    const ws = createTrainingWS(sessionId)
    wsRef.current = ws

    // Track final status across messages so onclose can read it
    let finalStatus = 'running'

    ws.onmessage = (evt) => {
      try {
        const msg = JSON.parse(evt.data)
        if (msg.heartbeat) return  // ignore keepalive pings
        if (msg.log) {
          setLogs(prev => [...prev, msg.log])
        }
        if (msg.metrics && Object.keys(msg.metrics).length > 0) {
          setLiveMetrics(msg.metrics)
          if (msg.status === 'completed' || msg.status === 'failed') {
            setFinalMetrics(msg.metrics)
          }
        }
        if (msg.status) {
          finalStatus = msg.status
          setTrainingStatus(msg.status)
          if (msg.best_model) setBestModel(msg.best_model)
          // DO NOT close WS here — let the server drain all error/completion
          // messages and close naturally. onclose handles cleanup.
        }
      } catch (_) {
        setLogs(prev => [...prev, evt.data])
      }
    }

    ws.onclose = () => {
      // Server has closed the connection after draining all messages
      setTraining(false)
      setTrainingDone(finalStatus === 'completed')
      // If status never moved past 'running' but WS closed → treat as failed
      if (finalStatus === 'running') {
        setTrainingStatus('failed')
        setLogs(prev => [...prev, 'Connection closed unexpectedly — check server logs'])
      }
    }

    ws.onerror = () => {
      setLogs(prev => [...prev, 'WebSocket connection error — is the backend running?'])
    }

    // Trigger autonomous training with user-selected (or auto-detected) task
    try {
      const taskType = selectedTask || datasetInfo?.task_type || undefined
      const res = await autoTrain(sessionId, taskType)
      const result = res.data?.data
      if (result?.best_model) setBestModel(result.best_model)
      if (result?.metrics) setFinalMetrics(result.metrics)
    } catch (err) {
      const msg = err.response?.data?.detail || err.response?.data?.message || err.message
      setLogs(prev => [...prev, `Training error: ${msg}`])
      setTrainingStatus('failed')
      setTraining(false)
    }
  }, [sessionId, preprocessDone, datasetInfo])

  // ── Derive task key for meta lookup (user selection takes priority) ──────────

  const taskKey = (() => {
    const raw = selectedTask || datasetInfo?.task_type || ''
    if (raw.includes('segmentation')) return 'segmentation'
    if (raw.includes('detection')) return 'object_detection'
    if (raw.includes('regression')) return 'regression'
    return 'classification'
  })()

  const taskMeta = TASK_META[taskKey]
  const metricsToShow = finalMetrics || liveMetrics

  // ── Render ───────────────────────────────────────────────────────────────────

  return (
    <div className="min-h-screen bg-gray-950 text-white">
      {/* Header */}
      <header className="border-b border-gray-800 px-6 py-3 flex items-center justify-between">
        <div className="flex items-center gap-6">
          <div>
            <h1 className="text-lg font-bold text-white leading-tight">AutoVision MLOps</h1>
            <p className="text-xs text-gray-500">Autonomous ML Engineer</p>
          </div>
          {/* Tab switcher */}
          <nav className="flex gap-1">
            {[['train', 'Train'], ['predict', 'Predict'], ['history', 'History']].map(([id, label]) => (
              <button
                key={id}
                onClick={() => setActiveTab(id)}
                className={`px-4 py-1.5 rounded-md text-sm font-medium transition
                  ${activeTab === id
                    ? 'bg-gray-700 text-white'
                    : 'text-gray-400 hover:text-white hover:bg-gray-800'
                  }`}
              >
                {label}
              </button>
            ))}
          </nav>
        </div>
        <div className="flex items-center gap-2 text-xs">
          <span className={`w-2 h-2 rounded-full ${llmOnline ? 'bg-green-400' : 'bg-gray-600'}`} />
          <span className="text-gray-400">Ollama {llmOnline ? 'Online' : 'Offline'}</span>
        </div>
      </header>

      {/* History tab */}
      {activeTab === 'history' && <HistoryPanel currentSessionId={sessionId} />}

      {/* Predict tab */}
      {activeTab === 'predict' && <PredictPanel />}

      {/* Train tab */}
      {activeTab !== 'history' && <div className="max-w-3xl mx-auto px-6 py-8 space-y-10">

        {/* ── SECTION 1: Upload Dataset ─────────────────────────────────── */}
        <section>
          <SectionHeader
            step="1"
            title="Upload Dataset"
            done={preprocessDone}
            locked={false}
          />

          <div className="bg-gray-900 border border-gray-800 rounded-xl p-6 space-y-4">
            {/* File picker */}
            <div>
              <label className="block text-sm text-gray-400 mb-2">
                Select a file or folder (ZIP, CSV, or image folder)
              </label>
              <input
                type="file"
                accept=".csv,.zip,.jpg,.jpeg,.png"
                onChange={handleFileChange}
                className="block w-full text-sm text-gray-300 file:mr-3 file:py-2 file:px-4
                           file:rounded-lg file:border-0 file:text-sm file:font-semibold
                           file:bg-blue-700 file:text-white hover:file:bg-blue-600 cursor-pointer"
              />
              {file && (
                <p className="mt-1 text-xs text-gray-500">{file.name} ({(file.size / 1024).toFixed(1)} KB)</p>
              )}
            </div>

            {/* Upload button */}
            {file && !datasetInfo && (
              <button
                onClick={handleUpload}
                disabled={uploading}
                className="px-5 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-700
                           disabled:cursor-not-allowed rounded-lg text-sm font-semibold transition"
              >
                {uploading ? 'Uploading…' : 'Upload Dataset'}
              </button>
            )}

            {/* Dataset info + task selector */}
            {datasetInfo && (
              <div className="space-y-4">
                {/* Stats row */}
                <div className="grid grid-cols-3 gap-3 text-sm">
                  <div className="bg-gray-800 rounded-lg p-3">
                    <div className="text-gray-400 text-xs mb-1">Samples</div>
                    <div className="font-semibold">{datasetInfo.num_samples > 0 ? datasetInfo.num_samples : '—'}</div>
                  </div>
                  <div className="bg-gray-800 rounded-lg p-3">
                    <div className="text-gray-400 text-xs mb-1">Classes</div>
                    <div className="font-semibold">{datasetInfo.num_classes > 0 ? datasetInfo.num_classes : '—'}</div>
                  </div>
                  <div className="bg-gray-800 rounded-lg p-3">
                    <div className="text-gray-400 text-xs mb-1">Format</div>
                    <div className="font-semibold capitalize">{(datasetInfo.format || '—').replace('_', ' ')}</div>
                  </div>
                </div>

                {/* Class names */}
                {datasetInfo.class_names?.length > 0 && (
                  <div className="flex flex-wrap gap-2">
                    {datasetInfo.class_names.map(c => (
                      <span key={c} className="px-2 py-0.5 bg-gray-700 text-gray-300 rounded text-xs font-mono">
                        {c}
                      </span>
                    ))}
                  </div>
                )}

                {/* Task selector — auto-detected + user can override */}
                {!preprocessDone && (
                  <div>
                    <label className="block text-sm text-gray-400 mb-2">
                      Task type
                      <span className="ml-2 text-xs text-gray-600">(auto-detected — change if wrong)</span>
                    </label>
                    <div className="grid grid-cols-2 gap-2">
                      {TASK_OPTIONS.map(opt => (
                        <button
                          key={opt.value}
                          onClick={() => setSelectedTask(opt.value)}
                          className={`px-3 py-2 rounded-lg text-sm font-medium border transition text-left
                            ${selectedTask === opt.value
                              ? 'bg-blue-700 border-blue-500 text-white'
                              : 'bg-gray-800 border-gray-700 text-gray-300 hover:border-gray-500'
                            }`}
                        >
                          {opt.label}
                        </button>
                      ))}
                    </div>
                  </div>
                )}

                {/* Confirmed task badge after preprocess */}
                {preprocessDone && (
                  <div className="flex items-center gap-3 flex-wrap">
                    <span className="text-sm text-gray-400">Task:</span>
                    <TaskBadge taskType={taskKey} />
                  </div>
                )}

                {/* Preprocess button */}
                {!preprocessDone && (
                  <button
                    onClick={handlePreprocess}
                    disabled={preprocessing || !selectedTask}
                    className="w-full px-5 py-2.5 bg-emerald-600 hover:bg-emerald-700 disabled:bg-gray-700
                               disabled:cursor-not-allowed rounded-lg text-sm font-semibold transition"
                  >
                    {preprocessing ? 'Preprocessing…' : `Preprocess as ${TASK_OPTIONS.find(o => o.value === selectedTask)?.label ?? '…'}`}
                  </button>
                )}
                {preprocessDone && (
                  <p className="text-emerald-400 text-sm font-semibold">✓ Preprocessing complete</p>
                )}
              </div>
            )}

            {/* Error display */}
            {uploadError && (
              <div className="bg-red-950 border border-red-800 rounded-lg p-3 text-red-300 text-sm">
                {uploadError}
              </div>
            )}
          </div>
        </section>

        {/* ── SECTION 2: Start Training ─────────────────────────────────── */}
        <section className={preprocessDone ? '' : 'opacity-40 pointer-events-none'}>
          <SectionHeader
            step="2"
            title="Start Training"
            done={trainingDone}
            locked={!preprocessDone}
          />

          <div className="bg-gray-900 border border-gray-800 rounded-xl p-6 space-y-4">
            <p className="text-sm text-gray-400">
              The Autonomous ML Engineer will detect the task, select the correct model,
              and train it. No configuration needed.
            </p>

            {/* Start / retry buttons */}
            {!training && !trainingDone && (
              <button
                onClick={handleStartTraining}
                disabled={!preprocessDone || training}
                className="px-6 py-2.5 bg-violet-600 hover:bg-violet-700 disabled:bg-gray-700
                           disabled:cursor-not-allowed rounded-lg text-sm font-semibold transition"
              >
                {trainingStatus === 'failed' ? 'Retry Training' : 'Start Autonomous Training'}
              </button>
            )}
            {training && (
              <button
                onClick={() => {
                  wsRef.current?.close()
                  setTraining(false)
                  setTrainingStatus('failed')
                }}
                className="px-6 py-2.5 bg-red-700 hover:bg-red-600 rounded-lg text-sm font-semibold transition"
              >
                Stop
              </button>
            )}

            {/* Status indicator */}
            {trainingStatus && (
              <div className="flex items-center gap-2 text-sm">
                <span className={`w-2 h-2 rounded-full ${
                  trainingStatus === 'running'   ? 'bg-yellow-400 animate-pulse' :
                  trainingStatus === 'completed' ? 'bg-green-400' : 'bg-red-400'
                }`} />
                <span className={
                  trainingStatus === 'running'   ? 'text-yellow-300' :
                  trainingStatus === 'completed' ? 'text-green-300' : 'text-red-300'
                }>
                  {trainingStatus.toUpperCase()}
                </span>
              </div>
            )}

            {/* Live log panel */}
            {logs.length > 0 && (
              <LogPanel logs={logs} status={trainingStatus} />
            )}
          </div>
        </section>

        {/* ── SECTION 3: View Results ───────────────────────────────────── */}
        <section className={(trainingDone || (trainingStatus === 'failed')) ? '' : 'opacity-40 pointer-events-none'}>
          <SectionHeader
            step="3"
            title="View Results"
            done={false}
            locked={!trainingDone && trainingStatus !== 'failed'}
          />

          <div className="bg-gray-900 border border-gray-800 rounded-xl p-6 space-y-4">
            {!trainingDone && trainingStatus !== 'failed' && (
              <p className="text-sm text-gray-500">Training results will appear here.</p>
            )}

            {bestModel && (
              <div className="text-sm">
                <span className="text-gray-400">Best model: </span>
                <span className="font-semibold text-white">{bestModel}</span>
              </div>
            )}

            {/* Task-appropriate metrics */}
            {metricsToShow && Object.keys(metricsToShow).length > 0 && taskMeta && (
              <div>
                <p className="text-xs text-gray-500 mb-3 uppercase tracking-wide">
                  {taskMeta.label} metrics
                </p>
                <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
                  {taskMeta.metrics.map(key => (
                    metricsToShow[key] !== undefined && (
                      <MetricCard
                        key={key}
                        label={taskMeta.metricLabels[key] || key}
                        value={metricsToShow[key]}
                      />
                    )
                  ))}
                </div>
              </div>
            )}

            {/* All metrics (debug / extra) */}
            {metricsToShow && Object.keys(metricsToShow).length > 0 && (
              <details className="mt-2">
                <summary className="text-xs text-gray-500 cursor-pointer hover:text-gray-300">
                  All metrics
                </summary>
                <pre className="mt-2 p-3 bg-gray-800 rounded-lg text-xs text-gray-300 overflow-auto">
                  {JSON.stringify(metricsToShow, null, 2)}
                </pre>
              </details>
            )}

            {trainingStatus === 'failed' && (
              <p className="text-red-400 text-sm">Training failed. Check logs above for details.</p>
            )}
          </div>
        </section>

      </div>
    }

    </div>
  )
}

// ── History Panel sub-component ───────────────────────────────────────────────

function HistoryPanel({ currentSessionId }) {
  const [sessions, setSessions] = useState([])
  const [loading, setLoading] = useState(true)
  const [expanded, setExpanded] = useState(null)       // session_id of expanded card
  const [summaries, setSummaries] = useState({})       // session_id → summary object
  const [summaryLoading, setSummaryLoading] = useState({})
  const [logExpanded, setLogExpanded] = useState({})   // `${sid}:${model_type}` → bool

  useEffect(() => {
    setLoading(true)
    listSessions()
      .then(r => setSessions(r.data?.data || []))
      .catch(() => setSessions([]))
      .finally(() => setLoading(false))
  }, [])

  const handleExpand = async (sid) => {
    if (expanded === sid) {
      setExpanded(null)
      return
    }
    setExpanded(sid)
    if (!summaries[sid]) {
      setSummaryLoading(prev => ({ ...prev, [sid]: true }))
      try {
        const r = await getSessionSummary(sid)
        setSummaries(prev => ({ ...prev, [sid]: r.data?.data }))
      } catch (_) {}
      setSummaryLoading(prev => ({ ...prev, [sid]: false }))
    }
  }

  const toggleLog = (sid, modelType) => {
    const key = `${sid}:${modelType}`
    setLogExpanded(prev => ({ ...prev, [key]: !prev[key] }))
  }

  const fmtDate = (iso) => {
    if (!iso) return ''
    try { return new Date(iso).toLocaleString() } catch (_) { return iso }
  }

  const fmtSeconds = (s) => {
    if (!s) return '—'
    return s < 60 ? `${s}s` : `${Math.floor(s / 60)}m ${s % 60}s`
  }

  const primaryMetric = (metrics) => {
    if (!metrics) return null
    const keys = ['accuracy', 'iou', 'map50', 'r2', 'f1']
    for (const k of keys) {
      if (metrics[k] !== undefined) return { label: k.toUpperCase(), value: metrics[k] }
    }
    const first = Object.entries(metrics)[0]
    return first ? { label: first[0], value: first[1] } : null
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-48 text-gray-500 text-sm">
        Loading sessions…
      </div>
    )
  }

  if (!sessions.length) {
    return (
      <div className="flex items-center justify-center h-48 text-gray-500 text-sm">
        No training sessions yet. Upload a dataset and train a model to see history here.
      </div>
    )
  }

  return (
    <div className="max-w-4xl mx-auto px-6 py-8 space-y-4">
      <h2 className="text-lg font-semibold text-white mb-2">Training History</h2>

      {sessions.map(s => {
        const isCurrent = s.session_id === currentSessionId
        const isExpanded = expanded === s.session_id
        const summary = summaries[s.session_id]
        const loadingSum = summaryLoading[s.session_id]

        return (
          <div
            key={s.session_id}
            className={`bg-gray-900 border rounded-xl overflow-hidden transition
              ${isCurrent ? 'border-blue-700' : 'border-gray-800'}`}
          >
            {/* Card header — always visible */}
            <button
              onClick={() => handleExpand(s.session_id)}
              className="w-full text-left px-5 py-4 flex items-center gap-4 hover:bg-gray-800/50 transition"
            >
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2 flex-wrap">
                  <span className="font-semibold text-white truncate">{s.dataset || 'Unnamed Dataset'}</span>
                  {isCurrent && (
                    <span className="px-2 py-0.5 bg-blue-800 text-blue-300 rounded text-xs font-medium">
                      current
                    </span>
                  )}
                  {s.task_type && s.task_type !== 'unknown' && (
                    <TaskBadge taskType={s.task_type} />
                  )}
                </div>
                <div className="text-xs text-gray-500 mt-0.5">
                  {s.num_samples > 0 ? `${s.num_samples} samples` : 'samples unknown'} ·{' '}
                  {s.num_training_runs} training run{s.num_training_runs !== 1 ? 's' : ''} ·{' '}
                  {fmtDate(s.created_at)}
                </div>
              </div>
              <span className="text-gray-500 text-lg flex-shrink-0">{isExpanded ? '▲' : '▼'}</span>
            </button>

            {/* Expanded detail */}
            {isExpanded && (
              <div className="border-t border-gray-800 px-5 py-4 space-y-4">
                {loadingSum && (
                  <p className="text-sm text-gray-500">Loading details…</p>
                )}

                {summary && (
                  <>
                    {/* Dataset info */}
                    {summary.dataset && (
                      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-sm">
                        {[
                          ['Format', (summary.dataset.format || '—').replace('_', ' ')],
                          ['Samples', summary.dataset.num_samples || '—'],
                          ['Classes', summary.dataset.classes?.length || '—'],
                          ['Preprocessed', summary.dataset.preprocessed_at
                            ? fmtDate(summary.dataset.preprocessed_at) : 'not yet'],
                        ].map(([label, val]) => (
                          <div key={label} className="bg-gray-800 rounded-lg p-3">
                            <div className="text-gray-400 text-xs mb-1">{label}</div>
                            <div className="font-semibold text-white capitalize truncate">{val}</div>
                          </div>
                        ))}
                      </div>
                    )}

                    {/* Class names */}
                    {summary.dataset?.classes?.length > 0 && (
                      <div className="flex flex-wrap gap-2">
                        {summary.dataset.classes.map(c => (
                          <span key={c} className="px-2 py-0.5 bg-gray-700 text-gray-300 rounded text-xs font-mono">
                            {c}
                          </span>
                        ))}
                      </div>
                    )}

                    {/* Training runs */}
                    {summary.training_runs?.length > 0 ? (
                      <div className="space-y-3">
                        <h3 className="text-sm font-semibold text-gray-300 uppercase tracking-wide">
                          Training Runs
                        </h3>
                        {summary.training_runs.map((run, idx) => {
                          const logKey = `${s.session_id}:${run.model_type}`
                          const showLog = logExpanded[logKey]
                          const pm = primaryMetric(run.metrics)
                          return (
                            <div key={idx} className="bg-gray-800 rounded-lg overflow-hidden">
                              {/* Run header */}
                              <div className="flex items-center gap-3 px-4 py-3 flex-wrap">
                                <span className="font-mono text-sm text-white font-semibold">
                                  {run.model_type}
                                </span>
                                {run.pilot && (
                                  <span className="px-2 py-0.5 bg-violet-900 text-violet-300 rounded text-xs">
                                    auto
                                  </span>
                                )}
                                {pm && (
                                  <span className="text-sm text-emerald-400 font-semibold">
                                    {pm.label}: {typeof pm.value === 'number' ? pm.value.toFixed(4) : pm.value}
                                  </span>
                                )}
                                <span className="text-xs text-gray-500 ml-auto">
                                  {fmtSeconds(run.training_time_s)} · {fmtDate(run.trained_at)}
                                </span>
                              </div>

                              {/* All metrics */}
                              {run.metrics && Object.keys(run.metrics).length > 0 && (
                                <div className="px-4 pb-3 grid grid-cols-3 sm:grid-cols-4 gap-2">
                                  {Object.entries(run.metrics).map(([k, v]) => (
                                    <div key={k} className="bg-gray-750 bg-gray-900 rounded p-2 text-center">
                                      <div className="text-xs text-gray-400">{k}</div>
                                      <div className="text-sm font-semibold text-white">
                                        {typeof v === 'number' ? v.toFixed(4) : String(v)}
                                      </div>
                                    </div>
                                  ))}
                                </div>
                              )}

                              {/* Log toggle */}
                              {run.log_tail && (
                                <div className="border-t border-gray-700">
                                  <button
                                    onClick={() => toggleLog(s.session_id, run.model_type)}
                                    className="w-full text-left px-4 py-2 text-xs text-gray-500 hover:text-gray-300 transition"
                                  >
                                    {showLog ? '▲ Hide log' : '▼ Show training log'}
                                  </button>
                                  {showLog && (
                                    <pre className="px-4 pb-4 text-xs font-mono text-green-300 whitespace-pre-wrap
                                                    max-h-64 overflow-y-auto bg-gray-950">
                                      {run.log_tail}
                                    </pre>
                                  )}
                                </div>
                              )}
                            </div>
                          )
                        })}
                      </div>
                    ) : (
                      <p className="text-sm text-gray-500">No training runs recorded for this session.</p>
                    )}
                  </>
                )}
              </div>
            )}
          </div>
        )
      })}
    </div>
  )
}

// ── Predict Panel sub-component ───────────────────────────────────────────────

function PredictPanel() {
  const [sessions, setSessions] = useState([])
  const [loading, setLoading] = useState(true)
  const [deleting, setDeleting] = useState({})
  const [allModels, setAllModels] = useState([]) // array of {sessionId, sessionName, modelType, metrics, taskType}
  const [selectedModel, setSelectedModel] = useState('')
  const [predictFile, setPredictFile] = useState(null)
  const [predicting, setPredicting] = useState(false)
  const [predictionResult, setPredictionResult] = useState(null)
  const [predictionError, setPredictionError] = useState(null)

  useEffect(() => {
    loadSessions()
  }, [])

  const loadSessions = async () => {
    setLoading(true)
    try {
      const r = await listSessions()
      const sess = r.data?.data || []
      setSessions(sess)
      const modelsList = []
      for (const s of sess) {
        if (s.task_type === 'classification') {
          try {
            const m = await listModels(s.session_id)
            const mods = m.data?.data || []
            mods.forEach(mod => {
              modelsList.push({
                sessionId: s.session_id,
                sessionName: s.dataset || 'Unnamed Dataset',
                modelType: mod.model_type,
                metrics: mod.metrics,
                taskType: s.task_type,
              })
            })
          } catch (_) {}
        }
      }
      setAllModels(modelsList)
    } catch (_) {
      setSessions([])
      setAllModels([])
    } finally {
      setLoading(false)
    }
  }

  const handleDelete = async (sid) => {
    if (!confirm('Are you sure you want to delete this dataset? This action cannot be undone.')) return
    setDeleting(prev => ({ ...prev, [sid]: true }))
    try {
      await deleteDataset(sid)
      setSessions(prev => prev.filter(s => s.session_id !== sid))
      setAllModels(prev => prev.filter(m => m.sessionId !== sid))
      if (selectedModel && selectedModel.startsWith(sid + ':')) {
        setSelectedModel('')
      }
    } catch (err) {
      alert(`Delete failed: ${err.response?.data?.detail || err.message}`)
    } finally {
      setDeleting(prev => ({ ...prev, [sid]: false }))
    }
  }

  const handlePredict = async () => {
    if (!selectedModel || !predictFile) return
    const [sessionId, modelType] = selectedModel.split(':')
    const model = allModels.find(m => m.sessionId === sessionId && m.modelType === modelType)
    if (!model) return
    setPredicting(true)
    setPredictionError(null)
    setPredictionResult(null)
    try {
      const fd = new FormData()
      fd.append('session_id', sessionId)
      fd.append('model_type', modelType)
      fd.append('task_type', model.taskType)
      fd.append('file', predictFile)
      const r = await predict(fd)
      setPredictionResult(r.data?.data)
    } catch (err) {
      setPredictionError(err.response?.data?.detail || err.message)
    } finally {
      setPredicting(false)
      setPredictFile(null)
    }
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-48 text-gray-500 text-sm">
        Loading datasets…
      </div>
    )
  }

  return (
    <div className="max-w-4xl mx-auto px-6 py-8 space-y-8">
      <h2 className="text-lg font-semibold text-white mb-2">Predict with Trained Models</h2>

      {/* Uploaded Datasets */}
      <section>
        <h3 className="text-md font-semibold text-white mb-4">Uploaded Datasets</h3>
        {sessions.length === 0 ? (
          <p className="text-sm text-gray-500">No datasets uploaded yet.</p>
        ) : (
          <div className="space-y-3">
            {sessions.map(s => (
              <div key={s.session_id} className="bg-gray-900 border border-gray-800 rounded-xl p-4 flex items-center justify-between">
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 flex-wrap">
                    <span className="font-semibold text-white truncate">{s.dataset || 'Unnamed Dataset'}</span>
                    {s.task_type && (
                      <TaskBadge taskType={s.task_type} />
                    )}
                  </div>
                  <div className="text-xs text-gray-500 mt-1">
                    {s.num_samples > 0 ? `${s.num_samples} samples` : 'samples unknown'} · 
                    {s.num_training_runs} training run{s.num_training_runs !== 1 ? 's' : ''} · 
                    {new Date(s.created_at).toLocaleString()}
                  </div>
                </div>
                <button
                  onClick={() => handleDelete(s.session_id)}
                  disabled={deleting[s.session_id]}
                  className="px-3 py-1 bg-red-700 hover:bg-red-600 disabled:bg-gray-700 text-xs font-semibold rounded transition"
                >
                  {deleting[s.session_id] ? 'Deleting…' : 'Delete'}
                </button>
              </div>
            ))}
          </div>
        )}
      </section>

      {/* Image Classification Prediction */}
      <section>
        <h3 className="text-md font-semibold text-white mb-4">Run Prediction</h3>
        {allModels.length === 0 ? (
          <p className="text-sm text-gray-500">No classification models available. Train a model first.</p>
        ) : (
          <div className="bg-gray-900 border border-gray-800 rounded-xl p-4 space-y-4">
            <div>
              <label className="block text-sm text-gray-400 mb-2">Select Model</label>
              <select
                value={selectedModel}
                onChange={(e) => setSelectedModel(e.target.value)}
                className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-white text-sm focus:outline-none focus:border-blue-500"
              >
                <option value="">Choose a model...</option>
                {allModels.map((m, idx) => (
                  <option key={idx} value={`${m.sessionId}:${m.modelType}`}>
                    {m.modelType} (from {m.sessionName})
                  </option>
                ))}
              </select>
            </div>
            <div>
              <label className="block text-sm text-gray-400 mb-2">Upload Image for Prediction</label>
              <input
                type="file"
                accept="image/*"
                onChange={(e) => setPredictFile(e.target.files[0])}
                className="block w-full text-sm text-gray-300 file:mr-3 file:py-2 file:px-4
                           file:rounded-lg file:border-0 file:text-sm file:font-semibold
                           file:bg-blue-700 file:text-white hover:file:bg-blue-600 cursor-pointer"
              />
              {predictFile && (
                <p className="mt-1 text-xs text-gray-500">{predictFile.name}</p>
              )}
            </div>
            <button
              onClick={handlePredict}
              disabled={!selectedModel || !predictFile || predicting}
              className="px-6 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-700
                         disabled:cursor-not-allowed rounded-lg text-sm font-semibold transition"
            >
              {predicting ? 'Predicting…' : 'Run Prediction'}
            </button>
          </div>
        )}
      </section>

      {/* Prediction Result */}
      {predictionResult && (
        <section>
          <h3 className="text-md font-semibold text-white mb-4">Prediction Result</h3>
          <div className="bg-gray-900 border border-gray-800 rounded-xl p-4">
            <div className="space-y-2">
              {predictionResult.label && (
                <div>
                  <span className="text-sm text-gray-400">Predicted Class: </span>
                  <span className="font-semibold text-white">{predictionResult.label}</span>
                </div>
              )}
              {predictionResult.confidence !== undefined && (
                <div>
                  <span className="text-sm text-gray-400">Confidence: </span>
                  <span className="font-semibold text-white">{predictionResult.confidence.toFixed(4)}</span>
                </div>
              )}
              {predictionResult.class_probabilities && (
                <div>
                  <span className="text-sm text-gray-400">Class Probabilities:</span>
                  <div className="mt-1 space-y-1">
                    {Object.entries(predictionResult.class_probabilities).map(([cls, prob]) => (
                      <div key={cls} className="text-xs text-gray-300">
                        {cls}: {prob}
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        </section>
      )}

      {predictionError && (
        <div className="bg-red-950 border border-red-800 rounded-lg p-3 text-red-300 text-sm">
          Prediction failed: {predictionError}
        </div>
      )}
    </div>
  )
}

// ── Log Panel sub-component ────────────────────────────────────────────────────

function LogPanel({ logs, status }) {
  const endRef = useRef(null)

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [logs])

  const statusColor = {
    running: 'text-yellow-400',
    completed: 'text-green-400',
    failed: 'text-red-400',
  }[status] || 'text-gray-400'

  return (
    <div className="border border-gray-700 rounded-xl overflow-hidden">
      <div className="flex items-center gap-2 px-4 py-2 border-b border-gray-700 bg-gray-800">
        <span className="text-xs font-mono text-gray-400">Training Log</span>
        {status && (
          <span className={`ml-auto text-xs font-semibold ${statusColor}`}>
            {status.toUpperCase()}
          </span>
        )}
      </div>
      <div className="h-72 overflow-y-auto p-4 font-mono text-xs text-green-300 space-y-0.5 bg-gray-950">
        {logs.map((line, i) => (
          <div key={i} className="leading-relaxed whitespace-pre-wrap">
            <span className="text-gray-600 mr-2 select-none">[{String(i + 1).padStart(3, '0')}]</span>
            {line}
          </div>
        ))}
        <div ref={endRef} />
      </div>
    </div>
  )
}
