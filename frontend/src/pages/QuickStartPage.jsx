import React, { useState, useRef, useEffect } from 'react'
import {
  Upload, Zap, Brain, TrendingUp, Loader2, CheckCircle, AlertCircle,
  Cpu, ChevronRight, Image, FileText, Database, Target, BarChart2,
  Layers, RefreshCw, X
} from 'lucide-react'
import { uploadDataset, previewDataset, preprocessDataset, autoTrain,
         getTrainingStatus, getTrainingResults } from '../api/client'

// ── Task type config ──────────────────────────────────────────────────────────

const TASK_TYPES = [
  {
    id: 'classification',
    label: 'Image Classification',
    icon: <Image className="w-6 h-6" />,
    desc: 'Classify images into categories (cats vs dogs, disease detection…)',
    color: 'border-blue-500 bg-blue-500/10 text-blue-400',
    badge: 'bg-blue-500/20 text-blue-300',
  },
  {
    id: 'object_detection',
    label: 'Object Detection',
    icon: <Target className="w-6 h-6" />,
    desc: 'Locate and identify multiple objects within images (YOLO)',
    color: 'border-orange-500 bg-orange-500/10 text-orange-400',
    badge: 'bg-orange-500/20 text-orange-300',
  },
  {
    id: 'segmentation',
    label: 'Segmentation',
    icon: <Layers className="w-6 h-6" />,
    desc: 'Pixel-level classification — segment regions in an image',
    color: 'border-purple-500 bg-purple-500/10 text-purple-400',
    badge: 'bg-purple-500/20 text-purple-300',
  },
  {
    id: 'regression',
    label: 'Regression',
    icon: <BarChart2 className="w-6 h-6" />,
    desc: 'Predict continuous values from tabular or structured data',
    color: 'border-green-500 bg-green-500/10 text-green-400',
    badge: 'bg-green-500/20 text-green-300',
  },
  {
    id: 'tabular_classification',
    label: 'Tabular Classification',
    icon: <FileText className="w-6 h-6" />,
    desc: 'Classify rows in CSV/tabular data using ML models',
    color: 'border-teal-500 bg-teal-500/10 text-teal-400',
    badge: 'bg-teal-500/20 text-teal-300',
  },
]

// ── Helpers ───────────────────────────────────────────────────────────────────

function fmtSize(bytes) {
  if (!bytes) return ''
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`
}

function guessDatasetName(files) {
  if (!files || !files.length) return 'Dataset'
  const first = files[0]
  // If folder upload, the first file path contains the folder name
  if (first.webkitRelativePath) {
    const parts = first.webkitRelativePath.split('/')
    return parts[0] || first.name.replace(/\.[^.]+$/, '')
  }
  return first.name.replace(/\.[^.]+$/, '').replace(/[-_]/g, ' ')
}

function wsBase() {
  const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
  return `${proto}//${window.location.host}`
}

// ── Step indicator ────────────────────────────────────────────────────────────

const STEPS = ['Upload', 'Task', 'Mode', 'Training', 'Results']

function StepBar({ step }) {
  return (
    <div className="flex items-center justify-center gap-0 mb-10">
      {STEPS.map((label, idx) => {
        const s = idx + 1
        const done = s < step
        const active = s === step
        return (
          <React.Fragment key={s}>
            <div className="flex flex-col items-center">
              <div className={`w-9 h-9 rounded-full flex items-center justify-center text-sm font-bold border-2 transition-all
                ${done ? 'bg-accent-500 border-accent-500 text-white'
                  : active ? 'bg-dark-700 border-accent-400 text-accent-400'
                  : 'bg-dark-800 border-dark-600 text-gray-500'}`}>
                {done ? <CheckCircle className="w-4 h-4" /> : s}
              </div>
              <span className={`text-xs mt-1 font-medium ${active ? 'text-accent-400' : done ? 'text-gray-400' : 'text-gray-600'}`}>
                {label}
              </span>
            </div>
            {idx < STEPS.length - 1 && (
              <div className={`h-0.5 w-12 mb-5 mx-1 rounded transition-all ${done ? 'bg-accent-500' : 'bg-dark-600'}`} />
            )}
          </React.Fragment>
        )
      })}
    </div>
  )
}

// ── Main page ─────────────────────────────────────────────────────────────────

export default function QuickStartPage() {
  const [step, setStep] = useState(1)
  const [sessionId, setSessionId] = useState(null)
  const [datasetName, setDatasetName] = useState('')
  const [datasetInfo, setDatasetInfo] = useState(null)  // preview data
  const [detectedTaskType, setDetectedTaskType] = useState(null)
  const [selectedTask, setSelectedTask] = useState(null)
  const [trainingMode, setTrainingMode] = useState(null)
  const [loading, setLoading] = useState(false)
  const [uploadProgress, setUploadProgress] = useState(0)
  const [logs, setLogs] = useState([])
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [experiments, setExperiments] = useState([])
  const logsEndRef = useRef(null)
  const uploadRef = useRef(null)
  const folderRef = useRef(null)
  const wsRef = useRef(null)

  // Auto-scroll logs
  useEffect(() => {
    logsEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [logs])

  // Cleanup WS on unmount
  useEffect(() => () => wsRef.current?.close(), [])

  // ── Step 1: Upload ──────────────────────────────────────────────────────────

  const handleFiles = async (files) => {
    if (!files || !files.length) return
    const fileList = Array.from(files)
    const sid = `session-${Date.now()}`
    setSessionId(sid)
    setDatasetName(guessDatasetName(fileList))
    setLoading(true)
    setError(null)
    setUploadProgress(0)

    try {
      // Folder upload (multiple files) uses upload-folder endpoint
      if (fileList.length > 1 || fileList[0].webkitRelativePath) {
        const fd = new FormData()
        fd.append('session_id', sid)
        fileList.forEach(f => fd.append('files', f, f.webkitRelativePath || f.name))
        await fetch('/api/v1/dataset/upload-folder', { method: 'POST', body: fd })
      } else {
        await uploadDataset(fileList[0], sid)
      }

      setUploadProgress(50)
      // Preview to get dataset info + detected task type
      const prev = await previewDataset(sid)
      const info = prev.data.data
      setDatasetInfo(info)

      // Detect task from format
      const detected = info.type === 'images'
        ? (info.classes?.length > 0 ? 'classification' : 'classification')
        : 'tabular_classification'
      setDetectedTaskType(detected)
      setSelectedTask(detected)
      setUploadProgress(100)
      setStep(2)
    } catch (err) {
      setError(`Upload failed: ${err.response?.data?.detail || err.message}`)
    } finally {
      setLoading(false)
    }
  }

  const onFileChange = (e) => handleFiles(e.target.files)
  const onFolderChange = (e) => handleFiles(e.target.files)
  const onDrop = (e) => {
    e.preventDefault()
    handleFiles(e.dataTransfer.files)
  }

  // ── Step 2: Task selection ──────────────────────────────────────────────────

  const confirmTask = () => {
    if (!selectedTask) return
    setStep(3)
  }

  // ── Step 3: Mode selection ──────────────────────────────────────────────────

  const selectMode = (mode) => {
    setTrainingMode(mode)
    if (mode === 'autonomous') {
      handleAutonomousStart()
    } else {
      handleStandardStart()
    }
  }

  // ── Standard mode ───────────────────────────────────────────────────────────

  const handleStandardStart = async () => {
    setStep(4)
    setLogs([`Starting standard training for "${datasetName}"…`])
    setLoading(true)
    setError(null)

    try {
      await preprocessDataset({ session_id: sessionId })
      addLog('✓ Preprocessing complete')

      const res = await autoTrain(sessionId)
      addLog(`✓ Training started: ${res.data?.data?.model_type || ''}`)

      // WS for live logs
      const ws = new WebSocket(`${wsBase()}/api/v1/training/ws/${sessionId}`)
      wsRef.current = ws

      ws.onmessage = (e) => {
        const d = JSON.parse(e.data)
        if (d.log) addLog(d.log)
        if (d.heartbeat) return
      }

      // Poll for completion
      pollStatus()
    } catch (err) {
      setError(`Error: ${err.response?.data?.detail || err.message}`)
      setLoading(false)
    }
  }

  const pollStatus = async () => {
    try {
      const s = await getTrainingStatus(sessionId)
      const status = s.data.data
      if (status?.status === 'completed') {
        const r = await getTrainingResults(sessionId)
        setResult(r.data.data)
        addLog('✅ Training complete!')
        setLoading(false)
        setStep(5)
      } else if (status?.status === 'failed') {
        setError('Training failed — check logs above')
        setLoading(false)
      } else {
        setTimeout(pollStatus, 3000)
      }
    } catch {
      setTimeout(pollStatus, 3000)
    }
  }

  // ── Autonomous mode ─────────────────────────────────────────────────────────

  const handleAutonomousStart = async () => {
    setStep(4)
    setLogs([`🤖 Starting Autonomous ML Engineer for "${datasetName}"…`])
    setLoading(true)
    setError(null)

    try {
      const params = new URLSearchParams({
        session_id: sessionId,
        max_experiments: 3,
        task_type: selectedTask || 'classification',
        dataset_name: datasetName,
      })

      const resp = await fetch(`/api/v1/training/autonomous?${params}`, { method: 'POST' })
      if (!resp.ok) {
        const body = await resp.json().catch(() => ({}))
        throw new Error(body.detail || `HTTP ${resp.status}`)
      }

      // Connect WS
      const ws = new WebSocket(`${wsBase()}/api/v1/training/ws/${sessionId}`)
      wsRef.current = ws

      ws.onmessage = (e) => {
        const d = JSON.parse(e.data)
        if (d.heartbeat) return
        if (d.log) {
          addLog(d.log)
          // Parse experiment accuracy lines
          const m = d.log.match(/accuracy=([\d.]+)/)
          if (m) setExperiments(prev => [...prev, { accuracy: parseFloat(m[1]) }])

          if (d.log.includes('AUTONOMOUS TRAINING COMPLETE')) {
            setLoading(false)
            setStep(5)
          }
          if (d.log.includes('AUTONOMOUS PIPELINE FAILED')) {
            setLoading(false)
            setError('Autonomous training failed — see logs for details')
          }
        }
      }

      ws.onerror = () => {
        addLog('⚠ WebSocket error — training may still be running on backend')
        setLoading(false)
      }

      ws.onclose = () => {
        setLoading(false)
      }

    } catch (err) {
      setError(`Failed to start: ${err.message}`)
      setLoading(false)
    }
  }

  const addLog = (msg) => setLogs(prev => [...prev, msg])

  // ── Reset ────────────────────────────────────────────────────────────────────

  const handleReset = () => {
    wsRef.current?.close()
    setStep(1)
    setSessionId(null)
    setDatasetName('')
    setDatasetInfo(null)
    setDetectedTaskType(null)
    setSelectedTask(null)
    setTrainingMode(null)
    setLoading(false)
    setLogs([])
    setResult(null)
    setError(null)
    setExperiments([])
    setUploadProgress(0)
    if (uploadRef.current) uploadRef.current.value = ''
    if (folderRef.current) folderRef.current.value = ''
  }

  // ── Render ────────────────────────────────────────────────────────────────────

  const taskDef = TASK_TYPES.find(t => t.id === selectedTask)

  return (
    <div className="min-h-screen bg-dark-900 px-4 py-8">
      <div className="max-w-3xl mx-auto">

        {/* Header */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-white flex items-center gap-3">
            <Zap className="text-accent-400" size={32} />
            QuickStart
          </h1>
          <p className="text-gray-400 mt-1 text-sm">Upload a dataset → choose task → train with AI</p>
        </div>

        <StepBar step={step} />

        {/* ── STEP 1: Upload ── */}
        {step === 1 && (
          <div className="bg-dark-800 rounded-2xl border border-dark-600 p-8">
            <h2 className="text-xl font-semibold text-white mb-6 flex items-center gap-2">
              <Upload className="w-5 h-5 text-accent-400" /> Upload Dataset
            </h2>

            <div
              onDrop={onDrop}
              onDragOver={e => e.preventDefault()}
              onClick={() => uploadRef.current?.click()}
              className="border-2 border-dashed border-dark-500 hover:border-accent-500 rounded-xl p-12 text-center cursor-pointer transition-colors group"
            >
              <Upload className="w-12 h-12 mx-auto text-gray-500 group-hover:text-accent-400 mb-3 transition-colors" />
              <p className="text-white font-semibold mb-1">Drop files here or click to browse</p>
              <p className="text-gray-500 text-sm">CSV, ZIP, JPG, PNG — any format</p>
              <input ref={uploadRef} type="file" multiple className="hidden" onChange={onFileChange} />
            </div>

            <div className="flex items-center gap-3 mt-4">
              <div className="flex-1 h-px bg-dark-600" />
              <span className="text-gray-500 text-sm">or</span>
              <div className="flex-1 h-px bg-dark-600" />
            </div>

            <button
              onClick={() => folderRef.current?.click()}
              className="w-full mt-4 py-3 rounded-xl border border-dark-500 hover:border-accent-400 text-gray-300 hover:text-white text-sm font-medium transition-colors flex items-center justify-center gap-2"
            >
              <Database className="w-4 h-4" />
              Browse Folder (image dataset)
            </button>
            <input
              ref={folderRef}
              type="file"
              className="hidden"
              // @ts-ignore
              webkitdirectory=""
              multiple
              onChange={onFolderChange}
            />

            {loading && (
              <div className="mt-6">
                <div className="flex items-center gap-2 text-accent-400 mb-2">
                  <Loader2 className="w-4 h-4 animate-spin" />
                  <span className="text-sm">Uploading… {uploadProgress}%</span>
                </div>
                <div className="w-full bg-dark-700 rounded-full h-1.5">
                  <div className="bg-accent-500 h-1.5 rounded-full transition-all" style={{ width: `${uploadProgress}%` }} />
                </div>
              </div>
            )}

            {error && <ErrorBox message={error} onDismiss={() => setError(null)} />}
          </div>
        )}

        {/* ── STEP 2: Task Type Selection ── */}
        {step === 2 && (
          <div className="bg-dark-800 rounded-2xl border border-dark-600 p-8">
            <h2 className="text-xl font-semibold text-white mb-2 flex items-center gap-2">
              <Brain className="w-5 h-5 text-accent-400" /> What task do you want to train?
            </h2>
            <p className="text-gray-400 text-sm mb-6">
              Dataset: <span className="text-white font-medium">"{datasetName}"</span>
              {datasetInfo && (
                <span className="ml-2 text-gray-500">
                  ({datasetInfo.count || datasetInfo.rows || '?'} samples
                  {datasetInfo.classes?.length ? `, ${datasetInfo.classes.length} classes` : ''})
                </span>
              )}
              {detectedTaskType && (
                <span className="ml-2 text-accent-400 text-xs">
                  · auto-detected: {detectedTaskType.replace('_', ' ')}
                </span>
              )}
            </p>

            <div className="grid grid-cols-1 gap-3">
              {TASK_TYPES.map(task => (
                <button
                  key={task.id}
                  onClick={() => setSelectedTask(task.id)}
                  className={`flex items-center gap-4 p-4 rounded-xl border-2 text-left transition-all
                    ${selectedTask === task.id
                      ? task.color + ' shadow-lg'
                      : 'border-dark-600 hover:border-dark-400 text-gray-400'}`}
                >
                  <div className={`flex-shrink-0 ${selectedTask === task.id ? '' : 'text-gray-500'}`}>
                    {task.icon}
                  </div>
                  <div>
                    <div className={`font-semibold text-sm ${selectedTask === task.id ? 'text-white' : 'text-gray-300'}`}>
                      {task.label}
                    </div>
                    <div className="text-xs text-gray-500 mt-0.5">{task.desc}</div>
                  </div>
                  {selectedTask === task.id && (
                    <CheckCircle className="w-5 h-5 ml-auto flex-shrink-0 text-current" />
                  )}
                </button>
              ))}
            </div>

            <button
              onClick={confirmTask}
              disabled={!selectedTask}
              className="mt-6 w-full py-3 bg-accent-600 hover:bg-accent-700 disabled:bg-dark-600 disabled:text-gray-500
                text-white font-semibold rounded-xl transition-colors flex items-center justify-center gap-2"
            >
              Continue <ChevronRight className="w-4 h-4" />
            </button>
          </div>
        )}

        {/* ── STEP 3: Training Mode ── */}
        {step === 3 && (
          <div className="bg-dark-800 rounded-2xl border border-dark-600 p-8">
            <h2 className="text-xl font-semibold text-white mb-2 flex items-center gap-2">
              <Zap className="w-5 h-5 text-accent-400" /> Choose Training Mode
            </h2>
            <p className="text-gray-400 text-sm mb-6">
              <span className="text-white">"{datasetName}"</span>
              {taskDef && <span className="ml-2 px-2 py-0.5 rounded text-xs font-medium bg-accent-500/20 text-accent-400">{taskDef.label}</span>}
            </p>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {/* Standard */}
              <button
                onClick={() => selectMode('standard')}
                className="flex flex-col p-6 rounded-xl border-2 border-dark-500 hover:border-orange-500 hover:bg-orange-500/5 text-left transition-all group"
              >
                <div className="flex items-center gap-3 mb-3">
                  <Zap className="w-8 h-8 text-orange-400" />
                  <h3 className="text-white font-bold">Standard</h3>
                </div>
                <ul className="text-gray-400 text-xs space-y-1 mb-4">
                  <li>· Auto preprocessing</li>
                  <li>· LLM model selection</li>
                  <li>· Single best model</li>
                </ul>
                <div className="mt-auto py-2 px-4 bg-orange-500/20 text-orange-400 rounded-lg text-sm font-medium text-center group-hover:bg-orange-500 group-hover:text-white transition-colors">
                  Start Standard
                </div>
              </button>

              {/* Autonomous */}
              <button
                onClick={() => selectMode('autonomous')}
                className="flex flex-col p-6 rounded-xl border-2 border-accent-500 bg-accent-500/5 text-left transition-all group hover:bg-accent-500/10"
              >
                <div className="flex items-center gap-3 mb-3">
                  <Cpu className="w-8 h-8 text-accent-400" />
                  <h3 className="text-white font-bold">🤖 Autonomous</h3>
                </div>
                <ul className="text-gray-400 text-xs space-y-1 mb-4">
                  <li>· AI analyzes your dataset</li>
                  <li>· Runs multiple experiments</li>
                  <li>· Auto improves & selects best</li>
                  <li className="text-accent-400 font-medium">✨ Recommended</li>
                </ul>
                <div className="mt-auto py-2 px-4 bg-accent-600 text-white rounded-lg text-sm font-medium text-center group-hover:bg-accent-700 transition-colors">
                  Start Autonomous
                </div>
              </button>
            </div>
          </div>
        )}

        {/* ── STEP 4: Training Logs ── */}
        {step === 4 && (
          <div className="bg-dark-800 rounded-2xl border border-dark-600 p-6">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-lg font-semibold text-white flex items-center gap-2">
                {loading
                  ? <Loader2 className="w-5 h-5 text-accent-400 animate-spin" />
                  : <CheckCircle className="w-5 h-5 text-green-400" />}
                {trainingMode === 'autonomous' ? '🤖 Autonomous Training' : 'Training'}
              </h2>
              <span className="text-xs text-gray-500 px-2 py-1 bg-dark-700 rounded">
                "{datasetName}"
              </span>
            </div>

            {/* Log terminal */}
            <div className="bg-gray-950 rounded-xl border border-dark-700 p-4 h-72 overflow-y-auto font-mono text-xs">
              {logs.length === 0
                ? <p className="text-gray-600">Waiting for logs…</p>
                : logs.map((line, i) => (
                    <div key={i} className={`leading-5 whitespace-pre-wrap break-all
                      ${line.includes('✅') || line.includes('✓') ? 'text-green-400'
                        : line.includes('❌') || line.includes('✗') ? 'text-red-400'
                        : line.includes('⚠') ? 'text-yellow-400'
                        : line.includes('🤖') || line.includes('🧠') || line.includes('📊') ? 'text-accent-400'
                        : line.startsWith('=') || line.startsWith('-') ? 'text-gray-600'
                        : 'text-green-300'}`}>
                      {line}
                    </div>
                  ))
              }
              <div ref={logsEndRef} />
            </div>

            {/* Experiment progress */}
            {experiments.length > 0 && (
              <div className="mt-4 p-3 bg-dark-700 rounded-xl">
                <p className="text-xs text-gray-400 mb-2">Experiments ({experiments.length}):</p>
                <div className="space-y-1.5">
                  {experiments.map((exp, i) => (
                    <div key={i} className="flex items-center gap-2">
                      <span className="text-xs text-gray-500 w-8">#{i + 1}</span>
                      <div className="flex-1 bg-dark-600 rounded-full h-1.5">
                        <div
                          className="h-1.5 rounded-full bg-accent-500"
                          style={{ width: `${Math.min(100, exp.accuracy * 100)}%` }}
                        />
                      </div>
                      <span className="text-xs text-gray-300 w-14 text-right">
                        {(exp.accuracy * 100).toFixed(1)}%
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {error && <ErrorBox message={error} onDismiss={() => setError(null)} />}
          </div>
        )}

        {/* ── STEP 5: Results ── */}
        {step === 5 && (
          <div className="bg-dark-800 rounded-2xl border border-dark-600 p-8">
            <div className="flex items-center gap-3 mb-6">
              <div className="w-12 h-12 rounded-full bg-green-500/20 flex items-center justify-center">
                <CheckCircle className="w-6 h-6 text-green-400" />
              </div>
              <div>
                <h2 className="text-xl font-bold text-white">Training Complete!</h2>
                <p className="text-gray-400 text-sm">"{datasetName}"</p>
              </div>
            </div>

            {/* Summary from logs */}
            {experiments.length > 0 && (
              <div className="grid grid-cols-3 gap-4 mb-6">
                <MetricCard label="Experiments" value={experiments.length} />
                <MetricCard
                  label="Best Accuracy"
                  value={`${(Math.max(...experiments.map(e => e.accuracy)) * 100).toFixed(1)}%`}
                  highlight
                />
                <MetricCard label="Task" value={taskDef?.label || selectedTask} />
              </div>
            )}

            {result && (
              <div className="bg-dark-700 rounded-xl p-4 mb-6">
                <p className="text-xs text-gray-400 uppercase tracking-wide mb-3">Best Model</p>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <p className="text-gray-500 text-xs">Model</p>
                    <p className="text-white font-semibold">{result.best_model?.model_name || 'RandomForest'}</p>
                  </div>
                  {result.best_model?.accuracy != null && (
                    <div>
                      <p className="text-gray-500 text-xs">Accuracy</p>
                      <p className="text-green-400 font-bold text-lg">
                        {(result.best_model.accuracy * 100).toFixed(1)}%
                      </p>
                    </div>
                  )}
                </div>
              </div>
            )}

            {/* Final logs snippet */}
            <div className="bg-gray-950 rounded-xl p-3 max-h-40 overflow-y-auto font-mono text-xs mb-6">
              {logs.slice(-15).map((line, i) => (
                <div key={i} className={`leading-5
                  ${line.includes('✅') || line.includes('✓') ? 'text-green-400'
                    : line.includes('❌') ? 'text-red-400'
                    : 'text-gray-400'}`}>
                  {line}
                </div>
              ))}
            </div>

            <div className="flex gap-3">
              <button
                onClick={handleReset}
                className="flex-1 py-3 bg-dark-700 hover:bg-dark-600 text-gray-300 font-semibold rounded-xl transition-colors flex items-center justify-center gap-2"
              >
                <RefreshCw className="w-4 h-4" /> New Training
              </button>
              <a
                href="/inference"
                className="flex-1 py-3 bg-accent-600 hover:bg-accent-700 text-white font-semibold rounded-xl transition-colors flex items-center justify-center gap-2 text-center no-underline"
              >
                <TrendingUp className="w-4 h-4" /> Go to Inference
              </a>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

// ── Reusable sub-components ───────────────────────────────────────────────────

function MetricCard({ label, value, highlight }) {
  return (
    <div className="bg-dark-700 rounded-xl p-4 text-center">
      <p className="text-gray-500 text-xs mb-1">{label}</p>
      <p className={`font-bold text-lg ${highlight ? 'text-green-400' : 'text-white'}`}>{value}</p>
    </div>
  )
}

function ErrorBox({ message, onDismiss }) {
  return (
    <div className="mt-4 p-4 bg-red-900/20 border border-red-800/50 rounded-xl flex items-start gap-3">
      <AlertCircle className="w-5 h-5 text-red-400 flex-shrink-0 mt-0.5" />
      <p className="text-red-300 text-sm flex-1">{message}</p>
      {onDismiss && (
        <button onClick={onDismiss} className="text-red-500 hover:text-red-300">
          <X className="w-4 h-4" />
        </button>
      )}
    </div>
  )
}
