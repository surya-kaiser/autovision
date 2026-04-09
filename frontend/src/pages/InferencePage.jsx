import React, { useState, useRef, useEffect, useCallback } from 'react'
import { predict, listModels, chatWithLLM } from '../api/client'
import api from '../api/client'
import { useToast } from '../components/Toast'
import {
  Loader2, Upload, Image, Zap, MessageSquare, Send,
  CheckCircle, XCircle, ChevronDown, RefreshCw, Database,
  BarChart2, FileText, Sparkles
} from 'lucide-react'

// ── Helpers ───────────────────────────────────────────────────────────────────

const IMAGE_EXTS = ['jpg', 'jpeg', 'png', 'bmp', 'webp', 'gif', 'tiff']

function isImageFile(name) {
  return IMAGE_EXTS.includes((name || '').split('.').pop().toLowerCase())
}

function pctColor(p) {
  if (p >= 0.7) return 'bg-green-500'
  if (p >= 0.4) return 'bg-yellow-500'
  return 'bg-red-500'
}

// ── Image drop zone ───────────────────────────────────────────────────────────

function ImageDropZone({ onImage, preview, loading }) {
  const inputRef = useRef(null)
  const [dragging, setDragging] = useState(false)

  const handleDrop = useCallback((e) => {
    e.preventDefault()
    setDragging(false)
    const f = e.dataTransfer.files[0]
    if (f && isImageFile(f.name)) onImage(f)
  }, [onImage])

  const handleFile = (e) => {
    const f = e.target.files[0]
    if (f) onImage(f)
    e.target.value = ''
  }

  return (
    <div
      onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
      onDragLeave={() => setDragging(false)}
      onDrop={handleDrop}
      onClick={() => !preview && inputRef.current?.click()}
      className={`relative rounded-2xl border-2 transition-all duration-200 overflow-hidden
        ${dragging ? 'border-accent-400 bg-accent-500/10' : preview ? 'border-dark-600' : 'border-dashed border-dark-600 hover:border-accent-500/50 cursor-pointer'}
        ${loading ? 'opacity-60 pointer-events-none' : ''}`}
      style={{ minHeight: 260 }}
    >
      <input ref={inputRef} type="file" accept="image/*" className="hidden" onChange={handleFile} />

      {preview ? (
        <div className="relative">
          <img src={preview} alt="Input" className="w-full object-contain max-h-72 rounded-2xl" />
          <button
            onClick={(e) => { e.stopPropagation(); onImage(null) }}
            className="absolute top-2 right-2 w-7 h-7 bg-dark-900/80 rounded-full flex items-center justify-center text-gray-400 hover:text-white"
          >×</button>
          {loading && (
            <div className="absolute inset-0 bg-dark-900/60 rounded-2xl flex items-center justify-center">
              <Loader2 className="w-8 h-8 text-accent-400 animate-spin" />
            </div>
          )}
        </div>
      ) : (
        <div className="flex flex-col items-center justify-center py-14 px-4 text-center">
          <div className="w-16 h-16 rounded-2xl bg-dark-700 border border-dark-600 flex items-center justify-center mb-4">
            <Image className="w-7 h-7 text-gray-500" />
          </div>
          <p className="text-sm font-semibold text-white mb-1">Drop image here</p>
          <p className="text-xs text-gray-500">JPG, PNG, BMP, WEBP</p>
          <button
            type="button"
            onClick={(e) => { e.stopPropagation(); inputRef.current?.click() }}
            className="mt-4 px-4 py-2 bg-accent-500 hover:bg-accent-400 text-white rounded-xl text-sm font-medium transition-colors"
          >
            Browse Image
          </button>
        </div>
      )}
    </div>
  )
}

// ── Prediction result card ────────────────────────────────────────────────────

function PredictionResult({ result }) {
  if (!result) return null

  const { label, confidence, class_probabilities, segmented_image } = result

  // ── Segmentation output ───────────────────────────────────────────────────
  if (segmented_image) {
    return (
      <div className="space-y-4 h-full flex flex-col">
        {label && (
          <div className="bg-dark-700 rounded-2xl p-4 flex items-center gap-3 shrink-0">
            <CheckCircle className="w-5 h-5 text-accent-400 shrink-0" />
            <span className="text-sm font-semibold text-white">{label}</span>
          </div>
        )}
        <div className="bg-dark-800 border border-dark-700 rounded-2xl overflow-hidden flex flex-col flex-1">
          <div className="px-4 py-3 border-b border-dark-700 text-xs font-semibold text-gray-400 shrink-0">
            Segmentation Output
          </div>
          <div className="flex-1 flex items-center justify-center overflow-auto p-4">
            <img
              src={segmented_image}
              alt="Segmentation result"
              className="max-w-full max-h-full object-contain"
            />
          </div>
        </div>
      </div>
    )
  }

  // Sort probabilities descending
  const probs = class_probabilities
    ? Object.entries(class_probabilities).sort((a, b) => b[1] - a[1])
    : []

  return (
    <div className="space-y-4">
      {/* Top prediction */}
      {label && (
        <div className="bg-dark-700 rounded-2xl p-5 flex items-center gap-4">
          <div className="w-12 h-12 rounded-xl bg-accent-500/20 flex items-center justify-center shrink-0">
            <CheckCircle className="w-6 h-6 text-accent-400" />
          </div>
          <div className="flex-1 min-w-0">
            <div className="text-xs text-gray-500 mb-0.5">Predicted class</div>
            <div className="text-2xl font-bold text-white capitalize truncate">{label}</div>
          </div>
          {confidence != null && (
            <div className="text-right shrink-0">
              <div className="text-2xl font-bold text-accent-400">{(confidence * 100).toFixed(1)}%</div>
              <div className="text-xs text-gray-500">confidence</div>
            </div>
          )}
        </div>
      )}

      {/* All class probabilities */}
      {probs.length > 0 && (
        <div className="bg-dark-800 border border-dark-700 rounded-2xl p-4 space-y-3">
          <div className="flex items-center gap-2 mb-1">
            <BarChart2 className="w-4 h-4 text-accent-400" />
            <span className="text-xs font-semibold text-gray-400">All Classes</span>
          </div>
          {probs.map(([cls, prob]) => (
            <div key={cls}>
              <div className="flex items-center justify-between text-xs mb-1">
                <span className={`font-medium ${cls === label ? 'text-accent-400' : 'text-gray-300'} capitalize`}>
                  {cls}
                  {cls === label && <span className="ml-1.5 px-1 py-0.5 bg-accent-500/20 text-accent-400 rounded text-[10px]">top</span>}
                </span>
                <span className="text-gray-500 font-mono">{(prob * 100).toFixed(2)}%</span>
              </div>
              <div className="h-2 bg-dark-600 rounded-full overflow-hidden">
                <div
                  className={`h-2 rounded-full transition-all duration-500 ${pctColor(prob)}`}
                  style={{ width: `${prob * 100}%` }}
                />
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Numeric regression result */}
      {result.value != null && (
        <div className="bg-dark-700 rounded-2xl p-5 text-center">
          <div className="text-xs text-gray-500 mb-1">Predicted value</div>
          <div className="text-3xl font-bold text-accent-400">{result.value.toFixed(4)}</div>
        </div>
      )}
    </div>
  )
}

// ── LLM Chat panel ────────────────────────────────────────────────────────────

function ChatPanel({ sessionId }) {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const endRef = useRef(null)

  useEffect(() => { endRef.current?.scrollIntoView({ behavior: 'smooth' }) }, [messages])

  async function send() {
    if (!input.trim()) return
    const userMsg = { role: 'user', content: input }
    setMessages(prev => [...prev, userMsg])
    setInput('')
    setLoading(true)
    try {
      const res = await chatWithLLM(sessionId || 'default', input, messages.slice(-6))
      setMessages(prev => [...prev, { role: 'assistant', content: res.data.data.response }])
    } catch (e) {
      setMessages(prev => [...prev, { role: 'assistant', content: 'Error: ' + (e.message || 'unknown') }])
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="bg-dark-800 border border-dark-700 rounded-2xl flex flex-col" style={{ height: 420 }}>
      <div className="flex items-center gap-2 px-4 py-3 border-b border-dark-700 shrink-0">
        <Sparkles className="w-4 h-4 text-accent-400" />
        <span className="text-xs font-semibold text-gray-300">LLM Assistant</span>
        <span className="ml-auto text-xs text-gray-600">Powered by Ollama</span>
      </div>
      <div className="flex-1 overflow-y-auto p-4 space-y-3">
        {messages.length === 0 && (
          <p className="text-xs text-gray-600 text-center mt-8">
            Ask about model choices, accuracy, hyperparameters, or next steps…
          </p>
        )}
        {messages.map((m, i) => (
          <div key={i} className={`flex ${m.role === 'user' ? 'justify-end' : 'justify-start'}`}>
            <div className={`max-w-[85%] rounded-2xl px-3 py-2 text-sm leading-relaxed
              ${m.role === 'user' ? 'bg-accent-500 text-white' : 'bg-dark-700 text-gray-200'}`}>
              {m.content}
            </div>
          </div>
        ))}
        {loading && (
          <div className="flex justify-start">
            <div className="bg-dark-700 rounded-2xl px-3 py-2">
              <Loader2 className="w-4 h-4 animate-spin text-gray-400" />
            </div>
          </div>
        )}
        <div ref={endRef} />
      </div>
      <div className="flex gap-2 p-3 border-t border-dark-700 shrink-0">
        <input
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && !loading && send()}
          placeholder="Ask a question…"
          className="flex-1 bg-dark-700 border border-dark-600 rounded-xl px-3 py-2 text-sm text-white focus:outline-none focus:border-accent-500"
        />
        <button
          onClick={send}
          disabled={loading || !input.trim()}
          className="p-2 bg-accent-500 hover:bg-accent-400 disabled:opacity-50 rounded-xl text-white transition-colors"
        >
          <Send className="w-4 h-4" />
        </button>
      </div>
    </div>
  )
}

// ── Main page ─────────────────────────────────────────────────────────────────

export default function InferencePage({ sessionId }) {
  const toast = useToast()

  const [sessions, setSessions] = useState([])
  const [activeSession, setActiveSession] = useState(sessionId)
  const [models, setModels] = useState([])
  const [selectedModel, setSelectedModel] = useState('')
  const [taskType, setTaskType] = useState('classification')

  const [imageFile, setImageFile] = useState(null)
  const [imagePreview, setImagePreview] = useState(null)
  const [dataJson, setDataJson] = useState('')
  const [csvFile, setCsvFile] = useState(null)

  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const [mode, setMode] = useState('image') // 'image' | 'tabular'

  useEffect(() => { loadSessions() }, [])
  useEffect(() => { if (activeSession) loadModels(activeSession) }, [activeSession])
  useEffect(() => { if (sessionId) setActiveSession(sessionId) }, [sessionId])

  async function loadSessions() {
    try {
      const r = await api.get('/dataset/sessions')
      setSessions(r.data.data || [])
    } catch {}
  }

  async function loadModels(sid) {
    try {
      const r = await api.get(`/inference/models/${sid}`)
      const m = r.data.data || []
      setModels(m)
      if (m.length > 0 && !selectedModel) setSelectedModel(m[0].model_type)
    } catch {}
  }

  function handleImageSelect(file) {
    if (!file) {
      setImageFile(null)
      setImagePreview(null)
      setResult(null)
      return
    }
    setImageFile(file)
    setResult(null)
    const reader = new FileReader()
    reader.onload = e => setImagePreview(e.target.result)
    reader.readAsDataURL(file)
  }

  async function handlePredict() {
    if (!activeSession) { setError('Select a dataset session first'); return }
    if (!selectedModel) { setError('No trained model found. Train a model first.'); return }

    setLoading(true)
    setError(null)
    setResult(null)

    const fd = new FormData()
    fd.append('session_id', activeSession)
    fd.append('model_type', selectedModel)
    fd.append('task_type', taskType)

    if (mode === 'image' && imageFile) {
      fd.append('file', imageFile)
    } else if (mode === 'tabular') {
      if (dataJson) fd.append('data_json', dataJson)
      if (csvFile) fd.append('file', csvFile)
    }

    try {
      const res = await predict(fd)
      setResult(res.data.data)
      toast('Prediction complete', 'success')
    } catch (e) {
      const msg = e.response?.data?.detail || e.message
      setError(msg)
      toast('Prediction failed: ' + msg, 'error')
    } finally {
      setLoading(false)
    }
  }

  const canPredict = activeSession && selectedModel && (
    (mode === 'image' && imageFile) ||
    (mode === 'tabular' && (dataJson || csvFile))
  )

  return (
    <div className="p-6 max-w-6xl mx-auto space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">Inference</h1>
          <p className="text-sm text-gray-500 mt-0.5">Run predictions on new data using trained models</p>
        </div>
      </div>

      {/* ── Session + Model selectors ─────────────────────────────────────── */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
        <div className="bg-dark-800 border border-dark-700 rounded-2xl p-4">
          <label className="text-xs font-semibold text-gray-400 mb-2 flex items-center gap-1.5">
            <Database className="w-3.5 h-3.5" /> Dataset Session
          </label>
          <select
            value={activeSession || ''}
            onChange={e => { setActiveSession(e.target.value); setSelectedModel(''); setResult(null) }}
            className="w-full bg-dark-700 border border-dark-600 rounded-xl px-3 py-2 text-sm text-white focus:outline-none focus:border-accent-500 mt-1"
          >
            {!activeSession && <option value="">Select session…</option>}
            {sessions.map(s => {
              const name = s.dataset && s.dataset !== 'unknown' ? s.dataset : null
              const task = s.task_type && s.task_type !== 'unknown' ? s.task_type : null
              const shortId = s.session_id.slice(0, 8)
              const label = name
                ? task ? `${name}  ·  ${task}` : name
                : `Dataset ${shortId}…`
              return (
                <option key={s.session_id} value={s.session_id}>{label}</option>
              )
            })}
          </select>
        </div>

        <div className="bg-dark-800 border border-dark-700 rounded-2xl p-4">
          <label className="text-xs font-semibold text-gray-400 mb-2 flex items-center gap-1.5">
            <Zap className="w-3.5 h-3.5" /> Trained Model
          </label>
          <select
            value={selectedModel}
            onChange={e => { setSelectedModel(e.target.value); setResult(null) }}
            className="w-full bg-dark-700 border border-dark-600 rounded-xl px-3 py-2 text-sm text-white focus:outline-none focus:border-accent-500 mt-1"
          >
            {models.length === 0 && <option value="">No models yet — train first</option>}
            {models.map(m => {
              let metricStr = ''
              if (m.metrics?.accuracy) metricStr = ` (acc: ${(m.metrics.accuracy * 100).toFixed(1)}%)`
              else if (m.metrics?.iou) metricStr = ` (IoU: ${(m.metrics.iou * 100).toFixed(1)}%)`
              else if (m.metrics?.map50) metricStr = ` (mAP50: ${(m.metrics.map50 * 100).toFixed(1)}%)`
              return (
                <option key={m.model_type} value={m.model_type}>
                  {m.model_type}{metricStr}
                </option>
              )
            })}
          </select>
        </div>
      </div>

      {/* ── Mode tabs ─────────────────────────────────────────────────────── */}
      <div className="flex gap-2">
        <button
          onClick={() => { setMode('image'); setResult(null) }}
          className={`flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-medium transition-colors
            ${mode === 'image' ? 'bg-accent-500 text-white' : 'bg-dark-800 border border-dark-700 text-gray-400 hover:text-white'}`}
        >
          <Image className="w-4 h-4" /> Image Prediction
        </button>
        <button
          onClick={() => { setMode('tabular'); setResult(null) }}
          className={`flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-medium transition-colors
            ${mode === 'tabular' ? 'bg-accent-500 text-white' : 'bg-dark-800 border border-dark-700 text-gray-400 hover:text-white'}`}
        >
          <FileText className="w-4 h-4" /> Tabular Prediction
        </button>
        <select
          value={taskType}
          onChange={e => setTaskType(e.target.value)}
          className="ml-auto bg-dark-800 border border-dark-700 rounded-xl px-3 py-2 text-sm text-gray-300 focus:outline-none focus:border-accent-500"
        >
          <option value="classification">Classification</option>
          <option value="regression">Regression</option>
          <option value="object_detection">Object Detection</option>
          <option value="segmentation">Segmentation</option>
        </select>
      </div>

      {/* ── Main prediction layout ─────────────────────────────────────── */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">

        {/* LEFT: Input */}
        <div className="space-y-4">
          {mode === 'image' ? (
            <ImageDropZone
              onImage={handleImageSelect}
              preview={imagePreview}
              loading={loading}
            />
          ) : (
            <div className="bg-dark-800 border border-dark-700 rounded-2xl p-5 space-y-4">
              <h2 className="text-sm font-semibold text-gray-300">Input Data</h2>
              <div>
                <label className="block text-xs text-gray-400 mb-1">JSON input</label>
                <textarea
                  value={dataJson}
                  onChange={e => setDataJson(e.target.value)}
                  rows={5}
                  placeholder='{"feature1": 1.2, "feature2": 3.4}'
                  className="w-full bg-dark-700 border border-dark-600 rounded-xl px-3 py-2 text-sm text-white font-mono focus:outline-none focus:border-accent-500"
                />
              </div>
              <div>
                <label className="block text-xs text-gray-400 mb-1">Or upload CSV</label>
                <label className="flex items-center gap-3 px-4 py-3 bg-dark-700 border border-dark-600 rounded-xl cursor-pointer hover:border-accent-500/50 transition-colors">
                  <Upload className="w-4 h-4 text-gray-400" />
                  <span className="text-sm text-gray-400">{csvFile ? csvFile.name : 'Browse CSV…'}</span>
                  <input type="file" accept=".csv" className="hidden"
                    onChange={e => { setCsvFile(e.target.files[0]); e.target.value = '' }} />
                </label>
              </div>
            </div>
          )}

          {/* Predict button */}
          <button
            onClick={handlePredict}
            disabled={loading || !canPredict}
            className="w-full flex items-center justify-center gap-2 py-3 bg-accent-500 hover:bg-accent-400 disabled:opacity-40 text-white rounded-2xl text-sm font-semibold transition-colors shadow-lg shadow-accent-500/20"
          >
            {loading
              ? <><Loader2 className="w-4 h-4 animate-spin" /> Predicting…</>
              : <><Zap className="w-4 h-4" /> Predict</>
            }
          </button>

          {error && (
            <div className="bg-red-900/30 border border-red-700 rounded-xl p-3 text-sm text-red-300 flex items-start justify-between gap-3">
              <span>{error}</span>
              <button onClick={() => setError(null)} className="text-red-400 shrink-0">×</button>
            </div>
          )}
        </div>

        {/* RIGHT: Results */}
        <div className="min-h-96">
          {result ? (
            <PredictionResult result={result} />
          ) : (
            <div className="h-full min-h-48 flex flex-col items-center justify-center text-center rounded-2xl border-2 border-dashed border-dark-600 p-8">
              <BarChart2 className="w-10 h-10 text-gray-700 mb-3" />
              <p className="text-sm text-gray-600">
                {mode === 'image' ? 'Drop an image and click Predict' : 'Enter data and click Predict'}
              </p>
            </div>
          )}
        </div>
      </div>

      {/* ── LLM Chat ─────────────────────────────────────────────────────── */}
      <ChatPanel sessionId={activeSession} />
    </div>
  )
}
