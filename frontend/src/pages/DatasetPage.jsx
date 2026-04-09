import React, { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import FileUploader from '../components/FileUploader'
import PreprocessPanel from '../components/PreprocessPanel'
import { uploadDataset, previewDataset, preprocessDataset, getRecommendation, autoTrain, deleteDataset } from '../api/client'
import { useToast } from '../components/Toast'
import api from '../api/client'
import {
  Loader2, Sparkles, FolderOpen, File, CheckCircle, Database,
  PlayCircle, ChevronRight, X, Minus, ChevronDown, Image,
  FileText, Archive, MoreVertical, Clock, Layers, Trash2, Plus
} from 'lucide-react'

// ── Helpers ───────────────────────────────────────────────────────────────────

function fmtSize(bytes) {
  if (!bytes) return ''
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / 1024 / 1024).toFixed(1)} MB`
  return `${(bytes / 1024 / 1024 / 1024).toFixed(2)} GB`
}

function taskBadgeColor(type) {
  const map = {
    classification: 'bg-blue-500/20 text-blue-400',
    regression: 'bg-green-500/20 text-green-400',
    detection: 'bg-orange-500/20 text-orange-400',
    segmentation: 'bg-purple-500/20 text-purple-400',
    unknown: 'bg-gray-500/20 text-gray-400',
  }
  return map[type] || 'bg-gray-500/20 text-gray-400'
}

function datasetIcon(taskType) {
  if (taskType === 'classification' || taskType === 'detection' || taskType === 'segmentation')
    return <Image className="w-6 h-6 text-blue-400" />
  if (taskType === 'regression')
    return <FileText className="w-6 h-6 text-green-400" />
  return <Database className="w-6 h-6 text-accent-400" />
}

// ── Floating upload progress panel (Google Drive style) ────────────────────

function UploadPanel({ files, onClose }) {
  const [minimized, setMinimized] = useState(false)
  if (!files.length) return null

  const done = files.every(f => f.status === 'done' || f.status === 'error')
  const allDone = files.every(f => f.status === 'done')
  const anyError = files.some(f => f.status === 'error')

  const headerText = done
    ? allDone ? `${files.length} upload${files.length > 1 ? 's' : ''} complete`
    : `${files.filter(f => f.status === 'error').length} upload(s) failed`
    : `Uploading ${files.length} item${files.length > 1 ? 's' : ''}…`

  return (
    <div className="fixed bottom-4 right-4 w-80 bg-dark-800 border border-dark-600 rounded-2xl shadow-2xl z-50 overflow-hidden">
      {/* Header */}
      <div className={`flex items-center justify-between px-4 py-3 border-b border-dark-600
        ${allDone ? 'bg-green-900/20' : anyError ? 'bg-red-900/20' : 'bg-dark-700'}`}>
        <div className="flex items-center gap-2">
          {done
            ? allDone
              ? <CheckCircle className="w-4 h-4 text-green-400" />
              : <X className="w-4 h-4 text-red-400" />
            : <Loader2 className="w-4 h-4 text-accent-400 animate-spin" />
          }
          <span className="text-sm font-medium text-white">{headerText}</span>
        </div>
        <div className="flex items-center gap-1">
          <button
            onClick={() => setMinimized(m => !m)}
            className="p-1 text-gray-400 hover:text-white rounded transition-colors"
          >
            {minimized ? <ChevronDown className="w-4 h-4" /> : <Minus className="w-4 h-4" />}
          </button>
          {done && (
            <button
              onClick={onClose}
              className="p-1 text-gray-400 hover:text-white rounded transition-colors"
            >
              <X className="w-4 h-4" />
            </button>
          )}
        </div>
      </div>

      {/* File list */}
      {!minimized && (
        <div className="max-h-64 overflow-y-auto divide-y divide-dark-600/50">
          {files.map((f) => (
            <div key={f.id} className="flex items-start gap-3 px-4 py-3">
              <div className="shrink-0 mt-0.5">{f.icon}</div>
              <div className="flex-1 min-w-0">
                <div className="flex items-center justify-between gap-2">
                  <span className="text-xs text-white font-medium truncate">{f.name}</span>
                  <span className="text-xs text-gray-500 shrink-0">{fmtSize(f.size)}</span>
                </div>
                {f.status === 'uploading' && (
                  <>
                    <div className="text-xs text-gray-500 mt-0.5">{f.label || `${f.pct}%`}</div>
                    <div className="mt-1.5 h-1 bg-dark-600 rounded-full overflow-hidden">
                      <div
                        className="h-1 bg-accent-400 rounded-full transition-all duration-300"
                        style={{ width: `${f.pct}%` }}
                      />
                    </div>
                  </>
                )}
                {f.status === 'done' && (
                  <div className="text-xs text-green-400 mt-0.5 flex items-center gap-1">
                    <CheckCircle className="w-3 h-3" /> Upload complete
                  </div>
                )}
                {f.status === 'error' && (
                  <div className="text-xs text-red-400 mt-0.5">{f.error || 'Upload failed'}</div>
                )}
                {f.status === 'processing' && (
                  <div className="text-xs text-accent-400 mt-0.5 flex items-center gap-1">
                    <Loader2 className="w-3 h-3 animate-spin" /> {f.label || 'Processing…'}
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

// ── Dataset card (Google Drive style) ─────────────────────────────────────

function DatasetCard({ session, isActive, onSelect, onDelete }) {
  const [confirmDelete, setConfirmDelete] = useState(false)

  const handleDeleteClick = (e) => {
    e.stopPropagation()
    setConfirmDelete(true)
  }

  const handleConfirm = (e) => {
    e.stopPropagation()
    onDelete(session.session_id)
    setConfirmDelete(false)
  }

  const handleCancel = (e) => {
    e.stopPropagation()
    setConfirmDelete(false)
  }

  return (
    <div
      className={`relative group w-full text-left rounded-2xl border transition-all duration-150 overflow-hidden cursor-pointer
        ${isActive
          ? 'border-accent-500/60 bg-accent-500/10 ring-2 ring-accent-500/30'
          : 'border-dark-600 bg-dark-800 hover:border-dark-500 hover:bg-dark-700'}`}
      onClick={() => onSelect(session.session_id)}
    >
      {/* Top colour strip */}
      <div className={`h-1.5 w-full ${isActive ? 'bg-accent-500' : 'bg-dark-600 group-hover:bg-dark-500'}`} />

      <div className="p-4">
        {/* Icon + name + delete button */}
        <div className="flex items-start gap-3 mb-3">
          <div className={`w-10 h-10 rounded-xl flex items-center justify-center shrink-0
            ${isActive ? 'bg-accent-500/20' : 'bg-dark-700'}`}>
            {datasetIcon(session.task_type)}
          </div>
          <div className="flex-1 min-w-0">
            <div className="text-sm font-semibold text-white truncate">
              {session.dataset && session.dataset !== 'unknown'
                ? session.dataset
                : `Dataset ${session.session_id.slice(0, 6)}`}
            </div>
            <div className="text-xs text-gray-500 mt-0.5">
              {session.task_type && session.task_type !== 'unknown'
                ? session.task_type
                : `${session.num_samples || 0} samples`}
            </div>
          </div>
          {isActive
            ? <CheckCircle className="w-4 h-4 text-accent-400 shrink-0 mt-0.5" />
            : (
              <button
                onClick={handleDeleteClick}
                className="opacity-0 group-hover:opacity-100 p-1 rounded-lg hover:bg-red-500/20 text-gray-600 hover:text-red-400 transition-all"
                title="Delete dataset"
              >
                <Trash2 className="w-3.5 h-3.5" />
              </button>
            )
          }
        </div>

        {/* Confirm delete overlay */}
        {confirmDelete && (
          <div
            className="absolute inset-0 bg-dark-900/90 backdrop-blur-sm flex flex-col items-center justify-center gap-3 p-4 z-10 rounded-2xl"
            onClick={e => e.stopPropagation()}
          >
            <p className="text-sm text-white font-semibold text-center">Delete this dataset?</p>
            <p className="text-xs text-gray-400 text-center">This cannot be undone</p>
            <div className="flex gap-2 w-full">
              <button
                onClick={handleCancel}
                className="flex-1 py-2 bg-dark-700 hover:bg-dark-600 text-gray-300 rounded-xl text-xs font-medium transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={handleConfirm}
                className="flex-1 py-2 bg-red-600 hover:bg-red-500 text-white rounded-xl text-xs font-semibold transition-colors"
              >
                Delete
              </button>
            </div>
          </div>
        )}

        {/* Meta */}
        <div className="flex flex-wrap gap-1.5 mb-3">
          {session.task_type && (
            <span className={`px-2 py-0.5 rounded-full text-xs font-medium ${taskBadgeColor(session.task_type)}`}>
              {session.task_type.replace('_', ' ')}
            </span>
          )}
          {session.num_training_runs > 0 && (
            <span className="px-2 py-0.5 rounded-full text-xs bg-dark-600 text-gray-400">
              {session.num_training_runs} run{session.num_training_runs !== 1 ? 's' : ''}
            </span>
          )}
        </div>

        {/* Footer */}
        <div className="flex items-center justify-between text-xs text-gray-600">
          <span className="flex items-center gap-1">
            <Layers className="w-3 h-3" />
            {session.num_samples ? session.num_samples.toLocaleString() + ' samples' : 'No preprocess yet'}
          </span>
          <ChevronRight className="w-3.5 h-3.5 text-gray-600 group-hover:text-gray-400 transition-colors" />
        </div>
      </div>
    </div>
  )
}

// ── Main page ─────────────────────────────────────────────────────────────────

export default function DatasetPage({ sessionId, setSessionId, setDatasetInfo, setPendingRecommendation }) {
  const toast = useToast()
  const navigate = useNavigate()

  // Upload panel state
  const [uploadFiles, setUploadFiles] = useState([])
  const panelOpen = uploadFiles.length > 0

  const [preprocessing, setPreprocessing] = useState(false)
  const [preview, setPreview] = useState(null)
  const [report, setReport] = useState(null)
  const [recommendation, setRecommendation] = useState(null)
  const [error, setError] = useState(null)
  const [taskType, setTaskType] = useState('')
  const [showUploader, setShowUploader] = useState(false)

  const [savedSessions, setSavedSessions] = useState([])

  useEffect(() => { loadSavedSessions() }, [])

  async function loadSavedSessions() {
    try {
      const r = await api.get('/dataset/sessions')
      setSavedSessions(r.data.data || [])
    } catch {}
  }

  async function handleDeleteSession(sid) {
    try {
      await deleteDataset(sid)
      setSavedSessions(prev => prev.filter(s => s.session_id !== sid))
      if (sessionId === sid) {
        setSessionId(null)
        setPreview(null)
        setReport(null)
        setRecommendation(null)
      }
      toast('Dataset deleted', 'success')
    } catch (e) {
      toast(e.response?.data?.detail || 'Delete failed', 'error')
    }
  }

  // ── Panel helpers ─────────────────────────────────────────────────────────

  function addPanelFile(id, name, size, icon) {
    setUploadFiles(prev => [...prev, { id, name, size, icon, pct: 0, status: 'uploading', label: '' }])
  }

  function updatePanelFile(id, patch) {
    setUploadFiles(prev => prev.map(f => f.id === id ? { ...f, ...patch } : f))
  }

  // ── Select saved session ──────────────────────────────────────────────────

  async function selectSavedSession(sid) {
    setError(null)
    setPreview(null)
    setReport(null)
    setRecommendation(null)
    setSessionId(sid)
    setShowUploader(false)
    try {
      const prev = await previewDataset(sid)
      setPreview(prev.data.data)
      if (setDatasetInfo) setDatasetInfo({ session_id: sid })
      toast('Dataset loaded', 'success')
    } catch (e) {
      setError('Could not load session: ' + (e.response?.data?.detail || e.message))
    }
  }

  // ── File upload ───────────────────────────────────────────────────────────

  async function handleFile(file) {
    const id = crypto.randomUUID()
    const icon = getFileIcon(file.name)
    addPanelFile(id, file.name, file.size, icon)
    setError(null)
    setPreview(null)
    setReport(null)
    setRecommendation(null)
    setShowUploader(false)

    try {
      const fd = new FormData()
      fd.append('file', file)
      if (sessionId) fd.append('session_id', sessionId)

      const res = await api.post('/dataset/upload', fd, {
        onUploadProgress: (e) => {
          const pct = e.total ? Math.round((e.loaded / e.total) * 100) : 0
          updatePanelFile(id, { pct, label: `${pct}%` })
        },
      })

      const { session_id, task_type } = res.data.data
      setSessionId(session_id)
      setTaskType(task_type)
      if (setDatasetInfo) setDatasetInfo(res.data.data)

      updatePanelFile(id, { pct: 100, status: 'processing', label: 'Loading preview…' })
      const prev = await previewDataset(session_id)
      setPreview(prev.data.data)
      updatePanelFile(id, { status: 'done' })
      toast(`"${file.name}" uploaded`, 'success')
      loadSavedSessions()
    } catch (e) {
      updatePanelFile(id, { status: 'error', error: e.response?.data?.detail || e.message })
      setError(e.response?.data?.detail || e.message)
      toast('Upload failed', 'error')
    }
  }

  // ── Folder upload ─────────────────────────────────────────────────────────

  async function handleFolder(files) {
    if (!files.length) return
    const folderName = files[0].webkitRelativePath?.split('/')[0] || 'dataset'
    const totalSize = files.reduce((s, f) => s + f.size, 0)
    const sid = crypto.randomUUID()
    const id = crypto.randomUUID()

    addPanelFile(id, folderName, totalSize, <FolderOpen className="w-5 h-5 text-accent-400" />)
    setError(null)
    setPreview(null)
    setReport(null)
    setRecommendation(null)
    setSessionId(sid)
    setShowUploader(false)

    try {
      const BATCH = 50
      let uploaded = 0

      for (let i = 0; i < files.length; i += BATCH) {
        const batch = files.slice(i, i + BATCH)
        const fd = new FormData()
        fd.append('session_id', sid)
        for (const f of batch) {
          const relPath = f.webkitRelativePath || f.name
          fd.append('files', new File([f], relPath, { type: f.type }))
        }
        const batchStart = uploaded
        await api.post('/dataset/upload-folder', fd, {
          onUploadProgress: (e) => {
            const batchPct = e.total ? e.loaded / e.total : 1
            const totalDone = batchStart + batch.length * batchPct
            const pct = Math.min(Math.round((totalDone / files.length) * 100), 99)
            updatePanelFile(id, {
              pct,
              label: `${Math.round(totalDone).toLocaleString()} / ${files.length.toLocaleString()} files`,
            })
          },
        })
        uploaded += batch.length
        updatePanelFile(id, {
          pct: Math.round((uploaded / files.length) * 100),
          label: `${uploaded.toLocaleString()} / ${files.length.toLocaleString()} files`,
        })
      }

      updatePanelFile(id, { pct: 99, status: 'processing', label: 'Analysing structure…' })
      const detectRes = await api.post('/dataset/detect', { session_id: sid })
      const { task_type } = detectRes.data.data
      setTaskType(task_type)
      if (setDatasetInfo) setDatasetInfo({ session_id: sid, ...detectRes.data.data })

      updatePanelFile(id, { status: 'processing', label: 'Loading preview…' })
      const prev = await previewDataset(sid)
      setPreview(prev.data.data)
      updatePanelFile(id, { pct: 100, status: 'done' })
      toast(`"${folderName}" uploaded (${files.length} files)`, 'success')
      loadSavedSessions()
    } catch (e) {
      updatePanelFile(id, { status: 'error', error: e.response?.data?.detail || e.message })
      setError(e.response?.data?.detail || e.message || 'Upload failed')
      toast('Upload failed', 'error')
    }
  }

  function getFileIcon(name) {
    const ext = (name || '').split('.').pop().toLowerCase()
    if (['jpg','jpeg','png','gif','bmp','webp','tiff'].includes(ext))
      return <Image className="w-5 h-5 text-blue-400" />
    if (['zip','tar','gz','rar','7z'].includes(ext))
      return <Archive className="w-5 h-5 text-yellow-400" />
    if (['csv','tsv','json','xlsx','txt'].includes(ext))
      return <FileText className="w-5 h-5 text-green-400" />
    return <File className="w-5 h-5 text-gray-400" />
  }

  // ── Preprocess / recommend / auto-train ───────────────────────────────────

  async function handlePreprocess() {
    if (!sessionId) return
    setPreprocessing(true)
    setError(null)
    try {
      const config = {
        session_id: sessionId,
        scale_method: 'standard',
        handle_missing: 'auto',
        augmentation: true,
        train_ratio: 0.70,
        val_ratio: 0.15,
        test_ratio: 0.15,
      }
      const res = await preprocessDataset(config)
      setReport(res.data.data.report)
      toast('Preprocessing complete', 'success')
      loadSavedSessions()
    } catch (e) {
      setError(e.response?.data?.detail || e.message)
      toast('Preprocessing failed', 'error')
    } finally {
      setPreprocessing(false)
    }
  }

  async function handleRecommend() {
    if (!sessionId) return
    setError(null)
    try {
      const res = await getRecommendation(sessionId, taskType)
      setRecommendation(res.data.data)
      toast('Recommendation ready', 'info')
    } catch (e) {
      setError(e.response?.data?.detail || e.message)
    }
  }

  async function handleAutoTrain() {
    if (!sessionId) return
    try {
      const res = await autoTrain(sessionId)
      const modelName = res.data.data?.model_type || 'model'
      if (setPendingRecommendation) setPendingRecommendation(res.data.data?.recommendation || null)
      toast(`Auto-training started with ${modelName}`, 'success', 5000)
      navigate('/training')
    } catch (e) {
      toast(e.response?.data?.detail || 'Auto-train failed', 'error')
    }
  }

  const activeUpload = uploadFiles.some(f => f.status === 'uploading' || f.status === 'processing')

  return (
    <div className="p-6 max-w-5xl mx-auto space-y-6">

      {/* ── Page header ──────────────────────────────────────────────────── */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">Datasets</h1>
          <p className="text-sm text-gray-500 mt-0.5">Upload and manage your training datasets</p>
        </div>
        <button
          onClick={() => setShowUploader(u => !u)}
          disabled={activeUpload}
          className="flex items-center gap-2 px-4 py-2 bg-accent-500 hover:bg-accent-400 disabled:opacity-50 text-white rounded-xl text-sm font-semibold transition-colors shadow-lg shadow-accent-500/20"
        >
          <Plus className="w-4 h-4" />
          New Upload
        </button>
      </div>

      {/* ── Upload panel (collapsible) ────────────────────────────────────── */}
      {showUploader && (
        <div className="bg-dark-800 border border-dark-700 rounded-2xl p-5">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-sm font-semibold text-gray-300">Upload Dataset</h2>
            <button onClick={() => setShowUploader(false)} className="text-gray-600 hover:text-gray-400">
              <X className="w-4 h-4" />
            </button>
          </div>
          <FileUploader onFile={handleFile} onFolder={handleFolder} />
          <p className="mt-3 text-xs text-gray-600">
            Supported: CSV, ZIP archive, image folders (classification/detection), JSON, Excel
          </p>
        </div>
      )}

      {/* ── Saved datasets grid ───────────────────────────────────────────── */}
      {savedSessions.length > 0 ? (
        <div>
          <h2 className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-3">
            My Datasets ({savedSessions.length})
          </h2>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
            {savedSessions.map(s => (
              <DatasetCard
                key={s.session_id}
                session={s}
                isActive={sessionId === s.session_id}
                onSelect={selectSavedSession}
                onDelete={handleDeleteSession}
              />
            ))}
          </div>
        </div>
      ) : (
        !showUploader && (
          <div
            onClick={() => setShowUploader(true)}
            className="flex flex-col items-center justify-center py-20 border-2 border-dashed border-dark-600 rounded-2xl cursor-pointer hover:border-accent-500/50 hover:bg-dark-800/50 transition-all group"
          >
            <div className="w-16 h-16 rounded-2xl bg-dark-800 border border-dark-700 flex items-center justify-center mb-4 group-hover:border-accent-500/30 transition-colors">
              <Database className="w-7 h-7 text-gray-600 group-hover:text-accent-400 transition-colors" />
            </div>
            <p className="text-sm font-semibold text-gray-400 group-hover:text-white transition-colors">No datasets yet</p>
            <p className="text-xs text-gray-600 mt-1">Click to upload your first dataset</p>
          </div>
        )
      )}

      {/* ── Active session details ────────────────────────────────────────── */}
      {sessionId && !showUploader && (
        <div className="space-y-5">
          {/* Preview */}
          {preview && (
            <div className="bg-dark-800 border border-dark-700 rounded-2xl p-5">
              {preview.type === 'csv' ? (
                <>
                  <h2 className="text-sm font-semibold text-gray-300 mb-3">
                    Preview — {preview.rows?.toLocaleString()} rows × {preview.columns?.length} columns
                  </h2>
                  <div className="overflow-x-auto">
                    <table className="min-w-full text-xs text-gray-300">
                      <thead>
                        <tr className="border-b border-dark-700">
                          {preview.columns?.map(c => (
                            <th key={c} className="text-left py-1.5 px-2 text-gray-500 font-medium">{c}</th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {preview.sample?.slice(0, 8).map((row, i) => (
                          <tr key={i} className="border-b border-dark-700/50 hover:bg-dark-700/30">
                            {preview.columns?.map(c => (
                              <td key={c} className="py-1.5 px-2 truncate max-w-24">{String(row[c] ?? '')}</td>
                            ))}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </>
              ) : (
                <>
                  <h2 className="text-sm font-semibold text-gray-300 mb-3 flex items-center gap-2">
                    <CheckCircle className="w-4 h-4 text-green-400" />
                    {preview.count?.toLocaleString()} images
                    {preview.classes?.length > 0 && ` · ${preview.classes.length} classes`}
                  </h2>
                  {preview.class_counts && (
                    <div className="grid grid-cols-2 sm:grid-cols-3 gap-2 mb-4">
                      {Object.entries(preview.class_counts).map(([cls, count]) => (
                        <div key={cls} className="bg-dark-700 rounded-xl px-3 py-2 flex items-center justify-between">
                          <div className="flex items-center gap-2">
                            <FolderOpen className="w-3.5 h-3.5 text-accent-400" />
                            <span className="text-sm text-white font-medium">{cls}</span>
                          </div>
                          <span className="text-xs text-gray-400">{count.toLocaleString()}</span>
                        </div>
                      ))}
                    </div>
                  )}
                  <div className="flex flex-wrap gap-1.5">
                    {preview.sample_names?.slice(0, 15).map(n => (
                      <span key={n} className="flex items-center gap-1 px-2 py-0.5 bg-dark-700 rounded text-xs text-gray-400">
                        <File className="w-2.5 h-2.5 shrink-0" />{n}
                      </span>
                    ))}
                  </div>
                </>
              )}
            </div>
          )}

          {/* Actions */}
          {preview && (
            <div className="flex gap-3 flex-wrap">
              <button
                onClick={handlePreprocess}
                disabled={preprocessing}
                className="flex items-center gap-2 px-4 py-2 bg-accent-500 hover:bg-accent-400 disabled:opacity-50 text-white rounded-xl text-sm font-medium transition-colors"
              >
                {preprocessing ? <Loader2 className="w-4 h-4 animate-spin" /> : null}
                {preprocessing ? 'Preprocessing…' : 'Auto Preprocess'}
              </button>
              <button
                onClick={handleRecommend}
                className="flex items-center gap-2 px-4 py-2 bg-dark-700 hover:bg-dark-600 border border-dark-600 text-white rounded-xl text-sm font-medium transition-colors"
              >
                <Sparkles className="w-4 h-4 text-accent-400" />
                Get LLM Recommendation
              </button>
            </div>
          )}

          {report && (
            <div className="bg-dark-800 border border-dark-700 rounded-2xl p-5">
              <PreprocessPanel report={report} />
            </div>
          )}

          {recommendation && (
            <div className="bg-dark-800 border border-accent-500/30 rounded-2xl p-5 space-y-4">
              <div className="flex items-center gap-2">
                <Sparkles className="w-4 h-4 text-accent-400" />
                <h2 className="text-sm font-semibold text-accent-400">LLM Recommendation</h2>
              </div>
              <p className="text-sm text-gray-300">{recommendation.explanation}</p>
              <div className="grid grid-cols-2 gap-3 text-xs">
                <div><span className="text-gray-500">Model: </span><span className="text-white font-medium">{recommendation.model_type}</span></div>
                <div><span className="text-gray-500">Est. training: </span><span className="text-white font-medium">{recommendation.estimated_training_minutes} min</span></div>
                <div className="col-span-2"><span className="text-gray-500">Strategy: </span><span className="text-white">{recommendation.preprocessing_strategy}</span></div>
              </div>
              {recommendation.hyperparams && (
                <div className="bg-dark-900 rounded-xl p-3 font-mono text-xs text-gray-300">
                  {JSON.stringify(recommendation.hyperparams, null, 2)}
                </div>
              )}
              <button
                onClick={handleAutoTrain}
                className="flex items-center gap-2 px-4 py-2.5 bg-green-600 hover:bg-green-500 text-white rounded-xl text-sm font-medium transition-colors w-full justify-center"
              >
                <PlayCircle className="w-4 h-4" />
                Auto Train Recommended Model
              </button>
            </div>
          )}
        </div>
      )}

      {error && (
        <div className="bg-red-900/30 border border-red-700 rounded-2xl p-4 text-sm text-red-300 flex items-start justify-between gap-3">
          <span>{error}</span>
          <button onClick={() => setError(null)} className="text-red-400 hover:text-red-200 shrink-0 text-lg leading-none">×</button>
        </div>
      )}

      {/* ── Floating upload progress panel ───────────────────────────────── */}
      <UploadPanel
        files={uploadFiles}
        onClose={() => setUploadFiles([])}
      />
    </div>
  )
}
