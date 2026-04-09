import axios from 'axios'

const BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000/api/v1'

const api = axios.create({
  baseURL: BASE_URL,
  timeout: 60000,
})

// Dataset
export const uploadDataset = (file, sessionId) => {
  const fd = new FormData()
  fd.append('file', file)
  if (sessionId) fd.append('session_id', sessionId)
  return api.post('/dataset/upload', fd)
}

export const previewDataset = (sessionId) =>
  api.get(`/dataset/preview/${sessionId}`)

export const preprocessDataset = (config) =>
  api.post('/dataset/preprocess', config)

export const getRecommendation = (sessionId, taskType) =>
  api.get(`/dataset/recommend/${sessionId}`, { params: { task_type: taskType } })

export const deleteDataset = (sessionId) =>
  api.delete(`/dataset/${sessionId}`)

export const listSessions = () =>
  api.get('/dataset/sessions')

export const getSessionSummary = (sessionId) =>
  api.get(`/dataset/session/${sessionId}/summary`)

export const getTrainingLog = (sessionId, modelType) =>
  api.get(`/training/log/${sessionId}/${modelType}`)

// Training
export const startTraining = (config) =>
  api.post('/training/start', config)

export const stopTraining = (sessionId) =>
  api.post(`/training/stop/${sessionId}`)

export const autoTrain = (sessionId, taskTypeOverride) =>
  api.post('/training/auto-start', null, {
    params: {
      session_id: sessionId,
      ...(taskTypeOverride ? { task_type: taskTypeOverride } : {}),
    },
  })

export const getTrainingStatus = (sessionId) =>
  api.get(`/training/status/${sessionId}`)

export const getTrainingResults = (sessionId) =>
  api.get(`/training/results/${sessionId}`)

export const compareModels = (sessionId) =>
  api.get(`/training/compare/${sessionId}`)

// Pilot
export const startPilot = (config) =>
  api.post('/pilot/run', config)

// Inference
export const predict = (formData) =>
  api.post('/inference/predict', formData)

export const listModels = (sessionId) =>
  api.get(`/inference/models/${sessionId}`)

export const chatWithLLM = (sessionId, message, history = []) =>
  api.get('/inference/chat', {
    params: {
      session_id: sessionId,
      message,
      history_json: JSON.stringify(history),
    },
  })

// System
export const getSystemInfo = () => api.get('/system/info')
export const getLLMStatus = () => api.get('/llm/status')

// WebSocket helpers — use window.location so the connection routes through
// whatever proxy is in front (Vite dev proxy on :3000, nginx in Docker, etc.)
function _wsBase() {
  const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
  return `${proto}//${window.location.host}`
}

export const createTrainingWS = (sessionId) =>
  new WebSocket(`${_wsBase()}/api/v1/training/ws/${sessionId}`)

export const createPilotWS = (sessionId) =>
  new WebSocket(`${_wsBase()}/api/v1/pilot/ws/${sessionId}`)

export default api
