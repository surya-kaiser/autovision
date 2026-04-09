import React, { createContext, useContext, useState, useCallback } from 'react'
import { CheckCircle, XCircle, AlertCircle, X } from 'lucide-react'

const ToastContext = createContext(null)

export function ToastProvider({ children }) {
  const [toasts, setToasts] = useState([])

  const addToast = useCallback((message, type = 'info', duration = 3500) => {
    const id = Date.now() + Math.random()
    setToasts(prev => [...prev, { id, message, type }])
    setTimeout(() => setToasts(prev => prev.filter(t => t.id !== id)), duration)
  }, [])

  const remove = useCallback((id) => setToasts(prev => prev.filter(t => t.id !== id)), [])

  return (
    <ToastContext.Provider value={addToast}>
      {children}
      <div className="fixed bottom-5 right-5 space-y-2 z-50 pointer-events-none">
        {toasts.map(t => (
          <ToastItem key={t.id} toast={t} onRemove={remove} />
        ))}
      </div>
    </ToastContext.Provider>
  )
}

function ToastItem({ toast, onRemove }) {
  const cfg = {
    success: { icon: <CheckCircle className="w-4 h-4 text-green-400 shrink-0" />, border: 'border-green-700' },
    error:   { icon: <XCircle   className="w-4 h-4 text-red-400 shrink-0" />,   border: 'border-red-700' },
    info:    { icon: <AlertCircle className="w-4 h-4 text-accent-400 shrink-0" />, border: 'border-accent-500/40' },
  }
  const { icon, border } = cfg[toast.type] || cfg.info
  return (
    <div className={`pointer-events-auto flex items-center gap-3 bg-dark-800 border ${border} rounded-xl px-4 py-3 shadow-xl min-w-64 max-w-sm`}>
      {icon}
      <span className="text-sm text-white flex-1">{toast.message}</span>
      <button onClick={() => onRemove(toast.id)} className="text-gray-500 hover:text-gray-300 ml-1">
        <X className="w-3.5 h-3.5" />
      </button>
    </div>
  )
}

export function useToast() {
  const ctx = useContext(ToastContext)
  if (!ctx) throw new Error('useToast must be used inside ToastProvider')
  return ctx
}
