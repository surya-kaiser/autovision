import React from 'react'
import { ToastProvider } from './components/Toast'
import MainPage from './pages/MainPage'

export default function App() {
  return (
    <ToastProvider>
      <MainPage />
    </ToastProvider>
  )
}
