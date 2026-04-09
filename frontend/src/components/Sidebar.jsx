import React from 'react'
import { NavLink } from 'react-router-dom'
import {
  LayoutDashboard, Database, Cpu, BarChart2,
  Zap, MessageSquare, Activity, Rocket
} from 'lucide-react'

const links = [
  { to: '/quickstart', label: '⚡ Quick Start', Icon: Rocket },
  { to: '/', label: 'Dashboard', Icon: LayoutDashboard },
  { to: '/dataset', label: 'Dataset', Icon: Database },
  { to: '/training', label: 'Training', Icon: Cpu },
  { to: '/results', label: 'Results', Icon: BarChart2 },
  { to: '/inference', label: 'Inference', Icon: Zap },
]

export default function Sidebar({ llmOnline }) {
  return (
    <aside className="w-56 min-h-screen bg-dark-800 border-r border-dark-700 flex flex-col py-6 px-3">
      {/* Logo */}
      <div className="flex items-center gap-2 px-3 mb-8">
        <Activity className="text-accent-400 w-6 h-6" />
        <span className="font-bold text-lg tracking-tight text-white">AutoVision</span>
      </div>

      {/* Nav links */}
      <nav className="flex flex-col gap-1 flex-1">
        {links.map(({ to, label, Icon }) => (
          <NavLink
            key={to}
            to={to}
            end={to === '/'}
            className={({ isActive }) =>
              `flex items-center gap-3 px-3 py-2 rounded-lg text-sm font-medium transition-colors
              ${isActive
                ? 'bg-accent-500 text-white'
                : 'text-gray-400 hover:bg-dark-700 hover:text-white'}`
            }
          >
            <Icon className="w-4 h-4 shrink-0" />
            {label}
          </NavLink>
        ))}
      </nav>

      {/* LLM status */}
      <div className="px-3 pt-4 border-t border-dark-700">
        <div className="flex items-center gap-2 text-xs text-gray-500">
          <span className={`w-2 h-2 rounded-full ${llmOnline ? 'bg-green-400' : 'bg-yellow-400'}`} />
          <span>{llmOnline ? 'Ollama online' : 'Fallback mode'}</span>
        </div>
      </div>
    </aside>
  )
}
