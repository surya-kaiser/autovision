import React from 'react'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  Legend, ResponsiveContainer
} from 'recharts'

export default function MetricsChart({ data = [], keys = ['loss', 'accuracy'] }) {
  if (!data.length) {
    return (
      <div className="h-48 flex items-center justify-center text-gray-600 text-sm">
        No metrics to display yet
      </div>
    )
  }

  const colors = ['#818cf8', '#34d399', '#fb923c', '#f472b6']

  return (
    <ResponsiveContainer width="100%" height={240}>
      <LineChart data={data} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#2e3346" />
        <XAxis
          dataKey="epoch"
          stroke="#4b5563"
          tick={{ fill: '#9ca3af', fontSize: 11 }}
          label={{ value: 'Epoch', position: 'insideBottom', offset: -2, fill: '#6b7280', fontSize: 11 }}
        />
        <YAxis stroke="#4b5563" tick={{ fill: '#9ca3af', fontSize: 11 }} />
        <Tooltip
          contentStyle={{ background: '#1a1d27', border: '1px solid #2e3346', borderRadius: 8 }}
          labelStyle={{ color: '#e5e7eb' }}
        />
        <Legend wrapperStyle={{ paddingTop: 10, fontSize: 12 }} />
        {keys.map((k, i) => (
          <Line
            key={k}
            type="monotone"
            dataKey={k}
            stroke={colors[i % colors.length]}
            strokeWidth={2}
            dot={false}
            activeDot={{ r: 4 }}
          />
        ))}
      </LineChart>
    </ResponsiveContainer>
  )
}
