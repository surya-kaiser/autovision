import React from 'react'
import { CheckCircle, AlertTriangle, Info } from 'lucide-react'

export default function PreprocessPanel({ report }) {
  if (!report) return null

  return (
    <div className="space-y-3">
      <h3 className="text-sm font-semibold text-gray-300">Preprocessing Report</h3>

      {/* Steps applied */}
      <div className="space-y-1">
        {report.steps_applied?.map((step, i) => (
          <div key={i} className="flex items-start gap-2 text-xs text-gray-300">
            <CheckCircle className="w-3.5 h-3.5 text-green-400 shrink-0 mt-0.5" />
            {step}
          </div>
        ))}
      </div>

      {/* Stats */}
      {(report.train_size || report.val_size || report.test_size) ? (
        <div className="grid grid-cols-3 gap-2 mt-2">
          {[
            { label: 'Train', value: report.train_size },
            { label: 'Val', value: report.val_size },
            { label: 'Test', value: report.test_size },
          ].map(({ label, value }) => (
            <div key={label} className="bg-dark-700 rounded-lg p-2 text-center">
              <div className="text-lg font-bold text-accent-400">{value ?? '—'}</div>
              <div className="text-xs text-gray-500">{label}</div>
            </div>
          ))}
        </div>
      ) : null}

      {/* Warnings */}
      {report.warnings?.length > 0 && (
        <div className="space-y-1 mt-2">
          {report.warnings.map((w, i) => (
            <div key={i} className="flex items-start gap-2 text-xs text-yellow-400">
              <AlertTriangle className="w-3.5 h-3.5 shrink-0 mt-0.5" />
              {w}
            </div>
          ))}
        </div>
      )}

      {/* Augmentations */}
      {report.augmentations?.length > 0 && (
        <div className="flex flex-wrap gap-1 mt-1">
          {report.augmentations.map((a) => (
            <span key={a} className="px-2 py-0.5 bg-accent-500/20 text-accent-400 rounded text-xs">
              {a}
            </span>
          ))}
        </div>
      )}
    </div>
  )
}
