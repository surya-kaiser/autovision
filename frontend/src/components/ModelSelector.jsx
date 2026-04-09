import React from 'react'

const MODELS = {
  classification: [
    { value: 'random_forest', label: 'Random Forest' },
    { value: 'xgboost', label: 'XGBoost' },
    { value: 'lightgbm', label: 'LightGBM' },
    { value: 'cnn', label: 'CNN' },
    { value: 'resnet', label: 'ResNet (transfer)' },
  ],
  object_detection: [
    { value: 'yolov8n', label: 'YOLOv8n (nano)' },
    { value: 'yolov8s', label: 'YOLOv8s (small)' },
    { value: 'yolov8m', label: 'YOLOv8m (medium)' },
  ],
  regression: [
    { value: 'linear_regression', label: 'Linear Regression' },
    { value: 'ridge', label: 'Ridge' },
    { value: 'xgboost', label: 'XGBoost' },
    { value: 'lightgbm', label: 'LightGBM' },
  ],
}

export default function ModelSelector({ taskType = 'classification', value, onChange }) {
  const options = MODELS[taskType] || MODELS.classification

  return (
    <div>
      <label className="block text-xs font-medium text-gray-400 mb-1">Model Architecture</label>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="w-full bg-dark-700 border border-dark-600 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-accent-500"
      >
        {options.map((o) => (
          <option key={o.value} value={o.value}>{o.label}</option>
        ))}
      </select>
    </div>
  )
}
