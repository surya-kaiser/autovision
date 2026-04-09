import React, { useCallback, useRef, useState } from 'react'
import { useDropzone } from 'react-dropzone'
import { Upload, FolderOpen, FileText, Image, Archive, File } from 'lucide-react'

function fileIcon(name) {
  const ext = (name || '').split('.').pop().toLowerCase()
  if (['jpg', 'jpeg', 'png', 'gif', 'bmp', 'webp', 'tiff'].includes(ext))
    return <Image className="w-5 h-5 text-blue-400" />
  if (['zip', 'tar', 'gz', 'rar', '7z'].includes(ext))
    return <Archive className="w-5 h-5 text-yellow-400" />
  if (['csv', 'tsv', 'json', 'xlsx', 'txt'].includes(ext))
    return <FileText className="w-5 h-5 text-green-400" />
  return <File className="w-5 h-5 text-gray-400" />
}

function fmtSize(bytes) {
  if (!bytes) return ''
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / 1024 / 1024).toFixed(1)} MB`
  return `${(bytes / 1024 / 1024 / 1024).toFixed(2)} GB`
}

export default function FileUploader({ onFile, onFolder }) {
  const folderRef = useRef(null)
  const [queued, setQueued] = useState(null) // { name, size, type }

  const onDrop = useCallback((accepted) => {
    if (!accepted[0]) return
    setQueued({ name: accepted[0].name, size: accepted[0].size })
    onFile(accepted[0])
  }, [onFile])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    maxFiles: 1,
    // Accept everything — no restriction
  })

  function handleFolderChange(e) {
    const files = Array.from(e.target.files)
    if (!files.length) return
    const name = files[0].webkitRelativePath?.split('/')[0] || 'folder'
    const totalSize = files.reduce((s, f) => s + f.size, 0)
    setQueued({ name, size: totalSize, isFolder: true, count: files.length })
    onFolder(files)
    e.target.value = ''
  }

  return (
    <div className="space-y-3">
      {/* Drop zone */}
      <div
        {...getRootProps()}
        className={`relative rounded-2xl border-2 border-dashed transition-all duration-200 cursor-pointer
          ${isDragActive
            ? 'border-accent-400 bg-accent-500/10 scale-[1.01]'
            : 'border-dark-600 hover:border-accent-500/60 bg-dark-900/40'}`}
      >
        <input {...getInputProps()} />

        {/* Drag overlay */}
        {isDragActive && (
          <div className="absolute inset-0 rounded-2xl bg-accent-500/5 flex items-center justify-center z-10">
            <div className="text-center">
              <Upload className="w-12 h-12 mx-auto text-accent-400 animate-bounce" />
              <p className="mt-2 text-accent-300 font-medium">Drop to upload</p>
            </div>
          </div>
        )}

        <div className="p-10 text-center">
          <div className="w-16 h-16 rounded-2xl bg-dark-700 border border-dark-600 flex items-center justify-center mx-auto mb-4">
            <Upload className="w-7 h-7 text-accent-400" />
          </div>
          <p className="text-base font-semibold text-white mb-1">
            Drag and drop your dataset here
          </p>
          <p className="text-sm text-gray-500 mb-5">
            CSV, ZIP, images, JSON, Excel — any format accepted
          </p>

          {/* Action buttons */}
          <div className="flex items-center justify-center gap-3">
            <button
              type="button"
              onClick={(e) => { e.stopPropagation(); document.getElementById('_gdrive_file_input')?.click() }}
              className="flex items-center gap-2 px-4 py-2 bg-accent-500 hover:bg-accent-400 text-white rounded-lg text-sm font-medium transition-colors shadow-lg shadow-accent-500/20"
            >
              <File className="w-4 h-4" />
              Browse Files
            </button>
            <button
              type="button"
              onClick={(e) => { e.stopPropagation(); folderRef.current?.click() }}
              className="flex items-center gap-2 px-4 py-2 bg-dark-700 hover:bg-dark-600 border border-dark-600 text-white rounded-lg text-sm font-medium transition-colors"
            >
              <FolderOpen className="w-4 h-4 text-accent-400" />
              Browse Folder
            </button>
          </div>
        </div>

        {/* Hidden inputs */}
        <input
          id="_gdrive_file_input"
          type="file"
          className="hidden"
          onChange={(e) => { if (e.target.files[0]) onDrop([e.target.files[0]]); e.target.value = '' }}
        />
        <input
          ref={folderRef}
          type="file"
          webkitdirectory=""
          directory=""
          multiple
          className="hidden"
          onChange={handleFolderChange}
        />
      </div>

      {/* Queued file chip */}
      {queued && (
        <div className="flex items-center gap-3 px-3 py-2.5 bg-dark-700 border border-dark-600 rounded-xl text-sm">
          {queued.isFolder
            ? <FolderOpen className="w-5 h-5 text-accent-400 shrink-0" />
            : fileIcon(queued.name)
          }
          <div className="flex-1 min-w-0">
            <div className="text-white font-medium truncate">{queued.name}</div>
            <div className="text-xs text-gray-500">
              {queued.isFolder ? `${queued.count?.toLocaleString()} files · ` : ''}{fmtSize(queued.size)}
            </div>
          </div>
          <button
            type="button"
            onClick={() => setQueued(null)}
            className="text-gray-600 hover:text-gray-400 text-lg leading-none shrink-0"
          >×</button>
        </div>
      )}
    </div>
  )
}
