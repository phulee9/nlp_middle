export default function PdfUpload({ onUpload, fileInfo }) {
  function handleChange(e) {
    const file = e.target.files?.[0];
    if (file) onUpload(file);
  }

  function handleDrop(e) {
    e.preventDefault();
    const file = e.dataTransfer.files?.[0];
    if (file && file.type === 'application/pdf') onUpload(file);
  }

  function handleDragOver(e) {
    e.preventDefault();
  }

  return (
    <div className="sidebar-section">
      <label className="section-label">Document</label>
      <label
        className="pdf-dropbox"
        onDrop={handleDrop}
        onDragOver={handleDragOver}
      >
        <input type="file" accept="application/pdf" onChange={handleChange} />

        <div className="pdf-icon">PDF</div>

        <div className="pdf-title">
          {fileInfo ? fileInfo.originalName : 'Upload PDF'}
        </div>

        <div className="pdf-subtitle">
          {fileInfo
            ? '✓ Ready to ask questions'
            : 'Click or drag & drop'}
        </div>
      </label>
    </div>
  );
}
