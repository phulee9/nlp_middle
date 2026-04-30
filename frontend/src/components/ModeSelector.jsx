const modes = [
  { value: 'bm25', label: 'BM25', desc: 'Keyword matching' },
  { value: 'embedding', label: 'Embedding', desc: 'Semantic search' },
  { value: 'hybrid', label: 'Hybrid', desc: 'BM25 + Embedding' },
  { value: 'hybrid_rerank', label: 'Hybrid + Rerank', desc: 'With reranker' },
  { value: 'full', label: 'Full Pipeline', desc: 'Best accuracy' },
];

export default function ModeSelector({ value, onChange }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
      <select
        className="mode-select"
        value={value}
        onChange={(e) => onChange(e.target.value)}
      >
        {modes.map((m) => (
          <option key={m.value} value={m.value}>
            {m.label} — {m.desc}
          </option>
        ))}
      </select>
    </div>
  );
}
