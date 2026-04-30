function scoreOf(c) {
  return c.reranker_score ?? c.hybrid_score ?? c.cosine_score ?? c.bm25_score ?? c.bm25_raw ?? '';
}

export default function ContextList({ contexts = [] }) {
  if (!contexts || !contexts.length) return null;

  return (
    <div className="contexts">
      <h3>Retrieved Contexts ({contexts.length})</h3>
      {contexts.map((c, idx) => {
        const score = scoreOf(c);
        const label = c.id || `chunk-${idx + 1}`;
        return (
          <details key={c.id || idx} className="context-item">
            <summary>
              <span style={{ flex: 1 }}>{label}</span>
              {score !== '' && (
                <span style={{
                  fontSize: '10px',
                  background: 'var(--accent-dim)',
                  color: 'var(--accent)',
                  padding: '2px 6px',
                  borderRadius: '4px',
                  fontWeight: 600,
                  letterSpacing: '0.03em',
                  marginLeft: 'auto',
                  flexShrink: 0,
                }}>
                  {Number(score).toFixed(4)}
                </span>
              )}
            </summary>
            <pre>{c.text}</pre>
          </details>
        );
      })}
    </div>
  );
}
