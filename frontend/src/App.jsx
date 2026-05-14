import { useState } from 'react';
import PdfUpload from './components/PdfUpload.jsx';
import ModeSelector from './components/ModeSelector.jsx';
import ChatBox from './components/ChatBox.jsx';
import ContextList from './components/ContextList.jsx';
import { uploadPdf, askQuestion } from './services/api.js';

export default function App() {
  const [fileInfo, setFileInfo] = useState(null);
  const [mode, setMode] = useState('hybrid');
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [light, setLight] = useState(false);

  async function handleUpload(file) {
    setLoading(true);
    try {
      const uploaded = await uploadPdf(file);
      setFileInfo(uploaded);
      setMessages([]);
    } finally {
      setLoading(false);
    }
  }

  async function handleAsk(question) {
    if (!fileInfo) return;
    const start = Date.now();

    const userMsg = {
      role: 'user',
      content: question,
      time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    };

    setMessages((prev) => [...prev, userMsg]);
    setLoading(true);

    try {
      const result = await askQuestion({ filename: fileInfo.filename, question, mode, topK: 5 });
      const end = Date.now();

      const botMsg = {
        role: 'assistant',
        content: result.answer,
        contexts: result.contexts,
        meta: result.mode,
        time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        duration: ((end - start) / 1000).toFixed(2),
      };

      setMessages((prev) => [...prev, botMsg]);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className={`app ${light ? 'light' : ''}`}>

      {/* ── Sidebar ── */}
      <aside className={`sidebar ${sidebarOpen ? '' : 'collapsed'}`}>
        <div className="sidebar-brand">
          <div className="brand-icon">AI</div>
          <div>
            <h2>RAG Chat</h2>
            <p>PDF Intelligence</p>
          </div>
        </div>

        <PdfUpload onUpload={handleUpload} fileInfo={fileInfo} />

        <div className="sidebar-section">
          <label className="section-label">Retrieval Mode</label>
          <ModeSelector value={mode} onChange={setMode} />
        </div>

        <button className="theme-btn" onClick={() => setLight(!light)}>
          {light ? '🌙  Dark mode' : '☀️  Light mode'}
        </button>
      </aside>

      {/* ── Main ── */}
      <main className="main">

        {/* Topbar */}
        <div className="topbar">
          <button className="topbar-toggle" onClick={() => setSidebarOpen(!sidebarOpen)}>
            ☰
          </button>
          <h1>Chat with your PDF</h1>
          <div className={`topbar-status ${fileInfo ? 'online' : ''}`}>
            <span className="dot" />
            {fileInfo ? fileInfo.originalName : 'No document'}
          </div>
        </div>

        {/* Messages */}
        <div className="messages">
          {messages.length === 0 && !loading && (
            <div className="empty-state">
              <div className="empty-icon">💬</div>
              <h3>Start a conversation</h3>
              <p>Upload a PDF and ask anything about it</p>
            </div>
          )}

          {messages.map((m, i) => (
            <div key={i} className={`message ${m.role}`}>
              <div className="msg-row">
                {/* <div className={`avatar ${m.role === 'user' ? 'user-avatar' : 'bot-avatar'}`}>
                  {m.role === 'user' ? '' : 'AI'}
                </div> */}

                <div className="bubble">
                  <div className="msg-header">
                    <span className="msg-name">{m.role === 'user' ? 'You' : 'Assistant'}</span>
                    <div className="meta">
                      {m.meta && <span className="meta-badge">{m.meta}</span>}
                      {m.duration && <span>⏱ {m.duration}s</span>}
                      <span>{m.time}</span>
                    </div>
                  </div>

                  <div className="answer-text">{m.content}</div>

                  {m.role === 'assistant' && <ContextList contexts={m.contexts} />}
                </div>
              </div>
            </div>
          ))}
        </div>

        {/* Loading indicator */}
        {loading && (
          <div className="loading-bar">
            <div className="loading-dots">
              <span /><span /><span />
            </div>
            Thinking…
          </div>
        )}

        {/* Input */}
        <ChatBox disabled={!fileInfo || loading} onAsk={handleAsk} />
      </main>
    </div>
  );
}
