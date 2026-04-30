require('dotenv').config({ path: '../.env' });

const express = require('express');
const cors = require('cors');
const path = require('path');
const ragRoutes = require('./routes/rag.routes');

const app = express();
const PORT = Number(process.env.PORT || 3002);

app.use(cors({ origin: ['http://localhost:5173', 'http://127.0.0.1:5173'] }));
app.use(express.json({ limit: '2mb' }));
app.use('/uploads', express.static(path.join(__dirname, '..', 'uploads')));

app.get('/health', (_req, res) => res.json({ ok: true }));
app.use('/api/v1/rag', ragRoutes);

app.use((err, _req, res, _next) => {
  console.error(err);
  const status = err.statusCode || 500;
  res.status(status).json({ success: false, error: err.message || 'Internal server error' });
});

app.listen(PORT, () => {
  console.log(`Hybrid RAG backend listening on http://localhost:${PORT}`);
});
