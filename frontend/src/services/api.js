import axios from 'axios';

const api = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL || 'http://localhost:3002/api/v1/rag',
});

export async function uploadPdf(file) {
  const formData = new FormData();
  formData.append('pdf', file);
  const res = await api.post('/upload', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  return res.data.file;
}

export async function askQuestion({ filename, question, mode, topK = 5 }) {
  const res = await api.post('/ask', { filename, question, mode, topK });
  return res.data.data;
}
