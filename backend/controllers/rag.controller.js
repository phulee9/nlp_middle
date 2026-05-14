const path = require('path');
const { spawn } = require('child_process');

const PROJECT_ROOT = path.join(__dirname, '..', '..');
const PYTHON = process.env.PYTHON_BIN || 'python';

function uploadPdf(req, res) {
  if (!req.file) {
    return res.status(400).json({ success: false, error: 'No PDF uploaded' });
  }
  return res.json({
    success: true,
    file: {
      originalName: req.file.originalname,
      filename: req.file.filename,
      path: req.file.path,
    },
  });
}

function runPythonRag({ pdfPath, question, mode, topK }) {
  return new Promise((resolve, reject) => {
    const args = [
      '-m', 'rag.cli',
      '--pdf', pdfPath,
      '--question', question,
      '--mode', mode || 'hybrid_rerank',
      '--top-k', String(topK || 5),
    ];

    const child = spawn(PYTHON, args, {
      cwd: PROJECT_ROOT,
      env: {
        ...process.env,
        PYTHONIOENCODING: 'utf-8',
        PYTHONUTF8: '1',
        TESSDATA_PREFIX: 'C:\\Users\\train\\miniconda3\\envs\\ttt_env\\share\\tessdata',
      },
      shell: false,
    });

    let stdout = '';
    let stderr = '';
    child.stdout.on('data', (data) => { stdout += data.toString(); });
    child.stderr.on('data', (data) => { stderr += data.toString(); });

    child.on('close', (code) => {
      const lines = stdout.trim().split('\n').filter(Boolean);
      const lastLine = lines[lines.length - 1] || '{}';
      let parsed;
      try {
        parsed = JSON.parse(lastLine);
      } catch (e) {
        return reject(new Error(`Invalid Python output. stderr=${stderr} stdout=${stdout}`));
      }
      if (code !== 0 || parsed.error) {
        return reject(new Error(parsed.error || stderr || `Python exited with code ${code}`));
      }
      resolve(parsed);
    });

    child.on('error', reject);
  });
}

async function askQuestion(req, res, next) {
  try {
    const { filename, question, mode = 'hybrid_rerank', topK = 5 } = req.body;
    if (!filename) return res.status(400).json({ success: false, error: 'filename is required' });
    if (!question) return res.status(400).json({ success: false, error: 'question is required' });

    const pdfPath = path.join(PROJECT_ROOT, 'uploads', filename);
    const result = await runPythonRag({ pdfPath, question, mode, topK });
    return res.json({ success: true, data: result });
  } catch (err) {
    next(err);
  }
}

module.exports = { uploadPdf, askQuestion };
