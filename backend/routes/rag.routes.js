const express = require('express');
const multer = require('multer');
const path = require('path');
const { uploadPdf, askQuestion } = require('../controllers/rag.controller');

const router = express.Router();

const storage = multer.diskStorage({
  destination: (_req, _file, cb) => cb(null, path.join(__dirname, '..', '..', 'uploads')),
  filename: (_req, file, cb) => {
    const safe = file.originalname.replace(/[^a-zA-Z0-9._-]/g, '_');
    cb(null, `${Date.now()}-${safe}`);
  },
});

const upload = multer({
  storage,
  fileFilter: (_req, file, cb) => {
    if (file.mimetype !== 'application/pdf') return cb(new Error('Only PDF files are allowed'));
    cb(null, true);
  },
  limits: { fileSize: 50 * 1024 * 1024 },
});

router.post('/upload', upload.single('pdf'), uploadPdf);
router.post('/ask', askQuestion);

module.exports = router;
