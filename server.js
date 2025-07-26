const express = require('express');
const path = require('path');
const cors = require('cors');

const app = express();
const PORT = 3000;

// Enable CORS
app.use(cors());

// Serve static files from current directory
app.use(express.static(__dirname));

// Serve PyTorch model files
app.use('/pytorch-minillm', express.static(path.join(__dirname, '..', 'pytorch-minillm')));

// Default route
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'index.html'));
});

// Start server
app.listen(PORT, () => {
    console.log(`Mini LLM TensorFlow.js app running at http://localhost:${PORT}`);
    console.log(`Model files should be in: ${path.join(__dirname, '..', 'pytorch-minillm', 'tfjs_model')}`);
});
