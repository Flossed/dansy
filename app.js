// app.js - Main application logic

let model = null;
let isGenerating = false;

// DOM elements
const statusDiv = document.getElementById('status');
const progressBar = document.getElementById('progress-bar');
const modelInfoDiv = document.getElementById('model-info');
const modelDetailsSpan = document.getElementById('model-details');
const generateBtn = document.getElementById('generate-btn');
const loadModelBtn = document.getElementById('load-model-btn');
const promptInput = document.getElementById('prompt');
const outputText = document.getElementById('output-text');
const maxLengthSlider = document.getElementById('max-length');
const maxLengthValue = document.getElementById('max-length-value');
const temperatureSlider = document.getElementById('temperature');
const temperatureValue = document.getElementById('temperature-value');

// Update slider displays
maxLengthSlider.addEventListener('input', (e) => {
    maxLengthValue.textContent = e.target.value;
});

temperatureSlider.addEventListener('input', (e) => {
    temperatureValue.textContent = e.target.value;
});

// Load model on page load
window.addEventListener('load', () => {
    loadModel();
});

// Load model function
async function loadModel() {
    try {
        updateStatus('Loading model...', 'loading');
        progressBar.style.width = '0%';
        generateBtn.disabled = true;
        
        // Create model instance
        model = new MiniLLM();
        
        // Update progress
        progressBar.style.width = '30%';
        
        // Load the model - updated path for dansy location
        await model.loadModel('../pytorch-minillm/tfjs_model/');
        
        progressBar.style.width = '100%';
        
        // Update UI
        updateStatus('Model loaded successfully!', 'ready');
        generateBtn.disabled = false;
        
        // Show model info
        modelInfoDiv.style.display = 'block';
        modelDetailsSpan.textContent = `
            Parameters: ${(model.metadata.total_params / 1e6).toFixed(1)}M | 
            Vocab: ${model.vocabSize} | 
            Seq Length: ${model.maxSeqLen} | 
            Layers: ${model.metadata.num_blocks}
        `;
        
    } catch (error) {
        console.error('Error loading model:', error);
        updateStatus(`Error loading model: ${error.message}`, 'error');
        generateBtn.disabled = true;
    }
}

// Generate text function
async function generateText() {
    if (!model || isGenerating) return;
    
    try {
        isGenerating = true;
        generateBtn.disabled = true;
        generateBtn.textContent = 'Generating...';
        
        const prompt = promptInput.value.trim();
        if (!prompt) {
            alert('Please enter a prompt');
            return;
        }
        
        const maxLength = parseInt(maxLengthSlider.value);
        const temperature = parseFloat(temperatureSlider.value);
        
        outputText.innerHTML = '<em>Generating...</em>';
        
        // Generate text
        const startTime = performance.now();
        const generatedText = await model.generate(prompt, maxLength, temperature);
        const endTime = performance.now();
        
        // Display result
        outputText.innerHTML = `
            <strong>Prompt:</strong> ${escapeHtml(prompt)}<br>
            <strong>Generated:</strong> ${escapeHtml(generatedText)}<br>
            <small>Generation time: ${((endTime - startTime) / 1000).toFixed(2)}s</small>
        `;
        
    } catch (error) {
        console.error('Error generating text:', error);
        outputText.innerHTML = `<span style="color: red;">Error: ${error.message}</span>`;
    } finally {
        isGenerating = false;
        generateBtn.disabled = false;
        generateBtn.textContent = 'Generate Text';
    }
}

// Event listeners
generateBtn.addEventListener('click', generateText);
loadModelBtn.addEventListener('click', loadModel);

// Enter key to generate
promptInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter' && !isGenerating && model) {
        generateText();
    }
});

// Helper functions
function updateStatus(message, type) {
    statusDiv.textContent = message;
    statusDiv.className = `status ${type}`;
    
    if (type !== 'loading') {
        setTimeout(() => {
            progressBar.style.width = '0%';
        }, 1000);
    }
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Keyboard shortcuts
document.addEventListener('keydown', (e) => {
    // Ctrl/Cmd + Enter to generate
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter' && !isGenerating && model) {
        generateText();
    }
    
    // Ctrl/Cmd + R to reload model
    if ((e.ctrlKey || e.metaKey) && e.key === 'r') {
        e.preventDefault();
        loadModel();
    }
});

// Sample prompts
const samplePrompts = [
    "The cat",
    "A bird in the",
    "Time flies",
    "Better late than",
    "All that glitters",
    "The early bird",
    "Actions speak louder",
    "When in Rome"
];

// Add sample prompts on double-click
promptInput.addEventListener('dblclick', () => {
    const randomPrompt = samplePrompts[Math.floor(Math.random() * samplePrompts.length)];
    promptInput.value = randomPrompt;
});
