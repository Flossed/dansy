# Dansy - Mini LLM TensorFlow.js

A browser-based language model using TensorFlow.js, trained with PyTorch and exported for web deployment.

## Features

- 🚀 Runs entirely in the browser using TensorFlow.js
- 🤖 49.7M parameter transformer model
- ⚡ Real-time text generation
- 🎛️ Adjustable generation parameters (temperature, length)
- 📦 Loads PyTorch-trained weights

## Architecture

- **Model**: Mini LLM with simplified attention mechanism
- **Parameters**: 49.7M
- **Vocabulary**: 30,000 tokens
- **Max Sequence Length**: 128
- **Layers**: 6 transformer blocks

## Quick Start

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Flossed/dansy.git
   cd dansy
   ```

2. **Install dependencies**:
   ```bash
   npm install
   ```

3. **Start the server**:
   ```bash
   npm start
   ```
   Or on Windows:
   ```cmd
   start.bat
   ```

4. **Open your browser** to: http://localhost:3000

## Model Files

The app expects model files in the following structure:
```
../pytorch-minillm/tfjs_model/
├── model.json
├── weights.bin
├── metadata.json
└── vocabulary.json
```

## Usage

1. Enter a text prompt (e.g., "The cat", "A bird in the")
2. Adjust temperature (0.1-2.0) for creativity
3. Set max length (10-100) for output length
4. Click "Generate Text" or press Enter

### Keyboard Shortcuts

- **Enter**: Generate text
- **Ctrl+Enter**: Generate text
- **Ctrl+R**: Reload model

### Sample Prompts

Double-click the input field to get a random sample prompt.

## Technical Details

### Custom Layers

The implementation includes custom TensorFlow.js layers:
- `SimplifiedAttention`: Q+K softmax attention mechanism
- `TransformerBlock`: Complete transformer block with residuals

### Model Export

The model was trained in PyTorch and exported to TensorFlow.js format using a custom export script that:
- Maps PyTorch layer names to TF.js conventions
- Transposes weight matrices for compatibility
- Preserves the simplified attention mechanism

## Development

### Project Structure
```
dansy/
├── index.html      # User interface
├── model.js        # TensorFlow.js model implementation
├── app.js          # Application logic
├── server.js       # Express server
├── package.json    # Dependencies
└── start.bat       # Windows starter
```

### Building from Source

1. Train the PyTorch model (see pytorch-minillm)
2. Export weights to TF.js format
3. Place exported files in the expected directory
4. Run the application

## Requirements

- Node.js 14+
- Modern web browser with WebGL support
- ~200MB for model weights

## License

MIT

## Acknowledgments

- Built with TensorFlow.js
- Simplified attention mechanism for efficient browser execution
- Trained using PyTorch with AMD GPU optimization
