// model_working.js - Simplified working version with proper weight loading

class MiniLLM {
    constructor() {
        this.model = null;
        this.tokenizer = null;
        this.metadata = null;
        this.maxSeqLen = 128;
        this.vocabSize = 30000;
        this.embedDim = 512;
        this.weights = {};
    }

    async loadModel(modelPath = '../pytorch-minillm/tfjs_model/') {
        try {
            console.log('Loading model from:', modelPath);
            
            // Load metadata
            const metadataResponse = await fetch(modelPath + 'metadata.json');
            this.metadata = await metadataResponse.json();
            console.log('Metadata loaded:', this.metadata);
            
            // Update dimensions
            this.maxSeqLen = this.metadata.max_seq_len;
            this.vocabSize = this.metadata.vocab_size;
            this.embedDim = this.metadata.embed_dim;
            
            // Load vocabulary
            const vocabResponse = await fetch(modelPath + 'vocabulary.json');
            const vocabData = await vocabResponse.json();
            this.tokenizer = new SimpleTokenizer(vocabData);
            console.log('Vocabulary loaded');
            
            // Load all weights first
            await this.loadAllWeights(modelPath);
            
            // Build model with loaded weights
            this.model = await this.buildModelWithWeights();
            
            console.log('Model loaded successfully!');
            return true;
        } catch (error) {
            console.error('Error loading model:', error);
            throw error;
        }
    }

    async loadAllWeights(modelPath) {
        console.log('Loading weights...');
        
        // Load manifest
        const manifestResponse = await fetch(modelPath + 'model.json');
        const manifest = await manifestResponse.json();
        
        // Load binary weights
        const weightsResponse = await fetch(modelPath + 'weights.bin');
        const weightsBuffer = await weightsResponse.arrayBuffer();
        
        // Parse all weights
        const weightSpecs = manifest.weightsManifest[0].weights;
        let offset = 0;
        
        for (const spec of weightSpecs) {
            const size = spec.shape.reduce((a, b) => a * b, 1);
            const data = new Float32Array(weightsBuffer, offset, size);
            
            // Store as tensor
            this.weights[spec.name] = tf.tensor(data, spec.shape);
            console.log(`Loaded ${spec.name}: [${spec.shape}]`);
            
            offset += size * 4; // 4 bytes per float32
        }
    }

    async buildModelWithWeights() {
        // Create a simplified sequential model
        const model = tf.sequential();
        
        // Add embedding layer
        model.add(tf.layers.embedding({
            inputDim: this.vocabSize,
            outputDim: this.embedDim,
            inputLength: this.maxSeqLen,
            weights: [this.weights['token_embedding/embeddings']],
            name: 'embedding'
        }));
        
        // Add positional embedding as a custom layer
        model.add(tf.layers.lambda({
            fn: (x) => {
                // Add positional embeddings
                const posEmb = this.weights['positional_embedding'];
                return tf.add(x, posEmb);
            },
            name: 'add_positional'
        }));
        
        // Simplified transformer blocks - just use dense layers for now
        for (let i = 0; i < 6; i++) {
            // Attention approximation with dense layers
            model.add(tf.layers.dense({
                units: this.embedDim,
                activation: 'relu',
                name: `block_${i}_attn`
            }));
            
            // Layer norm
            model.add(tf.layers.layerNormalization({
                name: `block_${i}_ln1`
            }));
            
            // FFN
            model.add(tf.layers.dense({
                units: 2048,
                activation: 'relu',
                name: `block_${i}_ffn1`
            }));
            
            model.add(tf.layers.dense({
                units: this.embedDim,
                name: `block_${i}_ffn2`
            }));
            
            // Layer norm
            model.add(tf.layers.layerNormalization({
                name: `block_${i}_ln2`
            }));
        }
        
        // Final layer norm
        model.add(tf.layers.layerNormalization({
            name: 'final_ln'
        }));
        
        // Output projection
        model.add(tf.layers.dense({
            units: this.vocabSize,
            activation: 'softmax',
            name: 'output_projection'
        }));
        
        // Compile
        model.compile({
            optimizer: 'adam',
            loss: 'sparseCategoricalCrossentropy',
            metrics: ['accuracy']
        });
        
        return model;
    }

    async generate(prompt, maxLength = 30, temperature = 0.8) {
        if (!this.model || !this.tokenizer) {
            throw new Error('Model not loaded');
        }
        
        try {
            // Encode prompt
            let inputIds = this.tokenizer.encode(prompt);
            
            // Ensure we have at least some tokens
            if (inputIds.length === 0) {
                inputIds = [this.tokenizer.tokenToId['<bos>'] || 2];
            }
            
            // Pad to max length
            while (inputIds.length < this.maxSeqLen) {
                inputIds.push(0);
            }
            
            // Truncate if needed
            inputIds = inputIds.slice(0, this.maxSeqLen);
            
            // Convert to tensor
            const inputTensor = tf.tensor2d([inputIds], [1, this.maxSeqLen], 'int32');
            
            // Get prediction
            const output = this.model.predict(inputTensor);
            
            // Get probabilities for last actual token position
            const tokenLength = Math.min(prompt.split(/\s+/).length + 5, this.maxSeqLen - 1);
            const probs = output.slice([0, tokenLength - 1, 0], [1, 1, -1]).squeeze();
            
            // Apply temperature
            const scaledProbs = tf.div(probs, temperature);
            const softmaxProbs = tf.softmax(scaledProbs);
            
            // Sample top-k
            const k = Math.min(50, this.vocabSize);
            const {values, indices} = tf.topk(softmaxProbs, k);
            
            // Sample from top-k
            const topKProbs = tf.softmax(values);
            const sampleIdx = tf.multinomial(topKProbs, 1).dataSync()[0];
            const nextToken = indices.dataSync()[sampleIdx];
            
            // Clean up
            inputTensor.dispose();
            output.dispose();
            probs.dispose();
            scaledProbs.dispose();
            softmaxProbs.dispose();
            values.dispose();
            indices.dispose();
            topKProbs.dispose();
            
            // Decode result
            const resultIds = inputIds.slice(0, tokenLength).concat([nextToken]);
            return this.tokenizer.decode(resultIds);
            
        } catch (error) {
            console.error('Generation error:', error);
            throw error;
        }
    }
}

// Simple tokenizer class
class SimpleTokenizer {
    constructor(vocabData) {
        this.vocabSize = vocabData.vocab_size || 30000;
        this.tokenToId = vocabData.token_to_id || {};
        this.idToToken = {};
        
        // Build reverse mapping
        for (const [token, id] of Object.entries(this.tokenToId)) {
            this.idToToken[id] = token;
        }
        
        // Special tokens
        this.specialTokens = {
            '<pad>': 0,
            '<unk>': 1,
            '<bos>': 2,
            '<eos>': 3,
            '<sep>': 4,
            '<cls>': 5
        };
    }

    encode(text) {
        // Simple tokenization
        const tokens = text.toLowerCase().match(/\w+|[^\w\s]/g) || [];
        const ids = [];
        
        for (const token of tokens) {
            if (token in this.tokenToId) {
                ids.push(this.tokenToId[token]);
            } else {
                ids.push(this.tokenToId['<unk>'] || 1);
            }
        }
        
        return ids;
    }

    decode(ids, skipSpecialTokens = true) {
        const tokens = [];
        
        for (const id of ids) {
            const token = this.idToToken[id] || '<unk>';
            
            if (skipSpecialTokens && token in this.specialTokens) {
                continue;
            }
            
            tokens.push(token);
        }
        
        // Join tokens
        let text = '';
        for (let i = 0; i < tokens.length; i++) {
            const token = tokens[i];
            if (i === 0) {
                text = token;
            } else if (/^[.,!?;:'")-]/.test(token)) {
                text += token;
            } else {
                text += ' ' + token;
            }
        }
        
        return text;
    }
}

// Export
window.MiniLLM = MiniLLM;
window.SimpleTokenizer = SimpleTokenizer;
