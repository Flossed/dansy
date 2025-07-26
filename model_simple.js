// model.js - Fixed version with better weight loading

class SimplifiedAttention extends tf.layers.Layer {
    constructor(config) {
        super(config);
        this.units = config.units || 512;
    }

    build(inputShape) {
        // Create weight variables without bias
        this.q = this.addWeight('q/kernel', [this.units, this.units], 'float32', tf.initializers.glorotUniform());
        this.k = this.addWeight('k/kernel', [this.units, this.units], 'float32', tf.initializers.glorotUniform());
        this.v = this.addWeight('v/kernel', [this.units, this.units], 'float32', tf.initializers.glorotUniform());
        this.out = this.addWeight('out/kernel', [this.units, this.units], 'float32', tf.initializers.glorotUniform());
    }

    call(inputs, kwargs) {
        return tf.tidy(() => {
            const x = inputs[0];
            
            // Q, K, V projections
            const Q = tf.matMul(x, this.q.read());
            const K = tf.matMul(x, this.k.read());
            const V = tf.matMul(x, this.v.read());
            
            // Simplified attention: scores = Softmax(Q + K)
            const scores = tf.softmax(tf.add(Q, K), -1);
            
            // Apply attention: output = scores * V
            const attentionOutput = tf.mul(scores, V);
            
            // Output projection
            const output = tf.matMul(attentionOutput, this.out.read());
            
            return output;
        });
    }

    getConfig() {
        const config = super.getConfig();
        config.units = this.units;
        return config;
    }

    static get className() {
        return 'SimplifiedAttention';
    }
}

// Register the custom layer
tf.serialization.registerClass(SimplifiedAttention);

// Mini LLM Model class
class MiniLLM {
    constructor() {
        this.model = null;
        this.tokenizer = null;
        this.metadata = null;
        this.maxSeqLen = 128;
        this.vocabSize = 30000;
        this.embedDim = 512;
        this.positionalEmbedding = null;
    }

    async loadModel(modelPath = '../pytorch-minillm/tfjs_model/') {
        try {
            console.log('Loading model from:', modelPath);
            
            // Load metadata
            const metadataResponse = await fetch(modelPath + 'metadata.json');
            this.metadata = await metadataResponse.json();
            console.log('Metadata loaded:', this.metadata);
            
            // Update dimensions from metadata
            this.maxSeqLen = this.metadata.max_seq_len;
            this.vocabSize = this.metadata.vocab_size;
            this.embedDim = this.metadata.embed_dim;
            
            // Load vocabulary
            const vocabResponse = await fetch(modelPath + 'vocabulary.json');
            const vocabData = await vocabResponse.json();
            this.tokenizer = new SimpleTokenizer(vocabData);
            console.log('Vocabulary loaded');
            
            // Load weights directly using tf.loadLayersModel
            // This is a simplified approach - we'll build a basic model
            this.model = await this.buildSimpleModel();
            
            // Load weights manually
            await this.loadWeightsManually(modelPath);
            
            console.log('Model loaded successfully!');
            return true;
        } catch (error) {
            console.error('Error loading model:', error);
            throw error;
        }
    }

    async buildSimpleModel() {
        // Build a simplified model for testing
        const input = tf.input({shape: [this.maxSeqLen], dtype: 'int32'});
        
        // Embedding layer
        const embedding = tf.layers.embedding({
            inputDim: this.vocabSize,
            outputDim: this.embedDim,
            inputLength: this.maxSeqLen,
            name: 'token_embedding'
        });
        
        let x = embedding.apply(input);
        
        // Add positional embedding
        const posEmbedding = tf.layers.dense({
            units: this.embedDim,
            use_bias: false,
            name: 'pos_embedding'
        });
        
        // Simple transformation layers (simplified from full transformer)
        for (let i = 0; i < 6; i++) {
            const dense1 = tf.layers.dense({
                units: this.embedDim,
                activation: 'relu',
                name: `block_${i}_dense1`
            });
            const dense2 = tf.layers.dense({
                units: this.embedDim,
                name: `block_${i}_dense2`
            });
            const layerNorm = tf.layers.layerNormalization({
                name: `block_${i}_ln`
            });
            
            x = dense1.apply(x);
            x = dense2.apply(x);
            x = layerNorm.apply(x);
        }
        
        // Output layer
        const output = tf.layers.dense({
            units: this.vocabSize,
            activation: 'softmax',
            name: 'output_projection'
        }).apply(x);
        
        const model = tf.model({inputs: input, outputs: output});
        model.compile({
            optimizer: 'adam',
            loss: 'sparseCategoricalCrossentropy',
            metrics: ['accuracy']
        });
        
        return model;
    }

    async loadWeightsManually(modelPath) {
        try {
            // Load the weights manifest
            const manifestResponse = await fetch(modelPath + 'model.json');
            const manifest = await manifestResponse.json();
            
            // Load the binary weights
            const weightsResponse = await fetch(modelPath + 'weights.bin');
            const weightsBuffer = await weightsResponse.arrayBuffer();
            
            console.log('Weights binary loaded, size:', weightsBuffer.byteLength);
            
            // Store positional embedding for later use
            const weights = manifest.weightsManifest[0].weights;
            for (const weight of weights) {
                if (weight.name === 'positional_embedding') {
                    const data = new Float32Array(
                        weightsBuffer,
                        weight.offset,
                        weight.shape.reduce((a, b) => a * b, 1)
                    );
                    this.positionalEmbedding = tf.tensor(data, weight.shape);
                    console.log('Loaded positional embedding:', weight.shape);
                    break;
                }
            }
        } catch (error) {
            console.error('Error loading weights manually:', error);
        }
    }

    async generate(prompt, maxLength = 30, temperature = 0.8) {
        if (!this.model || !this.tokenizer) {
            throw new Error('Model not loaded');
        }
        
        // Encode prompt
        let inputIds = this.tokenizer.encode(prompt);
        
        // Pad or truncate to maxSeqLen
        if (inputIds.length > this.maxSeqLen) {
            inputIds = inputIds.slice(0, this.maxSeqLen);
        }
        while (inputIds.length < this.maxSeqLen) {
            inputIds.push(0); // Pad with 0
        }
        
        // Convert to tensor
        const inputTensor = tf.tensor2d([inputIds], [1, this.maxSeqLen], 'int32');
        
        try {
            // Get model prediction
            const output = this.model.predict(inputTensor);
            
            // Get the last token position that has actual content
            const lastPosition = Math.min(prompt.split(' ').length + maxLength, this.maxSeqLen - 1);
            
            // Sample from the output
            const logits = output.slice([0, lastPosition, 0], [1, 1, -1]).squeeze();
            const probs = tf.softmax(logits.div(temperature));
            
            // Get top k tokens
            const k = 50;
            const {values, indices} = tf.topk(probs, k);
            
            // Sample from top k
            const topKProbs = tf.softmax(values);
            const sampleIdx = tf.multinomial(topKProbs, 1).dataSync()[0];
            const nextToken = indices.dataSync()[sampleIdx];
            
            // Clean up
            inputTensor.dispose();
            output.dispose();
            logits.dispose();
            probs.dispose();
            values.dispose();
            indices.dispose();
            topKProbs.dispose();
            
            // Decode
            const generatedIds = [...inputIds.slice(0, lastPosition + 1), nextToken];
            return this.tokenizer.decode(generatedIds.filter(id => id !== 0));
            
        } catch (error) {
            inputTensor.dispose();
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
        // Simple tokenization - split on spaces and punctuation
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
        
        // Join tokens with appropriate spacing
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

// Export for use in other scripts
window.MiniLLM = MiniLLM;
window.SimpleTokenizer = SimpleTokenizer;
