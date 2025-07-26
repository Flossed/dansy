// test-model.js - Simple test to verify model loading works

async function testModelLoading() {
    console.log('=== Testing Model Loading ===');
    
    // Test 1: Check if files are accessible
    const modelPath = '../pytorch-minillm/tfjs_model/';
    const files = ['metadata.json', 'vocabulary.json', 'model.json', 'weights.bin'];
    
    console.log('Checking file accessibility...');
    for (const file of files) {
        try {
            const response = await fetch(modelPath + file);
            if (response.ok) {
                console.log(`✓ ${file} - OK (${response.status})`);
            } else {
                console.log(`✗ ${file} - Failed (${response.status})`);
            }
        } catch (error) {
            console.log(`✗ ${file} - Error: ${error.message}`);
        }
    }
    
    // Test 2: Try to load weights directly with TF.js
    console.log('\nTrying to load weights with tf.loadLayersModel...');
    try {
        // This will fail because we have custom layers, but it will show if files are accessible
        const model = await tf.loadLayersModel(modelPath + 'model.json');
        console.log('✓ Model loaded (unexpected - we have custom layers!)');
    } catch (error) {
        console.log(`Expected error (custom layers): ${error.message}`);
    }
    
    // Test 3: Load and inspect model.json
    console.log('\nInspecting model.json structure...');
    try {
        const response = await fetch(modelPath + 'model.json');
        const modelJson = await response.json();
        
        console.log('Model format:', modelJson.format);
        console.log('Weights manifest:', modelJson.weightsManifest);
        console.log('Total weights:', modelJson.weightsManifest[0].weights.length);
        
        // Show first few weights
        console.log('\nFirst 5 weights:');
        modelJson.weightsManifest[0].weights.slice(0, 5).forEach(w => {
            console.log(`  ${w.name}: shape=${JSON.stringify(w.shape)}, size=${w.size}`);
        });
    } catch (error) {
        console.log('Error loading model.json:', error);
    }
    
    // Test 4: Create a minimal working model
    console.log('\nCreating minimal test model...');
    try {
        const input = tf.input({shape: [10], dtype: 'int32'});
        const embedding = tf.layers.embedding({
            inputDim: 1000,
            outputDim: 64
        }).apply(input);
        const dense = tf.layers.dense({
            units: 10,
            activation: 'softmax'
        }).apply(embedding);
        
        const model = tf.model({inputs: input, outputs: dense});
        console.log('✓ Test model created successfully');
        console.log('Model summary:');
        model.summary();
    } catch (error) {
        console.log('✗ Failed to create test model:', error);
    }
}

// Run the test
testModelLoading();
