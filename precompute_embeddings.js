(async () => {
    const fs = require('fs');
    const path = require('path');
    const datasetDir = path.join(__dirname, 'public', 'dataset');
    const embeddingsFile = path.join(__dirname, 'embeddings.json');
  
    const { AutoProcessor, CLIPVisionModelWithProjection, RawImage } = await import('@xenova/transformers');
  
    const quantized = false; // Đặt thành `true` nếu muốn sử dụng mô hình nhỏ hơn nhưng độ chính xác thấp hơn
  
    async function loadModels() {
        const imageProcessor = await AutoProcessor.from_pretrained('Xenova/clip-vit-base-patch16');
        const visionModel = await CLIPVisionModelWithProjection.from_pretrained('Xenova/clip-vit-base-patch16', { quantized });
        return { imageProcessor, visionModel };
    }
  
    function cosineSimilarity(A, B) {
        if (A.length !== B.length) throw new Error("A.length !== B.length");
        let dotProduct = 0, mA = 0, mB = 0;
        for (let i = 0; i < A.length; i++) {
            dotProduct += A[i] * B[i];
            mA += A[i] * A[i];
            mB += B[i] * B[i];
        }
        mA = Math.sqrt(mA);
        mB = Math.sqrt(mB);
        let similarity = dotProduct / (mA * mB);
        return similarity;
    }
  
    async function processImages() {
        const { imageProcessor, visionModel } = await loadModels();
        console.log('Models loaded successfully.');
  
        const files = fs.readdirSync(datasetDir).filter(file => {
            const ext = path.extname(file).toLowerCase();
            return ['.png', '.jpg', '.jpeg', '.bmp', '.gif'].includes(ext);
        });
  
        const embeddings = [];
  
        for (let i = 0; i < files.length; i++) {
            const file = files[i];
            const filePath = path.join(datasetDir, file);
            console.log(`Starting to process file: ${filePath}`); 
            try {
                const image = await RawImage.read(filePath);
  
                const imageInputs = await imageProcessor(image);

                console.log(imageInputs);
  
                const imageOutput = await visionModel({ pixel_values: imageInputs.pixel_values });

                const imageEmbedding = imageOutput.image_embeds.data; // Float32Array
  
                embeddings.push({
                    filename: file,
                    embedding: Array.from(imageEmbedding)
                });
  
                console.log(`Successfully processed file: ${file}`); // Log khi xử lý thành công

                if ((i + 1) % 100 === 0) {
                    console.log(`Processed ${i + 1}/${files.length} images.`);
                }
            } catch (error) {
                console.error(`Error processing ${file}:`, error);
            }
        }
  
        fs.writeFileSync(embeddingsFile, JSON.stringify(embeddings, null, 2));
        console.log(`Embeddings saved to ${embeddingsFile}`);
    }
  
    processImages();
})();
