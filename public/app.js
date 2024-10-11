// Đánh dấu file là module để sử dụng 'import'
import { 
    AutoProcessor, 
    CLIPVisionModelWithProjection, 
    RawImage
} from 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.5.4/dist/transformers.js';

// Biến kiểm soát lượng tử hóa mô hình
let quantized = false; // Đặt thành `true` để sử dụng mô hình nhỏ hơn nhưng độ chính xác thấp hơn

// Tải các mô hình và bộ xử lý
let imageProcessor, visionModel;

// Tải embeddings từ tệp embeddings.json
let datasetEmbeddings = [];

// Hàm tải các mô hình và bộ xử lý
async function loadModels() {
    try {
        // Tải bộ xử lý ảnh
        imageProcessor = await AutoProcessor.from_pretrained('Xenova/clip-vit-base-patch16');
        console.log('Image Processor Loaded');

        // Tải mô hình xử lý ảnh
        visionModel = await CLIPVisionModelWithProjection.from_pretrained('Xenova/clip-vit-base-patch16', {quantized});
        console.log('Vision Model Loaded');

        // Tải embeddings.json
        const response = await fetch('embeddings.json');
        if (!response.ok) {
            throw new Error('Failed to load embeddings.json');
        }
        datasetEmbeddings = await response.json();
        console.log('Embeddings Loaded:', datasetEmbeddings.length, 'images.');
        
        document.getElementById('status-message').innerText = 'Models and embeddings loaded successfully!';
    } catch (error) {
        console.error('Error loading models or embeddings:', error);
        document.getElementById('status-message').innerText = 'Error loading models or embeddings. Please check the console.';
    }
}

// Hàm tính cosine similarity
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

// Hàm hiển thị ảnh đã tải lên
function displayImage(file, container) {
    const img = document.createElement('img');
    img.src = URL.createObjectURL(file);
    container.innerHTML = ''; // Xóa ảnh cũ
    container.appendChild(img);
}

// Hàm xử lý tải lên ảnh input
document.getElementById('upload-input-image').addEventListener('change', async (event) => {
    const file = event.target.files[0];
    if (file && isValidFile(file)) {
        displayImage(file, document.getElementById('uploaded-input-image-container'));
    }
});

// Hàm kiểm tra tính hợp lệ của file
function isValidFile(file) {
    const MAX_FILE_SIZE = 5 * 1024 * 1024; // 5 MB
    const validTypes = ['image/jpeg', 'image/png', 'image/gif'];
    if (file.size > MAX_FILE_SIZE) {
        alert('File size exceeds the maximum limit of 5MB.');
        return false;
    }
    if (!validTypes.includes(file.type)) {
        alert('Unsupported file type. Please upload JPEG, PNG, or GIF images.');
        return false;
    }
    return true;
}

// Hàm xử lý khi nhấn nút tìm ảnh tương đồng
document.getElementById('find-similar').addEventListener('click', async () => {
    const fileInput = document.getElementById('upload-input-image');
    const resultContainer = document.getElementById('result');
    const processingMessage = document.getElementById('status-message');

    if (fileInput.files.length === 0) {
        alert('Please upload an input image.');
        return;
    }

    const file = fileInput.files[0];
    if (!isValidFile(file)) {
        return;
    }

    const reader = new FileReader();

    reader.onload = async function(e) {
        try {
            processingMessage.innerText = 'Computing similarity, please wait...';
            // Đọc ảnh từ Data URL
            let image = await RawImage.read(e.target.result);
            let imageInputs = await imageProcessor(image);
            let imageOutput = await visionModel.forward(imageInputs);
            let inputEmbedding = imageOutput.image_embeds.data; // Float32Array

            // Tính độ tương đồng với mỗi ảnh trong dataset
            let maxSimilarity = -Infinity;
            let bestMatch = null;

            for (let i = 0; i < datasetEmbeddings.length; i++) {
                const datasetEmbedding = datasetEmbeddings[i].embedding;
                const similarity = cosineSimilarity(inputEmbedding, datasetEmbedding);
                if (similarity > maxSimilarity) {
                    maxSimilarity = similarity;
                    bestMatch = datasetEmbeddings[i];
                }
            }

            if (bestMatch) {
                resultContainer.innerHTML = `
                    <h3>Cosine Similarity: ${maxSimilarity.toFixed(4)}</h3>
                    <div class="image-pair">
                        <div>
                            <p>Input Image:</p>
                            <img src="${e.target.result}" alt="Input Image">
                        </div>
                        <div>
                            <p>Most Similar Image:</p>
                            <img src="dataset/${bestMatch.filename}" alt="Most Similar Image">
                        </div>
                    </div>
                `;
            } else {
                resultContainer.innerHTML = `<p>No similar images found.</p>`;
            }

            processingMessage.innerText = 'Similarity computation completed.';
        } catch (error) {
            console.error('Error computing similarity:', error);
            alert('Error computing similarity.');
            processingMessage.innerText = 'Error computing similarity.';
        }
    };

    reader.readAsDataURL(file);
});

// Tải các mô hình và embeddings khi trang được tải
window.addEventListener('load', () => {
    loadModels();
});
