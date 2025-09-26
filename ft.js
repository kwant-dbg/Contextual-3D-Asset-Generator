document.addEventListener('DOMContentLoaded', () => {
    const fileInput = document.getElementById('file-input');
    const imageContainer = document.getElementById('image-container');
    const loader = document.querySelector('.loader');
    const loaderText = document.getElementById('loader-text');

    let currentDetections = [];
    let currentImageUrl = '';
    let originalImageWidth = 0;

    fileInput.addEventListener('change', async (event) => {
        const file = event.target.files[0];
        if (!file) return;

        clearContainer();
        showLoader('Uploading and detecting objects...');

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('/detect', { method: 'POST', body: formData });
            const data = await handleResponse(response);
            
            if (data.detections && data.detections.length > 0) {
                hideLoader();
                currentDetections = data.detections;
                currentImageUrl = data.image_url;
                displayImageAndBoxes(currentImageUrl, currentDetections);
            } else {
                showError('No furniture detected. Please try another image.');
                displayImageAndBoxes(data.image_url, []);
            }
        } catch (error) {
            showError(`Error: ${error.message}`);
        }
    });

    function displayImageAndBoxes(imageUrl, detections) {
        clearContainer();
        const img = new Image();
        img.src = imageUrl;
        img.onload = () => {
            originalImageWidth = img.naturalWidth;
            imageContainer.appendChild(img);

            const canvas = document.createElement('canvas');
            imageContainer.appendChild(canvas);

            const scale = img.width / originalImageWidth;
            canvas.width = img.width;
            canvas.height = img.height;
            const ctx = canvas.getContext('2d');

            detections.forEach(det => {
                const [x1, y1, x2, y2] = det.box.map(coord => coord * scale);
                ctx.strokeStyle = 'rgba(24, 119, 242, 0.9)';
                ctx.lineWidth = 3;
                ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

                ctx.fillStyle = 'rgba(24, 119, 242, 0.9)';
                ctx.font = 'bold 16px Arial';
                ctx.fillText(det.class_name, x1 + 5, y1 > 20 ? y1 - 5 : y1 + 20);
            });

            canvas.addEventListener('click', (event) => handleCanvasClick(event, canvas, scale));
        };
    }

    function handleCanvasClick(event, canvas, scale) {
        const rect = canvas.getBoundingClientRect();
        const x = event.clientX - rect.left;
        const y = event.clientY - rect.top;

        const clickedDetection = currentDetections.find(det => {
            const [x1, y1, x2, y2] = det.box.map(c => c * scale);
            return x >= x1 && x <= x2 && y >= y1 && y <= y2;
        });

        if (clickedDetection) {
            replaceObject(clickedDetection.box);
        }
    }

    async function replaceObject(box) {
        showLoader('Generating new model & compositing scene... This may take a minute.');
        
        try {
            const response = await fetch('/replace', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image_path: currentImageUrl, box: box })
            });
            const data = await handleResponse(response);
            
            hideLoader();
            const finalImage = new Image();
            finalImage.src = data.final_image_url;
            imageContainer.innerHTML = '';
            imageContainer.appendChild(finalImage);

        } catch (error) {
            showError(`Error: ${error.message}`);
        }
    }

    // --- UI Helper Functions ---
    function showLoader(message) {
        loaderText.textContent = message;
        loader.style.display = 'block';
        imageContainer.innerHTML = '';
    }

    function hideLoader() {
        loader.style.display = 'none';
    }
    
    function showError(message) {
        loaderText.textContent = message;
        loader.style.display = 'block';
    }

    function clearContainer() {
        imageContainer.innerHTML = '';
        hideLoader();
    }
    
    async function handleResponse(response) {
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
        }
        return response.json();
    }
});

