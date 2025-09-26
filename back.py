import os
import subprocess
import json
import cv2
import numpy as np
from sklearn.cluster import KMeans
from flask import Flask, request, render_template, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from ultralytics import YOLO

# --- Configuration ---
UPLOAD_FOLDER = 'static/uploads'
GENERATED_FOLDER = 'static/generated_renders'
BLENDER_SCRIPT_PATH = 'blender_generator.py'
# Ensure Blender is in your system's PATH, or provide the full path here.
BLENDER_PATH = "blender" 

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['GENERATED_FOLDER'] = GENERATED_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(GENERATED_FOLDER, exist_ok=True)

# Load the YOLO model once at startup
try:
    yolo_model = YOLO('yolov8n.pt')
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    yolo_model = None

# --- Computer Vision Functions ---
def extract_color_palette(image, k=5):
    """Extracts the k dominant colors from an image."""
    pixels = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).reshape(-1, 3)
    # Using a smaller sample for speed
    pixels = pixels[np.random.choice(pixels.shape[0], size=min(5000, pixels.shape[0]), replace=False)]
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto').fit(pixels)
    hex_colors = [f"#{int(c[0]):02x}{int(c[1]):02x}{int(c[2]):02x}" for c in kmeans.cluster_centers_]
    return hex_colors

def analyze_room_style(image):
    """
    SIMULATED: In a real system, this would be a trained CNN classifier.
    Here, we simulate it with a heuristic based on image properties.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clarity = cv2.Laplacian(gray, cv2.CV_64F).var()
    if clarity > 150:
        return "industrial"
    else:
        return "modern"

def inpaint_image(image, mask):
    """Uses a simple inpainting algorithm to 'erase' the old object."""
    # The TELEA algorithm is generally faster and good for this use case
    inpainted_image = cv2.inpaint(image, mask, inpaintRadius=10, flags=cv2.INPAINT_TELEA)
    return inpainted_image

# --- Flask Routes ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect_objects():
    if yolo_model is None:
        return jsonify({'error': 'YOLO model not loaded'}), 500

    file = request.files.get('file')
    if not file:
        return jsonify({'error': 'No file provided'}), 400

    filename = secure_filename(file.filename)
    img_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(img_path)

    results = yolo_model(img_path)
    detections = []
    # COCO class names for common furniture
    furniture_classes = ['chair', 'couch', 'bed', 'dining table', 'sofa', 'potted plant', 'bench']
    
    for r in results:
        for box in r.boxes:
            class_name = yolo_model.names[int(box.cls)]
            if class_name in furniture_classes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detections.append({
                    'box': [x1, y1, x2, y2],
                    'class_name': class_name,
                    'confidence': float(box.conf)
                })

    return jsonify({'image_url': f'/{img_path}', 'detections': detections})

@app.route('/replace', methods=['POST'])
def replace_object():
    data = request.json
    image_path = data['image_path'].lstrip('/')
    box = data['box']
    
    original_image = cv2.imread(image_path)
    
    # --- 1. Scene Analysis ---
    # Create a mask of the rest of the scene for analysis
    scene_mask = np.ones(original_image.shape[:2], dtype="uint8") * 255
    cv2.rectangle(scene_mask, (box[0], box[1]), (box[2], box[3]), 0, -1)
    scene_image = cv2.bitwise_and(original_image, original_image, mask=scene_mask)

    style = analyze_room_style(scene_image)
    colors = extract_color_palette(scene_image)

    # --- 2. Generative Pipeline (Simulated) ---
    generation_params = {
        'style': style,
        'colors': colors,
        'shape_asset_path': 'static/sample_assets/generated_chair.obj', # SIMULATED GAN OUTPUT
        'output_path': os.path.join(GENERATED_FOLDER, f'render_{os.path.basename(image_path)}.png')
    }

    # --- 3. Blender Rendering ---
    try:
        command = [BLENDER_PATH, "--background", "--python", BLENDER_SCRIPT_PATH, "--", json.dumps(generation_params)]
        print(f"Running Blender command: {' '.join(command)}")
        subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError:
        return jsonify({'error': f"Blender not found. Make sure '{BLENDER_PATH}' is correct and in your system PATH."}), 500
    except subprocess.CalledProcessError as e:
        print(f"Blender Error STDOUT: {e.stdout}")
        print(f"Blender Error STDERR: {e.stderr}")
        return jsonify({'error': f'Blender execution failed. Check console for details.'}), 500

    # --- 4. Compositing and In-Painting ---
    object_mask = np.zeros(original_image.shape[:2], dtype="uint8")
    cv2.rectangle(object_mask, (box[0], box[1]), (box[2], box[3]), 255, -1)

    inpainted_bg = inpaint_image(original_image, object_mask)

    rendered_object_rgba = cv2.imread(generation_params['output_path'], cv2.IMREAD_UNCHANGED)
    if rendered_object_rgba is None:
        return jsonify({'error': 'Rendered object could not be read.'}), 500

    box_w, box_h = box[2] - box[0], box[3] - box[1]
    rendered_object_resized = cv2.resize(rendered_object_rgba, (box_w, box_h))

    roi = inpainted_bg[box[1]:box[3], box[0]:box[2]]
    
    # Separate alpha channel and color channels
    alpha = rendered_object_resized[:, :, 3] / 255.0
    rendered_colors = rendered_object_resized[:, :, :3]

    for c in range(0, 3):
        roi[:, :, c] = roi[:, :, c] * (1 - alpha) + rendered_colors[:, :, c] * alpha

    final_image_filename = f'final_{os.path.basename(image_path)}'
    final_image_path = os.path.join(GENERATED_FOLDER, final_image_filename)
    cv2.imwrite(final_image_path, inpainted_bg)

    return jsonify({'final_image_url': f'/{final_image_path}'})

@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)

if __name__ == '__main__':
    app.run(debug=True)

