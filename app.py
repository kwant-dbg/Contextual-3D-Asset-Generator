# -*- coding: utf-8 -*-
"""
Flask Backend for the Contextual 3D Asset Generator

This script handles the web server logic, including:
- Serving the main HTML page.
- Receiving image uploads.
- Running the YOLOv8 object detection model.
- Orchestrating the AI asset generation and compositing pipeline.
- Calling the Blender rendering script.
"""

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
# Define paths for storing uploaded and generated files.
UPLOAD_FOLDER = 'static/uploads'
GENERATED_FOLDER = 'static/generated_renders'
# Path to the Blender script that will be executed.
BLENDER_SCRIPT_PATH = 'blender_generator.py'
# Command to execute Blender.
# Ensure Blender is in your system's PATH, or provide the full absolute path.
# Example for Windows: "C:\\Program Files\\Blender Foundation\\Blender 3.0\\blender.exe"
BLENDER_PATH = "blender" 

# --- Flask App Initialization ---
app = Flask(__name__, template_folder='templates', static_folder='static')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['GENERATED_FOLDER'] = GENERATED_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 # 16 MB upload limit

# Create necessary directories if they don't exist.
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(GENERATED_FOLDER, exist_ok=True)

# --- Model Loading ---
# Load the YOLOv8 model once at startup for efficiency.
try:
    yolo_model = YOLO('yolov8n.pt')
    print("YOLOv8 model loaded successfully.")
except Exception as e:
    print(f"CRITICAL: Error loading YOLO model: {e}")
    yolo_model = None

# --- Computer Vision & Analysis Functions ---

def extract_color_palette(image, k=5):
    """
    Extracts the k dominant colors from an image using K-Means clustering.
    
    Args:
        image (np.array): The input image in BGR format.
        k (int): The number of dominant colors to extract.
        
    Returns:
        list: A list of hex color strings.
    """
    # Convert image to RGB and reshape for K-Means
    pixels = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).reshape(-1, 3)
    
    # To speed up clustering, we'll use a random sample of pixels.
    sample_size = min(5000, pixels.shape[0])
    pixels_sample = pixels[np.random.choice(pixels.shape[0], size=sample_size, replace=False)]
    
    # Perform K-Means clustering
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto').fit(pixels_sample)
    
    # Convert cluster centers to hex codes
    hex_colors = [f"#{int(c[0]):02x}{int(c[1]):02x}{int(c[2]):02x}" for c in kmeans.cluster_centers_]
    return hex_colors

def analyze_room_style(image):
    """
    SIMULATED: In a real-world application, this would be a trained CNN
    for style classification (e.g., modern, industrial, vintage).
    
    Here, we simulate it with a simple heuristic based on image clarity/texture.
    A high variance in the Laplacian suggests more edges and texture (industrial),
    while a lower variance suggests smoother surfaces (modern).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clarity = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    if clarity > 150:
        return "industrial"
    else:
        return "modern"

def inpaint_image(image, mask):
    """
    Uses OpenCV's inpainting algorithm to "erase" the selected object from the scene.
    
    Args:
        image (np.array): The original image.
        mask (np.array): A binary mask where the object to be removed is white.
        
    Returns:
        np.array: The image with the masked area inpainted.
    """
    # The TELEA algorithm is generally fast and effective for this kind of task.
    inpainted_image = cv2.inpaint(image, mask, inpaintRadius=10, flags=cv2.INPAINT_TELEA)
    return inpainted_image

# --- Flask Routes ---

@app.route('/')
def index():
    """Serves the main page of the application."""
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect_objects():
    """
    Handles image uploads, runs object detection, and returns results.
    """
    if yolo_model is None:
        return jsonify({'error': 'Object detection model is not available.'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request.'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file.'}), 400

    if file:
        filename = secure_filename(file.filename)
        img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(img_path)

        try:
            # Run YOLOv8 model on the uploaded image
            results = yolo_model(img_path)
            detections = []
            
            # We are interested in common furniture items.
            # These are based on the COCO dataset class names.
            furniture_classes = ['chair', 'couch', 'bed', 'dining table', 'sofa', 'potted plant', 'bench', 'table']
            
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
            
            # Return the path to the image and the list of detections
            return jsonify({'image_url': f'/{img_path}', 'detections': detections})
        except Exception as e:
            print(f"Error during object detection: {e}")
            return jsonify({'error': 'Failed to process the image.'}), 500
            
    return jsonify({'error': 'An unknown error occurred.'}), 500

@app.route('/replace', methods=['POST'])
def replace_object():
    """
    Handles the main logic for replacing a selected object.
    """
    data = request.json
    image_path = data.get('image_path', '').lstrip('/')
    box = data.get('box')

    if not os.path.exists(image_path) or not box:
        return jsonify({'error': 'Invalid image path or bounding box.'}), 400
    
    original_image = cv2.imread(image_path)
    
    # --- 1. Scene Analysis ---
    # We analyze the scene *without* the object to be replaced.
    # A mask is created to isolate the rest of the room.
    scene_mask = np.ones(original_image.shape[:2], dtype="uint8") * 255
    cv2.rectangle(scene_mask, (box[0], box[1]), (box[2], box[3]), 0, -1)
    scene_image = cv2.bitwise_and(original_image, original_image, mask=scene_mask)

    style = analyze_room_style(scene_image)
    colors = extract_color_palette(scene_image)

    # --- 2. Generative Pipeline (Simulated) ---
    # In a real system, these parameters would feed into generative models.
    # Here, we use them to drive a procedural generation in Blender.
    output_filename = f'render_{os.path.basename(image_path)}'
    generation_params = {
        'style': style,
        'colors': colors,
        'shape_asset_path': 'static/sample_assets/generated_chair.obj', # SIMULATED: Path to a pre-generated mesh
        'output_path': os.path.join(app.config['GENERATED_FOLDER'], output_filename)
    }

    # --- 3. Blender Rendering ---
    # Execute the Blender script in the background with the generated parameters.
    try:
        command = [
            BLENDER_PATH, 
            "--background", 
            "--python", 
            BLENDER_SCRIPT_PATH, 
            "--", 
            json.dumps(generation_params)
        ]
        print(f"Executing Blender: {' '.join(command)}")
        # Using capture_output=True to get stdout/stderr, which is useful for debugging.
        result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=120)
        print(f"Blender STDOUT: {result.stdout}")
    except FileNotFoundError:
        error_msg = f"Blender executable not found at '{BLENDER_PATH}'. Please check the path in app.py."
        print(f"ERROR: {error_msg}")
        return jsonify({'error': error_msg}), 500
    except subprocess.CalledProcessError as e:
        error_msg = 'Blender script execution failed. See server console for details.'
        print(f"ERROR: Blender execution failed. Return code: {e.returncode}")
        print(f"Blender STDERR: {e.stderr}")
        return jsonify({'error': error_msg}), 500
    except subprocess.TimeoutExpired:
        error_msg = 'Blender rendering timed out. The scene might be too complex.'
        print("ERROR: Blender timeout.")
        return jsonify({'error': error_msg}), 500

    # --- 4. Compositing and In-Painting ---
    # Create a mask for the original object's location.
    object_mask = np.zeros(original_image.shape[:2], dtype="uint8")
    cv2.rectangle(object_mask, (box[0], box[1]), (box[2], box[3]), 255, -1)

    # Inpaint the background where the original object was.
    inpainted_bg = inpaint_image(original_image, object_mask)

    # Load the newly rendered object (it should have a transparent background).
    rendered_object_rgba = cv2.imread(generation_params['output_path'], cv2.IMREAD_UNCHANGED)
    if rendered_object_rgba is None:
        return jsonify({'error': 'Failed to read the rendered object image.'}), 500

    # Resize the rendered object to fit the original bounding box.
    box_w, box_h = box[2] - box[0], box[3] - box[1]
    rendered_object_resized = cv2.resize(rendered_object_rgba, (box_w, box_h))

    # Define the Region of Interest (ROI) in the background.
    roi = inpainted_bg[box[1]:box[3], box[0]:box[2]]
    
    # Alpha-blend the rendered object onto the inpainted background.
    alpha = rendered_object_resized[:, :, 3] / 255.0
    rendered_colors = rendered_object_resized[:, :, :3]

    for c in range(3):
        roi[:, :, c] = roi[:, :, c] * (1 - alpha) + rendered_colors[:, :, c] * alpha

    # Save the final composited image.
    final_image_filename = f'final_{os.path.basename(image_path)}'
    final_image_path = os.path.join(app.config['GENERATED_FOLDER'], final_image_filename)
    cv2.imwrite(final_image_path, inpainted_bg)

    return jsonify({'final_image_url': f'/{final_image_path}'})

# This route is used for serving static files like images and JS
@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)

if __name__ == '__main__':
    # Set debug=False for production environments
    app.run(debug=True, host='0.0.0.0')

