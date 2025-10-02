Contextual 3D Asset Generator

This is an advanced proof-of-concept for an AI-driven interior design tool. The system analyzes a user's room photo, allows them to select an existing piece of furniture, and uses a series of generative models to replace it with a unique, AI-generated 3D asset that is stylistically and contextually appropriate.

The project demonstrates an end-to-end pipeline, from computer vision scene understanding to generative 3D modeling, texturing, and final photorealistic composition.
Core Novelty & Technical Depth

This project goes far beyond simple 3D model placement. Its innovation lies in the synergy of multiple AI models to perform a creative task:

    Interactive Scene Analysis: Uses an object detection model (YOLOv8) to identify and isolate existing furniture, making the scene interactive.

    AI-Driven Shape Generation (Simulated): Instead of using pre-made models, the system's architecture is built to use a 3D Generative Adversarial Network (GAN) or Variational Autoencoder (VAE) to create novel mesh geometry based on a style prompt (e.g., "modern chair").

    Generative Texture Synthesis (Simulated): A StyleGAN-based texture model generates unique, seamless textures (e.g., fabric, wood) that harmonize with the color palette of the user's room.

    Contextual In-Painting: The original object is digitally removed from the photo, and the new, AI-generated object is rendered and composited back into the scene with appropriate lighting and shadows.

 Tech Stack

    Backend: Python, Flask

    Computer Vision: ultralytics (for YOLOv8 Object Detection), OpenCV, Scikit-learn

    3D Rendering Engine: Blender Python API (bpy)

    Frontend: JavaScript, HTML5

    Generative Models (Simulated): The architecture is designed for models like ShapeGAN (for meshes) and StyleGAN2 (for textures).

 How to Run
Prerequisites

    You must have Blender installed and its executable available in your system's PATH.

    You have a sample .obj file located at static/sample_assets/generated_chair.obj to simulate the output of the shape generation model.

Setup

    Clone the repository and navigate into it.

    Create and activate a Python virtual environment.

    Install dependencies:

    pip install -r requirements.txt

    Run the Flask application:

    python app.py

    Open your browser to http://127.0.0.1:5000. Upload a room image containing a chair or table to see the replacement workflow.

 Detailed Workflow

    Upload & Detect: The user uploads a room photo. The frontend sends it to the Flask backend, which runs a YOLOv8 model to detect furniture and returns bounding boxes.

    User Selection: The frontend draws the bounding boxes on the image, making them clickable. The user selects an object to replace.

    AI Generation Pipeline (Backend):

        Style Analysis: A classification model analyzes the scene (minus the selected object) to determine the room's style.

        Color Extraction: K-Means clustering extracts the dominant color palette.

        Shape Generation (Simulated): The system calls a (simulated) 3D GAN with the object type and style. For this demo, it loads generated_chair.obj.

        Texture Generation (Simulated): A (simulated) texture GAN is called with the style and color palette. For this demo, a procedural texture is created.

    Rendering & Compositing:

        The blender_generator.py script is executed. It loads the generated shape, applies the generated texture, and renders the object with realistic lighting.

        The app.py script uses the bounding box to create a mask, "in-paints" the original image to remove the old object, and then composites the newly rendered 3D object on top.

    Result: The final, modified image is sent back to the user, showing their room with a new, unique piece of AI-designed furniture.
