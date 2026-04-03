from flask import Flask, request, render_template, jsonify
from PIL import Image
import io
import torch
from fastai.vision.all import load_learner
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'

# Create uploads folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the trained model
try:
    learn = load_learner('eye_disease_model.pkl')
    model_loaded = True
except Exception as e:
    print(f"Error loading model: {e}")
    model_loaded = False

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not model_loaded:
        return jsonify({'error': 'Model not loaded'}), 500
    
    # Check if image is in request
    if 'image' not in request.files and 'image_path' not in request.form:
        return jsonify({'error': 'No image provided'}), 400
    
    try:
        # Handle file upload
        if 'image' in request.files:
            file = request.files['image']
            if file.filename == '':
                return jsonify({'error': 'No file selected'}), 400
            
            img = Image.open(file).convert('RGB')
        
        # Handle file path input
        elif 'image_path' in request.form:
            image_path = request.form['image_path']
            if not os.path.exists(image_path):
                return jsonify({'error': 'File not found'}), 400
            img = Image.open(image_path).convert('RGB')
        
        # Make prediction
        pred, pred_idx, probs = learn.predict(img)
        confidence = float(probs[pred_idx]) * 100
        
        # Get all predictions with probabilities
        classes = learn.dls.vocab
        predictions = []
        for i, prob in enumerate(probs):
            predictions.append({
                'disease': classes[i],
                'confidence': float(prob) * 100
            })
        
        # Sort by confidence descending
        predictions.sort(key=lambda x: x['confidence'], reverse=True)
        
        return jsonify({
            'disease': str(pred),
            'confidence': round(confidence, 2),
            'all_predictions': predictions[:3]  # Top 3
        })
    
    except Exception as e:
        return jsonify({'error': f'Error processing image: {str(e)}'}), 400

if __name__ == '__main__':
    print("Starting Eye Disease Classifier Web App...")
    print("Open http://localhost:5000 in your browser")
    app.run(debug=True, port=5000)
