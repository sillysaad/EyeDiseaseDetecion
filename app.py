import pathlib
import platform
from pathlib import Path

from flask import Flask, request, render_template, jsonify
from PIL import Image
from fastai.vision.all import load_learner

if platform.system() != 'Windows':
    pathlib.WindowsPath = pathlib.PosixPath

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Load the trained model
try:
    model_path = Path(__file__).resolve().parent / 'eye_disease_model.pkl'
    learn = load_learner(model_path)
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
    
    file = request.files.get('image')
    if file is None:
        return jsonify({'error': 'No image provided'}), 400
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        img = Image.open(file).convert('RGB')
        
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
    print("Open https://localhost:5000 in your browser")
    print("On your phone (same Wi-Fi): https://<your-PC-IP>:5000")
    app.run(debug=True, host='0.0.0.0', port=5000, ssl_context='adhoc')
