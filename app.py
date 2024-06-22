from flask import Flask, request, render_template, jsonify
from model import predict, load_model
from utils import save_uploaded_file
import os

app = Flask(__name__)

# Load the model once when the application starts
model = None

@app.before_request
def initialize():
    global model
    try:
        model = load_model()
    except Exception as e:
        app.logger.error(f"Error loading model: {str(e)}")
        raise

def handle_image_prediction(image):
    if image.filename == '':
        raise ValueError('No selected image file')
    
    image_path = save_uploaded_file(image)
    try:
        prediction = predict(image_path)  # Pass image_path to predict function
        return int(prediction)  # Convert prediction to int
    finally:
        os.remove(image_path)  # Remove the temporary file

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        return predict_route()
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_route():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file in request'}), 400
        
        image = request.files['image']
        prediction = handle_image_prediction(image)
        return jsonify({'prediction': prediction})
    except ValueError as ve:
        return jsonify({'error': str(ve)}), 400
    except Exception as e:
        app.logger.error(f"Error in predict route: {str(e)}")
        return jsonify({'error': 'An unexpected error occurred'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
