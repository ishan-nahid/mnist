from flask import Flask, request, render_template, jsonify
from model import predict, load_model
from utils import save_uploaded_file
import os

app = Flask(__name__)

model = None

@app.before_request
def initialize():
    global model
    try:
        model = load_model()
    except Exception as e:
        app.logger.error(f"Error loading model: {str(e)}")

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        return predict_route()
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_route():
    if 'image' not in request.files:
        return render_template('result.html', prediction="No image file uploaded")
    
    image = request.files['image']
    if image.filename == '':
        return render_template('result.html', prediction="No selected file")
    
    
    try:
        image_path = save_uploaded_file(image)
        
        verdict_map = predict(image_path)

        os.remove(image_path)

        return render_template('result.html', verdict_map=verdict_map)

    except Exception as e:
        app.logger.error(f"Error in predict route: {str(e)}")
        return render_template('result.html', prediction=str(e))

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)