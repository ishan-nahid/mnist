# from fastapi import FastAPI
# from pydantic import BaseModel
# # from app.model.model import predict_pipeline
# # from app.model.model import __version__ as model_version


# app = FastAPI()


# # class TextIn(BaseModel):
# #     text: str


# # class PredictionOut(BaseModel):
# #     language: str


# @app.get("/")
# def home():
#     return {"health_check": "OK"}


# # @app.post("/predict", response_model=PredictionOut)
# # def predict(payload: TextIn):
# #     language = predict_pipeline(payload.text)
# #     return {"language": language}



#########################################################################
# from flask import Flask, request, render_template
# from model import predict, load_model
# from utils import save_uploaded_file
# import os
# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# from flask import Flask, request, render_template
# from model import predict, load_model
# from utils import save_uploaded_file
# # Rest of the imports and code...


# app = Flask(__name__)

# # Initialize model on the first request
# model_initialized = False

# @app.before_request
# def initialize():
#     global model_initialized
#     if not model_initialized:
#         try:
#             load_model()
#             model_initialized = True
#         except Exception as e:
#             app.logger.error(f"Error loading model: {str(e)}")

# @app.route('/', methods=['GET'])
# def index():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict_route():
#     try:
#         if 'image' not in request.files:
#             return {'error': 'No image file in request'}, 400
        
#         image = request.files['image']
#         if image.filename == '':
#             return {'error': 'No selected image file'}, 400
        
#         image_path = save_uploaded_file(image)
#         prediction = predict(image_path)
#         os.remove(image_path)  # Remove the temporary file
#         return {'prediction': prediction}
#     except Exception as e:
#         app.logger.error(f"Error in predict_route: {str(e)}")
#         return {'error': str(e)}, 400

# if __name__ == '__main__':
#     app.run(debug=True)



############################################################################
# app.py

# app.py

from flask import Flask, request, render_template, jsonify
from model import predict, load_model
from utils import save_uploaded_file
import os

app = Flask(__name__)

@app.before_request
def initialize():
    try:
        load_model()
    except Exception as e:
        print(f"Error loading model: {str(e)}")

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            if 'image' not in request.files:
                return jsonify({'error': 'No image file in request'}), 400
            
            image = request.files['image']
            if image.filename == '':
                return jsonify({'error': 'No selected image file'}), 400
            
            image_path = save_uploaded_file(image)
            prediction = predict(image_path)  # Pass image_path to predict function
            os.remove(image_path)  # Remove the temporary file
            return jsonify({'prediction': int(prediction)})  # Convert prediction to int before jsonify
        except Exception as e:
            app.logger.error(f"Error in predict_route: {str(e)}")
            return jsonify({'error': str(e)}), 400
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_route():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file in request'}), 400
        
        image = request.files['image']
        if image.filename == '':
            return jsonify({'error': 'No selected image file'}), 400
        
        image_path = save_uploaded_file(image)
        prediction = predict(image_path)  # Pass image_path to predict function
        os.remove(image_path)  # Remove the temporary file
        return jsonify({'prediction': int(prediction)})  # Convert prediction to int before jsonify
    except Exception as e:
        app.logger.error(f"Error in predict route: {str(e)}")
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
