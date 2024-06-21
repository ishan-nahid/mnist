# import os
# import cv2
# import numpy as np
# import tensorflow as tf
# # from tensorflow.keras.utils import to_categorical
# # from keras.utils import to_categorical

# model = None

# def load_model():
#     global model
#     try:
#         model_path = os.path.join(os.path.dirname(__file__), 'model_filter.h5')
#         model = tf.keras.models.load_model(model_path)
#         print(f"Model loaded successfully from {model_path}")
#     except Exception as e:
#         print(f"Failed to load model: {str(e)}")
#         raise

# def preprocess_image(image_path, resize_dim=32):
#     """Preprocesses the image to be compatible with the model input.
    
#     Args:
#         image_path (str): Path to the image file.
#         resize_dim (int): The dimension to resize the image to.
    
#     Returns:
#         np.array: Preprocessed image array.
#     """
#     img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Load image in grayscale
#     if img is None:
#         raise ValueError(f"Image not found or unable to load: {image_path}")
    
#     if resize_dim is not None:
#         img = cv2.resize(img, (resize_dim, resize_dim), interpolation=cv2.INTER_AREA)  # Resize image

#     gaussian_3 = cv2.GaussianBlur(img, (9, 9), 10.0)  # Apply Gaussian blur
#     img = cv2.addWeighted(img, 1.5, gaussian_3, -0.5, 0, img)

#     kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])  # Sharpen filter
#     img = cv2.filter2D(img, -1, kernel)

#     img = img.astype('float32') / 255.0  # Normalize pixel values
#     img = np.expand_dims(img, axis=-1)  # Add channel dimension
#     img = np.expand_dims(img, axis=0)  # Add batch dimension
    
#     return img

# def predict(image_path):
#     global model
#     if model is None:
#         load_model()
    
#     # Preprocess the image
#     preprocessed_image = preprocess_image(image_path)
    
#     # Make prediction
#     prediction = model.predict(preprocessed_image)
#     predicted_class = np.argmax(prediction, axis=1)[0]
    
#     return predicted_class

# model.py

import tensorflow as tf
import numpy as np
import cv2

# Load your model and other necessary setup
def load_model():
    global model
    model = tf.keras.models.load_model('model_filter.h5')
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

def preprocess_image(image_path):
    # Load image and preprocess it here (similar to what you did in Jupyter notebook)

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    resized_img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA)
    processed_img = np.expand_dims(resized_img, axis=-1) / 255.0  # Normalize and add channel dimension

    return processed_img

def predict(image_path):
    try:
        # Preprocess the image
        processed_img = preprocess_image(image_path)

        # Perform prediction
        prediction = model.predict(np.array([processed_img]))  # Model expects a batch, so wrap in an array

        # Post-process prediction (assuming it's a classification task)
        predicted_class = np.argmax(prediction, axis=1)[0]  # Extract the predicted class index

        return predicted_class
    except Exception as e:
        raise RuntimeError(f"Error predicting image: {str(e)}")
