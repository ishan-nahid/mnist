import tensorflow as tf
import numpy as np
import cv2

# Load your model and other necessary setup
def load_model():
    global model
    model = tf.keras.models.load_model('model_filter.h5')
    # model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

def preprocess_image(image_path):
    # Load image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Resize image
    resize_dim = 32
    img = cv2.resize(img, (resize_dim, resize_dim), interpolation=cv2.INTER_AREA)
    
    # Apply Gaussian blur
    gaussian_3 = cv2.GaussianBlur(img, (9, 9), 10.0)
    
    # Sharpen image
    img = cv2.addWeighted(img, 1.5, gaussian_3, -0.5, 0, img)
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    img = cv2.filter2D(img, -1, kernel)
    
    # Normalize and add channel dimension
    img = np.expand_dims(img, axis=-1) / 255.0
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    
    return img

def predict(image_path):
    try:
        # Preprocess the image
        processed_img = preprocess_image(image_path)

        # Perform prediction
        # prediction = model.predict(np.array([processed_img]))  # Model expects a batch, so wrap in an array

        prediction = model.predict(processed_img)
        predicted_class = np.argmax(prediction, axis=1)[0]  # Extract the predicted class index

        # Post-process prediction (assuming it's a classification task)
        # predicted_class = np.argmax(prediction, axis=1)[0]  # Extract the predicted class index

        return predicted_class
    except Exception as e:
        raise RuntimeError(f"Error predicting image: {str(e)}")
