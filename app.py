import os
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import io

# Load the trained model
model = tf.keras.models.load_model('digit_classifier_cnn.h5')

app = Flask(__name__)

def preprocess_image(image):
    # Convert image to grayscale if it's not already
    image = image.convert('L')  # Convert image to grayscale
    
    # Resize to 28x28 pixels (MNIST size)
    image = image.resize((28, 28))
    
    # Convert image to numpy array
    image = np.array(image)
    
    # Normalize the image to [0, 1]
    image = image / 255.0
    
    # Add batch and channel dimensions: (1, 28, 28, 1)
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    image = np.expand_dims(image, axis=-1)  # Add channel dimension (28, 28, 1)
    
    return image


@app.route('/')
def index():
    return render_template('index.html')
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Preprocess the uploaded image
        image = Image.open(file.stream)
        image = preprocess_image(image)
        
        # Predict the digit
        predictions = model.predict(image)
        predicted_class = predictions.argmax(axis=1)[0]
        
        # Debugging: Check the predictions before returning
        print(f"Predictions: {predictions}")
        print(f"Predicted class: {predicted_class}")
        
        return jsonify({'prediction': str(predicted_class)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
