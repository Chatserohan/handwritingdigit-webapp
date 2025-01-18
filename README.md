
```markdown
# Digit Classifier Web App

This project implements a simple digit classifier web app using a Convolutional Neural Network (CNN) model trained on the MNIST dataset. The app allows users to either upload an image or draw a digit on the canvas, and it predicts the digit using a pre-trained model.

## Features

- **Canvas Drawing**: Users can draw digits directly on the canvas.
- **Image Upload**: Users can upload images of handwritten digits for prediction.
- **Predictions**: The app uses a trained CNN model to predict the digit and display the result.

## Requirements

To run this project, you'll need to have the following Python packages installed:

- `Flask` - Web framework to serve the app.
- `TensorFlow` - For the trained model and predictions.
- `Pillow` - For image processing (resizing, converting, etc.).
- `NumPy` - For numerical operations.
- `OpenCV` - For handling image operations (if you plan to include real-time camera functionality).

You can install the dependencies using `pip`:

```bash
pip install Flask tensorflow Pillow numpy opencv-python
```

## Model

This web app uses a **Convolutional Neural Network (CNN)** model trained on the **MNIST dataset**. The MNIST dataset contains 28x28 grayscale images of handwritten digits (0-9). The model predicts the digit by processing the image and returning the most likely class (digit).

### Training the Model

If you want to train the model yourself, you can use the following Python script:

```python
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Load and preprocess data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Add a channel dimension (grayscale images)
x_train = x_train[..., np.newaxis]
x_test = x_test[..., np.newaxis]

# Build the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')  # Output layer for 10 classes (digits 0-9)
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=5)

# Save the model
model.save('digit_classifier_cnn.h5')
```

Once the model is trained, save it to a file (`digit_classifier_cnn.h5`) and use it for predictions in the web app.

## Project Structure

The project structure is as follows:

```
digit-classifier-webapp/
├── app.py              # Main Flask app
├── static/
│   └── style.css       # CSS for styling the page
├── templates/
│   └── index.html      # HTML page for the web app
├── digit_classifier_cnn.h5  # Pre-trained model file
├── README.md           # This README file
└── requirements.txt    # List of required Python packages
```

### 1. `app.py`

This is the main Flask app that serves the frontend and handles prediction requests. It loads the pre-trained model and handles image preprocessing and prediction.

### 2. `templates/index.html`

This HTML file contains the frontend code for the web app, including the canvas for drawing digits, the image upload button, and the predict and clear buttons.

### 3. `static/style.css`

CSS styles for the web app layout and design. It includes styling for the image upload input, canvas, and buttons.

### 4. `requirements.txt`

List of required Python dependencies for the project. You can install the dependencies by running:

```bash
pip install -r requirements.txt
```

## How to Run

To run the web app, follow these steps:

1. Clone or download the project repository.
2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Start the Flask app:

   ```bash
   python app.py
   ```

4. Open your browser and navigate to `http://127.0.0.1:5000/` to access the web app.

5. You can now either draw a digit on the canvas or upload an image of a handwritten digit. The model will predict the digit and display the result on the page.

## Usage

- **Drawing**: You can draw a digit on the canvas using your mouse or touch screen. Click the "Predict Digit" button to get the prediction.
- **Image Upload**: Click the "Choose an Image" button to upload an image of a digit, and click the "Predict Digit" button to get the prediction.

## Troubleshooting

1. **Model is always predicting `1`**:
   - Ensure that the model is correctly trained and loaded.
   - Check that the input image is properly preprocessed (normalized and resized).
   
2. **Canvas drawing not working**:
   - Ensure that the canvas is properly set up and drawing is enabled with mouse events.
   - Check the `clearCanvas` functionality to ensure it only clears the canvas when clicked.

## License

This project is open source and available under the [MIT License](LICENSE).

---

Feel free to contribute to this project or modify it according to your needs!
```

### How to Use the README:

1. **Copy the contents of this README** into a `README.md` file in your project directory.
2. **Ensure all sections match your project structure**. For example, if you're using a different model or different file names, update the `README` accordingly.

Name: Rohan Chatse
Email: rohancrchatse@gmail.com
Website link: www.chatse.in
