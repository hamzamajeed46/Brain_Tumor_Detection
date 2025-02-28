import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image

# Load the trained CNN model
model = tf.keras.models.load_model("cnn_model.h5")

# Define class labels
labels = ['Glioma Tumor', 'Meningioma Tumor', 'No Tumor', 'Pituitary Tumor']

# Function to preprocess the image
def preprocess_image(image):
    image = np.array(image)  # Convert to NumPy array
    image = cv2.resize(image, (150, 150))  # Resize to match model input
    image = image / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Streamlit UI
st.title("🧠 Brain Tumor Classification")
st.write("Upload an MRI scan to classify the type of brain tumor.")

# Upload image
uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded MRI Scan", use_column_width=True)

    # Preprocess the image
    processed_image = preprocess_image(image)

    # Make prediction
    prediction = model.predict(processed_image)
    predicted_class = labels[np.argmax(prediction)]
    confidence = np.max(prediction) * 100  # Convert to percentage

    # Show result
    st.success(f"Prediction: **{predicted_class}**")
    st.info(f"Confidence: **{confidence:.2f}%**")

# Run using: streamlit run app.py
