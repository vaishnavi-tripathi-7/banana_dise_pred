import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os

# -----------------------------
# Model configuration
# -----------------------------
MODEL_FILE = "my_cnn_model4.keras"
GOOGLE_DRIVE_FILE_ID = "19ondqnTkzrM07XS1TCtLxuE44fE7BdYC"

# -----------------------------
# Download model if it doesn't exist
# -----------------------------
if not os.path.exists(MODEL_FILE):
    st.info("Downloading model...")
    url = f"https://drive.google.com/uc?export=download&id={GOOGLE_DRIVE_FILE_ID}"
    gdown.download(url, MODEL_FILE, quiet=False, fuzzy=True)
    st.success("Model downloaded!")

# -----------------------------
# Load the model
# -----------------------------
try:
    model = tf.keras.models.load_model(MODEL_FILE)
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Failed to load model. Error: {e}")
    st.stop()  # Stop app if model can't load

# -----------------------------
# Class names
# -----------------------------
CLASS_NAMES = [
    "Augmented Banana Black Sigatoka Disease",
    "Augmented Banana Bract Mosaic Virus Disease",
    "Augmented Banana Healthy Leaf",
    "Augmented Banana Insect Pest Disease",
    "Augmented Banana Moko Disease",
    "Augmented Banana Panama Disease",
    "Augmented Banana Yellow Sigatoka Disease"
]

# -----------------------------
# App title
# -----------------------------
st.title("🍌 Banana Leaf Disease Classifier")
st.write("Upload a banana leaf image and get a disease prediction.")

# -----------------------------
# Upload image
# -----------------------------
uploaded_file = st.file_uploader("Choose a banana leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open and display image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img_size = (224, 224)
    img_array = np.array(image.resize(img_size)) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Make prediction
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    # Display prediction
    st.subheader("Prediction")
    st.write(f"**Class:** {CLASS_NAMES[np.argmax(score)]}")
    st.write(f"**Confidence:** {100 * np.max(score):.2f}%")








