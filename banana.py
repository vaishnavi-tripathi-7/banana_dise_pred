import os
import gdown
import numpy as np
import tensorflow as tf
from PIL import Image
import streamlit as st

# Constants
MODEL_FILE = "my_cnn_model_v3.keras"
GOOGLE_DRIVE_FILE_ID = "1IDw8g1dZps0LwDQBn_j_DjJLjcYywqgI"
MODEL_SAVE_PATH = "/content/drive/MyDrive/my_cnn_model_v3.keras"

# Class names
CLASS_NAMES = [
    "Augmented Banana Black Sigatoka Disease",
    "Augmented Banana Bract Mosaic Virus Disease",
    "Augmented Banana Healthy Leaf",
    "Augmented Banana Insect Pest Disease",
    "Augmented Banana Moko Disease",
    "Augmented Banana Panama Disease",
    "Augmented Banana Yellow Sigatoka Disease"
]

# Function to download model if not exists
def download_model():
    if not os.path.exists(MODEL_FILE):
        st.info("Downloading model...")
        url = f"https://drive.google.com/uc?export=download&id={GOOGLE_DRIVE_FILE_ID}"
        gdown.download(url, MODEL_FILE, quiet=False, fuzzy=True)
        st.success("Model downloaded successfully!")

# Function to load the model
def load_model():
    try:
        model = tf.keras.models.load_model(MODEL_FILE)
        st.success("Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Failed to load model. Error: {e}")
        return None

# Streamlit UI
st.title("🍌 Banana Leaf Disease Classifier")
st.write("Upload a banana leaf image and get a disease prediction.")

# Upload image
uploaded_file = st.file_uploader("Choose a banana leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img_size = (224, 224)
    img_array = np.array(image.resize(img_size)) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Load model and predict
    download_model()
    model = load_model()
    if model:
        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])

        st.subheader("Prediction")
        st.write(f"**Class:** {CLASS_NAMES[np.argmax(score)]}")
        st.write(f"**Confidence:** {100 * np.max(score):.2f}%")
