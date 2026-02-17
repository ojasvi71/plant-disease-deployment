import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("plant_disease_cnn_model.h5")

# Image size used during training
img_size = (256, 256)

# Class names
class_names = [
    "Pepper__bell___Bacterial_spot",
    "Tomato__Target_Spot",
    "Tomato_healthy",
    "Potato___healthy",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Potato___Early_blight",
    "Tomato_Early_blight",
    "Tomato_Bacterial_spot",
    "Tomato_Leaf_Mold",
    "Tomato__Tomato_mosaic_virus"
]

st.title("🌿 Plant Disease Detection System")

uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, img_size)

    img_array = np.expand_dims(img, axis=0) / 255.0

    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)

    st.image(img, caption="Uploaded Image", use_column_width=True)
    st.success(f"Prediction: {predicted_class}")
    st.info(f"Confidence: {confidence:.2f}")
