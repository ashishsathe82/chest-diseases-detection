import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
import os
import gdown
from tensorflow.keras.models import Model
from keras.applications.densenet import DenseNet121
from keras.layers import Dense, GlobalAveragePooling2D
from PIL import Image
import matplotlib.pyplot as plt
from util import grad_cam

# Number of classes in the trained model
NUM_CLASSES = 14

# Define model architecture
def build_model():
    base_model = DenseNet121(weights=None, include_top=False, input_shape=(320, 320, 3))
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(NUM_CLASSES, activation="softmax")(x)
    model = Model(inputs=base_model.input, outputs=x)
    return model

MODEL_PATH = "trained_model.h5"
MODEL_URL = "https://drive.google.com/uc?id=1xLlHLjAiXmq5gDbTS7YOCC3E12v5ia5M"


if not os.path.exists(MODEL_PATH):
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

# Load the model and weights
model = build_model()
model.load_weights(MODEL_PATH)

LABELS = ['Cardiomegaly', 'Tuberculosis', 'Effusion', 'Hernia', 'Infiltration', 'Mass', 'Lung_Cancer',
          'Covid', 'Pneumothorax', 'Asthma', 'Pneumonia', 'Heart_Failure', 'Edema', 'Consolidation']

def is_xray_image(image):
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 30, 100)
    return np.mean(edges) < 50



st.title("Chest Disease Detection Using X-Ray")
st.write("Upload a chest X-ray image to detect potential diseases.")

uploaded_file = st.file_uploader("Choose a chest X-ray image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    img = img.resize((320, 320))
    
    if not is_xray_image(img):
        st.error("The uploaded image does not appear to be a valid X-ray. Please upload a proper chest X-ray.")
    else:
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        predictions = model.predict(img_array)[0]
        probabilities = (predictions * 100).round(2)
        sorted_indices = np.argsort(probabilities)[::-1]
        top_3_indices = sorted_indices[:3]

        st.image(img, caption="Uploaded X-Ray", use_column_width=True)

        st.write("### Disease Probabilities:")
        for i in sorted_indices:
            st.write(f"**{LABELS[i]}: {probabilities[i]}%**")

        st.write("## Top 3 Predicted Diseases with Grad-CAM:")
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for idx, disease_idx in enumerate(top_3_indices):
            cam = grad_cam(model, img_array, disease_idx, 'conv5_block16_concat')
            axes[idx].imshow(img, cmap='gray')
            axes[idx].imshow(cam, cmap='jet', alpha=0.5)
            axes[idx].set_title(f"{LABELS[disease_idx]} ({probabilities[disease_idx]}%)")
            axes[idx].axis('off')
        st.pyplot(fig)

st.write("Developed for Chest Disease Detection using DL")
