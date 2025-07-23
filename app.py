# Required Libraries
import os
import cv2
import numpy as np
import pandas as pd
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
import torch
from torchvision import models, transforms
from PIL import Image
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF

# Logging config
logging.basicConfig(level=logging.INFO)

# Dataset paths
SOIL_IMAGE_PATH = "D:/z 8th sem/final project/Soil Types datasets"
CORE_DATA_PATH = "D:/z 8th sem/final project/data_core_updated.csv"  # Updated dataset

# Pretrained VGG16 for feature extraction in PyTorch
vgg16 = models.vgg16(pretrained=True)
vgg16.eval()
feature_extractor = torch.nn.Sequential(*list(vgg16.features.children()))

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def preprocess_and_extract(img_path):
    try:
        img = Image.open(img_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0)
        with torch.no_grad():
            features = feature_extractor(img_tensor)
        return features.flatten().numpy()
    except Exception as e:
        logging.error(f"Image processing failed for {img_path}: {e}")
        return None


def load_dataset(main_dir):
    image_paths, labels = [], []
    for soil_type in os.listdir(main_dir):
        sub_dir = os.path.join(main_dir, soil_type)
        if os.path.isdir(sub_dir):
            for fname in os.listdir(sub_dir):
                if fname.lower().endswith(('.jpg', '.png')):
                    image_paths.append(os.path.join(sub_dir, fname))
                    labels.append(soil_type)
    logging.info(f"Loaded {len(image_paths)} images")
    return image_paths, labels


def train_crop_fertilizer_model(core_path):
    df = pd.read_csv(core_path)
    X = df[['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Phosphorous', 'Potassium', 'Soil Type']]
    y_crop = df['Crop Type']
    y_fert = df['Fertilizer Name']

    X['Soil Type'] = X['Soil Type'].astype(str).str.strip().str.lower()
    le_soil = LabelEncoder()
    X['Soil Type'] = le_soil.fit_transform(X['Soil Type'])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    le_crop = LabelEncoder()
    le_fert = LabelEncoder()
    y_crop_enc = le_crop.fit_transform(y_crop)
    y_fert_enc = le_fert.fit_transform(y_fert)

    crop_model = RandomForestClassifier()
    fert_model = RandomForestClassifier()

    crop_model.fit(X_scaled, y_crop_enc)
    fert_model.fit(X_scaled, y_fert_enc)

    return crop_model, fert_model, scaler, le_crop, le_fert, le_soil


def predict_crop_and_fertilizer(models, inputs):
    crop_model, fert_model, scaler, le_crop, le_fert, le_soil = models

    try:
        soil_input_normalized = str(inputs[-1]).strip().lower()
        if soil_input_normalized not in le_soil.classes_:
            st.error(f"Soil type '{inputs[-1]}' not found in training data. Please ensure soil type is consistent.")
            return "Unknown", "Unknown"
        encoded_soil = le_soil.transform([soil_input_normalized])[0]
    except ValueError:
        st.error(f"Soil type '{inputs[-1]}' not found in training data. Please ensure soil type is consistent.")
        return "Unknown", "Unknown"

    X_input = scaler.transform([inputs[:-1] + [encoded_soil]])
    crop_pred = le_crop.inverse_transform(crop_model.predict(X_input))[0]
    fert_pred = le_fert.inverse_transform(fert_model.predict(X_input))[0]
    return crop_pred, fert_pred


def generate_pdf(soil_type, crop, fert, location):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=14)
    pdf.cell(200, 10, txt="Soil Analysis Report", ln=True, align='C')
    pdf.ln(10)
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Location: {location}", ln=True)
    pdf.cell(200, 10, txt=f"Predicted Soil Type: {soil_type}", ln=True)
    pdf.cell(200, 10, txt=f"Recommended Crop: {crop}", ln=True)
    pdf.cell(200, 10, txt=f"Recommended Fertilizer: {fert}", ln=True)
    pdf.output("soil_report.pdf")


# Streamlit UI
st.title("🌱 Soil Type Classifier & Crop + Fertilizer Recommender 🌾")

uploaded_file = st.file_uploader("📤 Upload a soil image", type=['jpg', 'png'])
location = st.text_input("📍 Enter your location (City, State, or Area)", "")

temp = st.number_input("🌡️ Temperature (°C)", 0.0, 100.0)
hum = st.number_input("💧 Humidity (%)", 0.0, 100.0)
moist = st.number_input("🧪 Moisture (%)", 0.0, 100.0)
n = st.number_input("🧬 Nitrogen", 0, 1000)
p = st.number_input("🧬 Phosphorus", 0, 1000)
k = st.number_input("🧬 Potassium", 0, 1000)

if uploaded_file:
    with open("temp.jpg", "wb") as f:
        f.write(uploaded_file.read())
    st.image("temp.jpg", caption='Uploaded Image', use_column_width=True)

    feat = preprocess_and_extract("temp.jpg")
    if feat is not None:
        paths, labels = load_dataset(SOIL_IMAGE_PATH)
        X = [preprocess_and_extract(p) for p in paths if preprocess_and_extract(p) is not None]
        le = LabelEncoder()
        y = le.fit_transform(labels)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        model = RandomForestClassifier()
        model.fit(X_train, y_train)

        # NEW: Calculate accuracy
        soil_accuracy = model.score(X_test, y_test)

        feat_scaled = scaler.transform([feat])
        pred = model.predict(feat_scaled)
        soil_name = le.inverse_transform(pred)[0].lower()

        st.success(f"🌍 Predicted Soil Type: {soil_name}")

        # NEW: Show model accuracy
        st.info(f"✅ Soil Model Accuracy: {soil_accuracy * 100:.2f}%")

        models = train_crop_fertilizer_model(CORE_DATA_PATH)
        user_input = [temp, hum, moist, n, p, k, soil_name]
        crop, fert = predict_crop_and_fertilizer(models, user_input)

        st.info(f"🌾 Recommended Crop: {crop}")
        st.info(f"🧪 Recommended Fertilizer: {fert}")

        if st.button("📥 Download PDF Report"):
            generate_pdf(soil_name, crop, fert, location)
            with open("soil_report.pdf", "rb") as pdf_file:
                st.download_button(label="Download Report", data=pdf_file, file_name="soil_report.pdf")
