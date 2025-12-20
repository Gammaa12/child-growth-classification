import streamlit as st
import numpy as np
import cv2
import torch
import torch.nn as nn
from PIL import Image
import plotly.graph_objects as go
from torchvision import transforms

# ============================
# MOCKUP MODELS (Agar Kode Bisa Running)
# ============================
# Di lingkungan produksi, Anda akan memuat model asli Anda di sini.
def cnn_tf(img):
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    return tf(img)

# Placeholder untuk model-model Anda
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 1, 3)
        self.fc = nn.Linear(1, 2)
    def forward(self, x):
        return torch.randn(x.size(0), 2)

cnn_model = SimpleCNN()
eff_model = SimpleCNN()

# ============================
# PAGE CONFIG
# ============================
st.set_page_config(
    page_title="GrowthVision | Child Growth Classification",
    page_icon="🔬",
    layout="wide"
)

# ============================
# PROFESSIONAL ACADEMIC CSS
# ============================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main-title {
        font-size: 42px;
        font-weight: 800;
        letter-spacing: -1px;
        background: -webkit-linear-gradient(#0e1117, #4e5e7a);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .academic-sub {
        font-size: 15px;
        color: #4A5568;
        font-style: italic;
        margin-bottom: 25px;
        border-left: 4px solid #007BFF;
        padding-left: 15px;
    }

    .status-card {
        background-color: #ffffff;
        border: 1px solid #e9ecef;
        padding: 15px;
        border-radius: 12px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
    }

    .label-tag {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ============================
# SIDEBAR
# ============================
with st.sidebar:
    st.markdown("### 🔬 Research Panel")
    model_choice = st.selectbox(
        "Model Architecture",
        ["EfficientNet + LoRA", "CNN Fine-Tuning", "SVM (Classic)"]
    )
    
    uploaded_files = st.file_uploader(
        "Upload Subject Morphologies",
        type=["jpg","png","jpeg"],
        accept_multiple_files=True
    )
    
    run_btn = st.button("🚀 ANALYZE DATASET", use_container_width=True, type="primary")
    
    st.divider()
    st.caption("Developed for Academic Purposes | v1.0.2")

# ============================
# MAIN CONTENT
# ============================
row1_1, row1_2 = st.columns([2, 1])
with row1_1:
    st.markdown("<div class='main-title'>GrowthVision AI</div>", unsafe_allow_html=True)
    st.markdown("<div class='academic-sub'>Pediatric Growth Classification System based on Facial Morphometry</div>", unsafe_allow_html=True)

if uploaded_files and run_btn:
    raw_images = []
    processed_tensors = []

    for file in uploaded_files:
        img = Image.open(file).convert("RGB")
        raw_images.append(img)
        processed_tensors.append(cnn_tf(img))

    x = torch.stack(processed_tensors)

    # Inisialisasi Prediksi (Inference Logic)
    if model_choice == "CNN Fine-Tuning":
        with torch.no_grad():
            out = cnn_model(x)
            probs = torch.softmax(out, 1).numpy()
    elif model_choice == "EfficientNet + LoRA":
        with torch.no_grad():
            out = eff_model(x)
            probs = torch.softmax(out, 1).numpy()
    else: # SVM Mockup
        probs = np.random.dirichlet(np.ones(2), size=len(raw_images))

    preds = probs.argmax(axis=1)

    # Result Display
    st.markdown("### 📊 Classification Analysis")
    
    # Grid system
    cols_per_row = 4
    for i in range(0, len(raw_images), cols_per_row):
        cols = st.columns(cols_per_row)
        for j, col in enumerate(cols):
            idx = i + j
            if idx < len(raw_images):
                img = raw_images[idx]
                pred = preds[idx]
                prob = probs[idx]
                
                label = "VP-0 (Proportional)" if pred == 0 else "VP-1 (Linear)"
                color = "#28a745" if pred == 0 else "#007bff"
                conf = prob[pred] * 100

                with col:
                    st.markdown(f"""
                    <div class="status-card">
                        <div class="label-tag" style="background-color: {color}22; color: {color};">
                            Class: {label.split()[0]}
                        </div>
                    """, unsafe_allow_html=True)
                    
                    st.image(img, use_container_width=True)
                    
                    # Plotly Mini Gauge
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = conf,
                        number = {'suffix': "%", 'font': {'size': 16}},
                        gauge = {'axis': {'range': [0, 100]}, 'bar': {'color': color}}
                    ))
                    fig.update_layout(height=100, margin=dict(l=10, r=10, t=10, b=10))
                    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
                    
                    st.markdown(f"<p style='text-align:center; font-size:13px; font-weight:600;'>{label}</p>", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)

elif not uploaded_files:
    st.info("💡 Awaiting input. Please upload clinical images in the sidebar to begin analysis.")