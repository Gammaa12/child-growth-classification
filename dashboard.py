import streamlit as st
import numpy as np
import cv2
import torch
import torch.nn as nn
import os
from PIL import Image
import plotly.graph_objects as go
from torchvision import transforms

# ============================
# 1. MODEL PREPARATION (Mockup)
# ============================
# Catatan: Ganti bagian ini dengan load_state_dict model asli Anda
def cnn_tf(img):
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return tf(img)

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Linear(10, 2)
    def forward(self, x):
        # Output dummy logit [batch, 2]
        return torch.randn(x.size(0), 2)

cnn_model = SimpleModel()
eff_model = SimpleModel()
cnn_model.eval()
eff_model.eval()

# ============================
# 2. PAGE CONFIG
# ============================
st.set_page_config(
    page_title="GrowthVision | Pediatric Morphometry",
    page_icon="🔬",
    layout="wide"
)

# ============================
# 3. CUSTOM ACADEMIC CSS
# ============================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    
    .main-title {
        font-size: 42px; font-weight: 800; letter-spacing: -1px;
        background: -webkit-linear-gradient(#0e1117, #4e5e7a);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .academic-sub {
        font-size: 15px; color: #4A5568; font-style: italic;
        margin-bottom: 25px; border-left: 4px solid #007BFF; padding-left: 15px;
    }
    .status-card {
        background: #ffffff; border: 1px solid #e9ecef;
        padding: 15px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.02);
    }
    .label-tag {
        display: inline-block; padding: 4px 12px; border-radius: 20px;
        font-size: 11px; font-weight: 700; text-transform: uppercase; margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ============================
# 4. SIDEBAR CONTROL PANEL
# ============================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3069/3069155.png", width=70)
    st.markdown("### 🔬 Research Panel")
    
    model_choice = st.selectbox(
        "Architecture Selection",
        ["EfficientNet + LoRA", "CNN Fine-Tuning", "SVM (Classic)"]
    )
    
    st.divider()
    
    data_source = st.radio(
        "Data Input Method:",
        ["Gunakan Dataset Sampel", "Upload Manual"]
    )

    final_files = []
    if data_source == "Upload Manual":
        uploaded_files = st.file_uploader("Upload Images", type=["jpg","png","jpeg"], accept_multiple_files=True)
        if uploaded_files:
            final_files = uploaded_files
    else:
        sample_folder = "samples"
        if os.path.exists(sample_folder):
            sample_list = [os.path.join(sample_folder, f) for f in os.listdir(sample_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if sample_list:
                st.success(f"✅ {len(sample_list)} gambar sampel siap dianalisis.")
                final_files = sample_list
            else:
                st.warning("Folder 'samples' kosong.")
        else:
            st.error("Folder 'samples/' tidak ditemukan di GitHub.")

    st.divider()
    run_btn = st.button("🚀 RUN INFERENCE", use_container_width=True, type="primary")

# ============================
# 5. MAIN INTERFACE
# ============================
st.markdown("<div class='main-title'>GrowthVision AI</div>", unsafe_allow_html=True)
st.markdown("<div class='academic-sub'>Pediatric Growth Classification System: Morphological Assessment via Deep Learning</div>", unsafe_allow_html=True)

with st.expander("📖 Methodology & Model Architecture"):
    st.write("""
    Dashboard ini menggunakan model **Convolutional Neural Networks (CNN)** dan **EfficientNet-B0** yang telah diadaptasi menggunakan 
    teknik **LoRA (Low-Rank Adaptation)**. Pendekatan ini memungkinkan klasifikasi morfologi wajah anak (VP-0 vs VP-1) 
    dengan efisiensi parameter yang tinggi.
    """)
    

if final_files and run_btn:
    raw_images = []
    tensors = []

    # Progress bar untuk estetika profesional
    progress_bar = st.progress(0)
    
    for i, file in enumerate(final_files):
        img = Image.open(file).convert("RGB")
        raw_images.append(img)
        tensors.append(cnn_tf(img))
        progress_bar.progress((i + 1) / len(final_files))

    x = torch.stack(tensors)

    # --- INFERENCE LOGIC ---
    with torch.no_grad():
        if model_choice == "SVM (Classic)":
            # Mockup SVM probabilities
            probs = np.random.dirichlet(np.ones(2), size=len(raw_images))
        else:
            active_model = eff_model if "Efficient" in model_choice else cnn_model
            logits = active_model(x)
            probs = torch.softmax(logits, dim=1).numpy()

    preds = probs.argmax(axis=1)

    # --- DISPLAY RESULTS ---
    st.markdown("### 📊 Classification Analysis Results")
    
    cols_per_row = 4
    for i in range(0, len(raw_images), cols_per_row):
        cols = st.columns(cols_per_row)
        for j, col in enumerate(cols):
            idx = i + j
            if idx < len(raw_images):
                pred = preds[idx]
                prob = probs[idx]
                label = "VP-0 (Proportional)" if pred == 0 else "VP-1 (Linear)"
                color = "#28a745" if pred == 0 else "#007bff"
                conf = prob[pred] * 100

                with col:
                    st.markdown(f"""
                    <div class="status-card">
                        <div class="label-tag" style="background-color: {color}22; color: {color};">
                            {label.split()[0]}
                        </div>
                    """, unsafe_allow_html=True)
                    
                    st.image(raw_images[idx], use_container_width=True)
                    
                    # Plotly Mini Gauge
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = conf,
                        number = {'suffix': "%", 'font': {'size': 16}, 'color': '#444'},
                        gauge = {
                            'axis': {'range': [0, 100], 'tickwidth': 1},
                            'bar': {'color': color},
                            'bgcolor': "#f8f9fa"
                        }
                    ))
                    fig.update_layout(height=110, margin=dict(l=10, r=10, t=10, b=10))
                    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
                    
                    st.markdown(f"<p style='text-align:center; font-size:13px; font-weight:600; color:{color};'>{label}</p>", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)

elif not final_files and run_btn:
    st.warning("⚠️ Mohon upload gambar atau pastikan folder sampel tersedia.")

else:
    # Landing State
    st.info("💡 Pilih sumber data di sidebar dan klik 'Run Inference' untuk memulai analisis morfologi.")

# ============================
# 6. FOOTER
# ============================
st.markdown("<br><hr>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:grey; font-size:12px;'>© 2024 Pediatric AI Research Framework | Distributed under MIT License</p>", unsafe_allow_html=True)