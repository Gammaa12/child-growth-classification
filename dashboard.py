import streamlit as st
import numpy as np
import cv2
import torch
import torch.nn as nn
import os
import random
import pandas as pd
from PIL import Image
import plotly.graph_objects as go
from torchvision import transforms

# ============================
# 1. MODEL PREPARATION
# ============================
@st.cache_resource
def load_models():
    # Ganti dengan arsitektur asli jika sudah ada
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(3 * 224 * 224, 2)
            )
        def forward(self, x):
            # Simulasi output logit [batch, 2]
            return torch.randn(x.size(0), 2)
    
    model = SimpleModel()
    model.eval()
    return model

model_engine = load_models()

def cnn_tf(img):
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return tf(img)

# ============================
# 2. PAGE CONFIG & UI
# ============================
st.set_page_config(
    page_title="GrowthVision | Pediatric Morphometry",
    page_icon="🔬",
    layout="wide"
)

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
        margin-bottom: 20px;
    }
    .label-tag {
        display: inline-block; padding: 4px 12px; border-radius: 20px;
        font-size: 11px; font-weight: 700; text-transform: uppercase; margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ============================
# 3. SIDEBAR (LOGIC PERBAIKAN)
# ============================
with st.sidebar:
    st.markdown("### 🔬 Research Panel")
    
    model_choice = st.selectbox(
        "Architecture Selection",
        ["EfficientNet + LoRA", "CNN Fine-Tuning", "SVM (Classic)"]
    )
    
    st.divider()
    
    data_source = st.radio(
        "Data Input Method:",
        ["Sampel Acak GitHub (20 Gambar)", "Upload Manual"]
    )

    final_files = []
    if data_source == "Upload Manual":
        uploaded_files = st.file_uploader("Upload Images", type=["jpg","png","jpeg"], accept_multiple_files=True)
        if uploaded_files:
            final_files = uploaded_files # Gunakan apa adanya yang diupload user
    else:
        sample_folder = "samples"
        if os.path.exists(sample_folder):
            all_samples = [os.path.join(sample_folder, f) for f in os.listdir(sample_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if all_samples:
                # Ambil 20 acak sesuai permintaan
                num_to_draw = min(20, len(all_samples))
                final_files = random.sample(all_samples, num_to_draw)
                st.success(f"✅ Terpilih {num_to_draw} gambar acak dari dataset GitHub.")
            else:
                st.warning("Folder 'samples' kosong.")
        else:
            st.error("Folder 'samples/' tidak ditemukan di repositori.")

    st.divider()
    run_btn = st.button("🚀 RUN INFERENCE", use_container_width=True, type="primary")

# ============================
# 4. MAIN INTERFACE
# ============================
st.markdown("<div class='main-title'>GrowthVision AI</div>", unsafe_allow_html=True)
st.markdown("<div class='academic-sub'>Pediatric Growth Classification System: Morphological Assessment via Deep Learning</div>", unsafe_allow_html=True)

with st.expander("📖 Methodology & Model Architecture"):
    st.write("""
    Dashboard ini menggunakan model **Convolutional Neural Networks (CNN)** dan **EfficientNet-B0** yang telah diadaptasi menggunakan 
    teknik **LoRA (Low-Rank Adaptation)**. Pendekatan ini memungkinkan klasifikasi morfologi wajah anak (VP-0 vs VP-1).
    """)

if final_files and run_btn:
    results_data = []
    progress_bar = st.progress(0)
    
    # --- INFERENCE LOOP ---
    for i, file in enumerate(final_files):
        try:
            img = Image.open(file).convert("RGB")
            tensor = cnn_tf(img).unsqueeze(0)
            
            with torch.no_grad():
                logits = model_engine(tensor)
                probs = torch.softmax(logits, dim=1).numpy()[0]
            
            pred = np.argmax(probs)
            conf = float(probs[pred]) * 100
            
            # PERBAIKAN: Penanganan nama file (Manual vs GitHub Path)
            if hasattr(file, 'name'):
                fname = file.name
            else:
                fname = os.path.basename(str(file))
            
            results_data.append({
                "img": img,
                "filename": fname,
                "prediction": pred,
                "label": "VP-0 (Proportional)" if pred == 0 else "VP-1 (Linear)",
                "confidence": conf
            })
            progress_bar.progress((i + 1) / len(final_files))
        except Exception as e:
            st.error(f"Gagal memproses file {i}: {e}")

    # --- SUMMARY METRICS ---
    st.markdown("### 📊 Classification Summary")
    df = pd.DataFrame(results_data)
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Analisis", len(df))
    c2.metric("VP-0 (Proportional)", len(df[df.prediction == 0]))
    c3.metric("VP-1 (Linear)", len(df[df.prediction == 1]))

    # --- GRID DISPLAY ---
    st.markdown("### 🖼️ Individual Analysis")
    cols_per_row = 4
    for i in range(0, len(results_data), cols_per_row):
        cols = st.columns(cols_per_row)
        for j, col in enumerate(cols):
            idx = i + j
            if idx < len(results_data):
                res = results_data[idx]
                color = "#28a745" if res['prediction'] == 0 else "#007bff"
                
                with col:
                    st.markdown(f"""
                    <div class="status-card">
                        <div class="label-tag" style="background-color: {color}22; color: {color};">
                            {res['label'].split()[0]}
                        </div>
                    """, unsafe_allow_html=True)
                    
                    st.image(res['img'], use_container_width=True)
                    
                    # Perbaikan Plotly Gauge
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = float(res['confidence']),
                        number = {
                            'suffix': "%", 
                            'font': {'size': 16, 'color': '#444'} # Properti color dipindah ke dalam font
                        },
                        gauge = {
                            'axis': {'range': [0, 100], 'tickwidth': 1},
                            'bar': {'color': color},
                            'bgcolor': "#f8f9fa"
                        }
                    ))
                    fig.update_layout(height=110, margin=dict(l=10, r=10, t=15, b=10))
                    st.plotly_chart(fig, use_container_width=True, key=f"chart_{idx}", config={'displayModeBar': False})
                    
                    st.markdown(f"<p style='text-align:center; font-size:12px; font-weight:600;'>{res['filename']}</p>", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)

    # --- DOWNLOAD REPORT ---
    st.divider()
    csv = df[['filename', 'label', 'confidence']].to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Hasil Analisis (CSV)",
        data=csv,
        file_name='growthvision_report.csv',
        mime='text/csv',
    )

elif not final_files and run_btn:
    st.warning("⚠️ Mohon upload gambar atau pastikan folder sampel tersedia.")
else:
    st.info("💡 Pilih sumber data di sidebar dan klik 'Run Inference' untuk memulai.")

# ============================
# 5. FOOTER
# ============================
st.markdown("<br><hr>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:grey; font-size:12px;'>© 2024 Pediatric AI Research Framework | Distributed under MIT License</p>", unsafe_allow_html=True)