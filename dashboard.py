# ============================
# PAGE CONFIG
# ============================
st.set_page_config(
    page_title="Child Growth Classification",
    page_icon="🧒",
    layout="wide"
)

# ============================
# CUSTOM CSS
# ============================
st.markdown("""
<style>
.main-title {
    font-size:42px;
    font-weight:800;
}
.sub-title {
    font-size:18px;
    color:#6c757d;
}
.card {
    background:#ffffff;
    padding:20px;
    border-radius:16px;
    box-shadow:0 8px 24px rgba(0,0,0,0.08);
}
.pred-label {
    font-size:20px;
    font-weight:700;
    text-align:center;
}
.conf {
    text-align:center;
    color:#555;
}
hr {
    margin: 1.5rem 0;
}
</style>
""", unsafe_allow_html=True)

# ============================
# HEADER
# ============================
st.markdown("<div class='main-title'>🧒 Child Growth Classification</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Multi-Model Inference: SVM · CNN · EfficientNet + LoRA</div>", unsafe_allow_html=True)
st.write("")

# ============================
# SIDEBAR (CONTROL PANEL)
# ============================
with st.sidebar:
    st.markdown("## ⚙️ Control Panel")

    model_choice = st.selectbox(
        "🧠 Pilih Model",
        ["SVM (Classic)", "CNN Fine-Tuning", "EfficientNet + LoRA"]
    )

    uploaded_files = st.file_uploader(
        "📤 Upload Foto Wajah Anak",
        type=["jpg","png","jpeg"],
        accept_multiple_files=True
    )

    run_btn = st.button("🔍 Jalankan Prediksi", use_container_width=True)

    st.markdown("---")
    st.markdown("### ℹ️ Info Model")
    if model_choice == "SVM (Classic)":
        st.caption("✔ Grayscale\n✔ Fitur klasik\n✔ Cepat & ringan")
    elif model_choice == "CNN Fine-Tuning":
        st.caption("✔ CNN custom\n✔ Fine-tuned\n✔ End-to-end")
    else:
        st.caption("✔ EfficientNet-B0\n✔ LoRA Adaptation\n✔ Parameter Efficient")

# ============================
# MAIN CONTENT
# ============================
if uploaded_files and run_btn:

    images = []
    raw_images = []

    for file in uploaded_files:
        img = Image.open(file).convert("RGB")
        raw_images.append(img)
        images.append(cnn_tf(img))

    x = torch.stack(images)

    # ============================
    # PREVIEW
    # ============================
    st.markdown("## 🖼️ Preview Gambar")
    cols = st.columns(4)

    for i, (img, file) in enumerate(zip(raw_images, uploaded_files)):
        with cols[i % 4]:
            st.image(img, caption=file.name, use_container_width=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ============================
    # INFERENCE
    # ============================
    if model_choice == "CNN Fine-Tuning":
        with torch.no_grad():
            out = cnn_model(x)
            probs = torch.softmax(out, 1).numpy()

    elif model_choice == "EfficientNet + LoRA":
        with torch.no_grad():
            out = eff_model(x)
            probs = torch.softmax(out, 1).numpy()

    else:  # SVM
        probs = []
        for img in raw_images:
            img_cv = cv2.resize(np.array(img), (64,64))
            gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
            feat = svm_scaler.transform(gray.flatten().reshape(1,-1))
            probs.append(svm_model.predict_proba(feat)[0])
        probs = np.array(probs)

    preds = probs.argmax(axis=1)

    # ============================
    # RESULT
    # ============================
    st.markdown("## 📊 Hasil Prediksi")
    cols = st.columns(4)

    for i, (img, pred, prob) in enumerate(zip(raw_images, preds, probs)):
        label = "VP-0 (Visually Proportional)" if pred == 0 else "VP-1 (Visually Linear)"
        conf = prob[pred] * 100

        with cols[i % 4]:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.image(img, use_container_width=True)
            st.markdown(f"<div class='pred-label'>{label}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='conf'>Confidence: {conf:.2f}%</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

else:
    st.info("⬅️ Silakan upload gambar dan jalankan prediksi melalui sidebar.")
