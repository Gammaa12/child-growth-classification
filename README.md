# 🔍 VP-0 / VP-1 Face Classification  
### **EfficientNet + LoRA (Parameter-Efficient Fine-Tuning)**

---

## 📌 **Main Image / Cover**
*(Tambahkan gambar utama Anda di sini)*  
**Sumber Image:** *[Link Dataset Anda]*

---

# 📑 Table of Contents
- [Deskripsi Proyek](#deskripsi-proyek)
- [Latar Belakang](#latar-belakang)
- [Tujuan Pengembangan](#tujuan-pengembangan)
- [Sumber Dataset](#sumber-dataset)
- [Preprocessing dan Ekstraksi Fitur](#preprocessing-dan-ekstraksi-fitur)
- [Pemodelan](#pemodelan)
- [Arsitektur Model EfficientNet + LoRA](#arsitektur-model-efficientnet--lora)
- [Instalasi](#instalasi)
- [Menjalankan Pelatihan Model](#menjalankan-pelatihan-model)
- [Hasil dan Analisis](#hasil-dan-analisis)
  - [Evaluasi Model](#evaluasi-model)
  - [Confusion Matrix](#confusion-matrix)
  - [Learning Curve](#learning-curve)
  - [Analisis Error](#analisis-error)
- [Biodata](#biodata)

---

# 📚 **Deskripsi Proyek**

Proyek ini bertujuan untuk melakukan **klasifikasi wajah manusia** ke dalam dua kategori kondisi:

- **VP-0**  
- **VP-1**

Menggunakan pendekatan *Deep Learning* berbasis **EfficientNet B0** yang ditingkatkan menggunakan metode **LoRA (Low-Rank Adaptation)** untuk *parameter-efficient fine-tuning*.

Model ini dipilih karena:

- ringan  
- cepat dilatih  
- akurasi tinggi  
- efisien dalam penggunaan parameter  

---

# 🧠 **Latar Belakang**

Perbedaan kondisi wajah VP-0 dan VP-1 sering kali sangat halus dan sulit dideteksi manusia. Oleh karena itu dibutuhkan model deep learning yang mampu:

- menangkap pola tekstur  
- memahami pencahayaan  
- mengenali fitur wajah kompleks  

EfficientNet + LoRA memberikan solusi ideal karena dapat **melatih ulang layer penting saja** sehingga:

- lebih cepat  
- tidak membutuhkan GPU besar  
- tetap memiliki performa tinggi  

---

# 🎯 **Tujuan Pengembangan**

1. Membangun model klasifikasi VP-0 / VP-1 yang **presisi tinggi**.  
2. Menggunakan teknik **fine-tuning efisien** seperti LoRA.  
3. Menyediakan model yang **ringan, cepat dilatih**, dan mudah di-deploy.  
4. Menyediakan *pipeline* end-to-end mulai preprocessing → training → evaluasi.  

---

# 📊 **Sumber Dataset**

Dataset dibagi ke dalam dua folder utama:

