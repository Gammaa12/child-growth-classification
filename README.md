# CHILD GROWTH CLASSIFICATION  
---
 
**Sumber Image:** *[[Link Dataset Anda](https://share.google/images/DDfO5yEbx6us0Cj8h)]*

---

# 📑 Table of Contents
- [Deskripsi Proyek](#deskripsi-proyek)
- [Latar Belakang](#latar-belakang)
- [Tujuan Pengembangan](#tujuan-pengembangan)
- [Sumber Dataset](#sumber-dataset)
- [Preprocessing dan Pemodelan](#preprocessing-dan-ekstraksi-fitur)
  - [Preprocessing Data](#preprocessing)
  - [Pemodelan](#pemodelan)
- [Hasil & Evaluasi](#hasil)

---

# 📚 **Deskripsi Proyek**

Proyek ini bertujuan mengembangkan sistem berbasis deep learning yang dapat mengenali pola proporsi wajah anak-anak sebagai indikator awal status pertumbuhan menggunakan pendekatan **Visual Proxy (VP)**.
Alih-alih mendiagnosis kondisi medis seperti stunting, model hanya mempelajari **pola visual wajah**, yaitu:

- **VP-0 (Visually Proportional)** → proporsi wajah normal 
- **VP-1 (Visually Linear)** → indikasi ketidakseimbangan proporsi visual

Sistem dikembangkan menggunakan dua model utama:

- **EfficientNet-B0** (baseline, fine-tuning, dan fine-tuning + LoRA)
- **Vision Transformer (ViT)** sebagai pembanding

Model dengan performa terbaik adalah EfficientNet + Fine-Tuning + LoRA.

---

# 🧠 **Latar Belakang**

Masalah gangguan pertumbuhan seperti stunting masih menjadi isu kritis di berbagai negara berkembang. Deteksi dini merupakan langkah penting, namun metode tradisional masih bergantung pada peralatan medis, tenaga ahli, dan proses pengukuran manual yang dapat memakan waktu serta tidak selalu tersedia di lapangan.

Untuk itu, pendekatan Visual Proxy berbasis Deep Learning menjadi solusi non-medis yang cepat, efisien, serta tetap menjaga privasi data anak. Dengan hanya melihat pola visual wajah, sistem dapat membantu proses pemantauan awal tanpa intervensi medis langsung.

---

# 🎯 **Tujuan Pengembangan**

- Mengembangkan model klasifikasi citra wajah anak berdasarkan pola visual proporsional (VP-0) dan non-proporsional (VP-1).
- Mengimplementasikan dan membandingkan performa dua model deep learning modern: **EfficientNet & Vision Transformer (ViT)**.
- Menguji peningkatan performa menggunakan teknik **Fine-Tuning** dan **Low-Rank Adaptation (LoRA)**.
- Membangun sistem screening non-medis yang aman secara etika dan privasi. 

---

# 📊 **Sumber Dataset**

Dataset diperoleh dari platform Roboflow, berisi citra wajah anak-anak dengan dua kategori:

- **Healthy → VP-0**
- **Stunting → VP-1**

Dataset kemudian di-relabel ulang menggunakan format Visual Proxy, dan diproses ulang menjadi dataset wajah terpotong (face-cropped).

