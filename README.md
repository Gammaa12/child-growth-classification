<h1 align="center">CHILD GROWTH CLASSIFICATION</h1>
---
<p align="center">
  <img src="assets/images/Cover.jpg" width="70%">
</p>

<p align="center">
  Sumber Image : <a href="https://share.google/images/DDfO5yEbx6us0Cj8h">Access Here</a>
</p>

---

<h1 align="center">📑 Table of Contents 📑</h1>

- [Deskripsi Proyek](#deskripsi-proyek)
- [Latar Belakang](#latar-belakang)
- [Tujuan Pengembangan](#tujuan-pengembangan)
- [Sumber Dataset](#sumber-dataset)
- [Preprocessing dan Pemodelan](#preprocessing-dan-pemodelan)
  - [Preprocessing Data](#preprocessing-data)
  - [Pemodelan](#pemodelan)
- [Hasil & Evaluasi](#hasil--evaluasi)

---

<h1 id="deskripsi-proyek" align="center">📚 Deskripsi Proyek 📚</h1>

Proyek ini bertujuan mengembangkan sistem berbasis deep learning yang dapat mengenali pola proporsi wajah anak-anak sebagai indikator awal status pertumbuhan menggunakan pendekatan **Visual Proxy (VP)**.
Alih-alih mendiagnosis kondisi medis seperti stunting, model hanya mempelajari **pola visual wajah**, yaitu:

- **VP-0 (Visually Proportional)** → proporsi wajah normal 
- **VP-1 (Visually Linear)** → indikasi ketidakseimbangan proporsi visual

Sistem dikembangkan menggunakan dua model utama:

- **EfficientNet-B0** (baseline, fine-tuning, dan fine-tuning + LoRA)
- **Vision Transformer (ViT)** sebagai pembanding

Model dengan performa terbaik adalah EfficientNet + Fine-Tuning + LoRA.

---

<h1 id="latar-belakang" align="center">🧠 Latar Belakang 🧠</h1>

Masalah gangguan pertumbuhan seperti stunting masih menjadi isu kritis di berbagai negara berkembang. Deteksi dini merupakan langkah penting, namun metode tradisional masih bergantung pada peralatan medis, tenaga ahli, dan proses pengukuran manual yang dapat memakan waktu serta tidak selalu tersedia di lapangan.

Untuk itu, pendekatan Visual Proxy berbasis Deep Learning menjadi solusi non-medis yang cepat, efisien, serta tetap menjaga privasi data anak. Dengan hanya melihat pola visual wajah, sistem dapat membantu proses pemantauan awal tanpa intervensi medis langsung.

---

<h1 id="tujuan-pengembangan" align="center">🎯 Tujuan Pengembangan 🎯</h1>


- Mengembangkan model klasifikasi citra wajah anak berdasarkan pola visual proporsional (VP-0) dan non-proporsional (VP-1).
- Mengimplementasikan dan membandingkan performa dua model deep learning modern: **EfficientNet & Vision Transformer (ViT)**.
- Menguji peningkatan performa menggunakan teknik **Fine-Tuning** dan **Low-Rank Adaptation (LoRA)**.
- Membangun sistem screening non-medis yang aman secara etika dan privasi. 

---

<h1 id="sumber-dataset" align="center">📊 Sumber Dataset 📊</h1>

Dataset diperoleh dari platform Roboflow, berisi citra wajah anak-anak dengan dua kategori:

- **Healthy → VP-0**
- **Stunting → VP-1**

Dataset kemudian di-relabel ulang menggunakan format Visual Proxy, dan diproses ulang menjadi dataset wajah terpotong (face-cropped).

Link Original Dataset: 
1. *[**STUNTING Computer Vision Dataset**](https://universe.roboflow.com/test-bdpwd/stunting-onvws)*
2. *[**STUNTING Computer Vision Model**](https://universe.roboflow.com/mnt-bgmps/stunting-onvws-b12p5)*
3. *[**Deteksi Stunting Computer Vision Model**](https://universe.roboflow.com/database-ayu/deteksi-stunting)*
---

<h1 id="preprocessing-dan-pemodelan" align="center">🧼 Preprocessing dan Pemodelan 🧼</h1>
<h2 id="preprocessing-data" align="center">✨ Preprocessing Data ✨</h2>

Tahap preprocessing dimulai dengan memuat dataset dari direktori yang telah ditentukan. Setiap citra kemudian diproses menggunakan MTCNN untuk mendeteksi wajah dan melakukan cropping sehingga model hanya menerima area wajah yang relevan, bukan latar belakang. Setelah wajah terdeteksi, citra diubah ukurannya menjadi 224×224 piksel dan dinormalisasi menggunakan metode mean–std normalization untuk menyesuaikan standar input model pre-trained seperti EfficientNet dan ViT.

Untuk meningkatkan variasi data dan mencegah overfitting, diterapkan beberapa teknik data augmentation seperti rotasi acak, horizontal flip, random crop, dan penyesuaian brightness–contrast. Dataset kemudian dibagi menggunakan stratified split menjadi 80% data latih dan 20% data uji agar distribusi kelas VP-0 dan VP-1 tetap seimbang. Hasil akhir preprocessing menghasilkan dataset yang bersih, terstruktur, dan siap digunakan untuk pelatihan model.

<h2 id="pemodelan" align="center">🤖 Pemodelan 🤖</h2>
  
**A. EfficientNet-B0**
Model pertama yang digunakan adalah **EfficientNet-B0**, sebuah CNN modern yang mengombinasikan depth, width, dan resolution scaling untuk efisiensi maksimal.
Tiga eksperimen dilakukan:

- **1. Baseline**
   - Mengganti classification head menjadi Linear (1280 → 2)
   - Melatih selama 10 epoch
   - Optimizer Adam, LR = 1e-4

- **2. Fine-Tuning Standar**
   - Membuka 40 layer teratas untuk dilatih ulang
   - Hasilnya meningkat signifikan

- **3. Fine-Tuning + LoRA**
   - Menambahkan modul LoRA pada pointwise convolution
   - Melatih parameter kecil ber-rank rendah
   - Paling efisien dan paling akurat dalam pengujian

**B. Vision Transformer (ViT)**
Sebagai pembanding, digunakan arsitektur ViT-Base Patch16/224:

- Pre-trained ImageNet
- Head diganti menjadi Linear (768 → 2)
- Pelatihan dilakukan tanpa LoRA

ViT mendapatkan performa tinggi namun masih kalah dari EfficientNet+LoRA.

---

<h1 id="hasil--evaluasi" align="center">📊 Hasil & Evaluasi 📊</h1>

**Evaluasi Model**

Model dievaluasi menggunakan beberapa metrik, termasuk **classification report** dan **confusion matrix**.

**Classification Report**

Berikut adalah penjelasan tentang metrik yang digunakan dalam classification report:

- **Precision**: Mengukur proporsi prediksi positif yang benar.
- **Recall**: Mengukur proporsi sampel aktual positif yang berhasil diidentifikasi dengan benar.
- **F1-Score**: Rata-rata harmonis dari precision dan recall.
- **Accuracy**: Mengukur keseluruhan performa model.

**Tabel Perbandingan Classification Report**

Berikut adalah perbandingan metrik evaluasi untuk setiap model:

| Model                         | Algoritma   | Akurasi | Precision | Recall | F1-Score |
|------------------------------|-------------|---------|-----------|--------|----------|
| Baseline EfficientNet        | CNN         | 0.88    | 0.84      | 0.94   | 0.89     |
| EfficientNet + Fine-Tuning   | CNN         | 0.97    | 0.99      | 0.95   | 0.97     |
| EfficientNet + LoRA (Best)   | CNN         | 0.98    | 0.99      | 0.97   | 0.98     |
| Vision Transformer Baseline  | Transformer | 0.95    | 0.95      | 0.95   | 0.95     |

**Confusion Matrix** 🔴🟢

Di bawah ini adalah confusion matrix untuk setiap model.

<p align="center">
  <!-- EfficientNet Baseline -->
  <img src="assets/images/Confusion_Matrix_Baseline.PNG" alt="Confusion Matrix Baseline" width="30%" />
  
  <!-- EfficientNet + Fine-Tuning -->
  <img src="assets/images/Confusion_Matrix_FT_Standar.PNG" alt="Confusion Matrix FT Standar" width="30%" />
  
  <!-- EfficientNet + LoRA -->
  <img src="assets/images/Confusion_Matrix_FT_LoRA.PNG" alt="Confusion Matrix FT LoRA" width="30%" />

  <!-- Vision Transformer Baseline -->
  <img src="assets/images/Confusion_Matrix_ViT.PNG" alt="Confusion Matrix ViT" width="30%" />
</p>

**Learning Curves** 📈

Berikut adalah learning curves untuk model EfficientNet dan ViT yang menunjukkan bagaimana model belajar seiring berjalannya waktu:

<p align="center">
  <!-- EfficientNet -->
  <img src="assets/images/grafik_efficient.PNG" alt="Grafik EfficientNet" width="60%" />
  
  <!-- ViT -->
  <img src="assets/images/grafik_vit.PNG" alt="Grafik ViT" width="60%" />
  
</p>
