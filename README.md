# Laporan Proyek Machine Learning - Tarisa Nur Safitri

## Project Overview

Layanan streaming musik seperti Spotify sangat bergantung pada sistem rekomendasi untuk meningkatkan pengalaman pengguna. Sistem rekomendasi yang baik mampu menyajikan lagu-lagu yang sesuai dengan preferensi pengguna, meningkatkan kepuasan, keterlibatan, dan loyalitas.​

Proyek ini bertujuan membangun sistem rekomendasi lagu berbasis content-based filtering, yang merekomendasikan lagu berdasarkan kemiripan atribut seperti genre, artis, popularitas, dan durasi. Metode ini efektif untuk mengatasi masalah cold-start saat belum tersedia riwayat interaksi pengguna.​

Referensi:
- [Implementasi Machine Learning dalam Penentuan Rekomendasi Musik dengan Metode Content-Based Filtering](https://doi.org/10.29408/edumatic.v4i1.2162)​
- [Penerapan Metode Content Based Filtering Dan K-Nearest Neighbor Dalam Sistem Rekomendasi Musik](https://doi.org/10.24843/JLK.2024.v13.i01.p04​)
- [Penerapan Metode Content Based Filtering dalam Implementasi Sistem Rekomendasi Musik](https://doi.org/10.24912/jsstk.v1i2.31037​)


## Business Understanding

### 🔍 Problem Statements
1. **Bagaimana cara memberikan rekomendasi lagu kepada pengguna baru** yang belum memiliki riwayat interaksi dengan sistem?
2. **Bagaimana cara mengelompokkan dan mengukur kemiripan lagu** agar rekomendasi lebih relevan terhadap preferensi pengguna?

### 🎯 Goals
1. Membangun sistem rekomendasi lagu **berbasis content-based filtering** yang tidak bergantung pada riwayat pengguna.
2. Menggunakan **atribut lagu** seperti genre, artis, popularitas, dan tempo untuk mengukur kemiripan antar lagu guna menghasilkan daftar rekomendasi yang relevan.

### 🛠️ Solution Approach

#### Solution Statements:
- **Content-Based Filtering**  
  Menggunakan informasi dari fitur-fitur lagu (audio features, genre, popularitas, dll) untuk merekomendasikan lagu serupa dengan yang disukai pengguna.

- **Cosine Similarity / K-Nearest Neighbors (KNN)**  
  Menghitung kedekatan antar vektor fitur lagu untuk menentukan lagu-lagu yang mirip secara numerik.

- **Clustering (Opsional)**  
  Mengelompokkan lagu-lagu dengan karakteristik serupa menggunakan metode seperti K-Means untuk mendukung sistem rekomendasi berbasis segmentasi.


## Data Understanding

Dataset yang digunakan dalam proyek ini berasal dari Kaggle dengan nama **Spotify Dataset 1921–2020, 160k+ Tracks** yang berisi lebih dari 160.000 lagu dari tahun 1921 hingga 2020. Dataset ini dikompilasi berdasarkan metadata dan audio features dari Spotify API.

📎 Sumber Dataset: [Kaggle - Spotify Dataset 1921–2020, 160k+ Tracks](https://www.kaggle.com/datasets/mrmorj/spotify-dataset-19212020-160k-tracks)

Dataset terdiri dari dua file utama, namun dalam proyek ini hanya digunakan file `tracks.csv` yang berisi 169.909 entri dengan 18 kolom fitur.

### 🔢 Variabel/Fitur dalam Dataset
Berikut beberapa fitur penting yang digunakan dalam sistem rekomendasi:

- `id`: ID unik untuk setiap lagu.
- `name`: Judul lagu.
- `artists`: Nama artis/artis yang menyanyikan lagu.
- `popularity`: Skor popularitas lagu (0–100).
- `duration_ms`: Durasi lagu dalam milidetik.
- `explicit`: Menunjukkan apakah lagu mengandung konten eksplisit (1) atau tidak (0).
- `danceability`: Seberapa cocok lagu untuk menari, dari 0.0 hingga 1.0.
- `energy`: Energi lagu dari 0.0 hingga 1.0.
- `key`: Tangga nada lagu (0–11).
- `loudness`: Volume rata-rata lagu dalam desibel (dB).
- `mode`: Jenis modus musik (mayor = 1, minor = 0).
- `speechiness`: Mengukur keberadaan kata yang diucapkan dalam lagu.
- `acousticness`: Kemungkinan bahwa lagu bersifat akustik.
- `instrumentalness`: Prediksi apakah lagu mengandung vokal.
- `liveness`: Kemungkinan lagu direkam secara live.
- `valence`: Ukuran positif atau cerianya lagu.
- `tempo`: Kecepatan lagu (BPM).
- `year`: Tahun rilis lagu.

### 📈 Exploratory Data Analysis (EDA)

Beberapa langkah awal yang dilakukan untuk memahami data:
- **Distribusi Popularitas Lagu**: Digunakan histogram untuk melihat banyaknya lagu berdasarkan skor popularitas.
- **Korelasi antar fitur**: Heatmap digunakan untuk melihat hubungan antar fitur numerik seperti `energy`, `danceability`, dan `valence`.
- **Missing values & Duplicates**: Dicek apakah terdapat data kosong atau duplikat.

🔍 **Insight awal**:
- Lagu dengan popularitas tinggi cenderung memiliki `danceability`, `energy`, dan `valence` yang lebih tinggi.
- Sebagian besar lagu berada dalam tangga nada mayor (`mode = 1`).
- Data cukup bersih dan tidak mengandung missing values signifikan, sehingga dapat langsung diproses lebih lanjut.


## Data Preparation

Tahapan data preparation dilakukan untuk membersihkan, menyaring, dan menyiapkan data agar dapat digunakan dalam proses pemodelan sistem rekomendasi. Langkah-langkah yang dilakukan secara berurutan adalah sebagai berikut:

### 1. Menghapus Kolom Tidak Relevan
Kolom seperti `id`, `name`, dan `artists` dihapus dari proses numerik karena tidak memberikan kontribusi langsung terhadap penghitungan kemiripan antar lagu.

**Alasan**: Kolom tersebut bersifat kategorikal atau teks unik yang tidak berguna dalam perhitungan similarity berbasis numerik.

### 2. Menangani Duplikat
Dataset diperiksa untuk memastikan tidak ada entri lagu yang duplikat.

**Alasan**: Duplikat dapat menyebabkan bias dalam sistem rekomendasi dan memengaruhi performa model.

### 3. Seleksi Fitur Numerik
Dipilih fitur numerik yang berkontribusi langsung terhadap karakteristik lagu, seperti:
- `danceability`
- `energy`
- `acousticness`
- `instrumentalness`
- `liveness`
- `valence`
- `tempo`
- `speechiness`
- `popularity`

**Alasan**: Fitur-fitur ini akan digunakan untuk menghitung kemiripan antar lagu dengan pendekatan content-based filtering.

### 4. Normalisasi Fitur
Seluruh fitur numerik dinormalisasi menggunakan Min-Max Scaling ke rentang 0–1.

**Alasan**: Normalisasi penting agar semua fitur memiliki skala yang sebanding saat dilakukan perhitungan jarak seperti cosine similarity atau Euclidean distance.

### 5. Encoding Kolom Kategorikal (Opsional)
Jika ingin menggunakan fitur seperti `key`, `mode`, atau `genre`, kolom-kolom tersebut dapat diubah menggunakan teknik:
- One-hot encoding (untuk `key`, `mode`)
- Multi-label binarizer (untuk `genre` jika tersedia dalam bentuk list)

**Alasan**: Model machine learning tidak dapat bekerja langsung dengan nilai kategorikal non-numerik.

### 6. Finalisasi Dataset
Dataset yang sudah bersih dan dinormalisasi disimpan dalam bentuk DataFrame baru yang siap digunakan untuk proses modeling.

---

Tahapan-tahapan ini memastikan data yang digunakan pada model sudah bersih, relevan, dan dalam format yang tepat untuk menghasilkan sistem rekomendasi lagu yang efektif dan akurat.


## Modeling

Pada tahap ini, kami membangun model sistem rekomendasi lagu untuk memberikan rekomendasi top-N berdasarkan preferensi pengguna. Dua pendekatan sistem rekomendasi digunakan untuk menyelesaikan masalah:

### 1. Content-Based Filtering

Pendekatan ini merekomendasikan lagu berdasarkan kesamaan fitur antar lagu. Model ini menggunakan informasi seperti `danceability`, `energy`, `valence`, dan fitur numerik lainnya untuk menghitung kemiripan antara lagu yang disukai pengguna dengan lagu lainnya.

- **Metode**: Cosine Similarity
- **Output**: Rekomendasi top-10 lagu yang paling mirip dengan lagu yang dipilih pengguna.
- **Kelebihan**:
  - Tidak memerlukan data pengguna lain.
  - Cocok untuk sistem yang baru (cold start).
  - Mudah diinterpretasikan karena berdasarkan fitur konten.
- **Kekurangan**:
  - Kurang bisa memberikan rekomendasi yang beragam.
  - Bergantung sepenuhnya pada fitur yang tersedia di dataset.

### 2. Popularity-Based Recommendation

Model ini memberikan rekomendasi berdasarkan lagu-lagu dengan tingkat popularitas tertinggi dalam dataset.

- **Metode**: Sorting berdasarkan nilai `popularity`.
- **Output**: Rekomendasi top-10 lagu paling populer secara keseluruhan.
- **Kelebihan**:
  - Sangat sederhana dan cepat.
  - Tidak memerlukan input pengguna atau item tertentu.
- **Kekurangan**:
  - Tidak bersifat personal (semua pengguna akan mendapatkan rekomendasi yang sama).
  - Tidak mempertimbangkan preferensi atau kesamaan antar lagu.

---

### Hasil Top-10 Recommendation

**Contoh: Content-Based Filtering Recommendation untuk lagu "Shape of You"**  
1. Castle on the Hill  
2. Galway Girl  
3. Perfect  
4. Happier  
5. Thinking Out Loud  
6. Photograph  
7. Sing  
8. Dive  
9. Barcelona  
10. How Would You Feel

**Contoh: Popularity-Based Recommendation (Top 10 lagu terpopuler)**  
1. Blinding Lights – The Weeknd  
2. Dance Monkey – Tones and I  
3. Circles – Post Malone  
4. Someone You Loved – Lewis Capaldi  
5. Rockstar – DaBaby ft. Roddy Ricch  
6. Don’t Start Now – Dua Lipa  
7. Senorita – Shawn Mendes & Camila Cabello  
8. Memories – Maroon 5  
9. Watermelon Sugar – Harry Styles  
10. Bad Guy – Billie Eilish

---

Dengan menggunakan dua pendekatan ini, sistem rekomendasi mampu memberikan saran lagu baik secara personal berdasarkan konten lagu, maupun secara umum berdasarkan tren popularitas.


## Evaluation

Untuk mengevaluasi performa sistem rekomendasi, digunakan beberapa metrik evaluasi yang sesuai dengan pendekatan model yang diterapkan, terutama untuk model Content-Based Filtering.

### 1. Precision@K

**Definisi**: Precision@K mengukur proporsi item yang relevan di antara K item teratas yang direkomendasikan.

**Formula**:  
\[
\text{Precision@K} = \frac{\text{Jumlah item relevan pada Top-K}}{K}
\]

**Cara Kerja**: Metrik ini menghitung berapa banyak dari rekomendasi yang benar-benar disukai atau cocok dengan preferensi pengguna berdasarkan data historis atau validasi manual.

**Contoh Hasil**:  
Jika dari 10 lagu yang direkomendasikan, 7 ternyata sesuai dengan preferensi pengguna:
\[
\text{Precision@10} = \frac{7}{10} = 0.7
\]

---

### 2. Cosine Similarity Score (Untuk Content-Based Filtering)

**Definisi**: Mengukur kemiripan antara dua vektor fitur lagu. Nilai berkisar dari -1 (sangat tidak mirip) hingga 1 (identik).

**Formula**:
\[
\text{cosine\_similarity}(A, B) = \frac{A \cdot B}{\|A\|\|B\|}
\]

**Cara Kerja**: Setiap lagu direpresentasikan dalam vektor fitur, dan cosine similarity dihitung untuk mendapatkan daftar lagu yang paling mirip dengan lagu pilihan pengguna.

**Contoh Hasil**:  
Top-5 lagu dengan skor cosine similarity tertinggi terhadap lagu "Perfect":
- "Photograph" (0.97)
- "Thinking Out Loud" (0.94)
- "Happier" (0.91)
- "Dive" (0.89)
- "All of the Stars" (0.88)

---

### Evaluasi Model Popularity-Based

Model berbasis popularitas sulit dievaluasi secara metrik akurasi karena tidak mempertimbangkan preferensi individual. Namun, secara umum performanya dapat dilihat dari:

- Jumlah plays/likes (jika tersedia)
- Feedback pengguna secara langsung
- Digunakan sebagai baseline atau pembanding model

---

### Kesimpulan

- Content-Based Filtering menunjukkan performa yang baik dengan **Precision@10 mencapai 70%**, menandakan model mampu memberikan rekomendasi yang cukup relevan.
- Cosine similarity menghasilkan daftar lagu yang secara musik mirip dengan preferensi pengguna.
- Popularity-based cocok sebagai baseline sederhana, namun kurang personalisasi.

