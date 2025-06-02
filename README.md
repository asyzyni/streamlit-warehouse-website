# 📦 Prediksi Demand Gudang

Proyek ini merupakan aplikasi prediksi demand gudang berbasis machine learning yang dibangun menggunakan model **XGBoost**. Aplikasi ini dilengkapi dengan pipeline preprocessing yang mencakup **StandardScaler** untuk fitur numerik dan **OneHotEncoder** untuk fitur kategorikal.

## 🔍 Deskripsi Proyek

Aplikasi ini bertujuan untuk membantu proses pengambilan keputusan terkait manajemen stok dan gudang dengan cara memprediksi permintaan (demand) berdasarkan data historis dan fitur waktu seperti hari, minggu, bulan, dan lainnya.

Model dibangun dalam dua tahap utama:

1. **Pelatihan Model (`model.py`)**:
   - Melatih model XGBoost dengan data yang telah dipreprocessing.
   - Menyimpan model, encoder, scaler, dan informasi kolom ke dalam file pickle dan JSON.

2. **Prediksi (`main.py`)**:
   - Mengambil input baru (misalnya dari Streamlit).
   - Melakukan preprocessing dengan encoder dan scaler yang telah dilatih.
   - Menghasilkan prediksi demand berdasarkan input pengguna.

## ⚠️ Permasalahan Umum

### Error:
```
ValueError: X has 10 features, but StandardScaler is expecting 4 features as input.
```

### Penyebab:
Terjadi ketidaksesuaian jumlah kolom antara data input saat prediksi dan jumlah kolom yang digunakan saat pelatihan. `StandardScaler` dilatih hanya dengan 4 fitur numerik, tetapi saat prediksi menerima 10 kolom.

### Solusi:

#### ✅ Opsi 1: Koreksi Input
Periksa file `column_mapping.json`, pastikan hanya mencantumkan kolom numerik yang digunakan saat pelatihan, misalnya:
```json
"numerical_cols": [
  "Moving_Average", 
  "Event_Multiplier", 
  "Safety Percentage", 
  "Multiplier_Safety"
]
```

#### 🔁 Opsi 2: Tambah Kolom Baru?
Jika ingin menambahkan fitur baru ke dalam prediksi, lakukan retraining dengan fitur baru tersebut di `model.py`, agar scaler dan model mengenali fitur tersebut sejak awal.

## 🧪 Struktur Proyek

```
├── model.py               # Script pelatihan model
├── main.py                # Script prediksi
├── column_mapping.json    # Metadata kolom fitur numerik dan kategorikal
├── model.pkl              # Model XGBoost yang sudah dilatih
├── scaler.pkl             # StandardScaler yang sudah dilatih
├── encoder.pkl            # OneHotEncoder yang sudah dilatih
```

## 🚀 Cara Menjalankan

1. **Latih Model**:
   ```bash
   python model.py
   ```

2. **Prediksi (via aplikasi atau script)**:
   ```bash
   python main.py
   ```

3. **(Opsional)**: Jalankan aplikasi Streamlit jika tersedia:
   ```bash
   streamlit run app.py
   ```

## 📌 Catatan Penting

- Jangan menambahkan atau menghapus kolom pada data input tanpa memperbarui dan melatih ulang model serta scaler/encoder.
- Selalu pastikan struktur input data sesuai dengan `column_mapping.json`.

## 👤 Kontributor
- Asyifa Izayani Safari
- Delyana Ika Wardani 
- Muhammad Pandu Prapyusta
- Febycandra Carmelinda Ximenes
- Fadiya Zahra Qatranada