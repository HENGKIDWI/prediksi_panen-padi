# Sistem Cerdas Peramalan Produksi Padi — ELM / GA-ELM (Streamlit)

Aplikasi web peramalan produksi padi **total bulanan** dengan **Extreme Learning Machine (ELM)** dan **GA-ELM**
(bobot dioptimasi Genetic Algorithm). Port dari notebook `GA_ELM_PERAMALAN_TOTAL_Q.ipynb`; penamaan, label, dan hasil
angkanya sama dengan `App1.py`, hanya dipecah ke struktur modular.

## Menjalankan

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Buka alamat yang tampil (biasanya http://localhost:8501). Klik **Muat Data Demo** untuk mencoba tanpa file.

## Alur 9 tahap (dibuka bertahap lewat tombol *Lanjut*)

| Tahap | Modul | Isi |
|---|---|---|
| 0 | `views/beranda.py` | Ringkasan, status tahap, panduan metodologi |
| 1 Muat Data | `views/data.py` | Unggah Excel, pilih sheet, pemetaan kolom, cuplikan & distribusi |
| 2 Cek Anomali | `views/anomali.py` | NaN, non-numerik, negatif, nol |
| 3 Preprocessing | `views/pilih_data.py` | Data asli / cleaning (drop, imputasi median) |
| 4 Fitur Lag | `views/deret.py` | Deret total bulanan + lag 1 / lag 12 / keduanya |
| 5 Split Data | `views/split.py` | Per tahun (disarankan) atau rasio persen, terurut waktu |
| 6 Pilih Model | `views/model.py` | ELM atau GA-ELM |
| 7 Parameter Model | `views/parameter.py` | Manual atau cari otomatis (neuron, aktivasi, l2_reg, parameter GA) |
| 8 Pelatihan Model | `views/pelatihan.py` | Multi-seed, riwayat GA, uji ketahanan tahun lain |
| 9 Evaluasi & Peramalan | `views/evaluasi.py`, `views/peramalan.py` | Per bulan & per Q, ketahanan, ramalan Q1–Q4 (Ton & Kuintal), unduh CSV |

## Struktur

```
app.py            # titik masuk: state -> pipeline -> sidebar -> tahap aktif
state.py          # state sesi, cache, pipeline data (tahap 1-5), status tahap
ui.py             # komponen UI bersama (dropdown, header, navigasi, sidebar)
views/            # satu berkas per tahap, masing-masing `render(...)`
core/             # logika murni tanpa Streamlit (mudah diuji)
  config.py data.py elm.py ga.py search.py forecast.py plots.py
tests/            # pytest: logika inti + antarmuka (streamlit.testing)
assets/           # taruh logo1.png di sini (opsional; bila tidak ada dipakai ikon 🌾)
.streamlit/config.toml
```

## Pengujian

```bash
pytest -q
```

Mencakup logika inti (termasuk uji kebocoran data pada peramalan rekursif) dan alur 9 tahap lewat `AppTest`.

## Catatan metodologi

- Split terurut waktu (tanpa shuffle); normalisasi min-max dipelajari dari data latih saja.
- Uji & ramal bersifat **rekursif tanpa produksi aktual**: fitur = rata-rata historis per bulan, lag diambil dari
  riwayat aktual bila ada, selain itu dari hasil ramalan sebelumnya.
- Prediksi akhir = (1−w) × model + w × rata-rata produksi bulan yang sama (blend musiman), dirata-ratakan lintas seed.
- Seleksi parameter memakai validasi di ujung data latih; data uji tidak disentuh.
- Kolom target bernama `Produksi (Ha)` mengikuti file data; satuannya diasumsikan **ton**. 1 ton = 10 kuintal.

## Deploy

https://prediksipanen-padigit-knhvctueutbgzeztkomccs.streamlit.app/