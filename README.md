# AI Sortir Biji Kopi (Green Bean)

Sistem AI untuk membedakan **biji kopi hijau bagus** dari **17 jenis cacat (defect)** dan
menyortirnya ke saluran (bin) sesuai kategori mutu SCA. Dibangun dengan Python +
Ultralytics YOLO (klasifikasi gambar) + OpenCV.

```
Foto biji ──► deteksi tiap biji (OpenCV) ──► klasifikasi (AI) ──► keputusan bin
                                                                   ├─ LOLOS       (biji bagus, AI yakin ≥ 0,9)
                                                                   ├─ KATEGORI_2  (defect sekunder)
                                                                   ├─ KATEGORI_1  (defect primer, tolak)
                                                                   └─ CEK_MANUAL  (AI kurang yakin)
```

> **Penting:** foto biji bagus saat ini berasal dari dataset publik USK-Coffee yang
> berlisensi **non-komersial** dan difoto dengan kamera berbeda. Model ini cocok untuk
> prototipe/uji coba. Untuk dipakai sungguhan, ganti dengan foto biji bagus & cacat dari
> kamera Anda sendiri. Lihat [Catatan dataset](#catatan-dataset).

## Struktur proyek

| File / folder | Isi |
|---|---|
| `Data/` | Dataset asli 17 defect (tidak pernah diubah oleh script) |
| `Data_tambahan/` | Foto dari sumber lain: `Good/` dan `Other Defect/` (USK-Coffee), lihat `SUMBER.txt` |
| `config.py` | Daftar kelas, kategori SCA, bin sortir, threshold, rumus grade |
| `import_good_beans.py` | Impor & seragamkan foto dari dataset lain → `Data_tambahan/` |
| `prepare_dataset.py` | Buang duplikat & label konflik, split train/val/test, augmentasi → `dataset/` |
| `train.py` | Training model → `models/coffee_defect_cls.pt` |
| `evaluate.py` | Akurasi, keputusan sortir, tes jebakan, tabel threshold → `reports/` |
| `sort_beans.py` | Program sortir: mode `folder`, `tray`, `camera` → `output/` |
| `app.py` | Aplikasi web lokal (upload foto → hasil sortir di browser) |
| `demo/` | Foto tray contoh + label aslinya |
| `sumber_data/USK-Coffee/` | Dataset USK-Coffee asli (8.000 foto) untuk impor ulang & tes jebakan |

## Instalasi

Proyek ini memakai virtual environment `.venv` (Python 3.14, PyTorch GPU CUDA 13.0).

```powershell
git clone https://github.com/mfadlikurniawann-webdev/Green-Bean---Coffee-Sorter.git
cd Green-Bean---Coffee-Sorter
python -m venv .venv
.venv\Scripts\python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
.venv\Scripts\python -m pip install -r requirements.txt
```

Semua perintah di bawah memakai `.venv\Scripts\python`.

Model terlatih (`models/coffee_defect_cls.pt`) sudah ikut di repo, jadi `app.py` dan
`sort_beans.py` bisa langsung dipakai. **Dataset tidak ikut di repo** karena lisensinya milik
pihak ketiga. Untuk training ulang, unduh sendiri:

| Folder | Sumber |
|---|---|
| `Data/` | [Coffee Green Bean with 17 Defects (Kaggle)](https://www.kaggle.com/datasets/sujitraarw/coffee-green-bean-with-17-defects-original). Ekstrak supaya isinya `Data/Broken/`, `Data/Cut/`, dst. |
| `sumber_data/USK-Coffee/` | [USK-Coffee (Hugging Face)](https://huggingface.co/datasets/duvdar/usk-coffee), ekstrak zip-nya ke `sumber_data/`. Lalu buat `Data_tambahan/` dengan perintah langkah 0 di bawah. |

## Cara paling mudah: aplikasi web lokal

```powershell
.venv\Scripts\python app.py
```

Browser akan terbuka ke **http://127.0.0.1:7860** (kalau tidak, buka alamat itu manual).
Upload foto biji (satu atau banyak biji di alas putih), atau klik salah satu contoh, lalu
tekan **Analisis**. Hasilnya berupa foto dengan kotak warna per biji, jumlah per bin,
estimasi grade, dan tabel detail per biji. Batas keyakinan bisa digeser langsung. Tekan
`Ctrl+C` di terminal untuk menghentikan aplikasi.

## Cara pakai (command line)

```powershell
# 0. (Opsional) Impor foto dari dataset lain. Biji bagus DAN cacat harus dari sumber yang sama.
.venv\Scripts\python import_good_beans.py --src "sumber_data\USK-Coffee\train\premium" --n 300
.venv\Scripts\python import_good_beans.py --src "sumber_data\USK-Coffee\train\defect" --class-name "Other Defect" --n 300

# 1. Siapkan dataset (ulangi setiap kali isi Data/ atau Data_tambahan/ berubah)
.venv\Scripts\python prepare_dataset.py

# 2. Training (+-12 menit di RTX 3060)
.venv\Scripts\python train.py

# 3. Evaluasi di data test (+ tes jebakan bila punya foto dari sumber luar yang tidak dipakai training)
.venv\Scripts\python evaluate.py --external-defects "sumber_data\USK-Coffee\test\defect" --external-good "sumber_data\USK-Coffee\test\premium"

# 4. Sortir
.venv\Scripts\python sort_beans.py folder --input "D:\foto_biji"      # 1 biji per foto
.venv\Scripts\python sort_beans.py tray   --input demo\tray_campuran.jpg  # banyak biji dalam 1 foto
.venv\Scripts\python sort_beans.py camera --cam 0                     # webcam langsung
.venv\Scripts\python sort_beans.py camera --cam http://192.168.1.5:8080/video  # kamera HP (app IP Webcam)
```

Hasil sortir berbentuk folder seperti ini, ditambah `hasil_sortir.csv` dan `ringkasan.txt`
(jumlah per bin, per kelas, dan estimasi grade SCA):

```
output/tray_20261109_200000/
├── foto_tray_hasil.jpg          ← foto dengan kotak warna per biji
├── LOLOS/Good/...               ← potongan gambar tiap biji
├── KATEGORI_1/Full Black/...
├── KATEGORI_2/Broken/...
├── CEK_MANUAL/...
├── hasil_sortir.csv
└── ringkasan.txt
```

Dua batas keyakinan (default di `config.py`, bisa diubah lewat opsi):
- `--good-threshold 0.9`: biji baru masuk LOLOS kalau AI ≥90% yakin biji itu bagus.
  Biji cacat yang lolos adalah kesalahan paling mahal, jadi batas ini dibuat ketat.
- `--threshold 0.5`: tebakan defect di bawah batas ini masuk CEK_MANUAL.

## Kategori & bin sortir

Mengikuti *SCA Arabica Green Coffee Defect Handbook* (diatur di `config.py`):

| Bin | Kelas | Full defect = |
|---|---|---|
| **LOLOS** | Good | – |
| **KATEGORI_1** (primer) | Full Black, Full Sour, Dry Cherry, Fungus Damage | 1 biji |
| | Severe Insect Damage | 5 biji |
| **KATEGORI_2** (sekunder) | Partial Black, Partial Sour | 3 biji |
| | Parchment, Floater, Immature, Withered, Shell, Broken, Cut, Husk | 5 biji |
| | Slight Insect Damage | 10 biji |
| | Fade *(tidak ada di handbook SCA, diasumsikan kategori 2)* | 5 biji |
| | Other Defect *(cacat dari USK, jenisnya tidak dilabeli)* | 5 biji |

Grade SCA per sampel 350 g: Specialty = 0 full defect kategori 1 dan ≤5 total ·
Premium ≤8 · Exchange 9–23 · Below Standard 24–86 · Off Grade >86.

## Hasil

Model: YOLO11s-cls (5,4 juta parameter), input 320 px, 19 kelas (Good + 17 defect +
Other Defect). Training 52 epoch (terbaik di epoch 27), ±12 menit di RTX 3060 Laptop.
Laporan lengkap: `reports/evaluasi_test.txt`, confusion matrix: `reports/confusion_matrix_test/`.

### Data test (234 foto yang tidak pernah dilihat model)

| Metrik | Hasil |
|---|---|
| Akurasi jenis (19 kelas, top-1) | 86,3% |
| Jawaban benar ada di 3 tebakan teratas | 95,7% |
| **Akurasi bin sortir** | **93,2%** |
| **Defect yang lolos sebagai biji bagus** | **2 dari 189 (1,1%)** |
| Biji bagus masuk LOLOS / salah tolak / cek manual | 41 / 2 / 2 dari 45 |
| Defect kategori 1 masuk KATEGORI_1 | 82,2% (37/45) |

Kelas tersulit: Partial Sour (sering tertebak Full Sour), Withered, Cut, Slight vs Severe
Insect Damage.

### Tes jebakan: apakah AI menilai biji, atau asal fotonya?

Karena foto biji bagus berasal dari sumber lain (USK), ada risiko AI hanya belajar
"foto bergaya USK = bagus". Tiga tes di `evaluate.py` memeriksanya:

| Tes | Model awal (Good saja dari USK) | **Model final** (+ Other Defect dari USK, batas LOLOS 0,9) |
|---|---|---|
| A. Biji **cacat** USK (400, di luar training) dikira bagus | 98,8% ❌ | **8,8%** |
| B. Defect dataset utama diberi gaya foto USK, dikira bagus | 0% | 0,5% |
| C. Biji **bagus** USK (400, di luar training) dikenali bagus | 100% | 91,2% |

Model awal gagal total di tes A, sehingga ditambahkan kelas *Other Defect* (biji cacat dari
sumber yang sama) dan batas LOLOS dinaikkan. Sisa 8,8% di tes A menunjukkan membedakan biji
bagus dan cacat pada foto USK masih sulit dengan 300 contoh. Tabel trade-off batas LOLOS:

| Batas LOLOS | Cacat USK lolos (A) | Bagus USK lolos (C) |
|---|---|---|
| 0,5 | 17,8% | 96,8% |
| **0,9 (default)** | **8,8%** | **91,8%** |
| 0,95 | 6,2% | 87,8% |
| 0,99 | 4,2% | 75,5% |

Angka tabel ini diukur terpisah dari tabel tes jebakan di atas. Selisih kecil (mis. 91,8% vs
91,2%) berasal dari noise acak saat penyeragaman foto.

### Demo di laptop

- `demo/tray_campuran.jpg` (8 bagus + 12 cacat dari data test): **8/8 biji bagus LOLOS,
  0/12 biji cacat lolos** (11 ke bin defect, 1 ke CEK_MANUAL), jenis benar 18/20.
  Hasil: `output/demo_campuran/`.
- `demo/tray_contoh.jpg` (20 biji cacat): 20/20 terdeteksi, tidak ada yang lolos.

### Ukuran biji di foto menentukan akurasi

Diuji dengan model 17 kelas pada foto tray yang sama:

| Ukuran 1 biji di foto | Kelas benar |
|---|---|
| ±90 px | 15/20 |
| ±130 px | 14/20 |
| ±180 px | 20/20 |
| ±225 px | 20/20 |

Jadi usahakan **setiap biji minimal ±180 piksel** (sekitar 20 piksel/mm). Contoh: kamera
1080p mencakup area ±9×5 cm, kamera 4K atau kamera HP 12 MP mencakup area ±19×11 cm.

### Kecepatan (foto tray 2000×1600 berisi 20 biji, termasuk deteksi OpenCV ±130 ms)

| Perangkat | Klasifikasi AI per biji | Throughput total |
|---|---|---|
| GPU RTX 3060 Laptop | 3,6 ms | ±99 biji/detik |
| CPU i7-11800H saja | 18 ms | ±41 biji/detik |

## Catatan dataset

1. **Biji bagus dari sumber luar (sementara).** Dataset utama (Kaggle "Coffee Green Bean
   with 17 Defects", Mae Fah Luang University) tidak punya foto biji bagus. Foto `Good` dan
   `Other Defect` diambil dari **USK-Coffee** (Universitas Syiah Kuala), lalu diseragamkan
   warna, ukuran biji, dan resolusinya oleh `import_good_beans.py`.
   - Lisensi USK-Coffee menurut salinan Roboflow: **CC BY-NC-SA 4.0 (non-komersial)**.
   - Lisensi dataset utama di Kaggle tertulis "Unknown".
   - Untuk produk komersial: foto sendiri biji bagus & cacat dengan kamera yang akan dipakai.
     Simpan di `Data/Good/` (dan defect di folder masing-masing), hapus `Data_tambahan/`,
     lalu jalankan ulang `prepare_dataset.py` dan `train.py`.
2. **Duplikat:** 12 gambar `Immature` adalah salinan gambar lain dan otomatis dibuang
   supaya hasil test tidak menipu.
3. **Label konflik:** `Fade/Fade_08.jpg` dan `Partial Sour/Partial Sour_06.jpg` adalah foto
   yang sama dengan label berbeda. Keduanya dibuang. Mohon dicek mana label yang benar.
4. **Salah ketik nama folder** (`Fungus Damange`, `Severe Insect Damange`) diperbaiki
   otomatis di `config.py`. Folder asli tidak diubah.
5. **Data defect masih sedikit** (24–53 foto train per kelas). Target ideal ≥200 foto per
   kelas, difoto dari dua sisi biji.

Detail file yang dipakai/dibuang ada di `dataset/split_info.csv`.

## Rencana implementasi

### Tahap 1: Meja inspeksi (bisa dimulai sekarang, biaya rendah)
- Kotak foto (*lightbox*) dengan alas putih dan lampu LED merata, kamera di atas (webcam 1080p
  atau kamera HP).
- Sebar biji di alas (tidak saling menempel), lalu jalankan `sort_beans.py camera` atau `tray`.
- Layar menandai tiap biji dengan warna bin. Operator mengambil biji cacat sesuai tanda.
- Manfaat: data `ringkasan.txt` untuk grading per batch, dan **foto dari kamera sendiri bisa
  dikumpulkan untuk menggantikan data USK**.

### Tahap 2: Mesin sortir otomatis
```
Hopper → vibratory feeder (biji jalan satu-satu) → kamera (+cermin/2 kamera untuk 2 sisi)
       → PC mini / Jetson Orin Nano (model AI) → ESP32/Arduino → katup udara (air ejector)
       → 3 saluran: LOLOS | KATEGORI_2 | KATEGORI_1
```
- Semua kelas tetap dicatat untuk laporan, tetapi secara fisik cukup **3 saluran**. Biji
  CEK_MANUAL bisa diarahkan ke saluran KATEGORI_2 untuk diperiksa ulang.
- Kecepatan & pencahayaan harus dikunci. Kamera industri dengan *global shutter* dianjurkan
  untuk biji yang bergerak.

### Perangkat yang bisa menjalankan model

| Perangkat | Cocok untuk | Catatan |
|---|---|---|
| Laptop/PC (seperti sekarang) | Meja inspeksi, lab QC | Sudah teruji, ±41 biji/detik tanpa GPU |
| Mini PC (Intel NUC dsb.) | Meja inspeksi permanen | Ekspor ke OpenVINO agar lebih cepat di CPU Intel |
| NVIDIA Jetson Orin Nano | Mesin sortir otomatis | Ekspor ke TensorRT |
| Raspberry Pi 5 (+AI HAT) | Alat murah / portabel | Lebih lambat, cocok untuk volume kecil |
| HP Android | Cek mutu di lapangan (petani/pengepul) | Ekspor ke TFLite, perlu dibuatkan aplikasi |
| Server / web | Upload foto dari mana saja | Perlu dibuatkan API/web (mis. FastAPI) |

Ekspor: `.venv\Scripts\yolo export model=models/coffee_defect_cls.pt format=onnx`
(ganti `onnx` dengan `openvino`, `engine` (TensorRT), atau `tflite`).

### Tips foto agar AI akurat
- Kondisi foto harus **semirip mungkin dengan dataset**: alas putih polos, cahaya merata,
  biji mengisi ±40–50% bingkai bila difoto satu per satu.
- Pada foto tray, **setiap biji minimal ±180 piksel** (lihat tabel di bagian hasil).
- Biji jangan saling menempel. Biji yang menempel akan terbaca sebagai satu objek.
- Jika kamera/lampu di lapangan berbeda jauh, tambahkan foto dari setup tersebut ke dataset
  lalu latih ulang.
