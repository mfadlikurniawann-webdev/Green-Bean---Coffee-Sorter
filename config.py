"""Konfigurasi bersama: lokasi folder, nama kelas, dan aturan kategori/bin sortir."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW_DATA_DIR = ROOT / "Data"          # dataset asli (tidak pernah diubah oleh script)
EXTRA_DATA_DIR = ROOT / "Data_tambahan"  # data dari sumber lain (mis. biji bagus USK-Coffee)
DATASET_DIR = ROOT / "dataset"        # hasil prepare_dataset.py (train/val/test)
RUNS_DIR = ROOT / "runs"              # log & grafik training Ultralytics
REPORTS_DIR = ROOT / "reports"        # hasil evaluate.py
OUTPUT_DIR = ROOT / "output"          # hasil sort_beans.py
BEST_MODEL = ROOT / "models" / "coffee_defect_cls.pt"

# Perbaikan salah ketik nama folder di dataset asli.
CLASS_NAME_FIXES = {
    "Fungus Damange": "Fungus Damage",
    "Severe Insect Damange": "Severe Insect Damage",
}

# Nama folder yang dianggap biji BAGUS. Dataset saat ini belum punya kelas ini;
# cukup tambahkan folder Data/Good berisi foto biji normal lalu latih ulang.
GOOD_CLASSES = {"Good", "Normal", "Bagus"}

# Kategori defect mengikuti SCA Arabica Green Coffee Defect Handbook.
#   kategori 1 = primary defect (paling merusak rasa)  -> bin KATEGORI_1 (tolak)
#   kategori 2 = secondary defect                        -> bin KATEGORI_2 (grade rendah)
#   full_defect = jumlah biji cacat yang dihitung sebagai 1 "full defect" (dipakai untuk grade)
DEFECTS = {
    # nama kelas            (kategori, full_defect, nama Indonesia)
    "Full Black":            (1, 1,  "Hitam penuh"),
    "Full Sour":             (1, 1,  "Asam/sour penuh"),
    "Dry Cherry":            (1, 1,  "Gelondong kering"),
    "Fungus Damage":         (1, 1,  "Berjamur"),
    "Severe Insect Damage":  (1, 5,  "Lubang serangga parah"),
    "Partial Black":         (2, 3,  "Hitam sebagian"),
    "Partial Sour":          (2, 3,  "Asam/sour sebagian"),
    "Parchment":             (2, 5,  "Berkulit tanduk"),
    "Floater":               (2, 5,  "Biji mengambang"),
    "Immature":              (2, 5,  "Biji muda"),
    "Withered":              (2, 5,  "Biji keriput"),
    "Shell":                 (2, 5,  "Biji kerang (shell)"),
    "Broken":                (2, 5,  "Biji pecah"),
    "Cut":                   (2, 5,  "Biji terpotong"),
    "Husk":                  (2, 5,  "Kulit kering (husk)"),
    "Slight Insect Damage":  (2, 10, "Lubang serangga ringan"),
    # "Fade" tidak ada di handbook SCA; di sini DIASUMSIKAN kategori 2. Ubah bila perlu.
    "Fade":                  (2, 5,  "Pudar (fade)"),
    # Biji cacat dari USK-Coffee (Data_tambahan/) yang jenis cacatnya tidak dilabeli.
    # Tetap ditolak, tetapi jenisnya tidak diketahui -> perhitungan grade hanya perkiraan.
    "Other Defect":          (2, 5,  "Cacat lain (jenis tidak diketahui)"),
}

# Bin keluaran sortir. Di mesin fisik, tiap bin = satu saluran keluar / katup udara.
BIN_GOOD = "LOLOS"
BIN_CAT2 = "KATEGORI_2"
BIN_CAT1 = "KATEGORI_1"
BIN_MANUAL = "CEK_MANUAL"   # model kurang yakin -> diperiksa manusia
BINS = [BIN_GOOD, BIN_CAT2, BIN_CAT1, BIN_MANUAL]

# Kelas yang tidak dikenal di DEFECTS dianggap kategori 2 supaya tidak lolos sebagai biji bagus.
DEFAULT_CATEGORY = 2

# Prediksi dengan keyakinan di bawah ini masuk bin CEK_MANUAL.
# Lihat tabel threshold di output evaluate.py untuk memilih nilai yang pas.
DEFAULT_CONF_THRESHOLD = 0.5
# Untuk MELOLOSKAN biji sebagai bagus, AI harus jauh lebih yakin: biji cacat yang lolos
# adalah kesalahan paling mahal. Di bawah nilai ini, tebakan "Good" masuk CEK_MANUAL.
# Pada uji biji cacat USK: 0,5 -> 17,8% lolos, 0,9 -> 8,8% lolos (biji bagus lolos 96,8% -> 91,8%).
GOOD_CONF_THRESHOLD = 0.9


def is_good(name: str) -> bool:
    return name in GOOD_CLASSES


def category_of(name: str) -> int:
    """0 = bagus, 1 = defect primer, 2 = defect sekunder."""
    if is_good(name):
        return 0
    return DEFECTS.get(name, (DEFAULT_CATEGORY,))[0]


def bin_for_class(name: str) -> str:
    return {0: BIN_GOOD, 1: BIN_CAT1, 2: BIN_CAT2}[category_of(name)]


def sort_decision(name: str, conf: float, threshold: float = DEFAULT_CONF_THRESHOLD,
                  good_threshold: float = GOOD_CONF_THRESHOLD) -> str:
    """Keputusan bin mesin sortir untuk 1 biji."""
    if conf < (good_threshold if is_good(name) else threshold):
        return BIN_MANUAL
    return bin_for_class(name)


def display_name(name: str) -> str:
    if is_good(name):
        return "Biji bagus"
    return DEFECTS[name][2] if name in DEFECTS else name


def sca_grade(class_counts: dict) -> dict:
    """Hitung full defect & grade SCA dari jumlah biji per kelas.

    Catatan: standar SCA berlaku untuk sampel 350 g green bean (sekitar 2.000-2.500 biji),
    jadi hasil pada sampel yang lebih kecil hanya perkiraan.
    """
    full_cat1 = full_cat2 = 0
    for name, n in class_counts.items():
        if is_good(name) or name not in DEFECTS:
            continue
        cat, per_full, _ = DEFECTS[name]
        if cat == 1:
            full_cat1 += n // per_full
        else:
            full_cat2 += n // per_full
    total = full_cat1 + full_cat2
    if full_cat1 == 0 and total <= 5:
        grade = "Specialty Grade (Grade 1)"
    elif total <= 8:
        grade = "Premium Grade (Grade 2)"
    elif total <= 23:
        grade = "Exchange Grade (Grade 3)"
    elif total <= 86:
        grade = "Below Standard (Grade 4)"
    else:
        grade = "Off Grade (Grade 5)"
    return {"full_defect_cat1": full_cat1, "full_defect_cat2": full_cat2,
            "full_defect_total": total, "grade": grade}
