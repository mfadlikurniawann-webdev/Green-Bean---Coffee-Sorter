"""Sistem sortir biji kopi berbasis AI.

Mode:
  folder : setiap gambar berisi 1 biji (seperti dataset). Gambar disalin ke
           output/<nama>/<BIN>/<kelas>/ sesuai hasil klasifikasi.
  tray   : 1 foto berisi banyak biji di atas alas putih. Tiap biji dicari otomatis,
           diklasifikasi, ditandai kotak berwarna, dan potongannya disortir ke folder bin.
  camera : sama seperti tray, langsung dari webcam / kamera USB / IP camera HP.

Contoh:
  python sort_beans.py folder --input "D:/foto_biji"
  python sort_beans.py tray   --input tray.jpg
  python sort_beans.py camera --cam 0
"""
import argparse
import csv
import shutil
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from config import (BEST_MODEL, BIN_CAT1, BIN_CAT2, BIN_GOOD, BIN_MANUAL, BINS,
                    DEFAULT_CONF_THRESHOLD, GOOD_CONF_THRESHOLD, OUTPUT_DIR, display_name,
                    is_good, sca_grade, sort_decision)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
BIN_COLORS = {BIN_GOOD: (60, 180, 75), BIN_CAT2: (0, 165, 255),      # warna BGR
              BIN_CAT1: (40, 40, 220), BIN_MANUAL: (150, 150, 150)}
# Di dataset, satu biji mengisi +-44% sisi gambar -> potongan dibuat 2.2x ukuran biji.
CROP_SCALE = 2.2


# ---------------------------------------------------------------- klasifikasi
class BeanSorter:
    def __init__(self, model_path: Path, threshold: float = DEFAULT_CONF_THRESHOLD,
                 device=None, good_threshold: float = GOOD_CONF_THRESHOLD):
        if not Path(model_path).exists():
            raise SystemExit(f"Model tidak ditemukan: {model_path}\nJalankan train.py dulu.")
        self.model = YOLO(str(model_path))
        self.names = self.model.names
        self.threshold = threshold
        self.good_threshold = good_threshold
        self.kwargs = {"verbose": False, **({"device": device} if device else {})}
        self.has_good_class = any(is_good(n) for n in self.names.values())

    def classify(self, images):
        """images: list path gambar atau array BGR. Return list dict hasil per biji."""
        results = []
        for i in range(0, len(images), 32):
            for r in self.model.predict(images[i:i + 32], **self.kwargs):
                top5 = [self.names[int(k)] for k in r.probs.top5]
                name, conf = top5[0], float(r.probs.top1conf)
                results.append({
                    "kelas": name,
                    "nama_id": display_name(name),
                    "confidence": round(conf, 4),
                    "bin": sort_decision(name, conf, self.threshold, self.good_threshold),
                    "kandidat_2": top5[1],
                })
        return results


# ---------------------------------------------------------------- deteksi biji
def find_beans(img: np.ndarray, min_area_frac: float = 0.0003, work_size: int = 1024):
    """Cari biji di atas alas terang. Return (boxes, crops). box = (x, y, w, h).

    Posisi biji dicari pada versi foto yang diperkecil (sisi terpanjang = work_size) supaya
    cepat; potongan untuk AI tetap diambil dari foto resolusi penuh.
    """
    s = min(1.0, work_size / max(img.shape[:2]))
    work = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else img
    h, w = work.shape[:2]
    min_area = min_area_frac * h * w
    lab = cv2.cvtColor(cv2.GaussianBlur(work, (5, 5), 0), cv2.COLOR_BGR2LAB).astype(np.float32)

    # Tahap 1: warna alas = median piksel tepi foto -> perkiraan kasar posisi & ukuran biji.
    b = max(2, min(h, w) // 50)
    border = np.concatenate([lab[:b].reshape(-1, 3), lab[-b:].reshape(-1, 3),
                             lab[:, :b].reshape(-1, 3), lab[:, -b:].reshape(-1, 3)])
    diff = _color_diff(lab, np.median(border, axis=0))
    otsu, _ = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    rough = _bean_contours(diff > max(otsu, 30), min_area)
    if not rough:
        return [], []

    # Tahap 2: warna alas LOKAL (median filter selebar +-4 biji). Tahan terhadap cahaya
    # yang tidak rata, dan biji pucat (Floater/Fade) yang mirip alas tetap terdeteksi.
    bean_px = float(np.median([max(cv2.boundingRect(c)[2:]) for c in rough]))
    f = min(1.0, 8.0 / bean_px)  # kecilkan foto sampai 1 biji +-8 piksel supaya cepat
    small = cv2.resize(work, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    k = max(3, min(int(4 * bean_px * f) | 1, (min(small.shape[:2]) - 1) | 1))
    bg_bgr = cv2.resize(cv2.medianBlur(small, k), (w, h), interpolation=cv2.INTER_LINEAR)
    diff = _color_diff(lab, cv2.cvtColor(bg_bgr, cv2.COLOR_BGR2LAB).astype(np.float32))
    otsu, _ = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours = _bean_contours(diff > np.clip(otsu, 20, 30), min_area)

    boxes = [cv2.boundingRect(c) for c in contours]
    if boxes:  # urutkan seperti membaca: baris atas ke bawah, kiri ke kanan
        row_h = max(1, int(np.median([bh for _, _, _, bh in boxes])))
        boxes.sort(key=lambda bx: ((bx[1] + bx[3] // 2) // row_h, bx[0]))

    full_boxes, crops = [], []
    for x, y, bw, bh in boxes:
        fill = bg_bgr[y + bh // 2, x + bw // 2]
        x, y, bw, bh = round(x / s), round(y / s), round(bw / s), round(bh / s)  # ke resolusi penuh
        full_boxes.append((x, y, bw, bh))
        crops.append(square_crop(img, x + bw // 2, y + bh // 2, int(max(bw, bh) * CROP_SCALE), fill))
    return full_boxes, crops


# L (terang-gelap) diberi bobot kecil: bayangan tipis diabaikan, sedangkan biji pucat
# tetap terlihat karena warnanya (kekuningan/kehijauan) beda dengan alas putih.
LAB_WEIGHTS = np.array([0.5, 1.0, 1.0], np.float32)


def _color_diff(lab, bg_lab):
    d = np.sqrt((((lab - bg_lab) * LAB_WEIGHTS) ** 2).sum(axis=2)) * 2
    return np.clip(d, 0, 255).astype(np.uint8)


def _bean_contours(mask, min_area):
    mask = mask.astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) >= min_area]
    if contours:  # buang debu/kotoran yang jauh lebih kecil dari biji pada umumnya
        median_area = np.median([cv2.contourArea(c) for c in contours])
        contours = [c for c in contours if cv2.contourArea(c) >= 0.25 * median_area]
    return contours


def square_crop(img, cx, cy, side, fill):
    """Potong persegi berpusat di (cx, cy); bagian di luar foto diisi warna alas."""
    h, w = img.shape[:2]
    x0, y0 = cx - side // 2, cy - side // 2
    crop = np.empty((side, side, 3), dtype=img.dtype)
    crop[:] = fill
    sx0, sy0, sx1, sy1 = max(x0, 0), max(y0, 0), min(x0 + side, w), min(y0 + side, h)
    crop[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = img[sy0:sy1, sx0:sx1]
    return crop


def annotate(img, boxes, results):
    vis = img.copy()
    thick = max(2, min(img.shape[:2]) // 400)
    font = max(0.45, min(img.shape[:2]) / 1600)
    for i, ((x, y, w, h), r) in enumerate(zip(boxes, results), 1):
        color = BIN_COLORS[r["bin"]]
        cv2.rectangle(vis, (x, y), (x + w, y + h), color, thick)
        label = f"{i}. {r['kelas']} {r['confidence']:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font, 1)
        ty = max(y - 4, th + 4)
        tx = max(0, min(x, vis.shape[1] - tw - 4))  # jaga agar label tidak keluar dari foto
        cv2.rectangle(vis, (tx, ty - th - 4), (tx + tw + 4, ty + 2), color, -1)
        cv2.putText(vis, label, (tx + 2, ty - 2), cv2.FONT_HERSHEY_SIMPLEX, font, (255, 255, 255), 1, cv2.LINE_AA)
    return vis


# ---------------------------------------------------------------- laporan
def write_csv(path: Path, rows):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def summarize(results, sorter: BeanSorter, out_dir: Path):
    bins = Counter(r["bin"] for r in results)
    classes = Counter(r["kelas"] for r in results)
    total = len(results)
    lines = [f"Total biji: {total}", "", "Per bin sortir:"]
    lines += [f"  {b:<12} {bins.get(b, 0):>5}  ({bins.get(b, 0) / total:.1%})" for b in BINS]
    lines += ["", "Per kelas (tebakan teratas):"]
    lines += [f"  {c:<22} {display_name(c):<24} {n:>5}" for c, n in classes.most_common()]
    g = sca_grade(classes)
    lines += ["", "Estimasi grade SCA (berlaku untuk sampel 350 g):",
              f"  Full defect kategori 1: {g['full_defect_cat1']}",
              f"  Full defect kategori 2: {g['full_defect_cat2']}",
              f"  Total full defect     : {g['full_defect_total']}  ->  {g['grade']}"]
    if not sorter.has_good_class:
        lines += ["", "PERINGATAN: model belum punya kelas biji bagus (Good). Semua biji akan",
                  "dianggap cacat. Tambahkan folder Data/Good lalu jalankan prepare_dataset.py & train.py."]
    text = "\n".join(lines)
    print(text)
    (out_dir / "ringkasan.txt").write_text(text, encoding="utf-8")


def read_image(path: Path):
    return cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)  # aman untuk path non-ASCII


def write_image(path: Path, img):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imencode(path.suffix or ".jpg", img)[1].tofile(str(path))


# ---------------------------------------------------------------- mode
def run_folder(sorter: BeanSorter, input_dir: Path, out_dir: Path):
    files = sorted(p for p in input_dir.rglob("*") if p.suffix.lower() in IMG_EXTS)
    if not files:
        raise SystemExit(f"Tidak ada gambar di {input_dir}")
    print(f"Mengklasifikasi {len(files)} gambar ...")
    results = sorter.classify([str(f) for f in files])
    rows = []
    for f, r in zip(files, results):
        dst = out_dir / r["bin"] / r["kelas"] / f.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, dst)
        rows.append({"file": str(f.relative_to(input_dir)), **r})
    write_csv(out_dir / "hasil_sortir.csv", rows)
    summarize(results, sorter, out_dir)


def process_tray_image(sorter, img, out_dir: Path, stem: str):
    boxes, crops = find_beans(img)
    results = sorter.classify(crops) if crops else []
    vis = annotate(img, boxes, results)
    write_image(out_dir / f"{stem}_hasil.jpg", vis)
    rows = []
    for i, (box, crop, r) in enumerate(zip(boxes, crops, results), 1):
        write_image(out_dir / r["bin"] / r["kelas"] / f"{stem}_biji{i:03d}.jpg", crop)
        rows.append({"foto": stem, "biji": i, "x": box[0], "y": box[1], "w": box[2], "h": box[3], **r})
    return results, rows, vis


def run_tray(sorter: BeanSorter, input_path: Path, out_dir: Path):
    files = [input_path] if input_path.is_file() else sorted(
        p for p in input_path.rglob("*") if p.suffix.lower() in IMG_EXTS)
    all_results, all_rows = [], []
    for f in files:
        results, rows, _ = process_tray_image(sorter, read_image(f), out_dir, f.stem)
        print(f"{f.name}: {len(results)} biji terdeteksi")
        all_results += results
        all_rows += rows
    if not all_rows:
        raise SystemExit("Tidak ada biji terdeteksi. Pastikan alas berwarna terang dan biji tidak saling menempel.")
    write_csv(out_dir / "hasil_sortir.csv", all_rows)
    summarize(all_results, sorter, out_dir)


def run_camera(sorter: BeanSorter, cam, out_dir: Path):
    cap = cv2.VideoCapture(cam)
    if not cap.isOpened():
        raise SystemExit(f"Kamera {cam} tidak bisa dibuka.")
    print("Tekan 's' untuk menyimpan hasil frame saat ini, 'q' untuk keluar.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        boxes, crops = find_beans(frame)
        results = sorter.classify(crops) if crops else []
        vis = annotate(frame, boxes, results)
        counts = Counter(r["bin"] for r in results)
        info = " | ".join(f"{b}:{counts.get(b, 0)}" for b in BINS)
        cv2.putText(vis, info, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.imshow("Sortir Biji Kopi (s: simpan, q: keluar)", vis)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s") and results:
            stem = time.strftime("kamera_%Y%m%d_%H%M%S")
            _, rows, _ = process_tray_image(sorter, frame, out_dir, stem)
            write_csv(out_dir / f"{stem}.csv", rows)
            print(f"Disimpan: {out_dir / (stem + '_hasil.jpg')}")
    cap.release()
    cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["folder", "tray", "camera"])
    ap.add_argument("--input", type=Path, help="folder gambar (folder) atau foto/folder foto tray (tray)")
    ap.add_argument("--cam", default="0", help="index webcam (0, 1, ...) atau URL IP camera")
    ap.add_argument("--model", type=Path, default=BEST_MODEL)
    ap.add_argument("--good-threshold", type=float, default=GOOD_CONF_THRESHOLD,
                    help="keyakinan minimal untuk meloloskan biji sebagai bagus")
    ap.add_argument("--threshold", type=float, default=DEFAULT_CONF_THRESHOLD,
                    help="di bawah nilai ini biji masuk CEK_MANUAL")
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--device", default=None, help="contoh: 0 (GPU) atau cpu")
    args = ap.parse_args()

    sorter = BeanSorter(args.model, args.threshold, args.device, args.good_threshold)
    out_dir = args.output or OUTPUT_DIR / time.strftime(f"{args.mode}_%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "camera":
        run_camera(sorter, int(args.cam) if args.cam.isdigit() else args.cam, out_dir)
    else:
        if not args.input or not args.input.exists():
            raise SystemExit("--input wajib diisi dan harus ada.")
        (run_folder if args.mode == "folder" else run_tray)(sorter, args.input, out_dir)
    print(f"\nHasil disimpan di: {out_dir}")


if __name__ == "__main__":
    main()
