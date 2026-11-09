"""Aplikasi web lokal untuk mencoba AI sortir biji kopi di browser.

Jalankan:
  .venv\\Scripts\\python app.py
lalu buka http://127.0.0.1:7860

Upload foto berisi satu atau banyak biji di atas alas putih (atau pakai contoh di bawah).
Setiap biji dideteksi, diklasifikasi, dan diberi kotak warna sesuai bin sortir.
"""
from collections import Counter

import numpy as np

import cv2
import gradio as gr
import pandas as pd

from config import (BEST_MODEL, BINS, DEFAULT_CONF_THRESHOLD, GOOD_CONF_THRESHOLD, ROOT,
                    sca_grade, sort_decision)
from sort_beans import BeanSorter, annotate, find_beans

sorter = BeanSorter(BEST_MODEL)
BIN_LABEL = {"LOLOS": "🟢 LOLOS (bagus)", "KATEGORI_2": "🟠 KATEGORI 2 (cacat sekunder)",
             "KATEGORI_1": "🔴 KATEGORI 1 (cacat primer)", "CEK_MANUAL": "⚪ CEK MANUAL (AI ragu)"}
COLUMNS = ["No", "Jenis", "Keterangan", "Keyakinan", "Bin"]


def analyze(image_rgb, good_threshold, threshold):
    if image_rgb is None:
        return None, "Upload foto biji kopi dulu.", pd.DataFrame(columns=COLUMNS)
    img = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    boxes, crops = find_beans(img)
    if not crops:
        return image_rgb, ("**Tidak ada biji terdeteksi.** Pastikan biji di atas alas putih/terang, "
                           "tidak saling menempel, dan foto cukup tajam."), pd.DataFrame(columns=COLUMNS)

    results = sorter.classify(crops)
    for r in results:  # terapkan batas dari slider
        r["bin"] = sort_decision(r["kelas"], r["confidence"], threshold, good_threshold)

    vis = cv2.cvtColor(annotate(img, boxes, results), cv2.COLOR_BGR2RGB)
    bins = Counter(r["bin"] for r in results)
    n = len(results)
    bean_px = int(np.median([max(w, h) for _, _, w, h in boxes]))

    md = [f"### {n} biji terdeteksi", "| Bin | Jumlah | % |", "|---|---:|---:|"]
    md += [f"| {BIN_LABEL[b]} | {bins.get(b, 0)} | {bins.get(b, 0) / n:.0%} |" for b in BINS]
    g = sca_grade(Counter(r["kelas"] for r in results))
    md += ["", f"**Estimasi grade SCA:** {g['grade']} · full defect kategori 1 = {g['full_defect_cat1']}, "
               f"kategori 2 = {g['full_defect_cat2']}",
           "<small>Grade SCA resminya dihitung per sampel 350 g (±2.000 biji); di sini hanya perkiraan.</small>"]
    if bean_px < 150:
        md += ["", f"⚠️ Ukuran biji di foto rata-rata hanya ±{bean_px} piksel. Akurasi turun bila biji terlalu "
                   "kecil; dekatkan kamera atau pakai resolusi lebih tinggi (target ±180 piksel per biji)."]

    table = pd.DataFrame([[i, r["kelas"], r["nama_id"], f"{r['confidence']:.0%}", BIN_LABEL[r["bin"]]]
                          for i, r in enumerate(results, 1)], columns=COLUMNS)
    return vis, "\n".join(md), table


EXAMPLES = [str(ROOT / "demo" / "tray_campuran.jpg"), str(ROOT / "demo" / "tray_contoh.jpg")]

with gr.Blocks(title="AI Sortir Biji Kopi") as demo:
    gr.Markdown("# ☕ AI Sortir Biji Kopi\nUpload foto biji kopi hijau (green bean) di atas alas putih. "
                "Bisa satu biji atau banyak biji sekaligus.")
    with gr.Row():
        with gr.Column(scale=1):
            inp = gr.Image(type="numpy", label="Foto biji", sources=["upload", "clipboard", "webcam"])
            good_thr = gr.Slider(0.5, 0.99, value=GOOD_CONF_THRESHOLD, step=0.01,
                                 label="Batas keyakinan untuk LOLOS",
                                 info="Makin tinggi, makin sedikit biji cacat yang lolos (tapi lebih banyak cek manual)")
            thr = gr.Slider(0.1, 0.95, value=DEFAULT_CONF_THRESHOLD, step=0.05,
                            label="Batas keyakinan untuk tebakan cacat")
            btn = gr.Button("Analisis", variant="primary")
            gr.Examples(EXAMPLES, inputs=inp, label="Contoh foto (klik lalu tekan Analisis)")
        with gr.Column(scale=2):
            out_img = gr.Image(label="Hasil (kotak warna = bin sortir)")
            out_md = gr.Markdown()
    out_table = gr.Dataframe(label="Detail per biji", headers=COLUMNS, wrap=True)

    btn.click(analyze, [inp, good_thr, thr], [out_img, out_md, out_table])

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True)
