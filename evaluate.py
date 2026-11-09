"""Evaluasi model pada data test (data yang tidak pernah dilihat saat training).

Yang dilaporkan:
  - akurasi per kelas
  - akurasi BIN sortir: salah tebak antar kelas yang satu kategori (mis. Partial Black
    vs Partial Sour) tidak masalah bagi mesin sortir karena masuk saluran yang sama
  - kesalahan kritis: defect kategori 1 yang tidak masuk bin KATEGORI_1
  - bila ada kelas Good: defect yang LOLOS sebagai biji bagus (kesalahan paling mahal)
    dan biji bagus yang ikut ditolak
  - tes jebakan bila kelas Good berasal dari sumber luar (Data_tambahan/), untuk memastikan
    AI menilai BIJI-nya, bukan gaya/asal fotonya:
      B. foto defect test diubah bergaya sumber luar      -> tidak boleh jadi Good
      A. --external-defects DIR (defect dari sumber luar) -> tidak boleh jadi Good
      C. --external-good DIR (biji bagus sumber luar yang tidak dipakai training) -> harus Good
  - tabel threshold untuk memilih batas CEK_MANUAL
Hasil disimpan di reports/ (teks + confusion matrix dari Ultralytics).

Contoh:
  python evaluate.py
  python evaluate.py --external-defects "D:/USK-Coffee/test/defect" --external-good "D:/USK-Coffee/test/premium"
"""
import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report
from ultralytics import YOLO

from config import (BEST_MODEL, BIN_CAT1, BIN_GOOD, BIN_MANUAL, DATASET_DIR, DEFAULT_CONF_THRESHOLD,
                    EXTRA_DATA_DIR, GOOD_CONF_THRESHOLD, REPORTS_DIR, bin_for_class, is_good,
                    sort_decision)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, default=BEST_MODEL)
    ap.add_argument("--split", default="test", choices=["test", "val"])
    ap.add_argument("--threshold", type=float, default=DEFAULT_CONF_THRESHOLD)
    ap.add_argument("--good-threshold", type=float, default=GOOD_CONF_THRESHOLD)
    ap.add_argument("--external-defects", type=Path, help="folder foto defect dari sumber luar (tes A)")
    ap.add_argument("--external-good", type=Path, help="folder foto biji bagus sumber luar, di luar training (tes C)")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    model = YOLO(str(args.model))
    names = model.names
    idx_of = {v: k for k, v in names.items()}
    kw = {"verbose": False, **({"device": args.device} if args.device else {})}

    def predict(images):
        return np.concatenate([
            np.stack([r.probs.data.cpu().numpy() for r in model.predict(images[i:i + 32], **kw)])
            for i in range(0, len(images), 32)])

    files, y_true = [], []
    for cls_dir in sorted(p for p in (DATASET_DIR / args.split).iterdir() if p.is_dir()):
        for f in sorted(cls_dir.iterdir()):
            files.append(str(f))
            y_true.append(idx_of[cls_dir.name])
    y_true = np.array(y_true)

    probs = predict(files)
    y_pred, conf = probs.argmax(1), probs.max(1)
    top3 = (np.argsort(-probs, 1)[:, :3] == y_true[:, None]).any(1)

    to_bin = np.vectorize(lambda i: bin_for_class(names[int(i)]))
    true_bin, pred_bin = to_bin(y_true), to_bin(y_pred)
    sorted_bin = np.array([sort_decision(names[int(i)], c, args.threshold, args.good_threshold)
                           for i, c in zip(y_pred, conf)])  # keputusan mesin sortir
    cat1 = true_bin == BIN_CAT1
    label_names = [names[i] for i in sorted(names)]
    has_good = any(is_good(n) for n in names.values())

    lines = [f"Model : {args.model}", f"Data  : {args.split} ({len(files)} gambar)", "",
             f"Akurasi kelas (top-1)  : {(y_pred == y_true).mean():.1%}",
             f"Akurasi kelas (top-3)  : {top3.mean():.1%}",
             f"Akurasi bin sortir     : {(pred_bin == true_bin).mean():.1%}",
             f"Defect kategori 1 masuk KATEGORI_1: {(pred_bin[cat1] == BIN_CAT1).mean():.1%}"
             f" ({(pred_bin[cat1] == BIN_CAT1).sum()}/{cat1.sum()})"]

    if has_good:
        good, defect = true_bin == BIN_GOOD, true_bin != BIN_GOOD
        leak = defect & (sorted_bin == BIN_GOOD)
        lines += ["", f"Keputusan sortir (threshold {args.threshold}, batas LOLOS {args.good_threshold}):",
                  f"  Defect yang LOLOS sebagai biji bagus : {leak.sum()}/{defect.sum()} ({leak.mean() / defect.mean():.1%})"
                  f"  <- kesalahan paling mahal",
                  f"    termasuk defect kategori 1         : {(leak & cat1).sum()}/{cat1.sum()}",
                  f"  Biji bagus masuk bin LOLOS           : {(good & (sorted_bin == BIN_GOOD)).sum()}/{good.sum()}",
                  f"  Biji bagus ikut ditolak (salah tolak): {(good & (sorted_bin != BIN_GOOD) & (sorted_bin != BIN_MANUAL)).sum()}/{good.sum()}",
                  f"  Biji bagus ke CEK_MANUAL             : {(good & (sorted_bin == BIN_MANUAL)).sum()}/{good.sum()}"]

    lines += ["", classification_report(y_true, y_pred, labels=sorted(names), target_names=label_names,
                                        digits=3, zero_division=0),
              "Tabel threshold (prediksi di bawah threshold -> CEK_MANUAL):",
              f"  {'threshold':>9} {'cek manual':>11} {'akurasi kelas*':>15} {'akurasi bin*':>13}"
              + (f" {'defect lolos':>13}" if has_good else "")]
    for t in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        keep = conf >= t
        if keep.any():
            row = (f"  {t:>9.1f} {1 - keep.mean():>11.1%} {(y_pred[keep] == y_true[keep]).mean():>15.1%}"
                   f" {(pred_bin[keep] == true_bin[keep]).mean():>13.1%}")
            if has_good:
                row += f" {(keep & (true_bin != BIN_GOOD) & (pred_bin == BIN_GOOD)).sum():>13d}"
            lines.append(row)
    lines.append("  * dihitung hanya pada biji yang tidak masuk CEK_MANUAL")

    # ---------------- tes jebakan: apakah AI menilai biji, atau gaya/asal fotonya?
    if has_good and EXTRA_DATA_DIR.exists():
        from import_good_beans import harmonize, reference_background, to_external_style
        from sort_beans import read_image
        target_bg, rng = reference_background(), np.random.default_rng(0)

        def good_rate(images):
            p = predict(images)
            is_g = np.array([sort_decision(names[int(i)], c, args.threshold, args.good_threshold) == BIN_GOOD
                             for i, c in zip(p.argmax(1), p.max(1))])
            return is_g.sum(), len(images)

        lines += ["", f"Tes jebakan (AI harus menilai biji, bukan gaya/asal foto; batas LOLOS {args.good_threshold}):"]
        styled = [to_external_style(read_image(Path(f)), target_bg, rng)
                  for f, t in zip(files, y_true) if not is_good(names[int(t)])]
        styled = [s for s in styled if s is not None]
        n, total = good_rate(styled)
        lines.append(f"  B. Defect test diberi gaya sumber luar, dikira Good : {n}/{total} ({n / total:.1%})  [harus rendah]")
        for label, folder, want in [("A. Defect sumber luar, dikira Good", args.external_defects, "rendah"),
                                    ("C. Biji bagus sumber luar (di luar training) dikenali Good", args.external_good, "tinggi")]:
            if folder:
                imgs = [harmonize(read_image(f), target_bg, rng)
                        for f in sorted(folder.rglob("*")) if f.suffix.lower() in IMG_EXTS]
                imgs = [i for i in imgs if i is not None]
                n, total = good_rate(imgs)
                lines.append(f"  {label} : {n}/{total} ({n / total:.1%})  [harus {want}]")

    errors = [(Path(files[i]).name, names[y_true[i]], names[y_pred[i]], conf[i])
              for i in np.nonzero(y_pred != y_true)[0]]
    lines += ["", "Gambar yang salah tebak (file | label asli -> tebakan, confidence):"]
    lines += [f"  {f} | {t} -> {p} ({c:.2f})" for f, t, p, c in errors]

    text = "\n".join(lines)
    print(text)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORTS_DIR / f"evaluasi_{args.split}.txt"
    out.write_text(text, encoding="utf-8")

    # Confusion matrix (gambar) dibuat oleh validator bawaan Ultralytics.
    model.val(data=str(DATASET_DIR), split=args.split, project=str(REPORTS_DIR),
              name=f"confusion_matrix_{args.split}", exist_ok=True, plots=True, **kw)
    print(f"\nLaporan: {out}\nConfusion matrix: {REPORTS_DIR / f'confusion_matrix_{args.split}'}")


if __name__ == "__main__":
    main()
