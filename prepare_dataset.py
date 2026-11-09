"""Menyiapkan dataset untuk training dari folder Data/ (dan Data_tambahan/ bila ada).

Langkah:
  1. Membaca semua gambar per kelas (nama folder salah ketik diperbaiki, lihat config.py).
     Kelas dengan nama sama di Data/ dan Data_tambahan/ digabung.
  2. Membuang duplikat: gambar kembar di kelas yang sama disimpan satu saja; gambar kembar
     dengan label BERBEDA dibuang keduanya karena labelnya tidak bisa dipercaya.
  3. Split per kelas (stratified) menjadi train / val / test.
  4. Augmentasi untuk data train saja: rotasi (biji bisa jatuh dengan posisi apa pun) dan
     sebagian diturunkan resolusinya, supaya ketajaman foto tidak dipakai AI sebagai
     petunjuk (foto dari sumber lain biasanya beresolusi lebih rendah).
  5. Bila ada data dari Data_tambahan/, sebagian foto kelas lain diberi "gaya" foto sumber
     luar (lihat import_good_beans.to_external_style), supaya AI tidak bisa membedakan
     kelas hanya dari asal fotonya.

Folder Data/ tidak diubah. Hasilnya ditulis ke dataset/ dan bisa dibuat ulang kapan saja.
"""
import argparse
import csv
import random
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from config import CLASS_NAME_FIXES, DATASET_DIR, EXTRA_DATA_DIR, RAW_DATA_DIR

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MARKER = ".generated_by_prepare_dataset"


def dhash(path: Path, size: int = 16) -> np.ndarray:
    """Hash perseptual: gambar yang mirip secara visual punya hash yang mirip."""
    g = np.asarray(Image.open(path).convert("L").resize((size + 1, size)), dtype=np.int16)
    return (g[:, 1:] > g[:, :-1]).flatten()


def collect(raw_dirs):
    items = []
    for raw_dir in raw_dirs:
        for cls_dir in sorted(p for p in raw_dir.iterdir() if p.is_dir()):
            cls = CLASS_NAME_FIXES.get(cls_dir.name, cls_dir.name)
            for f in sorted(cls_dir.iterdir()):
                if f.suffix.lower() in IMG_EXTS:
                    items.append((f, cls))
    return items


def source_name(f: Path) -> str:
    """Contoh: Data/Broken/Broken_01.jpg"""
    return f"{f.parent.parent.name}/{f.parent.name}/{f.name}"


def thumb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L").resize((64, 64)), dtype=np.float32)


def find_duplicates(items, max_dist: int, max_pixel_diff: float = 2.0):
    """Cari foto kembar. Tahap 1: hash perseptual (cepat, tapi bisa alarm palsu pada foto
    dengan latar sangat seragam). Tahap 2: konfirmasi selisih piksel rata-rata (kembar asli +-0)."""
    hashes = np.array([dhash(f) for f, _ in items])
    excluded = {}  # index -> alasan
    for i in range(len(items)):
        dist = (hashes[i + 1:] != hashes[i]).sum(1)
        for k in np.nonzero(dist <= max_dist)[0]:
            j = i + 1 + int(k)
            (fi, ci), (fj, cj) = items[i], items[j]
            if np.abs(thumb(fi) - thumb(fj)).mean() > max_pixel_diff:
                continue  # mirip sekilas, tetapi foto berbeda
            if ci == cj:
                excluded.setdefault(j, f"duplikat dari {fi.parent.name}/{fi.name}")
            else:
                excluded[i] = f"label bertentangan dengan {fj.parent.name}/{fj.name}"
                excluded[j] = f"label bertentangan dengan {fi.parent.name}/{fi.name}"
    return excluded


def rotate_with_background(img: Image.Image, angle: float) -> Image.Image:
    """Putar gambar dan isi sudut kosong dengan warna latar (median piksel tepi)."""
    a = np.asarray(img)
    border = np.concatenate([a[:4].reshape(-1, 3), a[-4:].reshape(-1, 3),
                             a[:, :4].reshape(-1, 3), a[:, -4:].reshape(-1, 3)])
    fill = tuple(int(v) for v in np.median(border, axis=0))
    return img.rotate(angle, resample=Image.BICUBIC, fillcolor=fill)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, nargs="+", default=[RAW_DATA_DIR, EXTRA_DATA_DIR],
                    help="folder sumber (yang tidak ada dilewati)")
    ap.add_argument("--out", type=Path, default=DATASET_DIR)
    ap.add_argument("--val", type=float, default=0.15, help="porsi data validasi")
    ap.add_argument("--test", type=float, default=0.15, help="porsi data test")
    ap.add_argument("--rotations", type=int, default=3, help="jumlah salinan rotasi per gambar train")
    ap.add_argument("--lowres", type=float, default=0.3,
                    help="peluang salinan rotasi diturunkan resolusinya (0 = tidak)")
    ap.add_argument("--style-mix", type=float, default=0.35,
                    help="peluang foto train dari Data/ diberi salinan bergaya sumber luar")
    ap.add_argument("--dup-dist", type=int, default=10, help="batas jarak hash untuk dianggap duplikat")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.out.exists():
        if not (args.out / MARKER).exists():
            raise SystemExit(f"{args.out} sudah ada dan bukan buatan script ini. Hapus/pindahkan dulu secara manual.")
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True)
    (args.out / MARKER).touch()

    raw_dirs = [d for d in args.raw if d.is_dir()]
    print("Sumber:", ", ".join(str(d) for d in raw_dirs))
    items = collect(raw_dirs)
    excluded = find_duplicates(items, args.dup_dist)
    rng = random.Random(args.seed)

    extra_root = EXTRA_DATA_DIR.resolve()
    is_external = lambda f: extra_root in f.resolve().parents  # noqa: E731
    use_style = args.style_mix > 0 and any(is_external(f) for f, _ in items)
    if use_style:
        import cv2
        from import_good_beans import reference_background, to_external_style
        target_bg, np_rng = reference_background(), np.random.default_rng(args.seed)

    by_class = {}
    for idx, (f, cls) in enumerate(items):
        if idx not in excluded:
            by_class.setdefault(cls, []).append(f)

    rows = []
    print(f"{'Kelas':<24}{'train':>7}{'val':>6}{'test':>6}")
    for cls, files in sorted(by_class.items()):
        rng.shuffle(files)
        n_val = max(1, round(len(files) * args.val))
        n_test = max(1, round(len(files) * args.test))
        splits = {"test": files[:n_test], "val": files[n_test:n_test + n_val], "train": files[n_test + n_val:]}
        for split, split_files in splits.items():
            dst_dir = args.out / split / cls
            dst_dir.mkdir(parents=True, exist_ok=True)
            for f in split_files:
                shutil.copy2(f, dst_dir / f.name)
                rows.append([source_name(f), cls, split, ""])
                if split == "train":
                    img = Image.open(f).convert("RGB")
                    for _ in range(args.rotations):
                        angle = rng.uniform(20, 340)
                        aug = rotate_with_background(img, angle)
                        if rng.random() < args.lowres:
                            side = rng.randint(220, 360)
                            aug = aug.resize((side, side), Image.BILINEAR).resize(img.size, Image.BICUBIC)
                        aug.save(dst_dir / f"{f.stem}_rot{int(angle):03d}.jpg", quality=95)
                    if use_style and not is_external(f) and rng.random() < args.style_mix:
                        rotated = rotate_with_background(img, rng.uniform(0, 360))
                        styled = to_external_style(cv2.cvtColor(np.asarray(rotated), cv2.COLOR_RGB2BGR),
                                                   target_bg, np_rng)
                        if styled is not None:
                            Image.fromarray(cv2.cvtColor(styled, cv2.COLOR_BGR2RGB)).save(
                                dst_dir / f"{f.stem}_style.jpg", quality=95)
        print(f"{cls:<24}{len(splits['train']):>7}{len(splits['val']):>6}{len(splits['test']):>6}")

    for idx, reason in sorted(excluded.items()):
        f, cls = items[idx]
        rows.append([source_name(f), cls, "DIBUANG", reason])

    with open(args.out / "split_info.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["file_asli", "kelas", "split", "keterangan"])
        w.writerows(rows)

    print(f"\nTotal gambar: {len(items)} | dipakai: {len(items) - len(excluded)} | dibuang: {len(excluded)}")
    for idx, reason in sorted(excluded.items()):
        if "bertentangan" in reason:
            print(f"  [LABEL KONFLIK] {source_name(items[idx][0])} -> {reason}")
    print(f"Train diperbanyak x{args.rotations + 1} dengan rotasi"
          + (f" + salinan bergaya sumber luar ({args.style_mix:.0%} foto Data/)" if use_style else "")
          + f". Detail: {args.out / 'split_info.csv'}")


if __name__ == "__main__":
    main()
