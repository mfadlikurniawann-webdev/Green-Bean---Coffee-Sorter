"""Impor foto biji dari dataset lain ke Data_tambahan/<kelas>/ dengan penyeragaman foto.

Dipakai untuk kelas Good (biji bagus) dan Other Defect (biji cacat dari sumber yang sama).
Keduanya HARUS diimpor dari sumber yang sama. Kalau hanya biji bagus yang diambil dari
sumber luar, AI belajar "foto dari sumber itu = bagus" (sudah terbukti lewat tes jebakan A
di evaluate.py: 98,8% biji cacat USK lolos sebagai Good).

Foto dari kamera/dataset lain biasanya berbeda warna alas, ukuran biji di bingkai, dan
resolusi. Kalau dipakai mentah, AI bisa belajar membedakan *gaya foto*, bukan mutu biji.
Script ini menyeragamkan tiap foto:
  1. white balance memakai alas sebagai acuan putih (disamakan dengan dataset utama),
  2. memotong ulang supaya biji mengisi +-45% bingkai seperti dataset utama,
  3. menyimpan ukuran 500x500 px.
Foto yang tidak berisi tepat 1 biji dilewati.

Sumber yang dipakai saat ini: kelas 'premium' dan 'defect' dari USK-Coffee (Universitas
Syiah Kuala), lisensi CC BY-NC-SA 4.0 (NON-KOMERSIAL). Untuk pemakaian komersial, ganti
dengan foto biji milik sendiri.

Contoh:
  python import_good_beans.py --src "D:/USK-Coffee/train/premium" --n 300
  python import_good_beans.py --src "D:/USK-Coffee/train/defect" --class-name "Other Defect" --n 300
"""
import argparse
import random
from pathlib import Path

import cv2
import numpy as np

from config import EXTRA_DATA_DIR, RAW_DATA_DIR
from sort_beans import CROP_SCALE, find_beans, read_image, square_crop, write_image

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
OUT_SIZE = 500


def border_color(img: np.ndarray) -> np.ndarray:
    b = max(2, min(img.shape[:2]) // 50)
    return np.median(np.concatenate([img[:b].reshape(-1, 3), img[-b:].reshape(-1, 3),
                                     img[:, :b].reshape(-1, 3), img[:, -b:].reshape(-1, 3)]), axis=0)


def reference_background(n: int = 100, seed: int = 0) -> np.ndarray:
    """Warna alas rata-rata dataset utama (BGR)."""
    files = sorted(RAW_DATA_DIR.rglob("*.jpg"))
    files = random.Random(seed).sample(files, min(n, len(files)))
    return np.median([border_color(read_image(f)) for f in files], axis=0)


def harmonize(img: np.ndarray, target_bg: np.ndarray, rng=None):
    """Seragamkan 1 foto biji ke gaya dataset utama. Return None bila bukan tepat 1 biji."""
    rng = rng or np.random.default_rng(0)
    gain = target_bg / np.maximum(border_color(img), 1)
    img = np.clip(img.astype(np.float32) * gain, 0, 255).astype(np.uint8)
    boxes, _ = find_beans(img)
    if len(boxes) != 1:
        return None
    x, y, w, h = boxes[0]
    H, W = img.shape[:2]
    side = int(max(w, h) * CROP_SCALE)

    # Kanvas seukuran bingkai baru, diisi warna alas + sedikit noise agar teksturnya alami.
    canvas = np.empty((side, side, 3), np.float32)
    canvas[:] = border_color(img)
    canvas += rng.normal(0, 1.5, canvas.shape)

    # Tempel foto asli di tengah kanvas. Tepinya dipudarkan halus ke kanvas supaya tidak
    # muncul garis kotak (AI bisa salah belajar "ada kotak = biji bagus").
    margin = max(0, min(x, y, W - x - w, H - y - h))
    feather = max(3.0, 0.8 * margin)
    yy, xx = np.mgrid[0:H, 0:W]
    alpha = np.clip(np.minimum.reduce([xx, yy, W - 1 - xx, H - 1 - yy]) / feather, 0, 1)[..., None]
    ox, oy = side // 2 - (x + w // 2), side // 2 - (y + h // 2)  # posisi foto asli di kanvas
    cx0, cy0, cx1, cy1 = max(ox, 0), max(oy, 0), min(ox + W, side), min(oy + H, side)
    src = img[cy0 - oy:cy1 - oy, cx0 - ox:cx1 - ox].astype(np.float32)
    a = alpha[cy0 - oy:cy1 - oy, cx0 - ox:cx1 - ox]
    canvas[cy0:cy1, cx0:cx1] = a * src + (1 - a) * canvas[cy0:cy1, cx0:cx1]

    canvas = np.clip(canvas, 0, 255).astype(np.uint8)
    return cv2.resize(canvas, (OUT_SIZE, OUT_SIZE), interpolation=cv2.INTER_CUBIC)


def to_external_style(img: np.ndarray, target_bg: np.ndarray, rng, size: int = 256):
    """Kebalikan dari impor: ubah foto dataset utama agar mirip foto sumber luar (biji lebih
    besar di bingkai, resolusi 256 px), lalu seragamkan lagi dengan harmonize().

    Dipakai prepare_dataset.py sebagai augmentasi kelas defect, supaya jejak proses
    penyeragaman ada di SEMUA kelas dan tidak bisa dijadikan petunjuk "biji bagus".
    """
    boxes, _ = find_beans(img)
    if len(boxes) != 1:
        return None
    x, y, w, h = boxes[0]
    side = int(max(w, h) / rng.uniform(0.55, 0.75))  # biji mengisi 55-75% bingkai, seperti USK
    crop = square_crop(img, x + w // 2, y + h // 2, side, border_color(img))
    small = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
    return harmonize(small, target_bg, rng)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, required=True, help="folder foto biji bagus")
    ap.add_argument("--class-name", default="Good")
    ap.add_argument("--out", type=Path, default=None, help="default: Data_tambahan/<class-name>")
    ap.add_argument("--n", type=int, default=300, help="jumlah foto yang diambil (acak)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = args.out or EXTRA_DATA_DIR / args.class_name
    files = sorted(p for p in args.src.rglob("*") if p.suffix.lower() in IMG_EXTS)
    random.Random(args.seed).shuffle(files)
    target_bg = reference_background()
    rng = np.random.default_rng(args.seed)
    print(f"Warna alas acuan (BGR): {target_bg.round()}")

    saved = skipped = 0
    for f in files:
        if saved >= args.n:
            break
        result = harmonize(read_image(f), target_bg, rng)
        if result is None:
            skipped += 1
            continue
        write_image(out / f"{args.class_name}_{f.stem}.jpg", result)
        saved += 1
    print(f"Disimpan {saved} foto ke {out} (dilewati {skipped} foto yang tidak berisi tepat 1 biji)")


if __name__ == "__main__":
    main()
