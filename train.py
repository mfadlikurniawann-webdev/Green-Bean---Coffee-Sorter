"""Training model klasifikasi defect biji kopi (Ultralytics YOLO classification).

Jalankan prepare_dataset.py dulu. Model terbaik disalin ke models/coffee_defect_cls.pt.

Contoh:
  python train.py                               # default: yolo11s-cls, 320 px
  python train.py --model yolo11m-cls.pt --epochs 120
"""
import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO

from config import BEST_MODEL, DATASET_DIR, RUNS_DIR


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="yolo11s-cls.pt", help="model pretrained (n/s/m/l/x)")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default=None, help="0 untuk GPU pertama, 'cpu' untuk CPU")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--name", default="coffee_defect")
    ap.add_argument("--no-save", action="store_true", help="jangan salin hasil ke models/")
    args = ap.parse_args()

    if not (DATASET_DIR / "train").exists():
        raise SystemExit("Folder dataset/train belum ada. Jalankan: python prepare_dataset.py")

    model = YOLO(args.model)
    model.train(
        data=str(DATASET_DIR),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=str(RUNS_DIR),
        name=args.name,
        exist_ok=True,
        patience=25,
        seed=0,
        # Augmentasi. Rotasi sudah dibuat di prepare_dataset.py.
        fliplr=0.5,
        flipud=0.5,          # biji bisa terbalik ke arah mana pun
        scale=0.3,           # crop acak 70-100% supaya cacat di tepi biji tidak terpotong
        erasing=0.0,         # jangan hapus area gambar: bisa menghapus lubang serangga/bercak
        auto_augment=None,   # RandAugment mengubah warna terlalu kuat
        hsv_h=0.0,           # warna adalah ciri utama defect (hitam, sour, fade), jadi
        hsv_s=0.15,          # jitter warna dibuat ringan
        hsv_v=0.2,
    )

    best = Path(model.trainer.save_dir) / "weights" / "best.pt"
    print(f"\nModel terbaik: {best}")
    if not args.no_save:
        BEST_MODEL.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best, BEST_MODEL)
        print(f"Disalin ke   : {BEST_MODEL}")


if __name__ == "__main__":
    main()
