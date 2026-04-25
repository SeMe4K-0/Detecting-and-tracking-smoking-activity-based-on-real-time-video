"""
YOLO26n training for smoking activity detection.
Dataset: Smoking Person Detection (nc=2)
  Class 0 — person
  Class 1 — smoke (cigarette / smoke near mouth)

Detection logic in app:
  smoking = smoke bbox overlaps person bbox

Download dataset in YOLO26 format:
  https://universe.roboflow.com/project-i6bzi/smoking-person-detection-ec7ec
  → unzip to datasets/smoking_detection/

Usage:
  python train.py
"""

from pathlib import Path



DATASETS_DIR = Path("datasets")
EPOCHS = 100


def check_datasets():
    path = DATASETS_DIR / "smoking_detection"
    imgs = list(path.rglob("*.jpg")) + list(path.rglob("*.png"))
    if not imgs:
        print("[ERROR] Dataset not found: datasets/smoking_detection/")
        print("Download: https://universe.roboflow.com/project-i6bzi/smoking-person-detection-ec7ec")
        raise SystemExit(1)

    for split in ("train", "valid"):
        imgs = list((path / split / "images").glob("*"))
        lbls = list((path / split / "labels").glob("*.txt"))
        print(f"  smoking_detection/{split}: {len(imgs)} images, {len(lbls)} label files")
    print()


def train():
    model = YOLO("yolo26n.pt")

    pbar = tqdm(
        total=EPOCHS,
        desc="Training YOLO26n",
        unit="epoch",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} epochs "
                   "[{elapsed}<{remaining}, {rate_fmt}] {postfix}",
        colour="green",
    )

    def on_train_epoch_end(trainer):
        metrics = trainer.metrics or {}
        loss = float(getattr(trainer, "loss", 0) or 0)
        map50 = metrics.get("metrics/mAP50(B)", 0)
        map5095 = metrics.get("metrics/mAP50-95(B)", 0)
        pbar.set_postfix({
            "loss": f"{loss:.4f}",
            "mAP50": f"{map50:.3f}",
            "mAP50-95": f"{map5095:.3f}",
        }, refresh=True)
        pbar.update(1)

    def on_train_end(trainer):
        pbar.close()
        print(f"\nBest weights: {trainer.best}")
        print(f"Results saved to: {trainer.save_dir}")

    model.add_callback("on_train_epoch_end", on_train_epoch_end)
    model.add_callback("on_train_end", on_train_end)

    model.train(
        data="data.yaml",
        epochs=EPOCHS,
        imgsz=640,
        batch=16,
        device="mps",
        patience=40,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        mosaic=1.0,
        project="runs",
        name="cigarette_yolo26n",
        exist_ok=True,
        verbose=False,
    )


if __name__ == "__main__":
    DATASETS_DIR.mkdir(exist_ok=True)
    print("Dataset statistics:")
    check_datasets()
    train()
