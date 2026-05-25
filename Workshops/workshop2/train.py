"""
train.py — Fine-tune YOLOv8 on the Roboflow Logistics dataset.

The dataset is the one used in Lecture 08's test.ipynb:
    rf.workspace("large-benchmark-datasets").project("logistics-sz9jr").version(2)
Roboflow's YOLOv8 export creates a folder called ``Logistics-2/`` containing
``data.yaml`` plus ``train/``, ``valid/`` and ``test/`` subdirectories.

Run once to produce runs/logistics/y8s/weights/best.pt.
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data",    default="Logistics-2/data.yaml",
                   help="Path to the Roboflow data.yaml.")
    p.add_argument("--model",   default="yolov8s.pt",
                   help="Pretrained checkpoint to fine-tune from.")
    p.add_argument("--epochs",  type=int, default=50)
    p.add_argument("--imgsz",   type=int, default=640)
    p.add_argument("--batch",   type=int, default=16)
    p.add_argument("--project", default="runs/logistics")
    p.add_argument("--name",    default="y8s")
    return p.parse_args()


def main():
    args = parse_args()

    if not Path(args.data).exists():
        raise FileNotFoundError(
            f"{args.data} not found. Download the dataset first — see download_data.py "
            f"or the README."
        )

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name,
        # Optimizer / schedule — AdamW with cosine decay tends to outperform SGD
        # on small fine-tuning datasets (~1 k images).
        optimizer="AdamW",
        lr0=1e-3,
        lrf=0.01,
        weight_decay=5e-4,
        warmup_epochs=3,
        # Data augmentation tuned for warehouse imagery (dense small objects,
        # variable lighting, arbitrary orientation).
        mosaic=1.0,
        mixup=0.1,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        # Early stopping + plots required by the deliverable.
        patience=10,
        plots=True,
        verbose=True,
    )
    print(f"\nBest weights: {args.project}/{args.name}/weights/best.pt")


if __name__ == "__main__":
    main()
