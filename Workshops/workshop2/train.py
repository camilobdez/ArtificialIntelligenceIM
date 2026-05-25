"""
train.py — Fine-tune YOLOv8 on the Logistics dataset.
Run once to produce runs/logistics/y8n/weights/best.pt
"""

from ultralytics import YOLO

DATA_YAML   = "datasets/logistics/dataset.yaml"
MODEL_NAME  = "yolov8s.pt"   # yolov8s: better mAP than nano, still fast
EPOCHS      = 50
IMGSZ       = 640
BATCH       = 16
PROJECT     = "runs/logistics"
RUN_NAME    = "y8s"

def main():
    model = YOLO(MODEL_NAME)
    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        project=PROJECT,
        name=RUN_NAME,
        optimizer="AdamW",
        lr0=1e-3,
        lrf=0.01,          # final lr = lr0 * lrf (cosine schedule)
        weight_decay=5e-4,
        warmup_epochs=3,
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.0,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        patience=10,       # early stopping
        save_period=5,
        verbose=True,
    )
    print(f"\nBest weights: {PROJECT}/{RUN_NAME}/weights/best.pt")
    return results

if __name__ == "__main__":
    main()
