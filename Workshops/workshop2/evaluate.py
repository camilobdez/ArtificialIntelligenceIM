"""
evaluate.py — Compute and display all required metrics on val and test sets.
Run after train.py has produced best.pt.
"""

import json
from pathlib import Path
from ultralytics import YOLO

MODEL_PATH = "runs/logistics/y8s/weights/best.pt"
DATA_YAML  = "datasets/logistics/dataset.yaml"


def run_split(model: YOLO, split: str) -> dict:
    metrics = model.val(
        data=DATA_YAML,
        split=split,
        imgsz=640,
        batch=16,
        verbose=False,
        plots=True,         # saves confusion matrix, P-R curve, etc.
        save_json=True,
    )
    return metrics


def extract_metrics(metrics, split: str) -> dict:
    mp  = float(metrics.box.mp)   # mean Precision
    mr  = float(metrics.box.mr)   # mean Recall
    map50   = float(metrics.box.map50)
    map5095 = float(metrics.box.map)

    f1 = 2 * mp * mr / (mp + mr + 1e-9)

    per_class_ap   = metrics.box.ap50.tolist()    # AP@.50 per class
    per_class_ar   = [float(r) for r in metrics.box.r]  # Recall per class ~ AR proxy
    class_names    = list(metrics.names.values()) if hasattr(metrics, "names") else []

    return {
        "split": split,
        "mAP@50":      round(map50,   4),
        "mAP@50:95":   round(map5095, 4),
        "Precision":   round(mp, 4),
        "Recall":      round(mr, 4),
        "F1":          round(f1, 4),
        "AP_per_class": {
            cls: round(ap, 4)
            for cls, ap in zip(class_names, per_class_ap)
        },
        "AR_per_class": {
            cls: round(ar, 4)
            for cls, ar in zip(class_names, per_class_ar)
        },
    }


def print_table(data: dict):
    print(f"\n{'='*50}")
    print(f"  Split: {data['split'].upper()}")
    print(f"{'='*50}")
    print(f"  mAP@50       : {data['mAP@50']:.4f}")
    print(f"  mAP@50:95    : {data['mAP@50:95']:.4f}")
    print(f"  Precision    : {data['Precision']:.4f}")
    print(f"  Recall       : {data['Recall']:.4f}")
    print(f"  F1 Score     : {data['F1']:.4f}")
    print(f"\n  AP per class:")
    for cls, ap in data["AP_per_class"].items():
        print(f"    {cls:<20} AP@50={ap:.4f}   AR≈{data['AR_per_class'].get(cls, 0):.4f}")


def main():
    model = YOLO(MODEL_PATH)
    results = {}

    for split in ("val", "test"):
        metrics = run_split(model, split)
        data    = extract_metrics(metrics, split)
        print_table(data)
        results[split] = data

    Path("metrics.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False)
    )
    print("\n✅ Saved to metrics.json")


if __name__ == "__main__":
    main()
