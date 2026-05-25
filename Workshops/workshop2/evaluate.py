"""
evaluate.py — Compute and display all required metrics on val and test sets.

Reports the metrics referenced in Lecture 08 (mAP@50, mAP@75, mAP@50:95,
per-class APs) plus the precision / recall / F1 framing from Lecture 04
(Metricas.pdf).

Run after train.py has produced best.pt.
"""

import argparse
import json
from pathlib import Path
from ultralytics import YOLO


def run_split(model: YOLO, data_yaml: str, split: str):
    return model.val(
        data=data_yaml,
        split=split,
        imgsz=640,
        batch=16,
        conf=0.001,        # low conf for metric computation; Ultralytics default
        verbose=False,
        plots=True,        # writes confusion_matrix.png, PR_curve.png, F1_curve.png …
        save_json=True,
    )


def extract_metrics(metrics, split: str) -> dict:
    box = metrics.box

    mp      = float(box.mp)        # mean Precision
    mr      = float(box.mr)        # mean Recall
    map50   = float(box.map50)
    map75   = float(box.map75)
    map5095 = float(box.map)
    f1      = 2 * mp * mr / (mp + mr + 1e-9)

    class_names = list(metrics.names.values()) if hasattr(metrics, "names") else []

    per_class_ap50  = box.ap50.tolist()    # AP@50      per class
    per_class_map   = box.maps.tolist()    # mAP@50:95  per class
    per_class_p     = [float(x) for x in box.p]
    per_class_r     = [float(x) for x in box.r]

    return {
        "split":       split,
        "mAP@50":      round(map50,   4),
        "mAP@75":      round(map75,   4),
        "mAP@50:95":   round(map5095, 4),
        "Precision":   round(mp,      4),
        "Recall":      round(mr,      4),
        "F1":          round(f1,      4),
        "per_class": [
            {
                "name":      cls,
                "AP@50":     round(per_class_ap50[i], 4),
                "mAP@50:95": round(per_class_map[i],  4),
                "Precision": round(per_class_p[i],   4),
                "Recall":    round(per_class_r[i],   4),
            }
            for i, cls in enumerate(class_names)
        ],
    }


def print_table(data: dict):
    print(f"\n{'='*60}")
    print(f"  Split: {data['split'].upper()}")
    print(f"{'='*60}")
    print(f"  mAP@50        : {data['mAP@50']:.4f}")
    print(f"  mAP@75        : {data['mAP@75']:.4f}")
    print(f"  mAP@50:95     : {data['mAP@50:95']:.4f}")
    print(f"  Precision     : {data['Precision']:.4f}")
    print(f"  Recall        : {data['Recall']:.4f}")
    print(f"  F1 Score      : {data['F1']:.4f}")
    print(f"\n  Per-class breakdown:")
    print(f"    {'class':<24} {'AP@50':>8} {'mAP':>8} {'P':>8} {'R':>8}")
    for row in data["per_class"]:
        print(f"    {row['name'][:24]:<24} "
              f"{row['AP@50']:>8.4f} {row['mAP@50:95']:>8.4f} "
              f"{row['Precision']:>8.4f} {row['Recall']:>8.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="runs/logistics/y8s/weights/best.pt")
    parser.add_argument("--data",  default="Logistics-2/data.yaml")
    parser.add_argument("--out",   default="metrics.json")
    args = parser.parse_args()

    if not Path(args.model).exists():
        raise FileNotFoundError(f"{args.model} not found — train first.")

    model   = YOLO(args.model)
    results = {}
    for split in ("val", "test"):
        m    = run_split(model, args.data, split)
        data = extract_metrics(m, split)
        print_table(data)
        results[split] = data

    Path(args.out).write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\n✅ Saved to {args.out}")


if __name__ == "__main__":
    main()
