# Workshop 2 — YOLOv8 Fine-tuning & Deployment Report

## 1. Training Setup

### Model Choice
We chose **YOLOv8s** (small) instead of YOLOv8n (nano) because logistics images tend to contain densely packed, small objects (barcodes, labels, boxes). The `s` variant has roughly 4× more parameters (~11 M vs ~3 M), which translates into meaningfully higher mAP on small-object benchmarks at a negligible inference-latency cost on modern hardware.

### Hyperparameters

| Parameter        | Value       | Rationale                                              |
|------------------|-------------|--------------------------------------------------------|
| Model            | yolov8s.pt  | Better feature resolution for small/dense objects     |
| Epochs           | 50          | Enough for convergence; early stopping at patience=10 |
| Batch size       | 16          | Fits a 6 GB GPU; larger batches stabilize BN stats    |
| Image size       | 640         | YOLOv8 default; good balance speed vs. accuracy       |
| Optimizer        | AdamW       | Decoupled weight decay → slightly better convergence  |
| Initial LR       | 1e-3        | Standard AdamW starting point                         |
| Final LR ratio   | 0.01        | Cosine decay to 1e-5 at epoch 50                      |
| Weight decay     | 5e-4        | Regularization to avoid over-fitting on ~1 k images   |
| Warmup epochs    | 3           | Avoids early instability                              |
| Mosaic           | 1.0         | Strong augmentation mixing 4 images                   |
| MixUp            | 0.1         | Light soft-label mixing                               |
| Flip (L-R)       | 0.5         | Logistics items appear at any orientation             |
| HSV jitter       | h=0.015, s=0.7, v=0.4 | Handles variable warehouse lighting     |

---

## 2. Evaluation Metrics

> Metrics were computed by running `evaluate.py` after training.  
> Plots (confusion matrix, P-R curve, F1-confidence curve) are saved under  
> `runs/logistics/y8s/` by Ultralytics automatically.

### Validation Set

| Metric        | Value  |
|---------------|--------|
| mAP@.50       | —      |
| mAP@[.50:.95] | —      |
| Precision     | —      |
| Recall        | —      |
| F1 Score      | —      |

### Test Set

| Metric        | Value  |
|---------------|--------|
| mAP@.50       | —      |
| mAP@[.50:.95] | —      |
| Precision     | —      |
| Recall        | —      |
| F1 Score      | —      |

> **Note:** Fill in the actual values from `metrics.json` after running  
> `python train.py` followed by `python evaluate.py`.

### Per-class AP / AR (Validation)

See `metrics.json` for the full breakdown; an excerpt after training:

```
Class               AP@50    AR
─────────────────── ──────── ──────
<class_0>           0.xxxx   0.xxxx
...
```

### Confusion Matrix

The confusion matrix image is saved at  
`runs/logistics/y8s/val/confusion_matrix_normalized.png`.  
Key things to inspect: which classes are most confused with "background" (high false-negative rate).

---

## 3. Recommended Metrics

For a **logistics / supply-chain** use case the single most informative metric is **Recall** — specifically the per-class recall on high-value SKUs or hazardous-goods labels.

**Why Recall over Precision?**  
In a warehouse scanning pipeline a missed detection (false negative) is more costly than a spurious one (false positive). A false negative means a package passes uninspected; a false positive merely triggers a human re-check. Maximizing recall at an operationally acceptable precision (e.g., ≥ 0.80) is therefore the right operating point.

**Why mAP@50 over mAP@50:95?**  
Tight IoU thresholds (0.75, 0.95) penalize bounding-box localization heavily. In practice a bounding box that correctly identifies *what* object it is but whose boundary is off by a few pixels is still actionable. mAP@50 reflects detection quality as logistics operators experience it. mAP@50:95 is more useful when the downstream step requires precise cropping (e.g., OCR on the label interior).

**Supporting argument:** The F1 score at the optimal confidence threshold (read from the F1-confidence curve) is a useful single-number summary for stakeholders: it balances precision and recall and directly maps to the question "if I deploy this at threshold T, what fraction of real objects do I catch, and how noisy is the result?"

---

## 4. Deployment

### Setup

```bash
pip install ultralytics litserve fastapi uvicorn pillow
python train.py          # produces runs/logistics/y8s/weights/best.pt
python server.py         # starts API on :8000
```

### API Test

```bash
curl -X POST http://127.0.0.1:8000/predict \
     -H "Content-Type: multipart/form-data" \
     -F "image=@sample.jpg"
```

**Example response:**

```json
{
  "n_detections": 3,
  "detections": [
    {
      "class_id": 2,
      "class_name": "barcode",
      "confidence": 0.8731,
      "xyxy": [102.4, 55.1, 340.2, 198.6]
    },
    {
      "class_id": 0,
      "class_name": "box",
      "confidence": 0.7914,
      "xyxy": [10.0, 5.3, 620.1, 490.8]
    },
    {
      "class_id": 1,
      "class_name": "label",
      "confidence": 0.6248,
      "xyxy": [95.0, 50.0, 350.0, 205.0]
    }
  ]
}
```

The server auto-selects GPU if available (`accelerator="auto"`), falls back to CPU otherwise.

---

## 5. File Structure

```
workshop2/
├── datasets/
│   └── logistics/          # download from Roboflow (not committed)
│       ├── dataset.yaml
│       ├── train/
│       ├── valid/
│       └── test/
├── runs/                   # generated by Ultralytics (not committed)
│   └── logistics/y8s/
│       └── weights/best.pt
├── train.py                # fine-tuning script
├── evaluate.py             # metrics computation
├── server.py               # LitServe API
├── requirements.txt
└── report.md               # this document
```
