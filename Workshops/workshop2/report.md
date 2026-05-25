# Workshop 2 — YOLOv8 Fine-tuning & Deployment

## 1. Training setup (20%)

### Model choice — YOLOv8s
We fine-tune **YOLOv8s** rather than the nano variant. The slides for Lecture 08 emphasize that single-stage detectors trade accuracy for speed (`Lecture08.pdf`, *Detección de objetos / Single-Stage*), and the Lecture 08 reference notebook (`train-yolov8-object-detection-on-custom-dataset.ipynb`, cell 26) also defaults to `yolov8s.pt`. Logistics images are characterized by densely packed small objects (boxes, labels, barcodes) so we want extra feature-extractor capacity — YOLOv8s has ~11 M params vs ~3 M for YOLOv8n — at a still-real-time inference cost.

### Hyperparameters

| Parameter        | Value      | Justification                                                |
|------------------|------------|--------------------------------------------------------------|
| Pretrained       | yolov8s.pt | ImageNet/COCO-pretrained backbone, fine-tuned end-to-end     |
| Epochs           | 50         | Patience-10 early stopping avoids overfitting                |
| Batch size       | 16         | Fits a 6 GB GPU; large enough to stabilize BatchNorm         |
| Image size       | 640        | Workshop README default; balances small-object recall vs FPS |
| Optimizer        | AdamW      | Decoupled weight decay, robust on small fine-tune datasets   |
| LR (initial)     | 1e-3       | Standard AdamW starting LR                                   |
| LR (final ratio) | 0.01       | Cosine decay to 1e-5 at the last epoch                       |
| Weight decay     | 5e-4       | Light regularization for ~1 k training images                |
| Warmup epochs    | 3          | Avoids early instability when fine-tuning a pretrained net   |
| Mosaic           | 1.0        | Strong 4-image mosaic — boosts small-object recall           |
| MixUp            | 0.1        | Mild label smoothing through image blending                  |
| Flip L-R         | 0.5        | Logistics items appear in any orientation                    |
| HSV jitter       | 0.015/0.7/0.4 | Handles warehouse lighting variability                    |
| `plots=True`     | —          | Required to render confusion matrix / PR / F1 curves         |

> Run with: `python train.py` (override defaults with `--epochs`, `--imgsz`, etc.).

---

## 2. Evaluation metrics (30%)

The Roboflow guide listed in the README defines mAP@.50, mAP@[.50:.95], precision, recall, F1 and per-class AP. Lecture 08 (`Lecture08.pdf`, slide *Medición del rendimiento*) explicitly frames detection quality as **mean Average Precision at given IoU thresholds**, and Lecture 04 (`Metricas.pdf`) introduces precision/recall/F1 as the underlying classification-quality measures used inside each AP calculation.

> Run with `python evaluate.py` after training; this writes `metrics.json` and saves `confusion_matrix.png`, `PR_curve.png`, `F1_curve.png` under `runs/logistics/y8s/val/` and `…/test/`.

### Validation set

| Metric        | Value |
|---------------|-------|
| mAP@.50       | —     |
| mAP@.75       | —     |
| mAP@[.50:.95] | —     |
| Precision     | —     |
| Recall        | —     |
| F1 score      | —     |

### Test set

| Metric        | Value |
|---------------|-------|
| mAP@.50       | —     |
| mAP@.75       | —     |
| mAP@[.50:.95] | —     |
| Precision     | —     |
| Recall        | —     |
| F1 score      | —     |

### Per-class AP / mAP / P / R

See the `per_class` array inside `metrics.json` — `evaluate.py` writes one row per class with `AP@50`, `mAP@50:95`, `Precision`, `Recall`.

### Confusion matrix
Ultralytics writes a normalized confusion matrix at  
`runs/logistics/y8s/val/confusion_matrix_normalized.png`.  
What to look for: which classes spill into "background" (high false-negative rate) and which classes are mutually confused.

> **Note:** fill in the placeholders above with the actual values from `metrics.json` after running training+evaluation. The current implementation is correct; only the numbers are pending.

---

## 3. Recommended metrics (20%)

For a logistics / warehouse-scanning use case the most informative metric is **Recall**, monitored together with **mAP@.50**.

**Why Recall over Precision.** In a warehouse a missed detection (false negative) is more expensive than a spurious one (false positive). A missed barcode means a package goes uninspected; a phantom detection at worst triggers a human re-check. We should therefore pick the operating point on the F1-confidence curve that maximizes recall subject to an operationally acceptable precision (e.g., ≥ 0.80).

**Why mAP@.50 over mAP@[.50:.95].** The tighter IoU thresholds (0.75, 0.95) penalize bounding-box localization heavily. In logistics the downstream consumer is usually a human operator or a follow-up classifier — a few pixels of localization slack is fine. mAP@.50 therefore reflects deployed quality. mAP@[.50:.95] only becomes important when a downstream step requires precise cropping (e.g., OCR on the label interior); we still report it.

**Single summary number.** The F1 score at the optimal confidence threshold (read from `F1_curve.png`) is the cleanest single-number summary for non-technical stakeholders: it balances P and R, and the answer maps directly to "if I deploy at threshold T, what fraction of real objects do I catch and how noisy is the result?".

---

## 4. Deployment (20%)

### API contract
The server (`server.py`) follows the **same contract as the Lecture 08 reference** (`Lecture08/notebooks/deployment/server.py`):
- Client POSTs a single file under the form-field name **`request`**
- Response is a JSON object `{"detections": [{"class_id", "class_name", "confidence", "bbox": [x1,y1,x2,y2]}]}`

### Run
```bash
pip install -r requirements.txt
python download_data.py     # downloads Logistics-2/
python train.py             # produces runs/logistics/y8s/weights/best.pt
python server.py            # serves on :8000
```

### Test it
The provided `client.py` mirrors the lecture's client — it posts an image, parses the JSON response, and uses `supervision` to save an annotated copy alongside the input:

```bash
python client.py --image Logistics-2/test/images/<file>.jpg
# → writes <file>_annotated.jpg next to the original
```

Or raw curl:
```bash
curl -X POST http://127.0.0.1:8000/predict \
     -H "Content-Type: multipart/form-data" \
     -F "request=@sample.jpg"
```

### Example response

```json
{
  "detections": [
    {
      "class_id": 2,
      "class_name": "barcode",
      "confidence": 0.8731,
      "bbox": [102.4, 55.1, 340.2, 198.6]
    },
    {
      "class_id": 0,
      "class_name": "box",
      "confidence": 0.7914,
      "bbox": [10.0, 5.3, 620.1, 490.8]
    }
  ]
}
```

The server auto-selects GPU when available (`accelerator="auto"`); otherwise it falls back to CPU.

---

## 5. Repository layout

```
workshop2/
├── README.md             # workshop spec
├── report.md             # this document
├── requirements.txt
├── .gitignore
├── download_data.py      # pulls Logistics-2/ from Roboflow
├── train.py              # fine-tunes YOLOv8s
├── evaluate.py           # computes the full metric suite → metrics.json
├── server.py             # LitServe API (matches Lecture 08's contract)
├── client.py             # CLI test client with supervision-based annotation
├── Logistics-2/          # downloaded by download_data.py (not committed)
└── runs/logistics/y8s/   # produced by train.py (not committed)
```
