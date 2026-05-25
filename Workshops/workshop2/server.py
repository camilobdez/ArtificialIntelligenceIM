"""
server.py — LitServe deployment for the fine-tuned logistics YOLOv8 model.

Usage:
    python server.py

Test:
    curl -X POST http://127.0.0.1:8000/predict \
         -H "Content-Type: multipart/form-data" \
         -F "image=@sample.jpg"
"""

import io
from typing import List, Dict, Any

import litserve as ls
from fastapi import UploadFile
from PIL import Image
from ultralytics import YOLO

MODEL_PATH = "runs/logistics/y8s/weights/best.pt"
CONF_THRESH = 0.25


class LogisticsAPI(ls.LitAPI):
    def setup(self, device: str):
        self.model = YOLO(MODEL_PATH)
        self.model.to(device)
        print(f"✅ Model loaded on {device}")

    def decode_request(self, request: Dict[str, Any]) -> Image.Image:
        if isinstance(request, dict) and "image" in request:
            f: UploadFile = request["image"]
            return Image.open(io.BytesIO(f.file.read())).convert("RGB")
        raise ValueError("Expected multipart/form-data with field 'image'")

    def predict(self, image: Image.Image) -> List[Dict[str, Any]]:
        results = self.model.predict(
            image, conf=CONF_THRESH, imgsz=640, verbose=False
        )[0]

        detections = []
        for box in results.boxes:
            cls_id = int(box.cls[0].item())
            detections.append({
                "class_id":   cls_id,
                "class_name": results.names[cls_id],
                "confidence": round(float(box.conf[0].item()), 4),
                "xyxy":       [round(float(x), 2) for x in box.xyxy[0].tolist()],
            })
        return detections

    def encode_response(self, output: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "n_detections": len(output),
            "detections":   output,
        }


if __name__ == "__main__":
    api    = LogisticsAPI()
    server = ls.LitServer(api, accelerator="auto", max_batch_size=1)
    server.run(port=8000)
