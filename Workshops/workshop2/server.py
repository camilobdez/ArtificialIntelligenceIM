"""
server.py — LitServe deployment for the fine-tuned logistics YOLOv8 model.

API contract matches the one taught in Lecture 08 (see
Lecture08/notebooks/deployment/server.py): clients POST a file with the
form-field name ``request`` and receive a JSON object of the form
``{"detections": [{"class_id", "class_name", "confidence", "bbox"}]}``.

Usage:
    python server.py

Test (matches client.py):
    python client.py --image path/to/sample.jpg

Or raw curl:
    curl -X POST http://127.0.0.1:8000/predict \
         -H "Content-Type: multipart/form-data" \
         -F "request=@sample.jpg"
"""

from fastapi import UploadFile
from litserve import LitAPI, LitServer
from PIL import Image
from ultralytics import YOLO

MODEL_PATH  = "runs/logistics/y8s/weights/best.pt"
CONF_THRESH = 0.25


class LogisticsDetectionAPI(LitAPI):
    """Object detection on the logistics dataset."""

    def setup(self, device: str):
        self.model = YOLO(MODEL_PATH)
        print(f"✅ Model loaded ({MODEL_PATH})")

    def decode_request(self, request: UploadFile) -> Image.Image:
        with Image.open(request.file) as img:
            return img.convert("RGB")

    def predict(self, image: Image.Image):
        return self.model.predict(image, conf=CONF_THRESH, imgsz=640, verbose=False)

    def encode_response(self, results) -> dict:
        detections = []
        if not results:
            return {"detections": detections}

        r = results[0]
        boxes = r.boxes
        if boxes is None or len(boxes) == 0:
            return {"detections": detections}

        cls  = boxes.cls.cpu().numpy().astype(int)
        conf = boxes.conf.cpu().numpy()
        xyxy = boxes.xyxy.cpu().numpy()
        names = r.names

        for class_id, confidence, bbox in zip(cls, conf, xyxy):
            detections.append({
                "class_id":   int(class_id),
                "class_name": names[int(class_id)],
                "confidence": float(confidence),
                "bbox":       bbox.tolist(),
            })
        return {"detections": detections}


if __name__ == "__main__":
    server = LitServer(LogisticsDetectionAPI(), accelerator="auto")
    server.run(port=8000)
