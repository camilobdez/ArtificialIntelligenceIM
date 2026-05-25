"""
client.py — Tests the LogisticsDetectionAPI server and saves an annotated image.

Modeled on the reference client from Lecture 08
(Lecture08/notebooks/deployment/client.py).

Usage:
    python client.py --image path/to/image.jpg
    python client.py --image image.jpg --url http://127.0.0.1:8000/predict
"""

import argparse
import logging
import os

import numpy as np
import requests
import supervision as sv
from PIL import Image


def parse_arguments():
    parser = argparse.ArgumentParser(description="Client for LogisticsDetectionAPI.")
    parser.add_argument("-i", "--image", type=str, required=True,
                        help="Path to the image to send.")
    parser.add_argument("-u", "--url",   type=str,
                        default="http://127.0.0.1:8000/predict",
                        help="Server endpoint.")
    return parser.parse_args()


def send_image(image_path: str, url: str):
    if not os.path.isfile(image_path):
        logging.error("Image file not found: %s", image_path)
        return

    with open(image_path, "rb") as f:
        response = requests.post(url, files={"request": f})

    if response.status_code != 200:
        logging.error("Server error %s: %s", response.status_code, response.text)
        return

    payload    = response.json()
    detections = payload.get("detections", [])
    logging.info("Got %d detection(s)", len(detections))

    if not detections:
        logging.info("Nothing to annotate.")
        return

    sv_detections = sv.Detections(
        class_id   = np.array([d["class_id"]   for d in detections]),
        confidence = np.array([d["confidence"] for d in detections]),
        xyxy       = np.array([d["bbox"]       for d in detections]),
    )
    labels = [f"{d['class_name']} {d['confidence']:.2f}" for d in detections]

    image = Image.open(image_path)
    image = sv.BoxAnnotator().annotate(image, sv_detections)
    image = sv.LabelAnnotator().annotate(image, sv_detections, labels)

    base, _ = os.path.splitext(os.path.basename(image_path))
    out = os.path.join(os.path.dirname(image_path) or ".", f"{base}_annotated.jpg")
    image.save(out)
    logging.info("Annotated image saved to %s", out)


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_arguments()
    send_image(args.image, args.url)


if __name__ == "__main__":
    main()
