"""
download_data.py — Download the Logistics dataset from Roboflow.

Mirrors the pattern in Lecture08/notebooks/deployment/test.ipynb:
    rf.workspace("large-benchmark-datasets").project("logistics-sz9jr").version(2)
Outputs ``./Logistics-2/`` with ``data.yaml`` + ``train/``, ``valid/``, ``test/``.

Set ROBOFLOW_API_KEY in your environment (or a .env file) before running.
"""

import os
from dotenv import load_dotenv
from roboflow import Roboflow

load_dotenv()

api_key = os.getenv("ROBOFLOW_API_KEY")
if not api_key:
    raise SystemExit(
        "ROBOFLOW_API_KEY is not set. Get a free key at https://app.roboflow.com "
        "and add ROBOFLOW_API_KEY=... to a .env file in this folder."
    )

rf = Roboflow(api_key=api_key)
project = rf.workspace("large-benchmark-datasets").project("logistics-sz9jr")
dataset = project.version(2).download("yolov8")
print(f"\n✅ Dataset downloaded to {dataset.location}")
print(f"   data.yaml: {dataset.location}/data.yaml")
