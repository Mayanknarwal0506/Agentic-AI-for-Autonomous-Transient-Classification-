from mcp.server.fastmcp import FastMCP
from mcp import types

from ultralytics import YOLO
import cv2
import base64
import json

import torch
from PIL import Image

from transformers import CvtForImageClassification, AutoImageProcessor

# ---------------------------
# MCP INIT
# ---------------------------
mcp = FastMCP("yolo-server")

# ---------------------------
# YOLO MODEL
# ---------------------------
yolo_model = YOLO(r"C:\Users\lsant\Downloads\best.pt")

# ---------------------------
# DEVICE
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------
# LOAD CvT MODEL (HF)
# ---------------------------
MODEL_PATH = r"C:\Users\lsant\Downloads\Final_sem_project\models\cvt_real_bogus_model"  # <-- your saved model folder

cvt_model = CvtForImageClassification.from_pretrained(MODEL_PATH)
processor = AutoImageProcessor.from_pretrained(MODEL_PATH)

cvt_model.to(device)
cvt_model.eval()

@mcp.tool()
def get_object_type(pos : str) -> list[types.TextContent]:
    """
    to find the type of transient affter it's been classified as real by the classify tool, we can use this tool to find out if it's a supernova, asteroid, or something else."""

    return [
        types.TextContent(
            type="text",
            text=f"API returned supernova"
        )
    ]

# ---------------------------
# CLASSIFY TOOL (CvT)
# ---------------------------
@mcp.tool()
def classify(image_path: str) -> list[types.TextContent]:
    """
    Classify image as REAL or BOGUS using CvT model.
    """

    image = Image.open(image_path).convert("RGB")

    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = cvt_model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)

        conf, pred = torch.max(probs, dim=1)

    # 🔥 Automatically get label names if available
    if hasattr(cvt_model.config, "id2label"):
        label = cvt_model.config.id2label[pred.item()]
    else:
        label = str(pred.item())

    confidence = round(conf.item(), 4)

    return [
        types.TextContent(
            type="text",
            text=f"{label} (confidence: {confidence})"
        )
    ]


# ---------------------------
# DETECT TOOL (LAST 1/3 IMAGE)
# ---------------------------
@mcp.tool()
def detect(image_path: str) -> list[types.TextContent | types.ImageContent]:
    """
    Detect objects using YOLO on LAST THIRD of image.
    """

    img = cv2.imread(image_path)
    h, w, _ = img.shape

    # ---- Crop last third (right side) ----
    x_start = int(2 * w / 3)
    crop = img[:, x_start:w]

    # ---- YOLO inference ----
    results = yolo_model(crop)

    annotated = results[0].plot()

    names = results[0].names
    detections = []

    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        label = names[cls_id]
        conf = round(float(box.conf[0]), 3)

        x1, y1, x2, y2 = box.xyxy[0]
        x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)

        # ---- SHIFT BACK TO ORIGINAL IMAGE ----
        x1 += x_start
        x2 += x_start

        detections.append({
            "label": label,
            "confidence": conf,
            "location": {
                "x1": round(x1, 1),
                "y1": round(y1, 1),
                "x2": round(x2, 1),
                "y2": round(y2, 1)
            }
        })

    # ---- Encode annotated image ----
    _, buffer = cv2.imencode(".jpg", annotated)
    img_b64 = base64.b64encode(buffer).decode()

    # ---- Text output ----
    lines = [f"Found {len(detections)} object(s):\n"]

    for d in detections:
        loc = d["location"]
        lines.append(
            f"  - {d['label']} (conf: {d['confidence']}) "
            f"at top-left ({loc['x1']}, {loc['y1']}) "
            f"bottom-right ({loc['x2']}, {loc['y2']})"
        )

    lines.append(
        f"\nFull JSON:\n{json.dumps({'total': len(detections), 'detections': detections}, indent=2)}"
    )

    return [
        types.TextContent(type="text", text="\n".join(lines)),
        types.ImageContent(type="image", data=img_b64, mimeType="image/jpeg")
    ]


# ---------------------------
# RUN SERVER
# ---------------------------
if __name__ == "__main__":
    print("YOLO + CvT MCP SERVER STARTED")
    mcp.run()