import os
from typing import get_args
import cv2
import numpy as np
import streamlit as st
from PIL import Image
from shapely.geometry import box

from fast_alpr import ALPR
from fast_alpr.default_detector import PlateDetectorModel
from fast_alpr.default_ocr import OcrModel
from inference_sdk import InferenceHTTPClient


# ---------------------------
# Configuration
# ---------------------------

ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")  # ⚠ Move to env variable in production

CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=ROBOFLOW_API_KEY
)

DETECTOR_MODELS = list(get_args(PlateDetectorModel))
OCR_MODELS = list(get_args(OcrModel))

if "cct-s-v1-global-model" in OCR_MODELS:
    OCR_MODELS.remove("cct-s-v1-global-model")
    OCR_MODELS.insert(0, "cct-s-v1-global-model")


# ---------------------------
# Helper Functions
# ---------------------------

def extract_bbox_coords(bbox_obj):
    """
    Extract (x1, y1, x2, y2) from fast_alpr BoundingBox safely.
    """
    if all(hasattr(bbox_obj, attr) for attr in ["x", "y", "width", "height"]):
        x1 = bbox_obj.x
        y1 = bbox_obj.y
        x2 = bbox_obj.x + bbox_obj.width
        y2 = bbox_obj.y + bbox_obj.height
        return x1, y1, x2, y2

    if all(hasattr(bbox_obj, attr) for attr in ["x1", "y1", "x2", "y2"]):
        return bbox_obj.x1, bbox_obj.y1, bbox_obj.x2, bbox_obj.y2

    raise ValueError("Unknown BoundingBox structure")


def convert_rf_bbox(pred):
    """
    Convert Roboflow center-based bbox to (x1, y1, x2, y2)
    """
    x_center = pred["x"]
    y_center = pred["y"]
    width = pred["width"]
    height = pred["height"]

    x1 = x_center - width / 2
    y1 = y_center - height / 2
    x2 = x_center + width / 2
    y2 = y_center + height / 2

    return x1, y1, x2, y2


def plate_center_inside_vehicle(plate_bbox, vehicle_bbox):
    """
    More reliable matching:
    Match plate to vehicle if plate center lies inside vehicle bbox.
    """
    px = (plate_bbox[0] + plate_bbox[2]) / 2
    py = (plate_bbox[1] + plate_bbox[3]) / 2

    return (
        vehicle_bbox[0] <= px <= vehicle_bbox[2] and
        vehicle_bbox[1] <= py <= vehicle_bbox[3]
    )


# ---------------------------
# UI
# ---------------------------

st.title("License Plate + Wrong Parking Detection")
st.write("ALPR system with wrong parking vehicle detection.")

detector_model = st.sidebar.selectbox("Choose Detector Model", DETECTOR_MODELS)
ocr_model = st.sidebar.selectbox("Choose OCR Model", OCR_MODELS)

uploaded_file = st.file_uploader(
    "Upload an image of a vehicle",
    type=["jpg", "jpeg", "png"]
)


# ---------------------------
# Main Logic
# ---------------------------

if uploaded_file is not None:

    img = Image.open(uploaded_file)
    img_array = np.array(img.convert("RGB"))

    st.image(img, caption="Uploaded Image", use_container_width=True)

    # ---------------- ALPR ----------------
    st.write("Running ALPR...")
    alpr = ALPR(detector_model=detector_model, ocr_model=ocr_model)
    alpr_results = alpr.predict(img_array)

    plate_data = []

    for r in alpr_results:
        if r.ocr:
            bbox_coords = extract_bbox_coords(r.detection.bounding_box)

            plate_data.append({
                "bbox": bbox_coords,
                "text": r.ocr.text,
                "confidence": r.ocr.confidence
            })

    # ---------------- Wrong Parking ----------------
    st.write("Running Wrong Parking Detection...")

    rf_result = CLIENT.infer(
        img_array,
        model_id="wrong-parking-hha4a/2"
    )

    wrong_parking_vehicles = []

    if "predictions" in rf_result:
        for pred in rf_result["predictions"]:
            if pred["confidence"] >= 0.5:
                vehicle_bbox = convert_rf_bbox(pred)
                wrong_parking_vehicles.append({
                    "bbox": vehicle_bbox,
                    "confidence": pred["confidence"]
                })

    # ---------------- Association (FIXED LOGIC) ----------------
    matches = []

    for vehicle in wrong_parking_vehicles:

        best_plate = None

        for plate in plate_data:
            if plate_center_inside_vehicle(plate["bbox"], vehicle["bbox"]):
                best_plate = plate
                break  # Only one plate per vehicle

        if best_plate:
            matches.append({
                "plate_text": best_plate["text"],
                "plate_conf": best_plate["confidence"],
                "vehicle_conf": vehicle["confidence"]
            })

    # ---------------- Visualization ----------------
    annotated = img_array.copy()

    # Draw wrong parking vehicles (RED)
    for v in wrong_parking_vehicles:
        x1, y1, x2, y2 = map(int, v["bbox"])
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 0, 0), 3)
        cv2.putText(
            annotated,
            "Wrong Parking",
            (x1, max(0, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2
        )

    # Draw plates (GREEN)
    for p in plate_data:
        x1, y1, x2, y2 = map(int, p["bbox"])
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            annotated,
            p["text"],
            (x1, max(0, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    st.image(annotated, caption="Final Detection Output", use_container_width=True)

    # ---------------- Results ----------------
    if matches:
        st.success("🚨 Wrong Parking Vehicles Detected")

        for m in matches:
            st.write(
                f"- Plate: `{m['plate_text']}` | "
                f"Plate Conf: `{m['plate_conf']:.2f}` | "
                f"Vehicle Conf: `{m['vehicle_conf']:.2f}`"
            )
    else:
        st.info("No wrong parking vehicles linked with detected plates.")

else:
    st.write("Please upload an image to continue.")


