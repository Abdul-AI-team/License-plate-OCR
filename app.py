from typing import get_args

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from fast_alpr import ALPR
from fast_alpr.default_detector import PlateDetectorModel
from fast_alpr.default_ocr import OcrModel

# Default models
DETECTOR_MODELS = list(get_args(PlateDetectorModel))
OCR_MODELS = list(get_args(OcrModel))
# Put global OCR first
OCR_MODELS.remove("cct-s-v1-global-model")
OCR_MODELS.insert(0, "cct-s-v1-global-model")

st.title("License Plate Detection Demo")
st.write("An automatic license plate recognition (ALPR) system with customizable detector and OCR models.")


# Sidebar for selecting models
detector_model = st.sidebar.selectbox("Choose Detector Model", DETECTOR_MODELS)
ocr_model = st.sidebar.selectbox("Choose OCR Model", OCR_MODELS)

# Load image
uploaded_file = st.file_uploader("Upload an image of a vehicle with a license plate", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert uploaded file to a format compatible with OpenCV
    img = Image.open(uploaded_file)
    img_array = np.array(img.convert("RGB"))  # Convert to RGB if needed
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # Initialize ALPR with selected models
    alpr = ALPR(detector_model=detector_model, ocr_model=ocr_model)

    # Run ALPR on the uploaded image
    st.write("Processing...")
    results = alpr.predict(img_array)

    # Draw predictions on the image
    annotated_img_array = alpr.draw_predictions(img_array)

    # Convert the annotated image back to display in Streamlit
    annotated_img = Image.fromarray(annotated_img_array)
    st.image(annotated_img, caption="Annotated Image with OCR Results", use_container_width=True)

    # Display OCR results in text format for more detail
    if results:
        st.write("**OCR Results:**")
        for result in results:
            # Access the detection and OCR attributes from ALPRResult
            plate_text = result.ocr.text if result.ocr else "N/A"
            plate_confidence = result.ocr.confidence if result.ocr else 0.0
            st.write(f"- Detected Plate: `{plate_text}` with confidence `{plate_confidence:.2f}`")
    else:
        st.write("No license plate detected 😔.")
else:
    st.write("Please upload an image to continue.")