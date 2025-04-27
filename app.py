import streamlit as st

# Set page config must be the first Streamlit command
st.set_page_config(
    page_title="Arabic License Plate Detection",
    page_icon="🚗",
    layout="wide"
)

# Initialize session state
if "state" not in st.session_state:
    st.session_state["state"] = "Uploader"

# Then import other libraries
import numpy as np
from PIL import Image
from ultralytics import YOLO
import cv2
import pytesseract
import pandas as pd
import uuid
import os
from streamlit_webrtc import webrtc_streamer
import av
from pathlib import Path
from utils import set_background, format_arabic_text, create_directories, write_csv
from PIL import ImageFont, ImageDraw
import arabic_reshaper
from bidi.algorithm import get_display

# Set up paths and create necessary directories
BASE_DIR = create_directories()
MODELS_DIR = BASE_DIR / "models"
IMAGES_DIR = BASE_DIR / "images"
LICENSES_DIR = BASE_DIR / "licenses_plates_imgs_detected"
CSV_DIR = BASE_DIR / "csv_detections"

# Model paths
LICENSE_MODEL_DETECTION_DIR = MODELS_DIR / "best.pt"
COCO_MODEL_DIR = MODELS_DIR / "yolov8n.pt"

# Optional: specify tesseract location for Windows
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Set background
set_background(str(IMAGES_DIR / "background.jpg"))

# Load models with error handling
@st.cache_resource
def load_models():
    try:
        coco_model = YOLO(str(COCO_MODEL_DIR))
        license_plate_detector = YOLO(str(LICENSE_MODEL_DETECTION_DIR))
        return coco_model, license_plate_detector
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        st.error("Please make sure the model files exist in the models directory")
        return None, None

# Initialize models
coco_model, license_plate_detector = load_models()

# Constants
vehicles = [2]  # car class ID in COCO
threshold = 0.15

def load_image(image_path):
    try:
        if image_path.exists():
            return Image.open(image_path)
        else:
            st.error(f"Image not found: {image_path}")
            return None
    except Exception as e:
        st.error(f"Error loading image {image_path}: {str(e)}")
        return None

def read_license_plate(license_plate_crop, img):
    try:
        if len(license_plate_crop.shape) == 2:
            license_plate_crop = cv2.cvtColor(license_plate_crop, cv2.COLOR_GRAY2BGR)

        license_plate_crop = cv2.resize(license_plate_crop, None, fx=3, fy=3)
        gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        kernel = np.ones((1, 1), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        kernel = np.ones((2, 2), np.uint8)
        dilated = cv2.dilate(opening, kernel, iterations=1)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])

        characters = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w * h > 100:
                char_crop = dilated[y:y+h, x:x+w]
                text = pytesseract.image_to_string(char_crop, config='--psm 10 -l ara').strip()
                if text:
                    characters.append({'text': text, 'bbox': [x, y, x+w, y+h]})
                    cv2.rectangle(license_plate_crop, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    cv2.putText(license_plate_crop, text, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        final_text = ''.join(char['text'] for char in characters)
        data = pytesseract.image_to_data(dilated, config='--psm 7 -l ara', output_type=pytesseract.Output.DICT)
        confidence = float(data['conf'][0]) if data['conf'] else 0

        return final_text, confidence/100, characters, license_plate_crop
    except Exception as e:
        st.error(f"Error reading license plate: {str(e)}")
        return None, 0.0, [], None

def validate_arabic_plate(text):
    if not text:
        return None
    import re
    patterns = [r'^\d{1,4}$', r'^\d{1,4}[\u0621-\u064A]$', r'^[\u0621-\u064A]\d{1,4}$', r'^\d{1,4}[\u0621-\u064A]\d{1,4}$']
    for pattern in patterns:
        if re.match(pattern, text):
            return text
    return None

class VideoProcessor:
    def __init__(self):
        self.license_plate_detector = license_plate_detector

    def recv(self, frame):
        try:
            img = frame.to_ndarray(format="bgr24")
            img_to_an = cv2.cvtColor(img.copy(), cv2.COLOR_RGB2BGR)
            license_detections = self.license_plate_detector(img_to_an)[0]
            if len(license_detections.boxes.cls.tolist()) != 0:
                for license_plate in license_detections.boxes.data.tolist():
                    x1, y1, x2, y2, score, class_id = license_plate
                    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)
                    license_crop = img[int(y1):int(y2), int(x1):int(x2), :]
                    gray_crop = cv2.cvtColor(license_crop, cv2.COLOR_BGR2GRAY)
                    text, _, _, _ = read_license_plate(gray_crop, img)
                    if text:
                        cv2.rectangle(img, (int(x1) - 40, int(y1) - 40), (int(x2) + 40, int(y1)), (255, 255, 255), cv2.FILLED)
                        cv2.putText(img, text, (int((x1 + x2) / 2) - 70, int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
            return av.VideoFrame.from_ndarray(img, format="bgr24")
        except Exception as e:
            st.error(f"Error in video processing: {str(e)}")
            return frame
   
# Prediction function
def model_prediction(img):
    try:
        license_numbers = 0
        results = {}
        licenses_texts = []
        characters_list = []

        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        object_detections = coco_model(img)[0]
        license_detections = license_plate_detector(img)[0]

        xcar1, ycar1, xcar2, ycar2, car_score = 0, 0, 0, 0, 0
        if len(object_detections.boxes.cls.tolist()) != 0:
            for detection in object_detections.boxes.data.tolist():
                xcar1, ycar1, xcar2, ycar2, car_score, class_id = detection
                if int(class_id) in vehicles:
                    cv2.rectangle(img, (int(xcar1), int(ycar1)), (int(xcar2), int(ycar2)), (0, 0, 255), 3)

        license_plate_crops_total = []
        if len(license_detections.boxes.cls.tolist()) != 0:
            for license_plate in license_detections.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = license_plate
                padding = 10
                x1 = max(0, x1 - padding)
                y1 = max(0, y1 - padding)
                x2 = min(img.shape[1], x2 + padding)
                y2 = min(img.shape[0], y2 + padding)

                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)
                crop = img[int(y1):int(y2), int(x1):int(x2), :]

                if len(crop.shape) == 2:
                    crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
                elif crop.shape[2] == 3:
                    crop = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)

                text, text_score, characters, processed_crop = read_license_plate(crop, img)
                validated_text = validate_arabic_plate(text)
                licenses_texts.append(validated_text or text)

                img_name = f'{uuid.uuid1()}.jpg'
                cv2.imwrite(str(LICENSES_DIR / img_name), processed_crop if processed_crop is not None else crop)

                if text:
                    license_plate_crops_total.append(processed_crop if processed_crop is not None else crop)
                    characters_list.append(characters)
                    results[license_numbers] = {
                        'car': {'bbox': [xcar1, ycar1, xcar2, ycar2], 'car_score': car_score},
                        'license_plate': {
                            'bbox': [x1, y1, x2, y2],
                            'text': validated_text or text,
                            'bbox_score': score,
                            'text_score': text_score,
                            'characters': characters
                        }
                    }
                    license_numbers += 1

            if results:
                df = pd.DataFrame.from_dict(results, orient='index')
                csv_path = CSV_DIR / "detection_results.csv"
                df.to_csv(csv_path, index=False)
                st.success(f"Results saved to {csv_path}")

            return [cv2.cvtColor(img, cv2.COLOR_BGR2RGB), licenses_texts, license_plate_crops_total, characters_list]

        return [cv2.cvtColor(img, cv2.COLOR_BGR2RGB), [], [], []]

    except Exception as e:
        st.error(f"Error in model prediction: {str(e)}")
        return [cv2.cvtColor(img, cv2.COLOR_BGR2RGB), [], [], []]

# Display results
def display_results(prediction, texts, crops, characters_list):
    try:
        st.subheader("Detection Results ✅")
        st.image(prediction)

        if texts and crops and characters_list:
            st.subheader("License Cropped ✅")
            st.image(crops[0], width=350)

            st.markdown("### License Number:")
            formatted_text = format_arabic_text(texts[0])
            st.markdown(f"<div style='text-align: right; direction: rtl; font-size: 24px;'>{formatted_text}</div>", unsafe_allow_html=True)

            # st.subheader("Detected Characters:")
            # for i, char in enumerate(characters_list[0]):
            #     st.write(f"Character {i+1}: {char['text']}")

            csv_path = CSV_DIR / "detection_results.csv"
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                st.dataframe(df)
            else:
                st.warning("No detection results found in CSV file")
        else:
            st.warning("No valid detection results to display")

    except Exception as e:
        st.error(f"Error displaying results: {str(e)}")

# Helper: load image from path
def load_image(image_path):
    try:
        return Image.open(image_path)
    except:
        return None

# Main Streamlit app
def main():
    with st.container():
        _, col1, _ = st.columns([0.2, 1, 0.1])
        col1.title("💥 Arabic License Plate Detection 🚘")

        _, col0, _ = st.columns([0.15, 1, 0.1])
        test_bg = load_image(IMAGES_DIR / "test_background.jpeg")
        if test_bg:
            col0.image(test_bg, width=500)

        _, col4, _ = st.columns([0.1, 1, 0.2])
        col4.subheader("Computer Vision Detection with YoloV8 🧪")

        _, col, _ = st.columns([0.3, 1, 0.1])
        plate_test = load_image(IMAGES_DIR / "plate_test.jpeg")
        if plate_test:
            col.image(plate_test)

    with st.container():
        _, col1, _ = st.columns([0.1, 1, 0.2])
        col1.subheader("Try it out 📸")
        _, colb1, colb2, colb3 = st.columns([0.2, 0.7, 0.6, 1])

        if colb1.button("Upload an Image"):
            st.session_state["state"] = "Uploader"
        elif colb2.button("Take a Photo"):
            st.session_state["state"] = "Camera"
        elif colb3.button("Live Detection"):
            st.session_state["state"] = "Live"

        img = None
        if st.session_state.get("state") == "Uploader":
            img = st.file_uploader("Upload a Car Image: ", type=["png", "jpg", "jpeg"])
        elif st.session_state.get("state") == "Camera":
            img = st.camera_input("Take a Photo: ")
        elif st.session_state.get("state") == "Live":
            webrtc_streamer(key="sample", video_processor_factory=VideoProcessor)

        if img is not None:
            try:
                image = np.array(Image.open(img))
                st.image(image, width=400)

                if st.button("Apply Detection"):
                    results = model_prediction(image)

                    if len(results) == 4:
                        prediction, texts, crops, characters_list = results
                        if texts and crops and characters_list:
                            display_results(prediction, texts, crops, characters_list)
                        else:
                            st.warning("No license plates detected in the image")
                    else:
                        st.warning("Error in detection process")
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")

if __name__ == "__main__":
    main()