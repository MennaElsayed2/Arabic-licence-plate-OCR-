import os
from pathlib import Path
import streamlit as st
import arabic_reshaper
from bidi.algorithm import get_display
import base64
import cv2
import pytesseract
import numpy as np

def set_background(image_path):
    """Set background image for Streamlit app"""
    try:
        if not os.path.exists(image_path):
            st.error(f"Background image not found at: {image_path}")
            return

        with open(image_path, "rb") as f:
            background_image = base64.b64encode(f.read()).decode()
        
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url(data:image/jpeg;base64,{background_image});
                background-attachment: fixed;
                background-size: cover;
                background-position: center;
                background-repeat: no-repeat;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
    except Exception as e:
        st.error(f"Error setting background: {str(e)}")

def format_arabic_text(text):
    """Format Arabic text for proper display"""
    try:
        reshaped_text = arabic_reshaper.reshape(text)
        bidi_text = get_display(reshaped_text)
        return bidi_text
    except Exception as e:
        st.error(f"Error formatting Arabic text: {str(e)}")
        return text

def create_directories():
    """Create necessary directories for the project"""
    base_dir = Path("D:/Arabic license plate OCR")
    directories = [
        base_dir / "models",
        base_dir / "images",
        base_dir / "licenses_plates_imgs_detected",
        base_dir / "csv_detections"
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
    
    return base_dir

def write_csv(results, output_path):
    with open(output_path, 'w') as f:
        f.write('{},{},{},{},{},{},{}\n'.format('frame_nmr', 'car_id', 'car_bbox',
                                                'license_plate_bbox', 'license_plate_bbox_score', 'license_number',
                                                'license_number_score'))
        for frame_nmr in results:
            for car_id in results[frame_nmr]:
                data = results[frame_nmr][car_id]
                if 'car' in data and 'license_plate' in data and 'text' in data['license_plate']:
                    f.write('{},{},{},{},{},{},{}\n'.format(
                        frame_nmr,
                        car_id,
                        '[{} {} {} {}]'.format(*data['car']['bbox']),
                        '[{} {} {} {}]'.format(*data['license_plate']['bbox']),
                        data['license_plate']['bbox_score'],
                        data['license_plate']['text'],
                        data['license_plate']['text_score']
                    ))

def read_license_plate(license_plate_crop):
    """
    Args:
        license_plate_crop (PIL.Image.Image): Cropped image of the license plate.

    Returns:
        tuple: Arabic text detected and confidence (dummy since pytesseract doesn’t return score easily).
    """
    # Convert PIL image to OpenCV format
    open_cv_image = np.array(license_plate_crop.convert('RGB'))
    open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)

    # Convert to grayscale and apply thresholding to improve OCR
    gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Run Arabic OCR
    custom_config = '--psm 7 -l ara'
    text = pytesseract.image_to_string(thresh, config=custom_config)

    if text.strip():
        return text.strip(), "N/A"  # Pytesseract doesn't provide score by default
    else:
        return None, None