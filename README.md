# Arabic License Plate Detection and OCR

This project implements a real-time Arabic license plate detection and recognition system using YOLOv8 and Tesseract OCR.

## Features

- Real-time license plate detection
- Arabic text recognition
- Support for image upload, camera capture, and live video
- Results storage and visualization
- Arabic text validation and formatting

## Prerequisites

1. Python 3.8 or higher
2. Tesseract OCR with Arabic language support
3. YOLOv8 models

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd "Arabic license plate OCR"
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Install Tesseract OCR:
   - Windows: Download and install from https://github.com/UB-Mannheim/tesseract/wiki
   - Make sure to install the Arabic language pack
   - Set the Tesseract path in the code if needed

4. Download YOLO models:
   - Place `yolov8n.pt` in the `models` directory
   - Place `best.pt` in the `models` directory

## Usage

1. Run the application:
```bash
streamlit run app.py
```

2. Choose input method:
   - Upload an image
   - Take a photo
   - Live detection

3. View results:
   - Detected license plates
   - Recognized text
   - Confidence scores

## Project Structure
├── app.py # Main application file
├── requirements.txt # Python dependencies
├── README.md # Project documentation
├── models\ # YOLO model files
├── images\ # Background and example images
├── licenses_plates_imgs_detected\ # Detected license plate images
└── csv_detections\ # Detection results
