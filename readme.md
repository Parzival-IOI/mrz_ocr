# Streamlit Pass - MRZ OCR Application

A Streamlit-based application for extracting Machine Readable Zone (MRZ) data from passport images using OCR and image processing.

## Features

- Upload passport images
- Automatic MRZ detection and extraction
- OCR processing using Tesseract
- Extraction of Passport ID, Family Name, and Given Names
- Preprocessing visualization
- Execution time tracking

## Setup Instructions

### 1. Create Virtual Environment

```bash
python -m venv venv
```

### 2. Activate Virtual Environment

On Windows (bash):

```bash
source venv/Scripts/activate
```

On macOS/Linux:

```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Application

```bash
streamlit run main.py
```

The application will open in your default browser at `http://localhost:8501`

## Usage

1. Upload a passport image using the file uploader
2. Click the "Submit" button
3. The application will:
   - Display the uploaded image
   - Show the extracted MRZ text
   - Display extracted data (Passport ID, Family Name, Given Names) in JSON format
   - Show the preprocessed MRZ image
   - Display the execution time

## Requirements

- Python 3.7+
- See `requirements.txt` for dependencies

## Project Structure

```
.
├── main.py              # Streamlit application
├── mrz.py               # MRZ detection and OCR logic
├── requirements.txt     # Python dependencies
├── custom/
│   ├── best
│   │   └── mrz.traineddata  # Tesseract MRZ language data for best
│   └── fast
│       └── mrz.traineddata  # Tesseract MRZ language data for fast
└── readme.md            # This file
```

## How It Works

1. **Image Upload**: User uploads a passport image
2. **MRZ Detection**: Morphological operations detect the MRZ region
3. **Preprocessing**: Image is upscaled, normalized, and sharpened
4. **OCR**: Tesseract extracts text from the preprocessed MRZ
5. **Parsing**: MRZ data is parsed to extract structured information
