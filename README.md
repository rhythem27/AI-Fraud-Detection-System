# AI Document Fraud Detection System

An end-to-end system to detect tampering in documents (Salary Slips, IDs, Utility Bills) using Computer Vision and OCR.

## Features
- **OCR Extraction**: Uses EasyOCR to extract text and bounding boxes from documents.
- **Tampering Detection**: Implements Error Level Analysis (ELA) to identify localized compression anomalies (potential forgeries).
- **Interactive Dashboard**: Streamlit-based UI for uploading documents and visualizing results.
- **Modular Backend**: FastAPI server with pluggable services for fraud detection and OCR.

## Tech Stack
- **Backend**: FastAPI, Python
- **Frontend**: Streamlit
- **CV/ML**: OpenCV, EasyOCR, Pillow
- **Infrastructure**: Docker, Docker Compose

## Quick Start (with Docker)
1. Clone the repository.
2. Run the system:
   ```bash
   docker-compose up --build
   ```
3. Access the UI at `http://localhost:8501`.
4. Access the API documentation at `http://localhost:8000/docs`.

## Manual Setup
### Backend
```bash
cd backend
pip install -r requirements.txt
python main.py
```

### Frontend
```bash
cd frontend
pip install -r requirements.txt
streamlit run app.py
```

## How ELA Works
Error Level Analysis resaves an image at a known quality (90%) and calculates the difference with the original. Genuine images have a uniform distribution of error, whereas manipulated images show significantly higher error levels in the tampered regions.
