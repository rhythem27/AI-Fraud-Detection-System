import os
import io
from PIL import Image
from pdf2image import convert_from_path, convert_from_bytes
import PyPDF2
from datetime import datetime
from typing import Dict, Any, List, Optional
from pydantic import BaseModel

class PDFMetadata(BaseModel):
    author: Optional[str] = "Unknown"
    creator: Optional[str] = "Unknown"
    producer: Optional[str] = "Unknown"
    created: Optional[str] = "Unknown"
    modified: Optional[str] = "Unknown"
    is_suspicious: bool = False
    suspicious_reasons: List[str] = []

class PDFProcessor:
    SUSPICIOUS_SOFTWARE = [
        "photoshop", "illustrator", "gimp", "canva", "inkscape", 
        "quartz pdfcontext", "acrobat distill", "nitro pdf", "foxit"
    ]

    def convert_to_images(self, pdf_path: str) -> List[Image.Image]:
        """
        Converts PDF pages to PIL Images.
        """
        try:
            images = convert_from_path(pdf_path, dpi=200)
            return images
        except Exception as e:
            print(f"Error converting PDF to image: {e}")
            return []

    def extract_metadata(self, pdf_path: str) -> PDFMetadata:
        """
        Extracts metadata and performs forensic rule-based analysis.
        """
        metadata_dict = {}
        suspicious_reasons = []
        is_suspicious = False

        try:
            with open(pdf_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                info = reader.metadata
                if info:
                    metadata_dict = {
                        "author": info.get('/Author', 'Unknown'),
                        "creator": info.get('/Creator', 'Unknown'),
                        "producer": info.get('/Producer', 'Unknown'),
                        "created": info.get('/CreationDate', 'Unknown'),
                        "modified": info.get('/ModDate', 'Unknown')
                    }
        except Exception as e:
            print(f"Error extracting PDF metadata: {e}")
            return PDFMetadata(author="Error", creator="Error", producer="Error")

        # Heuristic Analysis
        creator = str(metadata_dict.get('creator', '')).lower()
        producer = str(metadata_dict.get('producer', '')).lower()

        # 1. Suspicious Software Check
        for software in self.SUSPICIOUS_SOFTWARE:
            if software in creator or software in producer:
                is_suspicious = True
                suspicious_reasons.append(f"Suspicious editing software detected: {software}")

        # 2. Date Anomaly Check
        creation_date = self._parse_pdf_date(metadata_dict.get('created'))
        mod_date = self._parse_pdf_date(metadata_dict.get('modified'))

        if creation_date and mod_date:
            # If modification is more than 1 hour after creation, it might be tampered
            diff = (mod_date - creation_date).total_seconds()
            if diff > 3600: # 1 hour
                is_suspicious = True
                suspicious_reasons.append(f"Document modified significantly after creation ({round(diff/3600, 1)} hours later)")

        return PDFMetadata(
            author=metadata_dict.get('author'),
            creator=metadata_dict.get('creator'),
            producer=metadata_dict.get('producer'),
            created=metadata_dict.get('created'),
            modified=metadata_dict.get('modified'),
            is_suspicious=is_suspicious,
            suspicious_reasons=suspicious_reasons
        )

    def _parse_pdf_date(self, date_str: str) -> Optional[datetime]:
        """
        Parses PDF date format (e.g., D:20230520120000+00'00')
        """
        if not date_str or not isinstance(date_str, str) or len(date_str) < 10:
            return None
        
        try:
            # Clean string: remove D: and timezone parts for basic parsing
            clean_date = date_str.replace("D:", "").split("+")[0].split("-")[0].split("Z")[0]
            # Format is usually YYYYMMDDHHMMSS
            return datetime.strptime(clean_date[:14], "%Y%m%d%H%M%S")
        except:
            return None

# Singleton
pdf_processor = PDFProcessor()
