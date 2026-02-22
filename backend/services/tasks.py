import os
import time
from core.celery_app import celery_app
from services.ocr_service import ocr_service
from services.fraud_detector import calculate_ela, image_to_base64
from services.layout_analyzer import layout_analyzer
from services.scoring_engine import calculate_final_score
from services.entity_extractor import entity_extractor
from services.dl_detector import dl_detector, dl_image_to_base64
from services.pdf_processor import pdf_processor

@celery_app.task(bind=True)
def analyze_document_task(self, file_path, original_filename):
    """
    Heavy ML processing task for document fraud detection.
    """
    file_id = os.path.basename(file_path).split('.')[0]
    extension = os.path.splitext(file_path)[1].lower()
    upload_dir = os.path.dirname(file_path)
    
    try:
        # Update state: Processing
        self.update_state(state='PROGRESS', meta={'message': 'Initializing analysis...'})
        
        # 1. PDF Handling
        pdf_metadata = None
        processing_path = file_path
        
        if extension == '.pdf':
            self.update_state(state='PROGRESS', meta={'message': 'Extracting PDF metadata...'})
            pdf_metadata = pdf_processor.extract_metadata(file_path)
            images = pdf_processor.convert_to_images(file_path)
            if not images:
                raise Exception("Failed to convert PDF to image.")
            
            # Use the first page for analysis
            temp_img_path = os.path.join(upload_dir, f"{file_id}_page1.jpg")
            images[0].save(temp_img_path, "JPEG")
            processing_path = temp_img_path

        # 2. OCR and Layout Analysis
        self.update_state(state='PROGRESS', meta={'message': 'Running OCR and Layout Analysis...'})
        ocr_results = ocr_service.extract_text(processing_path)
        layout_score = layout_analyzer.analyze_spatial_consistency(ocr_results)
        
        # 3. Visual Fraud Detection (ELA + DL)
        self.update_state(state='PROGRESS', meta={'message': 'Running Forensic Vision Models...'})
        ela_image, ela_score = calculate_ela(processing_path)
        heatmap_base64 = image_to_base64(ela_image)
        
        dl_image, dl_score = dl_detector.sliding_window_inference(processing_path)
        dl_heatmap_base64 = dl_image_to_base64(dl_image)
        
        # 4. Final Scoring
        final_score, classification = calculate_final_score(ela_score, layout_score, dl_score)
        
        # 5. NLP Entity Extraction
        self.update_state(state='PROGRESS', meta={'message': 'Extracting intelligent entities...'})
        extracted_entities = entity_extractor.extract(ocr_results)
        
        # Convert extracted entities to dict if it's a Pydantic model
        if hasattr(extracted_entities, "dict"):
            extracted_entities = extracted_entities.dict()
        
        # Convert pdf_metadata to dict if it exists
        if hasattr(pdf_metadata, "dict"):
            pdf_metadata = pdf_metadata.dict()

        # 6. Generate AI Explanation if needed
        ai_explanation_64 = None
        if dl_score > 0.2:
            self.update_state(state='PROGRESS', meta={'message': 'Generating AI Explainability Map...'})
            explanation_img = dl_detector.generate_explanation(processing_path)
            ai_explanation_64 = dl_image_to_base64(explanation_img)

        result = {
            "filename": original_filename,
            "final_score": final_score,
            "classification": classification,
            "ela_score": round(float(ela_score), 4),
            "layout_score": round(float(layout_score), 4),
            "dl_score": round(float(dl_score), 4),
            "is_fraud": classification != "Authentic" or (pdf_metadata['is_suspicious'] if pdf_metadata else False),
            "ocr_data": ocr_results,
            "heatmap_base64": heatmap_base64,
            "dl_heatmap_base64": dl_heatmap_base64,
            "extracted_entities": extracted_entities,
            "pdf_metadata": pdf_metadata,
            "ai_explanation_64": ai_explanation_64
        }
        
        return result

    except Exception as e:
        self.update_state(state='FAILURE', meta={'error': str(e)})
        raise e
    finally:
        # Cleanup page image if it was created
        if extension == '.pdf' and 'temp_img_path' in locals() and os.path.exists(temp_img_path):
            os.remove(temp_img_path)
