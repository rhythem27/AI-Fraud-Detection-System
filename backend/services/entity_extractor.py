import spacy
from pydantic import BaseModel
from typing import Optional, List
import re

class ExtractedData(BaseModel):
    person_name: Optional[str] = "Unknown"
    address: Optional[str] = "Unknown"
    date: Optional[str] = "Unknown"

class EntityExtractor:
    def __init__(self):
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            # Fallback if model not loaded yet in this session
            self.nlp = None

    def extract(self, text_list: List[dict]) -> ExtractedData:
        """
        Extracts entities from a list of OCR results.
        """
        full_text = " ".join([item['text'] for item in text_list])
        
        if not self.nlp:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except:
                return ExtractedData(person_name="Model Loading", address="N/A", date="N/A")

        doc = self.nlp(full_text)
        
        entities = {
            "PERSON": [],
            "GPE": [], # Geopolitical entity (cities, states, etc.)
            "LOC": [], # Locations
            "DATE": [],
            "FAC": []  # Buildings, airports, highways, bridges, etc.
        }

        for ent in doc.ents:
            if ent.label_ in entities:
                entities[ent.label_].append(ent.text)

        # Simple logic to pick the best candidates
        name = entities["PERSON"][0] if entities["PERSON"] else "Unknown"
        
        # Address logic: Combine GPE, LOC, and FAC
        address_parts = entities["GPE"] + entities["LOC"] + entities["FAC"]
        address = ", ".join(address_parts[:3]) if address_parts else "Unknown"
        
        # If OCR text has something that looks like an address but spaCy missed it, 
        # we could add regex here, but for now we stick to NER.
        
        date = entities["DATE"][0] if entities["DATE"] else "Unknown"

        return ExtractedData(
            person_name=name,
            address=address,
            date=date
        )

# Singleton
entity_extractor = EntityExtractor()
