from thefuzz import fuzz
from pydantic import BaseModel
from typing import List, Dict
from .entity_extractor import ExtractedData

class ValidationResult(BaseModel):
    consistency_score: float
    mismatches: List[str]
    is_valid: bool

class KYCValidator:
    def __init__(self, threshold: int = 80):
        self.threshold = threshold

    def validate(self, doc_a_data: ExtractedData, doc_b_data: ExtractedData) -> ValidationResult:
        """
        Compares extracted data from two documents using fuzzy matching.
        """
        mismatches = []
        scores = []

        # 1. Compare Names
        name_score = fuzz.token_sort_ratio(doc_a_data.person_name, doc_b_data.person_name)
        scores.append(name_score)
        if name_score < self.threshold:
            mismatches.append(f"Name mismatch detected: '{doc_a_data.person_name}' vs '{doc_b_data.person_name}' ({name_score}%)")

        # 2. Compare Addresses
        # Addresses can be tricky, so we use a partial ratio or token set ratio
        addr_score = fuzz.token_set_ratio(doc_a_data.address, doc_b_data.address)
        scores.append(addr_score)
        if addr_score < self.threshold:
            mismatches.append(f"Address mismatch detected: '{doc_a_data.address}' vs '{doc_b_data.address}' ({addr_score}%)")

        # Calculate average consistency score
        avg_score = sum(scores) / len(scores) if scores else 0
        
        return ValidationResult(
            consistency_score=round(avg_score, 2),
            mismatches=mismatches,
            is_valid=avg_score >= self.threshold
        )

# Singleton
kyc_validator = KYCValidator()
