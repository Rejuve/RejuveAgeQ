# core/payload_parser.py
"""
Parse and validate incoming JSON payloads.
"""
import logging
from typing import Dict, Any, Tuple, List

logger = logging.getLogger(__name__)


class PayloadParser:
    """Validates and parses incoming request payloads."""
    
    @staticmethod
    def parse(payload: Dict) -> Tuple[List[Dict], Dict, List[str]]:
        """
        Parse payload into surveys, biometrics, and validation errors.
        
        Args:
            payload: Raw JSON payload
            
        Returns:
            Tuple of (surveys, biometrics, errors)
        """
        errors = []
        
        # Validate structure
        if not isinstance(payload, dict):
            errors.append("Payload must be a JSON object")
            return [], {}, errors
        
        # Extract surveys
        surveys = payload.get('surveys', [])
        if not isinstance(surveys, list) or  not surveys:
            errors.append("'surveys' must be a non-empty list in the payload")
            surveys = []
        
        # Extract biometrics
        biometrics = payload.get('biometrics', {})
        if not isinstance(biometrics, dict) or not biometrics:
            errors.append("'biometrics' must be a non-empty dict in the payload")
            biometrics = {}
        
        # Validate survey structure
        for idx, survey in enumerate(surveys):
            if not isinstance(survey, dict):
                errors.append(f"Survey item {idx} must be a dict")
                continue
            if 'ques_id' not in survey:
                errors.append(f"Survey item {idx} missing 'ques_id'")
            if 'answer' not in survey:
                errors.append(f"Survey item {idx} missing 'answer'")
        
        logger.info(
            f"Parsed payload: {len(surveys)} surveys, "
            f"{len(biometrics)} biometrics, {len(errors)} errors"
        )
        
        return surveys, biometrics, errors