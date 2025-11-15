# core/answer_mapper.py
"""
Map user answer codes to model-expected string values.
"""
import logging
from typing import Optional, Tuple

from config.question_config import get_question_config

logger = logging.getLogger(__name__)


class AnswerMapper:
    """Maps user answer codes to model-expected string values."""
    
    @staticmethod
    def map_answer(ques_id: str, answer_key: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Map user answer key to model string value.
        
        Args:
            ques_id: Question ID from payload (e.g., "10740")
            answer_key: Answer code from payload (e.g., "0", "1", "175")
            
        Returns:
            Tuple of (model_value, error_message)
            model_value is None if mapping fails or answer is null
        """
        # Handle null answers
        if answer_key is None or answer_key == '' or str(answer_key).lower() == 'null':
            return None, None
        
        # Convert to string
        answer_key = str(answer_key)
        
        # Get question config
        config = get_question_config(ques_id)
        if config is None:
            return None, f"Unknown question ID: {ques_id}"
        
        # Get Options
        options = config.get('options', {})
        
        if not options:
            logger.warning(f"No options defined for question {ques_id}")
            # Return answer_key as-is for questions without defined options
            return answer_key, None
        
        # Check if this is a numeric range question
        options_values = list(options.values())
        is_numeric_range = any('Range of Values' in str(v) for v in options_values)
        
        if is_numeric_range:
            # For numeric ranges, answer_key IS the numeric value
            # Example: {"ques_id": "10770", "answer": "175"}
            try:
                # Validate it's numeric
                float(answer_key)
                logger.debug(f"Numeric range for {ques_id}: value={answer_key}")
                return answer_key, None
            except ValueError:
                return None, f"Expected numeric value for {ques_id}, got: '{answer_key}'"
        
        # For categorical questions, validate key exists
        if answer_key not in options:
            valid_keys = list(options.keys())
            return None, (
                f"Invalid answer code '{answer_key}' for question {ques_id}. "
                f"Valid codes: {valid_keys}"
            )
        
        # Get the string value
        value = options[answer_key]
        
        logger.debug(f"Mapped {ques_id}: key '{answer_key}' → value '{value}'")
        
        return value, None