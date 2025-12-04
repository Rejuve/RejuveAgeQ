# config/question_config.py
"""
Load question configuration from CSV.
"""
import logging
import pandas as pd
import ast
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


def _safe_parse_dict(value: str, field_name: str, code: str) -> dict:
    """Safely parse a string representation of a dict."""
    if pd.isna(value) or not value.strip():
        return {}
    #Normalize smart quotes → ASCII
    value = (
        value.replace("‘", "'")
             .replace("’", "'")
             .replace("“", '"')
             .replace("”", '"')
    )
    try:
        result = ast.literal_eval(value)
        if not isinstance(result, dict):
            logger.warning(f"{field_name} for {code} is not a dict: {type(result)}")
            return {}
        return result
    except (ValueError, SyntaxError) as e:
        logger.warning(f"Failed to parse {field_name} for {code}: {e}")
        return {}

def load_question_config(csv_path: str = "data/final_questions.csv") -> Dict[str, Dict]:
    """
    Load question configuration from CSV.
    
    Returns:
        Dict of {nhanes_code: {options, dependencies, ...}}
    """
    try:
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df)} questions from {csv_path}")
    except FileNotFoundError:
        logger.error(f"CSV file not found: {csv_path}")
        return {}
    
    config = {}
    
    for _, row in df.iterrows():
        nhanes_code = row['code']
        
        # Parse Options (user-facing)
        try:
            options = _safe_parse_dict(row['Options'], 'Options', nhanes_code)

        except Exception as e:
            logger.warning(f"Failed to parse Options for {nhanes_code}: {e}")
            options = {}
        
        # Parse dependencies
        dependencies = None
        if pd.notna(row.get('Dependencies')):
            try:
                dependencies = _safe_parse_dict(row.get('Dependencies', ''), 'Dependencies', nhanes_code)
            except Exception as e:
                logger.warning(f"Failed to parse Dependencies for {nhanes_code}: {e}")
        
        config[nhanes_code] = {
            'nhanes_code': nhanes_code,
            'name': row.get('name', ''),
            'options': options,
            'dependencies': dependencies,
            'description': row.get('description', ''),
        }
    
    logger.info(f"Loaded configuration for {len(config)} questions")
    return config


# Load configuration at module import
QUESTION_CONFIG = load_question_config()
if not QUESTION_CONFIG:
    raise RuntimeError(
        "Failed to load question configuration. "
        "Ensure data/final_questions.csv exists and is valid."
    )


# Create reverse mapping: app_id → question config
from config.mappings import QUESTION_ID_MAPPING

APP_ID_TO_CONFIG = {}
for app_id, nhanes_code in QUESTION_ID_MAPPING.items():
    if nhanes_code in QUESTION_CONFIG:
        APP_ID_TO_CONFIG[str(app_id)] = QUESTION_CONFIG[nhanes_code]

logger.info(f"Created app ID mapping for {len(APP_ID_TO_CONFIG)} questions")


def get_question_config(ques_id: str) -> Optional[Dict[str, Any]]:
    """
    Get configuration for a question by app ID.
    
    Args:
        ques_id: Application question ID (e.g., "10740")
        
    Returns:
        Question config dict or None if not found
    """
    return APP_ID_TO_CONFIG.get(str(ques_id))