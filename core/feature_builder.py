# core/feature_builder.py
"""
Build feature DataFrame for AutoGluon prediction.
"""
import logging
from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np

from config.mappings import BIOMETRICS_MAPPING, get_nhanes_code
from core.answer_mapper import AnswerMapper

logger = logging.getLogger(__name__)


class FeatureBuilder:
    """Converts payload to AutoGluon-ready DataFrame."""
    
    def __init__(self, model_features: List[str]):
        """
        Args:
            model_features: List of feature names from predictor.features()
        """
        self.model_features = model_features
        self.mapper = AnswerMapper()
        
    def build(
        self,
        surveys: List[Dict],
        biometrics: Dict[str, Any]
    ) -> Tuple[pd.DataFrame, List[str], List[str]]:
        """
        Build feature DataFrame.
        
        Args:
            surveys: List of {ques_id, answer} dicts
            biometrics: Dict of biometric measurements
            
        Returns:
            Tuple of (DataFrame, warnings, errors)
        """
        warnings = []
        errors = []
        features = {}
        
        # Map survey questions
        for survey in surveys:
            ques_id = str(survey.get('ques_id', ''))
            answer_key = survey.get('answer')
            
            if not ques_id:
                continue
            
            # Map to NHANES code
            nhanes_code = get_nhanes_code(ques_id)
            
            if nhanes_code is None:
                warnings.append(f"Unknown question ID: {ques_id}")
                logger.warning(f"Unmapped question ID: {ques_id}")
                continue
            
            # Map answer code to model value
            model_value, error = self.mapper.map_answer(ques_id, answer_key)
            
            if error:
                errors.append(error)
                logger.error(error)
                continue
            
            # Store mapped value (or None for null answers)
            features[nhanes_code] = model_value if model_value is not None else np.nan
        
        # Map biometrics
        for bio_key, bio_value in biometrics.items():
            nhanes_code = BIOMETRICS_MAPPING.get(bio_key)
            
            if nhanes_code is None:
                warnings.append(f"Unknown biometric: {bio_key}")
                logger.warning(f"Unmapped biometric: {bio_key}")
                continue
            
            # Convert to appropriate type
            if bio_value is not None and bio_value != '':
                try:
                    # Try numeric conversion for biometrics
                    features[nhanes_code] = float(bio_value)
                except (ValueError, TypeError):
                    # Keep as string if not numeric
                    features[nhanes_code] = str(bio_value)
            else:
                features[nhanes_code] = np.nan
        
        # Calculate BMI if needed
        if 'BMXBMI' in self.model_features:
            if 'BMXWT' in features and 'BMXHT' in features:
                try:
                    weight_kg = float(features['BMXWT'])
                    height_cm = float(features['BMXHT'])
                    height_m = height_cm / 100.0
                    bmi = weight_kg / (height_m ** 2)
                    features['BMXBMI'] = round(bmi, 2)
                    logger.debug(f"Calculated BMI: {bmi:.2f}")
                except (ValueError, TypeError, ZeroDivisionError) as e:
                    features['BMXBMI'] = np.nan
                    warnings.append(f"Could not calculate BMI: {e}")
            else:
                features['BMXBMI'] = np.nan
        
        # Create DataFrame with all model features
        df = pd.DataFrame([features])
        
        # Reindex to match model features exactly (fills missing with NaN)
        df = df.reindex(columns=self.model_features)
        
        # Handle 'index' feature if present
        if 'index' in self.model_features:
            df['index'] = 0
        
        non_null_count = df.notna().sum().sum()
        logger.info(
            f"Built feature vector: {non_null_count}/{len(self.model_features)} "
            f"non-null features"
        )
        
        return df, warnings, errors