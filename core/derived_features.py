# core/derived_features.py
"""
Calculate derived features from existing values.
"""
import logging
from typing import Dict, Any
import numpy as np

logger = logging.getLogger(__name__)


class DerivedFeatureCalculator:
    """Calculates features derived from other features."""
    
    # Define all derivation rules
    DERIVATION_RULES = {
        'BMXBMI': {
            'requires': ['BMXWT', 'BMXHT'],
            'function': 'calculate_bmi',
            'description': 'BMI from weight and height'
        },
        'WHD010': {
            'requires': ['BMXHT'],
            'function': 'height_cm_to_inches',
            'description': 'Self-reported height in inches from measured height'
        },
        'WHD020': {
            'requires': ['BMXWT'],
            'function': 'weight_kg_to_pounds',
            'description': 'Self-reported weight in pounds from measured weight'
        },
        # Add more as discovered
    }
    
    @staticmethod
    def calculate_bmi(features: Dict[str, Any]) -> float:
        """Calculate BMI from weight (kg) and height (cm)."""
        try:
            weight_kg = float(features['BMXWT'])
            height_cm = float(features['BMXHT'])
            height_m = height_cm / 100.0
            bmi = weight_kg / (height_m ** 2)
            return round(bmi, 2)
        except (KeyError, ValueError, TypeError, ZeroDivisionError):
            return np.nan
    
    @staticmethod
    def height_cm_to_inches(features: Dict[str, Any]) -> float:
        """Convert height from cm to inches."""
        try:
            height_cm = float(features['BMXHT'])
            height_inches = height_cm / 2.54
            return round(height_inches, 1)
        except (KeyError, ValueError, TypeError):
            return np.nan
    
    @staticmethod
    def weight_kg_to_pounds(features: Dict[str, Any]) -> float:
        """Convert weight from kg to pounds."""
        try:
            weight_kg = float(features['BMXWT'])
            weight_lbs = weight_kg * 2.20462
            return round(weight_lbs, 1)
        except (KeyError, ValueError, TypeError):
            return np.nan
    
    def calculate_derived_features(
        self,
        features: Dict[str, Any],
        model_features: list
    ) -> Dict[str, Any]:
        """
        Calculate all derived features that the model expects.
        
        Args:
            features: Dict of existing features
            model_features: List of all features model expects
            
        Returns:
            Updated features dict with derived values added
        """
        updated_features = features.copy()
        
        for derived_feature in model_features:
            # Skip if already present
            if derived_feature in updated_features and updated_features[derived_feature] is not None:
                continue
            
            # Check if this is a derived feature
            if derived_feature not in self.DERIVATION_RULES:
                continue
            
            rule = self.DERIVATION_RULES[derived_feature]
            
            # Check if all required features are present
            required = rule['requires']
            if not all(req in updated_features for req in required):
                logger.debug(
                    f"Cannot derive {derived_feature}: missing {required}"
                )
                continue
            
            # Get the calculation function
            calc_func = getattr(self, rule['function'], None)
            if calc_func is None:
                logger.warning(f"No function found for {derived_feature}")
                continue
            
            # Calculate
            try:
                value = calc_func(updated_features)
                updated_features[derived_feature] = value
                logger.info(
                    f"Derived {derived_feature} = {value} ({rule['description']})"
                )
            except Exception as e:
                logger.error(f"Failed to derive {derived_feature}: {e}")
                updated_features[derived_feature] = np.nan
        
        return updated_features