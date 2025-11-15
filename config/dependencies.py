# config/dependencies.py
"""
Dependency validation logic.
"""
import logging
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd

from config.question_config import QUESTION_CONFIG

logger = logging.getLogger(__name__)


def build_dependency_map() -> Dict[str, Tuple[str, str]]:
    """
    Build dependency map from question configs.
    
    Returns:
        Dict of {dependent_code: (parent_code, required_key)}
    """
    dependencies = {}
    
    for nhanes_code, config in QUESTION_CONFIG.items():
        deps = config.get('dependencies')
        if deps and isinstance(deps, dict):
            # deps format: {'BPQ040A': '1'}
            # Means this question depends on BPQ040A having key '1'
            for parent_code, required_key in deps.items():
                dependencies[nhanes_code] = (parent_code, required_key)
                logger.debug(
                    f"Dependency: {nhanes_code} requires {parent_code}='{required_key}'"
                )
    
    logger.info(f"Built {len(dependencies)} dependency rules")
    return dependencies


DEPENDENCIES = build_dependency_map()


class DependencyHandler:
    """Handles conditional question logic."""
    
    def __init__(self):
        self.dependencies = DEPENDENCIES
        
    def apply_dependencies(
        self, 
        features: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], List[str]]:
        """
        Apply dependency rules to feature dictionary.
        
        Args:
            features: Dict of {nhanes_code: string_value}
            
        Returns:
            Tuple of (updated_features, warnings)
        """
        warnings = []
        updated_features = features.copy()
        
        for dependent, (parent, required_key) in self.dependencies.items():
            # Skip if dependent not in payload
            if dependent not in updated_features:
                continue
            
            # Get parent value
            parent_value = updated_features.get(parent)
            
            # Get parent's Options to find required string value
            parent_config = QUESTION_CONFIG.get(parent)
            
            if parent_config:
                parent_options = parent_config.get('options', {})
                required_value = parent_options.get(required_key)
            else:
                # If no config, use key as value (fallback)
                required_value = required_key
                logger.warning(
                    f"No config for parent {parent}, using key '{required_key}' as value"
                )
            
            # Check if dependency is met
            dependency_met = (parent_value == required_value)
            
            if not dependency_met:
                # Nullify dependent only if it has a non-null value
                if updated_features[dependent] is not None and not pd.isna(updated_features[dependent]):
                    logger.info(
                        f"Nullifying {dependent}: {parent}='{parent_value}' "
                        f"(required: '{required_value}')"
                    )
                    warnings.append(
                        f"Question {dependent} auto-nullified: "
                        f"requires {parent}='{required_value}'"
                    )
                    updated_features[dependent] = np.nan
        
        return updated_features, warnings