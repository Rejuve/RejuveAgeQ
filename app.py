# app.py
"""
Flask API for AutoGluon NHANES biological age prediction.
"""
import logging
from typing import Dict, Any
from flask import Flask, request, jsonify
from autogluon.tabular import TabularPredictor

from core.payload_parser import PayloadParser
from core.feature_builder import FeatureBuilder
from config.dependencies import DependencyHandler

# --- Setup ---
app = Flask(__name__)

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# --- Load model at startup ---
PREDICTOR_PATH = "checkpoint/survey_model"
logger.info(f"Loading AutoGluon predictor from {PREDICTOR_PATH}...")

try:
    PREDICTOR = TabularPredictor.load(PREDICTOR_PATH)
    logger.info(f"✓ Model loaded successfully. Features: {len(PREDICTOR.features())}")
except Exception as e:
    logger.error(f"✗ Failed to load model: {e}")
    PREDICTOR = None

# --- Initialize processors ---
parser = PayloadParser()
dependency_handler = DependencyHandler()
feature_builder = FeatureBuilder(
    model_features=PREDICTOR.features() if PREDICTOR else []
)


def process_payload(payload: Dict) -> Dict[str, Any]:
    """
    Process incoming payload through full pipeline.
    
    Returns:
        Dict with 'success', 'data', 'warnings', 'errors' keys
    """
    # Step 1: Parse payload
    surveys, biometrics, parse_errors = parser.parse(payload)
    
    if parse_errors:
        return {
            'success': False,
            'data': None,
            'warnings': [],
            'errors': parse_errors
        }
    
    # Step 2: Build features (includes answer mapping)
    df, build_warnings, build_errors = feature_builder.build(surveys, biometrics)
    
    if build_errors:
        return {
            'success': False,
            'data': None,
            'warnings': build_warnings,
            'errors': build_errors
        }
    
    # Step 3: Apply dependency logic
    # Convert DataFrame first row to dict for dependency checking
    features_dict = df.iloc[0].to_dict()
    updated_features, dep_warnings = dependency_handler.apply_dependencies(features_dict)
    
    # Update DataFrame with nullified dependencies
    for nhanes_code, value in updated_features.items():
        if nhanes_code in df.columns:
            df.at[0, nhanes_code] = value
    
    # Step 4: Predict
    try:
        predictions = PREDICTOR.predict(df)
        biological_age = float(predictions.iloc[0])
        
        all_warnings = build_warnings + dep_warnings
        
        return {
            'success': True,
            'data': {
                'biological_age': biological_age,
                'features_used': int(df.notna().sum().sum()),
                'total_features': len(PREDICTOR.features())
            },
            'warnings': all_warnings if all_warnings else [],
            'errors': []
        }
        
    except Exception as e:
        logger.exception("Prediction failed")
        return {
            'success': False,
            'data': None,
            'warnings': build_warnings + dep_warnings,
            'errors': [f"Prediction error: {str(e)}"]
        }


@app.route("/predict", methods=["POST"])
def predict_endpoint():
    """Main prediction endpoint."""
    logger.info("=" * 60)
    logger.info("Received prediction request")
    
    # Get payload
    payload = request.get_json(silent=True)
    
    if not payload:
        logger.warning("Invalid or empty JSON payload")
        return jsonify({
            "code": 400,
            "biological_age": None,
            "message": "Invalid JSON payload"
        }), 400
    
    # Check model loaded
    if PREDICTOR is None:
        logger.error("Model not loaded")
        return jsonify({
            "code": 500,
            "biological_age": None,
            "message": "Model not available"
        }), 500
    
    # Process
    result = process_payload(payload)
    
    # Format response
    if result['success']:
        response = {
            "code": 200,
            "biological_age": result['data']['biological_age'],
            "message": "Prediction successful",
            "metadata": {
                "features_used": result['data']['features_used'],
                "total_features": result['data']['total_features'],
                "warnings": result['warnings'] if result['warnings'] else None
            }
        }
        logger.info(
            f"✓ Prediction successful: {result['data']['biological_age']:.2f} years "
            f"({result['data']['features_used']}/{result['data']['total_features']} features)"
        )
        return jsonify(response), 200
    else:
        status_code = 400 if result['errors'] else 500
        response = {
            "code": status_code,
            "biological_age": None,
            "message": "Prediction failed",
            "errors": result['errors'],
            "warnings": result['warnings']
        }
        logger.error(f"✗ Prediction failed: {result['errors']}")
        return jsonify(response), status_code


@app.route("/validate", methods=["POST"])
def validate_endpoint():
    """
    Validate payload without running prediction.
    Useful for frontend validation.
    """
    payload = request.get_json(silent=True)
    
    if not payload:
        return jsonify({
            "valid": False,
            "errors": ["Invalid JSON payload"]
        }), 400
    
    # Parse only
    surveys, biometrics, errors = parser.parse(payload)
    
    # Check mappings (without full feature build)
    from config.mappings import get_nhanes_code
    warnings = []
    
    for survey in surveys:
        ques_id = str(survey.get('ques_id', ''))
        if ques_id and get_nhanes_code(ques_id) is None:
            warnings.append(f"Unknown question ID: {ques_id}")
    
    return jsonify({
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "surveys_count": len(surveys),
        "biometrics_count": len(biometrics)
    }), 200 if not errors else 400


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy" if PREDICTOR else "unhealthy",
        "model_loaded": PREDICTOR is not None,
        "features_count": len(PREDICTOR.features()) if PREDICTOR else 0
    }), 200


@app.route("/features", methods=["GET"])
def list_features():
    """List all expected model features."""
    if PREDICTOR is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    return jsonify({
        "features": PREDICTOR.features(),
        "count": len(PREDICTOR.features())
    }), 200


@app.route("/questions", methods=["GET"])
def list_questions():
    """List all available questions with their configurations."""
    from config.question_config import APP_ID_TO_CONFIG
    
    questions = []
    for ques_id, config in APP_ID_TO_CONFIG.items():
        questions.append({
            "ques_id": ques_id,
            "nhanes_code": config['nhanes_code'],
            "name": config['name'],
            "options": config['options'],
            "has_dependencies": config['dependencies'] is not None
        })
    
    return jsonify({
        "questions": questions,
        "count": len(questions)
    }), 200


if __name__ == "__main__":
    # For production, use gunicorn:
    # gunicorn app:app -w 4 -b 0.0.0.0:5001 --preload --timeout 120
    app.run(host="0.0.0.0", port=5001, debug=False)