"""
Prediction API for Healthcare ML Project
========================================
This module provides functions to make predictions on new patient data.

Author: Healthcare ML Project
Date: February 2026
"""

import pandas as pd
import numpy as np
import joblib
import logging
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from app.config import *
from training.utils import (
    load_model_artifacts,
    get_severity_label,
    get_risk_description,
    validate_input_features,
    setup_logger
)

# Setup logger
logger = setup_logger(__name__)


# ============================================================
# PREDICTION FUNCTIONS
# ============================================================

def predict_heart_disease(patient_data):
    """
    Predict heart disease severity for a patient.
    
    Args:
        patient_data (dict or pd.DataFrame): Patient clinical data
        
    Returns:
        dict: Prediction results
        
    Medical Context:
        Takes patient clinical measurements and predicts cardiovascular
        disease risk category using the trained XGBoost model.
        
    Example:
        >>> patient = {
        ...     'age': 63,
        ...     'sex': 1,
        ...     'cp': 1,
        ...     'trestbps': 145,
        ...     'chol': 233,
        ...     'fbs': 1,
        ...     'restecg': 2,
        ...     'thalach': 150,
        ...     'exang': 0,
        ...     'oldpeak': 2.3,
        ...     'slope': 3,
        ...     'ca': 0,
        ...     'thal': 6
        ... }
        >>> result = predict_heart_disease(patient)
        >>> print(result['severity'])
    """
    logger.info("Loading model artifacts...")
    
    # Load model artifacts
    try:
        artifacts = load_model_artifacts()
        model = artifacts['model']
        scaler = artifacts['scaler']
        feature_columns = artifacts['feature_columns']
    except FileNotFoundError as e:
        logger.error(f"Failed to load model artifacts: {e}")
        raise
    
    # Convert to DataFrame if dict
    if isinstance(patient_data, dict):
        df = pd.DataFrame([patient_data])
    else:
        df = patient_data.copy()
    
    logger.info(f"Making prediction for {len(df)} patient(s)...")
    
    # Get scaler's expected feature order
    scaler_features = list(scaler.feature_names_in_)
    
    # Validate features against scaler's expected features
    is_valid, missing, extra = validate_input_features(df, scaler_features)
    
    if not is_valid:
        error_msg = f"Missing required features: {missing}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    if extra:
        logger.warning(f"Extra features will be ignored: {extra}")
    
    # Reorder to match scaler's expected feature order
    df_for_scaling = df[scaler_features]
    
    # Scale features
    X_scaled = scaler.transform(df_for_scaling)
    
    # Create DataFrame with scaled values in model's expected order
    df_scaled = pd.DataFrame(X_scaled, columns=scaler_features)
    X_for_model = df_scaled[feature_columns]
    
    # Make prediction
    prediction = model.predict(X_for_model)[0]
    prediction_proba = model.predict_proba(X_for_model)[0]
    
    # Prepare result
    result = {
        'prediction': int(prediction),
        'severity': get_severity_label(prediction),
        'risk_description': get_risk_description(prediction),
        'confidence': float(prediction_proba[prediction]),
        'probabilities': {
            CLASS_LABELS[i]: float(prob) 
            for i, prob in enumerate(prediction_proba)
        }
    }
    
    logger.info(f"Prediction: {result['severity']} (confidence: {result['confidence']:.2%})")
    
    return result


def batch_predict(data_file_path):
    """
    Make predictions for multiple patients from a CSV file.
    
    Args:
        data_file_path (str): Path to CSV file with patient data
        
    Returns:
        pd.DataFrame: DataFrame with predictions
        
    Medical Context:
        Batch prediction for screening multiple patients or
        retrospective analysis of patient cohorts.
    """
    logger.info(f"Loading data from: {data_file_path}")
    
    # Load data
    df = pd.read_csv(data_file_path)
    
    logger.info(f"Loaded {len(df)} patient records")
    
    # Load model artifacts
    artifacts = load_model_artifacts()
    model = artifacts['model']
    scaler = artifacts['scaler']
    feature_columns = artifacts['feature_columns']
    
    # Validate and prepare features
    is_valid, missing, extra = validate_input_features(df, feature_columns)
    
    if not is_valid:
        error_msg = f"Missing required features: {missing}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    # Select and order features
    X = df[feature_columns]
    
    # Scale features
    X_scaled = scaler.transform(X)
    
    # Make predictions
    predictions = model.predict(X_scaled)
    probabilities = model.predict_proba(X_scaled)
    
    # Add predictions to DataFrame
    df_results = df.copy()
    df_results['prediction'] = predictions
    df_results['severity'] = [get_severity_label(p) for p in predictions]
    df_results['confidence'] = [probabilities[i, pred] for i, pred in enumerate(predictions)]
    
    logger.info(f"Predictions completed for {len(df_results)} patients")
    
    # Summary
    logger.info("\nPrediction Summary:")
    for severity_id, severity_name in CLASS_LABELS.items():
        count = (predictions == severity_id).sum()
        percentage = (count / len(predictions)) * 100
        logger.info(f"   {severity_name}: {count} ({percentage:.1f}%)")
    
    return df_results


# ============================================================
# EXAMPLE USAGE
# ============================================================

def example_prediction():
    """
    Example of how to use the prediction function.
    """
    print("\n" + "="*70)
    print("EXAMPLE PREDICTION")
    print("="*70 + "\n")
    
    # Example patient data
    patient = {
        'age': 63.0,
        'sex': 1,
        'cp': 1,
        'trestbps': 145.0,
        'chol': 233.0,
        'fbs': 1,
        'restecg': 2,
        'thalach': 150.0,
        'exang': 0,
        'oldpeak': 2.3,
        'slope': 3,
        'ca': 0.0,
        'thal': 6.0
    }
    
    print("Patient Data:")
    for key, value in patient.items():
        print(f"   {key}: {value}")
    
    print("\nMaking prediction...")
    
    try:
        result = predict_heart_disease(patient)
        
        print("\n" + "-"*70)
        print("PREDICTION RESULT")
        print("-"*70)
        print(f"Severity: {result['severity']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"\nRisk Assessment: {result['risk_description']}")
        print("\nProbability Distribution:")
        for severity, prob in result['probabilities'].items():
            print(f"   {severity:>15s}: {prob:.2%}")
        print("-"*70 + "\n")
        
    except Exception as e:
        print(f"Error: {e}")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    example_prediction()
