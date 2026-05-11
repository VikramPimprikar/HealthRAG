"""
Utility Functions for Healthcare ML Project
==========================================
This module contains helper functions used across the project.

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

# ============================================================
# LOGGING UTILITIES
# ============================================================

def setup_logger(name, level=logging.INFO):
    """
    Setup a logger with consistent formatting.
    
    Args:
        name (str): Logger name
        level: Logging level
        
    Returns:
        logging.Logger: Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Create handler if not exists
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(level)
        
        formatter = logging.Formatter(
            LOG_FORMAT,
            datefmt=DATE_FORMAT
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger


# ============================================================
# FILE UTILITIES
# ============================================================

def ensure_dir_exists(dir_path):
    """
    Create directory if it doesn't exist.
    
    Args:
        dir_path (Path): Directory path
    """
    dir_path = Path(dir_path)
    dir_path.mkdir(parents=True, exist_ok=True)


def load_pickle(file_path):
    """
    Load a pickle file.
    
    Args:
        file_path (Path): Path to pickle file
        
    Returns:
        object: Loaded object
    """
    return joblib.load(file_path)


def save_pickle(obj, file_path):
    """
    Save an object to pickle file.
    
    Args:
        obj: Object to save
        file_path (Path): Path to save file
    """
    ensure_dir_exists(file_path.parent)
    joblib.dump(obj, file_path)


# ============================================================
# DATA VALIDATION UTILITIES
# ============================================================

def validate_input_features(df, expected_features):
    """
    Validate that input data has expected features.
    
    Args:
        df (pd.DataFrame): Input data
        expected_features (list): List of expected feature names
        
    Returns:
        tuple: (is_valid, missing_features, extra_features)
        
    Medical Context:
        Ensures new patient data has all required clinical measurements
        before making predictions.
    """
    df_features = set(df.columns)
    expected_features_set = set(expected_features)
    
    missing_features = expected_features_set - df_features
    extra_features = df_features - expected_features_set
    
    is_valid = len(missing_features) == 0
    
    return is_valid, list(missing_features), list(extra_features)


def validate_feature_ranges(df, feature_ranges=None):
    """
    Validate that feature values are within expected ranges.
    
    Args:
        df (pd.DataFrame): Input data
        feature_ranges (dict): Dictionary of {feature: (min, max)}
        
    Returns:
        dict: Dictionary of out-of-range values
        
    Medical Context:
        Detects anomalous clinical measurements that may indicate
        data entry errors or unusual patient cases requiring review.
    """
    if feature_ranges is None:
        feature_ranges = NORMAL_RANGES
    
    out_of_range = {}
    
    for feature, (min_val, max_val) in feature_ranges.items():
        if feature in df.columns:
            below_min = df[df[feature] < min_val]
            above_max = df[df[feature] > max_val]
            
            if len(below_min) > 0 or len(above_max) > 0:
                out_of_range[feature] = {
                    'below_min': len(below_min),
                    'above_max': len(above_max)
                }
    
    return out_of_range


# ============================================================
# MODEL UTILITIES
# ============================================================

def load_model_artifacts():
    """
    Load all model artifacts (model, scaler, feature columns).
    
    Returns:
        dict: Dictionary containing model artifacts
        
    Medical Context:
        Loads all necessary components for making predictions
        on new patient data.
    """
    artifacts = {}
    
    # Load model
    if TRAINED_MODEL_FILE.exists():
        artifacts['model'] = joblib.load(TRAINED_MODEL_FILE)
    else:
        raise FileNotFoundError(f"Model not found: {TRAINED_MODEL_FILE}")
    
    # Load scaler
    if SCALER_FILE.exists():
        artifacts['scaler'] = joblib.load(SCALER_FILE)
    else:
        raise FileNotFoundError(f"Scaler not found: {SCALER_FILE}")
    
    # Load feature columns
    if FEATURE_COLUMNS_FILE.exists():
        artifacts['feature_columns'] = joblib.load(FEATURE_COLUMNS_FILE)
    else:
        raise FileNotFoundError(f"Feature columns not found: {FEATURE_COLUMNS_FILE}")
    
    return artifacts


# ============================================================
# PREDICTION UTILITIES
# ============================================================

def get_severity_label(prediction):
    """
    Convert numeric prediction to severity label.
    
    Args:
        prediction (int): Numeric prediction (0-4)
        
    Returns:
        str: Severity label
        
    Medical Context:
        Translates model output to clinically meaningful categories.
    """
    return CLASS_LABELS.get(prediction, f"Unknown ({prediction})")


def get_risk_description(prediction):
    """
    Get detailed risk description for a prediction.
    
    Args:
        prediction (int): Numeric prediction (0-4)
        
    Returns:
        str: Risk description
        
    Medical Context:
        Provides actionable clinical interpretation of predictions.
    """
    descriptions = {
        0: "No heart disease detected. Maintain healthy lifestyle.",
        1: "Mild heart disease. Recommend lifestyle modifications and monitoring.",
        2: "Moderate heart disease. Medical intervention recommended.",
        3: "Severe heart disease. Immediate medical attention advised.",
        4: "Very severe heart disease. Urgent medical intervention required."
    }
    
    return descriptions.get(prediction, "Unknown risk level. Consult physician.")


# ============================================================
# STATISTICS UTILITIES
# ============================================================

def calculate_statistics(df, column):
    """
    Calculate descriptive statistics for a column.
    
    Args:
        df (pd.DataFrame): DataFrame
        column (str): Column name
        
    Returns:
        dict: Statistics dictionary
    """
    stats = {
        'mean': df[column].mean(),
        'median': df[column].median(),
        'std': df[column].std(),
        'min': df[column].min(),
        'max': df[column].max(),
        'q25': df[column].quantile(0.25),
        'q75': df[column].quantile(0.75)
    }
    
    return stats


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    print("Healthcare ML Project - Utility Functions")
    print("=" * 60)
    print("This module provides helper functions for the project.")
    print("Import and use these functions in other modules.")
    print("=" * 60)
