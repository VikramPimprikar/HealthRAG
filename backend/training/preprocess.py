"""
Data Preprocessing Pipeline for Healthcare ML Project
====================================================
This module handles all data preprocessing tasks including:
- Loading raw data
- Data quality inspection
- Handling missing values
- Removing useless columns
- Feature encoding
- Feature scaling
- Data splitting

Author: Healthcare ML Project
Date: February 2026
"""

import pandas as pd
import numpy as np
import joblib
import logging
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from app.config import *

# ============================================================
# LOGGING SETUP
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# ============================================================
# DATA LOADING FUNCTIONS
# ============================================================

def load_data(file_path=None):
    """
    Load the heart disease dataset from CSV file.
    
    Args:
        file_path (str): Path to the CSV file. If None, uses config default.
        
    Returns:
        pd.DataFrame: Loaded dataset
        
    Medical Context:
        Loading clinical data for cardiovascular risk prediction.
        Dataset contains critical biomarkers and patient demographics.
    """
    if file_path is None:
        file_path = RAW_DATA_FILE
    
    logger.info(f"Loading data from: {file_path}")
    
    try:
        df = pd.read_csv(file_path)
        logger.info(f"✓ Data loaded successfully. Shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"✗ Error loading data: {e}")
        raise


def inspect_data(df):
    """
    Perform comprehensive inspection of the dataset.
    
    Args:
        df (pd.DataFrame): Dataset to inspect
        
    Medical Context:
        Data quality inspection is critical in healthcare ML to ensure
        reliable predictions. Missing values or data irregularities can
        lead to incorrect diagnoses.
    """
    logger.info("\n" + "="*70)
    logger.info("DATA INSPECTION REPORT")
    logger.info("="*70)
    
    # 1. Dataset shape
    logger.info(f"\n1. DATASET SHAPE")
    logger.info(f"   Rows (patients): {df.shape[0]}")
    logger.info(f"   Columns (features): {df.shape[1]}")
    
    # 2. Column names and types
    logger.info(f"\n2. COLUMN INFORMATION")
    logger.info(f"   Columns: {df.columns.tolist()}")
    logger.info(f"\n   Data Types:")
    for col in df.columns:
        logger.info(f"   - {col}: {df[col].dtype}")
    
    # 3. Missing values analysis
    logger.info(f"\n3. MISSING VALUES ANALYSIS")
    missing_counts = df.isnull().sum()
    missing_percentages = (missing_counts / len(df)) * 100
    
    if missing_counts.sum() == 0:
        logger.info("   ✓ No missing values found!")
    else:
        logger.info("   Missing values detected:")
        for col in df.columns:
            if missing_counts[col] > 0:
                logger.info(f"   - {col}: {missing_counts[col]} ({missing_percentages[col]:.2f}%)")
    
    # 4. Target distribution (critical for healthcare)
    if TARGET_COLUMN in df.columns:
        logger.info(f"\n4. TARGET VARIABLE DISTRIBUTION (Disease Severity)")
        target_dist = df[TARGET_COLUMN].value_counts().sort_index()
        logger.info("   Distribution:")
        for severity, count in target_dist.items():
            percentage = (count / len(df)) * 100
            severity_label = CLASS_LABELS.get(severity, f"Class {severity}")
            logger.info(f"   - {severity_label} (Class {severity}): {count} patients ({percentage:.1f}%)")
        
        # Check for class imbalance
        if target_dist.max() / target_dist.min() > 3:
            logger.warning("   ⚠ Class imbalance detected! Consider using stratified sampling.")
    
    # 5. Statistical summary
    logger.info(f"\n5. STATISTICAL SUMMARY")
    logger.info("\n" + str(df.describe().round(2)))
    
    # 6. Duplicate rows check
    duplicates = df.duplicated().sum()
    logger.info(f"\n6. DUPLICATE ROWS")
    if duplicates > 0:
        logger.warning(f"   ⚠ Found {duplicates} duplicate rows")
    else:
        logger.info(f"   ✓ No duplicate rows found")
    
    logger.info("="*70 + "\n")


# ============================================================
# DATA CLEANING FUNCTIONS
# ============================================================

def clean_data(df):
    """
    Clean the dataset by removing useless columns and handling data quality issues.
    
    Steps:
        1. Remove columns with excessive missing values (>40%)
        2. Remove columns with zero or near-zero variance
        3. Remove ID-like columns
        4. Handle duplicates
        
    Args:
        df (pd.DataFrame): Raw dataset
        
    Returns:
        pd.DataFrame: Cleaned dataset
        
    Medical Context:
        In healthcare, irrelevant or poor-quality features can introduce noise
        and reduce model reliability. We focus on clinically meaningful variables.
    """
    logger.info("Starting data cleaning process...")
    df_clean = df.copy()
    original_columns = df_clean.columns.tolist()
    columns_to_drop = []
    
    # 1. Remove columns with excessive missing values
    logger.info("\n1. Checking for columns with excessive missing values (>40%)...")
    missing_percentages = (df_clean.isnull().sum() / len(df_clean)) * 100
    
    for col in df_clean.columns:
        if col != TARGET_COLUMN:  # Never drop target
            if missing_percentages[col] > (MISSING_VALUE_THRESHOLD * 100):
                columns_to_drop.append(col)
                logger.info(f"   - Dropping '{col}': {missing_percentages[col]:.2f}% missing")
    
    # 2. Remove columns with near-zero variance
    logger.info("\n2. Checking for columns with near-zero variance...")
    for col in df_clean.columns:
        if col != TARGET_COLUMN and col not in columns_to_drop:
            if df_clean[col].dtype in ['float64', 'int64']:
                # Calculate coefficient of variation (std/mean)
                if df_clean[col].std() != 0 and df_clean[col].mean() != 0:
                    cv = df_clean[col].std() / abs(df_clean[col].mean())
                    if cv < ZERO_VARIANCE_THRESHOLD:
                        columns_to_drop.append(col)
                        logger.info(f"   - Dropping '{col}': near-zero variance (CV={cv:.4f})")
                elif df_clean[col].std() == 0:
                    columns_to_drop.append(col)
                    logger.info(f"   - Dropping '{col}': zero variance (constant value)")
    
    # 3. Remove duplicates
    logger.info("\n3. Removing duplicate rows...")
    duplicates_before = df_clean.duplicated().sum()
    if duplicates_before > 0:
        df_clean = df_clean.drop_duplicates()
        logger.info(f"   ✓ Removed {duplicates_before} duplicate rows")
    else:
        logger.info("   ✓ No duplicates found")
    
    # Drop identified columns
    if columns_to_drop:
        df_clean = df_clean.drop(columns=columns_to_drop)
        logger.info(f"\n✓ Dropped {len(columns_to_drop)} columns: {columns_to_drop}")
    else:
        logger.info("\n✓ No columns needed to be dropped")
    
    logger.info(f"\nCleaning summary:")
    logger.info(f"   Original columns: {len(original_columns)}")
    logger.info(f"   Remaining columns: {len(df_clean.columns)}")
    logger.info(f"   Original rows: {len(df)}")
    logger.info(f"   Remaining rows: {len(df_clean)}")
    
    return df_clean


def handle_missing_values(df):
    """
    Handle missing values in the dataset.
    
    Strategy:
        - Numerical columns: Fill with median (robust to outliers)
        - Categorical columns: Fill with mode (most frequent value)
        
    Args:
        df (pd.DataFrame): Dataset with potential missing values
        
    Returns:
        pd.DataFrame: Dataset with imputed missing values
        
    Medical Context:
        Missing clinical data is common in healthcare. We use median imputation
        for numerical features (robust to outliers) and mode for categorical.
        This preserves distributional properties while handling missingness.
    """
    logger.info("Handling missing values...")
    df_imputed = df.copy()
    
    missing_before = df_imputed.isnull().sum().sum()
    
    if missing_before == 0:
        logger.info("   ✓ No missing values to handle")
        return df_imputed
    
    logger.info(f"   Total missing values: {missing_before}")
    
    for col in df_imputed.columns:
        if col == TARGET_COLUMN:
            continue
            
        missing_count = df_imputed[col].isnull().sum()
        
        if missing_count > 0:
            # For numerical columns: use median
            if df_imputed[col].dtype in ['float64', 'int64']:
                median_value = df_imputed[col].median()
                df_imputed[col].fillna(median_value, inplace=True)
                logger.info(f"   - {col}: Filled {missing_count} missing values with median ({median_value:.2f})")
            
            # For categorical columns: use mode
            else:
                mode_value = df_imputed[col].mode()[0]
                df_imputed[col].fillna(mode_value, inplace=True)
                logger.info(f"   - {col}: Filled {missing_count} missing values with mode ({mode_value})")
    
    missing_after = df_imputed.isnull().sum().sum()
    logger.info(f"   ✓ Missing values handled. Before: {missing_before}, After: {missing_after}")
    
    return df_imputed


# ============================================================
# FEATURE ENCODING FUNCTIONS
# ============================================================

def encode_features(df):
    """
    Encode categorical features to numerical values.
    
    Note: In this dataset, most features are already numerical.
    This function ensures all features are properly encoded.
    
    Args:
        df (pd.DataFrame): Dataset with potential categorical features
        
    Returns:
        pd.DataFrame: Dataset with encoded features
        
    Medical Context:
        ML models require numerical input. We ensure all clinical categorical
        variables are properly encoded while preserving their relationships.
    """
    logger.info("Encoding categorical features...")
    df_encoded = df.copy()
    
    # Check for any remaining categorical columns
    categorical_columns = df_encoded.select_dtypes(include=['object']).columns.tolist()
    
    if TARGET_COLUMN in categorical_columns:
        categorical_columns.remove(TARGET_COLUMN)
    
    if len(categorical_columns) == 0:
        logger.info("   ✓ All features are already numerical")
        return df_encoded
    
    # For any categorical columns, use label encoding
    for col in categorical_columns:
        unique_values = df_encoded[col].nunique()
        logger.info(f"   - Encoding {col} ({unique_values} unique values)")
        df_encoded[col] = pd.Categorical(df_encoded[col]).codes
    
    logger.info(f"   ✓ Encoded {len(categorical_columns)} categorical features")
    
    return df_encoded


# ============================================================
# FEATURE SCALING FUNCTIONS
# ============================================================

def scale_features(df, scaler=None, fit=True):
    """
    Scale numerical features using StandardScaler.
    
    StandardScaler: Transforms features to have mean=0 and std=1.
    This is crucial for many ML algorithms and improves convergence.
    
    Args:
        df (pd.DataFrame): Dataset to scale
        scaler (StandardScaler): Pre-fitted scaler (for test data)
        fit (bool): Whether to fit the scaler (True for training, False for test)
        
    Returns:
        tuple: (scaled_dataframe, fitted_scaler)
        
    Medical Context:
        Clinical features have different scales (e.g., age: 0-100, cholesterol: 100-600).
        Standardization ensures all features contribute equally to the model,
        preventing bias toward high-magnitude features.
    """
    logger.info("Scaling features...")
    
    # Separate features and target
    if TARGET_COLUMN in df.columns:
        X = df.drop(columns=[TARGET_COLUMN])
        y = df[TARGET_COLUMN]
    else:
        X = df
        y = None
    
    feature_names = X.columns.tolist()
    
    # Scale features
    if fit:
        # Fit scaler on training data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        logger.info("   ✓ Scaler fitted and features scaled")
        
        # Log scaling statistics
        logger.info(f"   Feature means: {scaler.mean_[:3].round(2)}... (showing first 3)")
        logger.info(f"   Feature stds: {scaler.scale_[:3].round(2)}... (showing first 3)")
    else:
        # Transform using existing scaler (for test data)
        if scaler is None:
            raise ValueError("Scaler must be provided when fit=False")
        X_scaled = scaler.transform(X)
        logger.info("   ✓ Features scaled using existing scaler")
    
    # Convert back to DataFrame
    df_scaled = pd.DataFrame(X_scaled, columns=feature_names, index=X.index)
    
    # Add target back if it existed
    if y is not None:
        df_scaled[TARGET_COLUMN] = y
    
    return df_scaled, scaler


# ============================================================
# DATA SAVING FUNCTIONS
# ============================================================

def save_processed_data(df, file_path=None):
    """
    Save processed dataset to CSV file.
    
    Args:
        df (pd.DataFrame): Processed dataset
        file_path (str): Path to save file. If None, uses config default.
        
    Medical Context:
        Saving processed data ensures reproducibility and allows
        for auditing of the preprocessing pipeline.
    """
    if file_path is None:
        file_path = PROCESSED_DATA_FILE
    
    # Ensure directory exists
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving processed data to: {file_path}")
    df.to_csv(file_path, index=False)
    logger.info(f"   ✓ Data saved successfully. Shape: {df.shape}")


def save_scaler(scaler, file_path=None):
    """
    Save the fitted scaler to disk.
    
    Args:
        scaler (StandardScaler): Fitted scaler object
        file_path (str): Path to save scaler. If None, uses config default.
        
    Medical Context:
        The scaler must be saved to ensure new patient data is
        transformed consistently with training data.
    """
    if file_path is None:
        file_path = SCALER_FILE
    
    # Ensure directory exists
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving scaler to: {file_path}")
    joblib.dump(scaler, file_path)
    logger.info("   ✓ Scaler saved successfully")


def save_feature_columns(feature_columns, file_path=None):
    """
    Save the list of feature columns used in training.
    
    Args:
        feature_columns (list): List of feature column names
        file_path (str): Path to save. If None, uses config default.
        
    Medical Context:
        Saving feature names ensures we use the exact same features
        during prediction as during training.
    """
    if file_path is None:
        file_path = FEATURE_COLUMNS_FILE
    
    # Ensure directory exists
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving feature columns to: {file_path}")
    joblib.dump(feature_columns, file_path)
    logger.info(f"   ✓ Saved {len(feature_columns)} feature columns")


# ============================================================
# MAIN PREPROCESSING PIPELINE
# ============================================================

def preprocess_pipeline():
    """
    Execute the complete preprocessing pipeline.
    
    Pipeline Steps:
        1. Load raw data
        2. Inspect data quality
        3. Clean data (remove useless columns)
        4. Handle missing values
        5. Encode categorical features
        6. Scale numerical features
        7. Save processed data and artifacts
        
    Medical Context:
        This pipeline ensures data quality and consistency critical
        for reliable cardiovascular risk prediction.
    """
    logger.info("\n" + "="*70)
    logger.info("STARTING PREPROCESSING PIPELINE")
    logger.info("="*70 + "\n")
    
    # Step 1: Load data
    df = load_data()
    
    # Step 2: Inspect data
    inspect_data(df)
    
    # Step 3: Clean data
    df_clean = clean_data(df)
    
    # Step 4: Handle missing values
    df_imputed = handle_missing_values(df_clean)
    
    # Step 5: Encode features
    df_encoded = encode_features(df_imputed)
    
    # Step 6: Scale features
    df_scaled, scaler = scale_features(df_encoded, fit=True)
    
    # Step 7: Save processed data and artifacts
    save_processed_data(df_scaled)
    save_scaler(scaler)
    
    # Save feature columns (excluding target)
    feature_columns = [col for col in df_scaled.columns if col != TARGET_COLUMN]
    save_feature_columns(feature_columns)
    
    logger.info("\n" + "="*70)
    logger.info("PREPROCESSING PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("="*70)
    logger.info(f"Final dataset shape: {df_scaled.shape}")
    logger.info(f"Features: {len(feature_columns)}")
    logger.info(f"Samples: {len(df_scaled)}")
    logger.info("="*70 + "\n")
    
    return df_scaled, scaler, feature_columns


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    # Run preprocessing pipeline
    df_processed, scaler, features = preprocess_pipeline()
    
    print("\n✓ Preprocessing complete!")
    print(f"✓ Processed data saved to: {PROCESSED_DATA_FILE}")
    print(f"✓ Scaler saved to: {SCALER_FILE}")
    print(f"✓ Feature columns saved to: {FEATURE_COLUMNS_FILE}")
