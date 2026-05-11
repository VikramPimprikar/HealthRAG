"""
RAGChainMed Configuration Settings
===================================
Central configuration for the AI-based clinical decision support system.

Contains all settings for:
- Model paths and parameters
- Data directories
- Clinical data management
- Blockchain audit logging
- RAG pipeline settings
- Access control

Author: RAGChainMed
Date: May 2026
"""

import os
from pathlib import Path

# ============================================================
# PROJECT PATHS CONFIGURATION
# ============================================================

# Root directory of the project
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'

# Model directories
MODEL_DIR = PROJECT_ROOT / 'model'

# Backend directories
BACKEND_DIR = PROJECT_ROOT / 'backend'
TRAINING_DIR = BACKEND_DIR / 'training'

# RAGChainMed specific directories
CLINICAL_DATA_DIR = BACKEND_DIR / 'clinical_data'
KNOWLEDGE_BASE_DIR = BACKEND_DIR / 'knowledge_base'
AUDIT_LOGS_DIR = BACKEND_DIR / 'audit_logs'
VECTOR_DB_DIR = BACKEND_DIR / 'vector_db'

# ============================================================
# DATA FILES CONFIGURATION
# ============================================================

# Input data file
RAW_DATA_FILE = RAW_DATA_DIR / 'heart_disease.csv'

# Processed data files
PROCESSED_DATA_FILE = PROCESSED_DATA_DIR / 'processed_data.csv'
TRAIN_DATA_FILE = PROCESSED_DATA_DIR / 'train_data.csv'
TEST_DATA_FILE = PROCESSED_DATA_DIR / 'test_data.csv'
FINAL_PROCESSED_DATA = PROCESSED_DATA_DIR / 'final_processed_data.csv'

# ============================================================
# MODEL FILES CONFIGURATION
# ============================================================

# Trained model file
TRAINED_MODEL_FILE = MODEL_DIR / 'trained_model.pkl'

# Scaler file (StandardScaler for feature normalization)
SCALER_FILE = MODEL_DIR / 'scaler.pkl'

# Feature columns file (list of features used in training)
FEATURE_COLUMNS_FILE = MODEL_DIR / 'feature_columns.pkl'

# Feature importance file
FEATURE_IMPORTANCE_FILE = MODEL_DIR / 'feature_importance.pkl'

# ============================================================
# BLOCKCHAIN AUDIT CONFIGURATION
# ============================================================

# Audit log database
AUDIT_LOG_DB = AUDIT_LOGS_DIR / 'audit_logs.db'

# Ensure directories exist
CLINICAL_DATA_DIR.mkdir(exist_ok=True, parents=True)
KNOWLEDGE_BASE_DIR.mkdir(exist_ok=True, parents=True)
AUDIT_LOGS_DIR.mkdir(exist_ok=True, parents=True)
VECTOR_DB_DIR.mkdir(exist_ok=True, parents=True)

# ============================================================
# RAG PIPELINE CONFIGURATION
# ============================================================

# Embedding model to use
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Vector database path
VECTOR_DB_PATH = str(VECTOR_DB_DIR / 'medical_vector_db')

# Chunk size for document splitting
RAG_CHUNK_SIZE = 500
RAG_CHUNK_OVERLAP = 50

# Number of documents to retrieve by default
DEFAULT_RETRIEVE_K = 3

# ============================================================
# DATA PREPROCESSING CONFIGURATION
# ============================================================

# Target column name
TARGET_COLUMN = 'num'

# Features that should always be kept (clinical importance)
CRITICAL_FEATURES = [
    'age',        # Age of the patient - critical demographic factor
    'sex',        # Sex - known risk factor for heart disease
    'cp',         # Chest pain type - primary symptom indicator
    'trestbps',   # Resting blood pressure - key vital sign
    'chol',       # Serum cholesterol - major risk factor
    'thalach',    # Maximum heart rate - exercise capacity indicator
    'exang',      # Exercise induced angina - ischemia indicator
    'oldpeak',    # ST depression - ECG abnormality indicator
]

# Thresholds for data quality
MISSING_VALUE_THRESHOLD = 0.4  # Drop columns with >40% missing values
ZERO_VARIANCE_THRESHOLD = 0.01  # Drop columns with near-zero variance
HIGH_CORRELATION_THRESHOLD = 0.9  # Drop highly correlated features

# ============================================================
# FEATURE ENGINEERING CONFIGURATION  
# ============================================================

# Feature importance threshold (drop features below this)
FEATURE_IMPORTANCE_THRESHOLD = 0.01  # Keep features with >1% importance

# ============================================================
# MODEL TRAINING CONFIGURATION
# ============================================================

# Train-test split
TEST_SIZE = 0.2  # 20% test, 80% train
RANDOM_STATE = 42  # For reproducibility
STRATIFY = True  # Stratified split to maintain class distribution

# XGBoost hyperparameters
XGBOOST_PARAMS = {
    'n_estimators': 200,
    'learning_rate': 0.05,
    'max_depth': 5,
    'random_state': 42,
    'verbose': 0
}

# ============================================================
# CLASS LABELS CONFIGURATION
# ============================================================

# Disease severity classification
CLASS_LABELS = {
    0: "Healthy (No disease)",
    1: "Mild (0% narrowing)",
    2: "Moderate (50% narrowing)",
    3: "Severe (75% narrowing)",
    4: "Very Severe (>75% narrowing)"
}

# Disease severity descriptions for clinical use
SEVERITY_DESCRIPTIONS = {
    0: "No significant coronary artery disease detected",
    1: "Minimal coronary artery disease with <50% narrowing",
    2: "Moderate coronary artery disease with 50-75% narrowing",
    3: "Severe coronary artery disease with >75% narrowing",
    4: "Critical coronary artery disease requiring urgent intervention"
}

# Risk levels based on prediction
RISK_LEVELS = {
    0: "Low Risk",
    1: "Low-Moderate Risk",
    2: "Moderate Risk",
    3: "High Risk",
    4: "Critical Risk"
}

# ============================================================
# CLINICAL DECISION SUPPORT CONFIGURATION
# ============================================================

# Risk score thresholds
RISK_SCORE_CRITICAL = 0.8
RISK_SCORE_HIGH = 0.6
RISK_SCORE_MODERATE = 0.4
RISK_SCORE_LOW = 0.0

# ============================================================
# ACCESS CONTROL CONFIGURATION
# ============================================================

# Default users and roles
DEFAULT_USERS = [
    {
        'user_id': 'admin',
        'username': 'Admin User',
        'role': 'admin',
        'department': 'Administration'
    },
    {
        'user_id': 'system',
        'username': 'RAGChainMed System',
        'role': 'system',
        'department': 'System'
    }
]

# ============================================================
# DATA CLASSIFICATION CONFIGURATION
# ============================================================

# Data classification levels
DATA_CLASSIFICATIONS = {
    'public': 'Publicly available information',
    'internal': 'Internal use only',
    'confidential': 'Confidential patient information',
    'restricted': 'Highly restricted - requires special authorization'
}

# ============================================================
# MEDICAL DATA CONFIGURATION
# ============================================================

# Supported medical data types
SUPPORTED_DATA_TYPES = [
    'patient_demographics',
    'vital_signs',
    'laboratory_results',
    'imaging_results',
    'clinical_notes',
    'medication_history',
    'diagnosis',
    'treatment_plan'
]

# ============================================================
# API CONFIGURATION
# ============================================================

# API version
API_VERSION = "1.0.0"

# API settings
API_TITLE = "RAGChainMed"
API_DESCRIPTION = "AI-based Clinical Decision Support System"

# ============================================================
# LOGGING CONFIGURATION
# ============================================================

LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

# ============================================================
# ENVIRONMENT VARIABLES
# ============================================================

# OpenAI API Key
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')

# Debug mode
DEBUG_MODE = os.getenv('DEBUG_MODE', 'False').lower() == 'true'
XGBOOST_PARAMS = {
    'n_estimators': 200,          # Number of boosting rounds
    'learning_rate': 0.05,        # Learning rate (eta)
    'max_depth': 5,               # Maximum tree depth
    'subsample': 0.8,             # Subsample ratio of training instances
    'colsample_bytree': 0.8,      # Subsample ratio of columns
    'min_child_weight': 1,        # Minimum sum of instance weight in a child
    'gamma': 0,                   # Minimum loss reduction for split
    'reg_alpha': 0,               # L1 regularization
    'reg_lambda': 1,              # L2 regularization
    'random_state': RANDOM_STATE,
    'n_jobs': -1,                 # Use all CPU cores
    'objective': 'multi:softmax', # Multi-class classification
    'num_class': 5,               # 5 classes (0, 1, 2, 3, 4)
    'eval_metric': 'mlogloss',    # Multi-class log loss
}

# ============================================================
# MODEL EVALUATION CONFIGURATION
# ============================================================

# Class labels for heart disease severity
CLASS_LABELS = {
    0: 'Healthy',
    1: 'Mild',
    2: 'Moderate',
    3: 'Severe',
    4: 'Very Severe'
}

# Metrics to compute
EVALUATION_METRICS = [
    'accuracy',
    'precision_weighted',
    'recall_weighted',
    'f1_weighted',
    'roc_auc_ovr',  # One-vs-Rest ROC AUC for multi-class
]

# ============================================================
# LOGGING CONFIGURATION
# ============================================================

# Logging level
LOG_LEVEL = 'INFO'  # Options: DEBUG, INFO, WARNING, ERROR, CRITICAL

# Log format
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

# ============================================================
# CLINICAL FEATURE DESCRIPTIONS
# ============================================================

FEATURE_DESCRIPTIONS = {
    'age': 'Patient age in years',
    'sex': 'Patient sex (1=male, 0=female)',
    'cp': 'Chest pain type (1=typical angina, 2=atypical angina, 3=non-anginal, 4=asymptomatic)',
    'trestbps': 'Resting blood pressure in mm Hg',
    'chol': 'Serum cholesterol in mg/dl',
    'fbs': 'Fasting blood sugar > 120 mg/dl (1=true, 0=false)',
    'restecg': 'Resting ECG results (0=normal, 1=ST-T abnormality, 2=left ventricular hypertrophy)',
    'thalach': 'Maximum heart rate achieved during exercise',
    'exang': 'Exercise induced angina (1=yes, 0=no)',
    'oldpeak': 'ST depression induced by exercise relative to rest',
    'slope': 'Slope of peak exercise ST segment (1=upsloping, 2=flat, 3=downsloping)',
    'ca': 'Number of major vessels (0-3) colored by fluoroscopy',
    'thal': 'Thalassemia (3=normal, 6=fixed defect, 7=reversible defect)',
}

# ============================================================
# MEDICAL DOMAIN KNOWLEDGE
# ============================================================

# Normal ranges for clinical parameters (for reference)
NORMAL_RANGES = {
    'age': (0, 120),
    'trestbps': (90, 140),     # Normal resting BP: 90-140 mm Hg
    'chol': (0, 240),          # Desirable cholesterol: <200 mg/dl
    'thalach': (60, 220),      # Normal heart rate range
    'oldpeak': (0, 6.2),       # ST depression range
}

print("Configuration loaded successfully!")
print(f"Project Root: {PROJECT_ROOT}")
print(f"Raw Data File: {RAW_DATA_FILE}")
print(f"Model Directory: {MODEL_DIR}")
