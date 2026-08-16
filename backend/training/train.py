# ============================================================
# IMPORTS
# ============================================================

import pandas as pd
import numpy as np
import joblib
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import sys

# Add backend root to path
sys.path.append(str(Path(__file__).parent.parent))

from app.config import *

# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================
# LOAD DATA
# ============================================================

def load_processed_data(file_path=None):

    if file_path is None:
        file_path = PROCESSED_DATA_DIR / "final_processed_data.csv"

    logger.info(f"Loading data from: {file_path}")

    df = pd.read_csv(file_path)

    # Features and target
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    # Fix target classes
    y = np.round(y).astype(int)
    y = y - y.min()

    logger.info(f"Target Classes: {np.unique(y)}")

    return X, y

# ============================================================
# SPLIT DATA
# ============================================================

def split_data(X, y):

    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

# ============================================================
# TRAIN MODEL
# ============================================================

def train_model(X_train, y_train):

    model = XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        objective='multi:softmax',
        num_class=len(np.unique(y_train))
    )

    logger.info("Training model...")

    model.fit(X_train, y_train)

    logger.info("Model training completed")

    return model

# ============================================================
# SAVE MODEL
# ============================================================

def save_model(model):

    TRAINED_MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, TRAINED_MODEL_FILE)

    logger.info(f"Model saved at: {TRAINED_MODEL_FILE}")

# ============================================================
# LOAD TRAINED MODEL
# ============================================================

def load_trained_model():

    if not TRAINED_MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Model file not found at: {TRAINED_MODEL_FILE}"
        )

    logger.info(f"Loading trained model from: {TRAINED_MODEL_FILE}")

    model = joblib.load(TRAINED_MODEL_FILE)

    logger.info("Model loaded successfully")

    return model

# ============================================================
# TRAINING PIPELINE
# ============================================================

def training_pipeline():

    logger.info("🚀 STARTING TRAINING PIPELINE")

    # Load data
    X, y = load_processed_data()

    # Split
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Save train and test datasets
    train_df = X_train.copy()
    train_df[TARGET_COLUMN] = y_train
    train_df.to_csv(TRAIN_DATA_FILE, index=False)

    test_df = X_test.copy()
    test_df[TARGET_COLUMN] = y_test
    test_df.to_csv(TEST_DATA_FILE, index=False)

    # Train
    model = train_model(X_train, y_train)

    # Save
    save_model(model)

    logger.info("✅ TRAINING COMPLETE")

    return model, X_train, X_test, y_train, y_test

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    training_pipeline()