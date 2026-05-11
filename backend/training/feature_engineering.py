"""
Feature Engineering and Feature Selection
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import logging
from pathlib import Path
from xgboost import XGBClassifier
import sys

# Import config
sys.path.append(str(Path(__file__).parent.parent))
from app.config import *

# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================
# TARGET DETECTION (🔥 IMPORTANT FIX)
# ============================================================

def detect_target_column(df):
    possible_targets = ["target", "num", "output", "label"]
    
    for col in possible_targets:
        if col in df.columns:
            logger.info(f"✅ Using target column: {col}")
            return col
    
    raise ValueError(f"❌ No valid target column found! Columns: {list(df.columns)}")

# ============================================================
# CORRELATION
# ============================================================

def analyze_correlation(df, threshold=0.9):
    logger.info("Analyzing correlations...")
    
    target_col = detect_target_column(df)
    features = df.drop(columns=[target_col])

    corr_matrix = features.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    to_drop = []

    for col in upper.columns:
        if any(upper[col] > threshold):
            to_drop.append(col)

    logger.info(f"   Dropping {len(to_drop)} correlated features: {to_drop}")
    return to_drop

# ============================================================
# FEATURE IMPORTANCE
# ============================================================

def calculate_feature_importance(X, y):
    logger.info("Calculating feature importance...")

    # 🔥 FIX 1: convert to integer
    y = np.round(y).astype(int)

    # 🔥 FIX 2: shift labels to start from 0
    y = y - y.min()

    logger.info(f"Target classes after fix: {np.unique(y)}")

    model = XGBClassifier(
        n_estimators=100,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        objective='multi:softmax',
        num_class=len(np.unique(y))
    )

    model.fit(X, y)

    importance_df = pd.DataFrame({
        "feature": X.columns,
        "importance": model.feature_importances_
    }).sort_values(by="importance", ascending=False)

    return importance_df

# ============================================================
# MAIN PIPELINE
# ============================================================

def feature_engineering_pipeline():
    logger.info("\n🚀 STARTING FEATURE ENGINEERING\n")

    # Load data
    df = pd.read_csv(PROCESSED_DATA_FILE)
    logger.info(f"Loaded data shape: {df.shape}")

    # Detect target column automatically
    target_col = detect_target_column(df)

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Remove correlated features
    corr_drop = analyze_correlation(df)
    X = X.drop(columns=corr_drop)

    # Feature importance
    importance_df = calculate_feature_importance(X, y)

    # Select features
    selected_features = importance_df[
        importance_df["importance"] > FEATURE_IMPORTANCE_THRESHOLD
    ]["feature"].tolist()

    # Keep critical features
    for f in CRITICAL_FEATURES:
        if f in X.columns and f not in selected_features:
            selected_features.append(f)

    logger.info(f"Final selected features: {selected_features}")

    # Final dataset
    df_final = X[selected_features].copy()
    df_final[target_col] = y

    # Save outputs
    final_path = PROCESSED_DATA_DIR / "final_processed_data.csv"
    df_final.to_csv(final_path, index=False)

    joblib.dump(selected_features, FEATURE_COLUMNS_FILE)
    joblib.dump(importance_df, FEATURE_IMPORTANCE_FILE)

    logger.info("✅ Feature engineering completed successfully!")

    return df_final

# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    df_final = feature_engineering_pipeline()

    print("\n✅ DONE")
    print(f"Final shape: {df_final.shape}")