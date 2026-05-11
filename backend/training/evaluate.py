"""
Model Evaluation Module for Healthcare ML Project
================================================
This module evaluates the trained XGBoost model using:
- Accuracy, Precision, Recall, F1-Score
- ROC-AUC Score (One-vs-Rest for multi-class)
- Confusion Matrix
- Feature Importance Analysis
- Classification Report

Author: Healthcare ML Project
Date: February 2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import logging
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.preprocessing import label_binarize
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from app.config import *
from training.train import load_trained_model

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
# DATA LOADING
# ============================================================

def load_test_data():
    """
    Load the test dataset.
    
    Returns:
        tuple: (X_test, y_test)
        
    Medical Context:
        We evaluate on held-out test data to assess real-world
        performance on unseen patient cases.
    """
    logger.info(f"Loading test data from: {TEST_DATA_FILE}")
    
    df_test = pd.read_csv(TEST_DATA_FILE)
    
    X_test = df_test.drop(columns=[TARGET_COLUMN])
    y_test = df_test[TARGET_COLUMN]
    
    logger.info(f"   ✓ Test data loaded. Samples: {len(X_test)}")
    
    return X_test, y_test


# ============================================================
# MODEL EVALUATION METRICS
# ============================================================

def calculate_metrics(y_true, y_pred, y_pred_proba=None):
    """
    Calculate comprehensive evaluation metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities (for ROC-AUC)
        
    Returns:
        dict: Dictionary of metric scores
        
    Medical Context:
        Multiple metrics provide different perspectives on model performance:
        - Accuracy: Overall correctness
        - Precision: How many predicted cases are true positives
        - Recall: How many actual cases were correctly identified
        - F1-Score: Harmonic mean of precision and recall
        - ROC-AUC: Model's ability to distinguish between classes
    """
    logger.info("Calculating evaluation metrics...")
    
    metrics = {}
    
    # Accuracy
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    
    # Precision, Recall, F1 (weighted for multi-class)
    metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['f1'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # ROC-AUC (One-vs-Rest for multi-class)
    if y_pred_proba is not None:
        try:
            # Binarize labels for multi-class ROC-AUC
            y_true_bin = label_binarize(y_true, classes=[0, 1, 2, 3, 4])
            metrics['roc_auc'] = roc_auc_score(y_true_bin, y_pred_proba, average='weighted', multi_class='ovr')
        except Exception as e:
            logger.warning(f"Could not calculate ROC-AUC: {e}")
            metrics['roc_auc'] = None
    else:
        metrics['roc_auc'] = None
    
    return metrics


def print_metrics(metrics):
    """
    Display evaluation metrics in a formatted manner.
    
    Args:
        metrics (dict): Dictionary of metric scores
        
    Medical Context:
        Clear presentation of metrics helps clinicians understand
        model reliability and trustworthiness for patient care.
    """
    logger.info("\n" + "="*70)
    logger.info("MODEL EVALUATION METRICS")
    logger.info("="*70)
    
    logger.info(f"\n   Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    logger.info(f"   Precision: {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
    logger.info(f"   Recall:    {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
    logger.info(f"   F1-Score:  {metrics['f1']:.4f} ({metrics['f1']*100:.2f}%)")
    
    if metrics['roc_auc'] is not None:
        logger.info(f"   ROC-AUC:   {metrics['roc_auc']:.4f} ({metrics['roc_auc']*100:.2f}%)")
    
    logger.info("\n" + "="*70)
    
    # Medical interpretation
    logger.info("\nCLINICAL INTERPRETATION:")
    
    if metrics['accuracy'] >= 0.85:
        logger.info("   ✓ Excellent accuracy - Model shows strong predictive capability")
    elif metrics['accuracy'] >= 0.75:
        logger.info("   ✓ Good accuracy - Model is reliable for risk assessment")
    elif metrics['accuracy'] >= 0.65:
        logger.info("   ⚠ Moderate accuracy - Consider model improvements")
    else:
        logger.info("   ⚠ Low accuracy - Significant improvements needed")
    
    if metrics['precision'] >= 0.80:
        logger.info("   ✓ High precision - Few false positive diagnoses")
    else:
        logger.info("   ⚠ Consider improving precision to reduce false alarms")
    
    if metrics['recall'] >= 0.80:
        logger.info("   ✓ High recall - Successfully identifying most at-risk patients")
    else:
        logger.info("   ⚠ Consider improving recall to catch more at-risk patients")
    
    logger.info("="*70 + "\n")


# ============================================================
# CONFUSION MATRIX
# ============================================================

def plot_confusion_matrix(y_true, y_pred, save_path=None):
    """
    Create and visualize confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        save_path (str): Path to save plot
        
    Medical Context:
        Confusion matrix shows which disease severity levels
        are being confused by the model - critical for identifying
        where clinical review may be needed.
    """
    logger.info("Creating confusion matrix...")
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Create class labels
    class_names = [CLASS_LABELS[i] for i in sorted(CLASS_LABELS.keys())]
    
    # Create heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Number of Patients'})
    
    plt.title('Confusion Matrix\nHeart Disease Severity Prediction', fontsize=14, fontweight='bold')
    plt.ylabel('True Severity Level', fontsize=12)
    plt.xlabel('Predicted Severity Level', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"   ✓ Confusion matrix saved to: {save_path}")
    else:
        plt.savefig(MODEL_DIR / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        logger.info(f"   ✓ Confusion matrix saved to: {MODEL_DIR / 'confusion_matrix.png'}")
    
    plt.close()
    
    # Print confusion matrix analysis
    logger.info("\nConfusion Matrix Analysis:")
    logger.info(f"\n{cm}\n")
    
    # Calculate per-class accuracy
    logger.info("Per-Class Performance:")
    for i, class_name in enumerate(class_names):
        if cm[i].sum() > 0:
            class_accuracy = cm[i, i] / cm[i].sum()
            logger.info(f"   {class_name:>15s}: {cm[i, i]:>3d}/{cm[i].sum():>3d} correct ({class_accuracy*100:.1f}%)")


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

def print_classification_report(y_true, y_pred):
    """
    Print detailed classification report.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Medical Context:
        Classification report provides per-class precision, recall,
        and F1-scores - essential for understanding performance
        across different disease severity levels.
    """
    logger.info("\nDETAILED CLASSIFICATION REPORT:")
    logger.info("="*70)
    
    # Get class names
    target_names = [CLASS_LABELS[i] for i in sorted(CLASS_LABELS.keys())]
    
    # Generate report
    report = classification_report(y_true, y_pred, target_names=target_names, zero_division=0)
    
    logger.info("\n" + report)
    logger.info("="*70 + "\n")


# ============================================================
# FEATURE IMPORTANCE VISUALIZATION
# ============================================================

def plot_model_feature_importance(model, feature_names, save_path=None):
    """
    Visualize feature importance from the trained model.
    
    Args:
        model: Trained XGBoost model
        feature_names (list): List of feature names
        save_path (str): Path to save plot
        
    Medical Context:
        Feature importance reveals which clinical measurements
        are most influential in risk prediction, guiding
        clinical decision-making and resource allocation.
    """
    logger.info("Analyzing feature importance from trained model...")
    
    # Get feature importance
    importance_values = model.feature_importances_
    
    # Create DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance_values
    }).sort_values('importance', ascending=False)
    
    # Log feature importance
    logger.info("\nFeature Importance Ranking:")
    for idx, row in importance_df.iterrows():
        logger.info(f"   {row['feature']:>12s}: {row['importance']:.4f}")
    
    # Create bar plot
    plt.figure(figsize=(10, 6))
    bars = plt.barh(importance_df['feature'], importance_df['importance'])
    
    # Color critical features differently
    colors = ['#e74c3c' if f in CRITICAL_FEATURES else '#3498db' for f in importance_df['feature']]
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    plt.xlabel('Importance Score', fontsize=12)
    plt.title('Feature Importance - Trained Model\nHeart Disease Risk Prediction', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#e74c3c', label='Critical Medical Features'),
        Patch(facecolor='#3498db', label='Other Features')
    ]
    plt.legend(handles=legend_elements, loc='lower right')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"\n   ✓ Feature importance plot saved to: {save_path}")
    else:
        plt.savefig(MODEL_DIR / 'model_feature_importance.png', dpi=300, bbox_inches='tight')
        logger.info(f"\n   ✓ Feature importance plot saved to: {MODEL_DIR / 'model_feature_importance.png'}")
    
    plt.close()
    
    # Medical interpretation
    logger.info("\nMEDICAL INTERPRETATION OF TOP FEATURES:")
    logger.info("="*70)
    
    for idx, row in importance_df.head(5).iterrows():
        feature = row['feature']
        if feature in FEATURE_DESCRIPTIONS:
            logger.info(f"\n   {feature.upper()}:")
            logger.info(f"   - {FEATURE_DESCRIPTIONS[feature]}")
            logger.info(f"   - Importance: {row['importance']:.4f}")
    
    logger.info("\n" + "="*70 + "\n")


# ============================================================
# MAIN EVALUATION PIPELINE
# ============================================================

def evaluation_pipeline():
    """
    Execute the complete model evaluation pipeline.
    
    Pipeline Steps:
        1. Load trained model
        2. Load test data
        3. Make predictions
        4. Calculate metrics
        5. Display results
        6. Create visualizations
        
    Medical Context:
        Comprehensive evaluation ensures the model is safe and
        effective for clinical decision support.
    """
    logger.info("\n" + "="*70)
    logger.info("STARTING MODEL EVALUATION PIPELINE")
    logger.info("="*70 + "\n")
    
    # Step 1: Load trained model
    logger.info("Loading trained model...")
    model = load_trained_model()
    
    # Step 2: Load test data
    logger.info("\n" + "-"*70)
    X_test, y_test = load_test_data()
    
    # Step 3: Make predictions
    logger.info("\n" + "-"*70)
    logger.info("Making predictions on test set...")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    logger.info(f"   ✓ Predictions completed for {len(y_test)} samples")
    
    # Step 4: Calculate metrics
    logger.info("\n" + "-"*70)
    metrics = calculate_metrics(y_test, y_pred, y_pred_proba)
    
    # Step 5: Display results
    print_metrics(metrics)
    print_classification_report(y_test, y_pred)
    
    # Step 6: Create visualizations
    logger.info("-"*70)
    logger.info("Creating evaluation visualizations...")
    logger.info("-"*70 + "\n")
    
    plot_confusion_matrix(y_test, y_pred)
    plot_model_feature_importance(model, X_test.columns.tolist())
    
    # Summary
    logger.info("\n" + "="*70)
    logger.info("MODEL EVALUATION COMPLETED SUCCESSFULLY")
    logger.info("="*70)
    logger.info(f"Test Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    logger.info(f"F1-Score: {metrics['f1']:.4f}")
    if metrics['roc_auc']:
        logger.info(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    logger.info("="*70 + "\n")
    
    return metrics, y_test, y_pred, y_pred_proba


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    # Run evaluation pipeline
    metrics, y_test, y_pred, y_pred_proba = evaluation_pipeline()
    
    print("\n" + "="*70)
    print("✓ MODEL EVALUATION COMPLETE!")
    print("="*70)
    print(f"✓ Test Accuracy: {metrics['accuracy']*100:.2f}%")
    print(f"✓ Precision: {metrics['precision']*100:.2f}%")
    print(f"✓ Recall: {metrics['recall']*100:.2f}%")
    print(f"✓ F1-Score: {metrics['f1']*100:.2f}%")
    if metrics['roc_auc']:
        print(f"✓ ROC-AUC: {metrics['roc_auc']*100:.2f}%")
    print("="*70)
    print("\nVisualization files created:")
    print(f"   - {MODEL_DIR / 'confusion_matrix.png'}")
    print(f"   - {MODEL_DIR / 'model_feature_importance.png'}")
    print("="*70 + "\n")
