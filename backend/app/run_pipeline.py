"""
Main Entry Point for Healthcare ML Project
=========================================
This is the orchestration script that coordinates the entire ML pipeline.

Author: Healthcare ML Project
Date: February 2026
"""

import sys
from pathlib import Path
import argparse
import logging

# Add backend directory to path
sys.path.append(str(Path(__file__).parent.parent))

from app.config import *
from training.preprocess import preprocess_pipeline
from training.feature_engineering import feature_engineering_pipeline
from training.train import training_pipeline
from training.evaluate import evaluation_pipeline
from app.predict import predict_heart_disease, example_prediction

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# ============================================================
# PIPELINE ORCHESTRATION
# ============================================================

def run_full_pipeline():
    """
    Execute the complete ML pipeline from data preprocessing to evaluation.
    
    Steps:
        1. Data preprocessing
        2. Feature engineering
        3. Model training
        4. Model evaluation
        
    Medical Context:
        This end-to-end pipeline ensures consistent and reproducible
        model development for cardiovascular risk prediction.
    """
    logger.info("\n" + "="*70)
    logger.info("STARTING FULL ML PIPELINE")
    logger.info("="*70 + "\n")
    
    # Step 1: Preprocessing
    logger.info("="*70)
    logger.info("STEP 1: DATA PREPROCESSING")
    logger.info("="*70)
    df_processed, scaler, features = preprocess_pipeline()
    
    # Step 2: Feature Engineering
    logger.info("\n" + "="*70)
    logger.info("STEP 2: FEATURE ENGINEERING")
    logger.info("="*70)
    df_final, importance, selected_features = feature_engineering_pipeline()
    
    # Step 3: Model Training
    logger.info("\n" + "="*70)
    logger.info("STEP 3: MODEL TRAINING")
    logger.info("="*70)
    model, X_train, X_test, y_train, y_test = training_pipeline()
    
    # Step 4: Model Evaluation
    logger.info("\n" + "="*70)
    logger.info("STEP 4: MODEL EVALUATION")
    logger.info("="*70)
    metrics, y_test_eval, y_pred, y_pred_proba = evaluation_pipeline()
    
    # Pipeline Summary
    logger.info("\n" + "="*70)
    logger.info("FULL PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("="*70)
    logger.info(f"✓ Data preprocessing complete")
    logger.info(f"✓ Feature engineering complete")
    logger.info(f"✓ Model training complete")
    logger.info(f"✓ Model evaluation complete")
    logger.info(f"\nFinal Model Performance:")
    logger.info(f"   Accuracy: {metrics['accuracy']*100:.2f}%")
    logger.info(f"   F1-Score: {metrics['f1']*100:.2f}%")
    if metrics['roc_auc']:
        logger.info(f"   ROC-AUC: {metrics['roc_auc']*100:.2f}%")
    logger.info("="*70 + "\n")


def run_training_only():
    """
    Run only the training pipeline (assumes preprocessing is done).
    """
    logger.info("Running training pipeline only...")
    model, X_train, X_test, y_train, y_test = training_pipeline()
    return model


def run_evaluation_only():
    """
    Run only the evaluation pipeline (assumes model is trained).
    """
    logger.info("Running evaluation pipeline only...")
    metrics, y_test, y_pred, y_pred_proba = evaluation_pipeline()
    return metrics


def run_prediction_example():
    """
    Run an example prediction.
    """
    logger.info("Running prediction example...")
    example_prediction()


# ============================================================
# COMMAND LINE INTERFACE
# ============================================================

def main():
    """
    Main entry point with command line argument parsing.
    """
    parser = argparse.ArgumentParser(
        description='Healthcare ML Project - Heart Disease Risk Prediction',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example Commands:
  python main.py --full          # Run complete pipeline
  python main.py --preprocess    # Run preprocessing only
  python main.py --feature       # Run feature engineering only
  python main.py --train         # Run training only
  python main.py --evaluate      # Run evaluation only
  python main.py --predict       # Run prediction example
        """
    )
    
    parser.add_argument(
        '--full',
        action='store_true',
        help='Run the complete ML pipeline (preprocess, feature engineering, train, evaluate)'
    )
    
    parser.add_argument(
        '--preprocess',
        action='store_true',
        help='Run data preprocessing only'
    )
    
    parser.add_argument(
        '--feature',
        action='store_true',
        help='Run feature engineering only'
    )
    
    parser.add_argument(
        '--train',
        action='store_true',
        help='Run model training only'
    )
    
    parser.add_argument(
        '--evaluate',
        action='store_true',
        help='Run model evaluation only'
    )
    
    parser.add_argument(
        '--predict',
        action='store_true',
        help='Run prediction example'
    )
    
    args = parser.parse_args()
    
    # If no arguments, show help
    if not any(vars(args).values()):
        parser.print_help()
        return
    
    # Execute requested pipeline
    try:
        if args.full:
            run_full_pipeline()
        
        if args.preprocess:
            preprocess_pipeline()
        
        if args.feature:
            feature_engineering_pipeline()
        
        if args.train:
            run_training_only()
        
        if args.evaluate:
            run_evaluation_only()
        
        if args.predict:
            run_prediction_example()
        
        print("\n✓ Requested operations completed successfully!")
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}")
        raise


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
