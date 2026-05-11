"""
Data Conversion Script for UCI Heart Disease Dataset
====================================================
Converts the space-separated UCI Heart Disease dataset to CSV format.

The dataset uses multi-class classification:
    0 → healthy (no disease)
    1 → mild heart disease
    2 → moderate heart disease
    3 → severe heart disease
    4 → very severe heart disease

Author: Healthcare ML Project
Date: February 2026
"""

import pandas as pd
import os

def convert_heart_disease_to_csv():
    """
    Convert UCI Heart Disease dataset from space-separated format to CSV.
    
    The dataset contains 14 attributes:
        1. age: age in years
        2. sex: sex (1 = male; 0 = female)
        3. cp: chest pain type (1-4)
        4. trestbps: resting blood pressure (mm Hg)
        5. chol: serum cholesterol (mg/dl)
        6. fbs: fasting blood sugar > 120 mg/dl (1 = true; 0 = false)
        7. restecg: resting electrocardiographic results (0-2)
        8. thalach: maximum heart rate achieved
        9. exang: exercise induced angina (1 = yes; 0 = no)
        10. oldpeak: ST depression induced by exercise relative to rest
        11. slope: slope of the peak exercise ST segment (1-3)
        12. ca: number of major vessels (0-3) colored by flourosopy
        13. thal: thalassemia (3 = normal; 6 = fixed defect; 7 = reversible defect)
        14. target: diagnosis of heart disease (0-4)
    """
    
    # Define column names based on UCI documentation
    column_names = [
        'age',           # Age in years
        'sex',           # Sex (1=male, 0=female)
        'cp',            # Chest pain type (angina, abnang, notang, asympt)
        'trestbps',      # Resting blood pressure (mm Hg)
        'chol',          # Serum cholesterol (mg/dl)
        'fbs',           # Fasting blood sugar > 120 mg/dl (true/false)
        'restecg',       # Resting electrocardiographic results (norm, abn, hyper)
        'thalach',       # Maximum heart rate achieved
        'exang',         # Exercise induced angina (true/false)
        'oldpeak',       # ST depression induced by exercise
        'slope',         # Slope of peak exercise ST segment (up, flat, down)
        'ca',            # Number of major vessels colored by flourosopy
        'thal',          # Thalassemia (norm, fixed, reversible)
        'disease_class', # Disease classification (buff/sick)
        'target'         # Severity level (H, S1, S2, S3, S4)
    ]
    
    # Path to the original dataset
    data_path = r'E:\MajorProj\heart_disease\cleve.mod'
    
    # Read the data file
    print("Reading dataset from:", data_path)
    
    with open(data_path, 'r') as file:
        lines = file.readlines()
    
    # Skip the header comments (lines starting with %)
    data_lines = [line.strip() for line in lines if not line.strip().startswith('%')]
    
    # Parse the data
    data_rows = []
    for line in data_lines:
        if line:  # Skip empty lines
            # Split by whitespace
            parts = line.split()
            if len(parts) >= 14:  # Ensure we have all columns
                data_rows.append(parts)
    
    print(f"Total rows parsed: {len(data_rows)}")
    
    # Create DataFrame
    df = pd.DataFrame(data_rows, columns=column_names)
    
    # Convert categorical values to numerical for easier processing
    # Map sex: male=1, female=0
    df['sex'] = df['sex'].map({'male': 1, 'fem': 0})
    
    # Map chest pain type: angina=1, abnang=2, notang=3, asympt=4
    df['cp'] = df['cp'].map({
        'angina': 1,
        'abnang': 2,
        'notang': 3,
        'asympt': 4
    })
    
    # Map fasting blood sugar: true=1, false=0
    df['fbs'] = df['fbs'].map({'true': 1, 'fal': 0, 'false': 0})
    
    # Map resting ECG: norm=0, abn=1, hyper=2
    df['restecg'] = df['restecg'].map({
        'norm': 0,
        'abn': 1,
        'hyp': 2,
        'hyper': 2
    })
    
    # Map exercise induced angina: true=1, false=0
    df['exang'] = df['exang'].map({'true': 1, 'fal': 0, 'false': 0})
    
    # Map slope: up=1, flat=2, down=3
    df['slope'] = df['slope'].map({
        'up': 1,
        'flat': 2,
        'down': 3
    })
    
    # Map thalassemia: norm=3, fixed=6, reversible=7
    df['thal'] = df['thal'].map({
        'norm': 3,
        'fix': 6,
        'rev': 7,
        'fixed': 6,
        'reversible': 7
    })
    
    # Map target variable for multi-class classification
    # H (healthy) = 0, S1 = 1, S2 = 2, S3 = 3, S4 = 4
    target_mapping = {
        'H': 0,    # Healthy (no disease)
        'S1': 1,   # Mild heart disease
        'S2': 2,   # Moderate heart disease
        'S3': 3,   # Severe heart disease
        'S4': 4    # Very severe heart disease
    }
    df['target'] = df['target'].map(target_mapping)
    
    # Drop the disease_class column (we only need numeric target)
    df = df.drop('disease_class', axis=1)
    
    # Convert all columns to numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Display dataset info
    print("\n" + "="*60)
    print("DATASET CONVERSION SUMMARY")
    print("="*60)
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nData types:")
    print(df.dtypes)
    print("\nMissing values:")
    print(df.isnull().sum())
    print("\nTarget distribution:")
    print(df['target'].value_counts().sort_index())
    print("="*60)
    
    # Save to CSV
    output_path = r'E:\MajorProj\healthcare-ml-project\data\raw\heart_disease.csv'
    df.to_csv(output_path, index=False)
    print(f"\n✓ Dataset successfully saved to: {output_path}")
    
    return df


if __name__ == "__main__":
    df = convert_heart_disease_to_csv()
