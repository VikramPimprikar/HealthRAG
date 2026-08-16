"""
Prediction API for Healthcare ML Project
========================================
This module provides functions to make predictions on new patient data.

Author: Healthcare ML Project
"""

import pandas as pd
import numpy as np
import joblib
import logging
from pathlib import Path
import sys
from typing import Dict, Any

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
# FEATURE NORMALIZATION HELPER
# ============================================================

def normalize_patient_features(patient_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize clinical feature names and categorical values to numeric.
    """
    data = dict(patient_dict)

    # Normalize thalach alias
    if 'thalach' in data and 'thalch' not in data:
        data['thalch'] = data.pop('thalach')

    # Default mappings
    mappings = {
        'sex': {'male': 1, 'm': 1, 'female': 0, 'f': 0, '1': 1, '0': 0},
        'cp': {
            'typical angina': 1, 'atypical angina': 2, 'non-anginal': 3, 'asymptomatic': 4,
            '1': 1, '2': 2, '3': 3, '4': 4
        },
        'fbs': {'true': 1, 'false': 0, '1': 1, '0': 0, True: 1, False: 0},
        'restecg': {
            'normal': 0, 'st-t abnormality': 1, 'lv hypertrophy': 2,
            '0': 0, '1': 1, '2': 2
        },
        'exang': {'true': 1, 'false': 0, 'yes': 1, 'no': 0, '1': 1, '0': 0, True: 1, False: 0},
        'slope': {'upsloping': 1, 'flat': 2, 'downsloping': 3, '1': 1, '2': 2, '3': 3}
    }

    for key, mapper in mappings.items():
        if key in data and data[key] is not None:
            val = str(data[key]).strip().lower()
            if val in mapper:
                data[key] = mapper[val]
            else:
                try:
                    data[key] = float(data[key])
                except (ValueError, TypeError):
                    pass

    # Ensure all numerical values are float/int
    for key, val in data.items():
        if val is not None:
            try:
                data[key] = float(val)
            except (ValueError, TypeError):
                pass

    return data


# ============================================================
# PARAMETER CATEGORIZATION & EXPLAINABILITY HELPER
# ============================================================

def analyze_patient_parameters(normalized_data: Dict[str, Any], prediction: int, severity: str) -> Dict[str, Any]:
    """
    Evaluate on what clinical parameters and biomarkers the patient is categorized
    into their predicted severity level, including clinical risk status, reference ranges,
    model feature weights, and primary drivers.
    """
    feature_importance_map = {}
    try:
        if FEATURE_IMPORTANCE_FILE.exists():
            fi_df = joblib.load(FEATURE_IMPORTANCE_FILE)
            if isinstance(fi_df, pd.DataFrame) and 'feature' in fi_df.columns and 'importance' in fi_df.columns:
                feature_importance_map = dict(zip(fi_df['feature'], fi_df['importance']))
    except Exception as e:
        logger.warning(f"Could not load feature importance: {e}")

    if not feature_importance_map:
        feature_importance_map = {
            'cp': 0.2562,
            'sex': 0.1593,
            'exang': 0.1003,
            'oldpeak': 0.0741,
            'chol': 0.0643,
            'fbs': 0.0636,
            'age': 0.0635,
            'restecg': 0.0578,
            'slope': 0.0577,
            'thalch': 0.0519,
            'trestbps': 0.0513
        }

    parameters = []

    # 1. Chest Pain Type (cp)
    cp_val = normalized_data.get('cp', 1)
    cp_num = int(cp_val) if cp_val is not None else 1
    cp_names = {1: "Typical Angina", 2: "Atypical Angina", 3: "Non-Anginal Pain", 4: "Asymptomatic (Silent CAD)"}
    cp_name = cp_names.get(cp_num, f"Type {cp_num}")
    cp_weight = feature_importance_map.get('cp', 0.2562)
    
    if cp_num == 4:
        cp_status = "High Risk (Silent CAD)"
        cp_level = "danger"
        cp_finding = "Asymptomatic (Type 4) presentation is strongly correlated with silent myocardial ischemia and severe multivessel coronary disease."
        cp_driver = True
        cp_impact = "Primary Driver of Severe Categorization"
    elif cp_num == 1:
        cp_status = "Moderate Risk"
        cp_level = "warning"
        cp_finding = "Typical Angina (Type 1) - Classic substernal discomfort provoked by exertion and relieved by rest."
        cp_driver = prediction >= 1
        cp_impact = "Elevates Ischemic Risk"
    elif cp_num == 2:
        cp_status = "Borderline"
        cp_level = "info"
        cp_finding = "Atypical Angina (Type 2) — Intermediate ischemic probability."
        cp_driver = False
        cp_impact = "Moderate Risk Factor"
    else:
        cp_status = "Normal / Low Risk"
        cp_level = "success"
        cp_finding = "Non-Anginal Pain (Type 3) — Low cardiac specificity, favorable clinical profile."
        cp_driver = False
        cp_impact = "Low Cardiac Risk"

    parameters.append({
        "feature": "cp",
        "name": "Chest Pain Type (cp)",
        "value": f"{cp_name} (Type {cp_num})",
        "raw_value": cp_num,
        "normal_range": "Non-Anginal (Type 3)",
        "status": cp_status,
        "status_level": cp_level,
        "clinical_finding": cp_finding,
        "model_weight": f"{cp_weight * 100:.1f}%",
        "model_weight_num": float(cp_weight),
        "risk_impact": cp_impact,
        "is_primary_driver": cp_driver
    })

    # 2. ST Depression (oldpeak)
    oldpeak_val = float(normalized_data.get('oldpeak', 0.0))
    op_weight = feature_importance_map.get('oldpeak', 0.0741)
    if oldpeak_val >= 2.0:
        op_status = "High Risk / Critical"
        op_level = "danger"
        op_finding = f"Marked ST depression ({oldpeak_val:.1f} mm) exceeds the critical 2.0 mm clinical threshold, indicating severe exercise-induced myocardial ischemia."
        op_driver = True
        op_impact = "Major Driver of Ischemia Severity"
    elif oldpeak_val >= 1.0:
        op_status = "Moderate Risk"
        op_level = "warning"
        op_finding = f"Moderate ST depression ({oldpeak_val:.1f} mm) suggests subendocardial ischemia during exertional stress."
        op_driver = True
        op_impact = "Elevates Risk Score"
    else:
        op_status = "Normal"
        op_level = "success"
        op_finding = f"ST segment depression ({oldpeak_val:.1f} mm) is within physiological normal limits (< 1.0 mm)."
        op_driver = False
        op_impact = "Normal ST Repolarization"

    parameters.append({
        "feature": "oldpeak",
        "name": "ST Depression (oldpeak)",
        "value": f"{oldpeak_val:.1f} mm",
        "raw_value": oldpeak_val,
        "normal_range": "< 1.0 mm (Normal Baseline)",
        "status": op_status,
        "status_level": op_level,
        "clinical_finding": op_finding,
        "model_weight": f"{op_weight * 100:.1f}%",
        "model_weight_num": float(op_weight),
        "risk_impact": op_impact,
        "is_primary_driver": op_driver
    })

    # 3. Exercise Induced Angina (exang)
    exang_val = int(normalized_data.get('exang', 0))
    ex_weight = feature_importance_map.get('exang', 0.1003)
    if exang_val == 1:
        ex_status = "High Risk (Present)"
        ex_level = "danger"
        ex_finding = "Exercise-induced angina is present, confirming acute exertional coronary supply-demand mismatch."
        ex_driver = True
        ex_impact = "Major Risk Driver for CAD"
    else:
        ex_status = "Normal (Absent)"
        ex_level = "success"
        ex_finding = "No exercise-induced angina reported during exertion."
        ex_driver = False
        ex_impact = "Negative for Exertional Angina"

    parameters.append({
        "feature": "exang",
        "name": "Exercise Induced Angina (exang)",
        "value": "Yes (Present)" if exang_val == 1 else "No (Absent)",
        "raw_value": exang_val,
        "normal_range": "Absent (0 - No)",
        "status": ex_status,
        "status_level": ex_level,
        "clinical_finding": ex_finding,
        "model_weight": f"{ex_weight * 100:.1f}%",
        "model_weight_num": float(ex_weight),
        "risk_impact": ex_impact,
        "is_primary_driver": ex_driver
    })

    # 4. Serum Cholesterol (chol)
    chol_val = float(normalized_data.get('chol', 200.0))
    chol_weight = feature_importance_map.get('chol', 0.0643)
    if chol_val >= 240:
        chol_status = "High Hypercholesterolemia"
        chol_level = "danger"
        chol_finding = f"Hypercholesterolemia ({chol_val:.0f} mg/dL) exceeds 240 mg/dL threshold, substantially accelerating coronary atherosclerotic plaque buildup."
        chol_driver = True
        chol_impact = "Accelerates Atherogenesis"
    elif chol_val >= 200:
        chol_status = "Borderline High"
        chol_level = "warning"
        chol_finding = f"Borderline high cholesterol ({chol_val:.0f} mg/dL) between 200-239 mg/dL."
        chol_driver = False
        chol_impact = "Moderate Risk Contributor"
    else:
        chol_status = "Optimal / Desirable"
        chol_level = "success"
        chol_finding = f"Serum cholesterol ({chol_val:.0f} mg/dL) is within desirable clinical range (< 200 mg/dL)."
        chol_driver = False
        chol_impact = "Optimal Lipid Profile"

    parameters.append({
        "feature": "chol",
        "name": "Serum Cholesterol (chol)",
        "value": f"{chol_val:.0f} mg/dL",
        "raw_value": chol_val,
        "normal_range": "< 200 mg/dL (Desirable)",
        "status": chol_status,
        "status_level": chol_level,
        "clinical_finding": chol_finding,
        "model_weight": f"{chol_weight * 100:.1f}%",
        "model_weight_num": float(chol_weight),
        "risk_impact": chol_impact,
        "is_primary_driver": chol_driver
    })

    # 5. Resting Blood Pressure (trestbps)
    bp_val = float(normalized_data.get('trestbps', 120.0))
    bp_weight = feature_importance_map.get('trestbps', 0.0513)
    if bp_val >= 140:
        bp_status = "Stage 2 Hypertension"
        bp_level = "danger"
        bp_finding = f"Stage 2 Hypertension ({bp_val:.0f} mm Hg) significantly increases cardiac afterload and microvascular strain."
        bp_driver = True
        bp_impact = "Increases Cardiovascular Workload"
    elif bp_val >= 130:
        bp_status = "Stage 1 Hypertension"
        bp_level = "warning"
        bp_finding = f"Stage 1 Hypertension ({bp_val:.0f} mm Hg) increases cardiovascular risk over baseline."
        bp_driver = False
        bp_impact = "Moderate Hypertension Risk"
    elif bp_val >= 120:
        bp_status = "Elevated BP"
        bp_level = "info"
        bp_finding = f"Pre-hypertensive blood pressure ({bp_val:.0f} mm Hg)."
        bp_driver = False
        bp_impact = "Mild Risk Factor"
    else:
        bp_status = "Optimal / Normal"
        bp_level = "success"
        bp_finding = f"Resting blood pressure ({bp_val:.0f} mm Hg) is within healthy guideline limits (< 120 mm Hg)."
        bp_driver = False
        bp_impact = "Optimal Blood Pressure"

    parameters.append({
        "feature": "trestbps",
        "name": "Resting Blood Pressure (trestbps)",
        "value": f"{bp_val:.0f} mm Hg",
        "raw_value": bp_val,
        "normal_range": "90 - 120 mm Hg (Normal)",
        "status": bp_status,
        "status_level": bp_level,
        "clinical_finding": bp_finding,
        "model_weight": f"{bp_weight * 100:.1f}%",
        "model_weight_num": float(bp_weight),
        "risk_impact": bp_impact,
        "is_primary_driver": bp_driver
    })

    # 6. Fasting Blood Sugar (fbs)
    fbs_val = int(normalized_data.get('fbs', 0))
    fbs_weight = feature_importance_map.get('fbs', 0.0636)
    if fbs_val == 1:
        fbs_status = "Elevated (> 120 mg/dL)"
        fbs_level = "warning"
        fbs_finding = "Fasting blood sugar > 120 mg/dL indicates hyperglycemia or diabetic metabolism, promoting vascular endothelial dysfunction."
        fbs_driver = True
        fbs_impact = "Diabetic Metabolic Risk"
    else:
        fbs_status = "Normal (<= 120 mg/dL)"
        fbs_level = "success"
        fbs_finding = "Fasting blood glucose within normal euglycemic limits (<= 120 mg/dL)."
        fbs_driver = False
        fbs_impact = "Normal Glycemic Status"

    parameters.append({
        "feature": "fbs",
        "name": "Fasting Blood Sugar (fbs)",
        "value": "> 120 mg/dL (Elevated)" if fbs_val == 1 else "<= 120 mg/dL (Normal)",
        "raw_value": fbs_val,
        "normal_range": "<= 120 mg/dL (Euglycemia)",
        "status": fbs_status,
        "status_level": fbs_level,
        "clinical_finding": fbs_finding,
        "model_weight": f"{fbs_weight * 100:.1f}%",
        "model_weight_num": float(fbs_weight),
        "risk_impact": fbs_impact,
        "is_primary_driver": fbs_driver
    })

    # 7. Resting ECG (restecg)
    restecg_val = int(normalized_data.get('restecg', 0))
    ecg_weight = feature_importance_map.get('restecg', 0.0578)
    ecg_names = {0: "Normal", 1: "ST-T Wave Abnormality", 2: "Left Ventricular Hypertrophy (LVH)"}
    ecg_name = ecg_names.get(restecg_val, f"Code {restecg_val}")
    if restecg_val == 2:
        ecg_status = "High Risk (LVH)"
        ecg_level = "danger"
        ecg_finding = "Left Ventricular Hypertrophy (LVH) indicates chronic pressure overload and hypertensive cardiac remodeling."
        ecg_driver = True
        ecg_impact = "Structural Cardiac Remodeling"
    elif restecg_val == 1:
        ecg_status = "Abnormal Repolarization"
        ecg_level = "warning"
        ecg_finding = "ST-T wave abnormalities (T-wave inversion or ST shift > 0.05 mV) indicate myocardial irritability."
        ecg_driver = True
        ecg_impact = "Repolarization Disturbance"
    else:
        ecg_status = "Normal"
        ecg_level = "success"
        ecg_finding = "Resting electrocardiogram shows normal sinus rhythm without ischemic changes."
        ecg_driver = False
        ecg_impact = "Normal Cardiac Rhythm"

    parameters.append({
        "feature": "restecg",
        "name": "Resting ECG (restecg)",
        "value": f"{ecg_name} ({restecg_val})",
        "raw_value": restecg_val,
        "normal_range": "Normal Sinus (0)",
        "status": ecg_status,
        "status_level": ecg_level,
        "clinical_finding": ecg_finding,
        "model_weight": f"{ecg_weight * 100:.1f}%",
        "model_weight_num": float(ecg_weight),
        "risk_impact": ecg_impact,
        "is_primary_driver": ecg_driver
    })

    # 8. Maximum Heart Rate (thalch / thalach)
    thalch_val = float(normalized_data.get('thalch', normalized_data.get('thalach', 150.0)))
    hr_weight = feature_importance_map.get('thalch', 0.0519)
    age_val = float(normalized_data.get('age', 55.0))
    expected_max = max(100.0, 220.0 - age_val)
    if thalch_val < 120:
        hr_status = "Chronotropic Incompetence"
        hr_level = "danger"
        hr_finding = f"Peak heart rate ({thalch_val:.0f} bpm) is impaired relative to age-predicted maximum ({expected_max:.0f} bpm), signifying reduced coronary reserve."
        hr_driver = True
        hr_impact = "Blunted Exertional Reserve"
    elif thalch_val < 140:
        hr_status = "Suboptimal Reserve"
        hr_level = "warning"
        hr_finding = f"Achieved heart rate ({thalch_val:.0f} bpm) reflects reduced cardiovascular reserve during peak stress."
        hr_driver = False
        hr_impact = "Moderate Exertional Capacity"
    else:
        hr_status = "Good / Normal"
        hr_level = "success"
        hr_finding = f"Appropriate chronotropic response ({thalch_val:.0f} bpm) achieved under physical exertion."
        hr_driver = False
        hr_impact = "Adequate Cardiac Reserve"

    parameters.append({
        "feature": "thalch",
        "name": "Max Heart Rate (thalch)",
        "value": f"{thalch_val:.0f} bpm",
        "raw_value": thalch_val,
        "normal_range": f"> 140 bpm (Target ~{expected_max:.0f})",
        "status": hr_status,
        "status_level": hr_level,
        "clinical_finding": hr_finding,
        "model_weight": f"{hr_weight * 100:.1f}%",
        "model_weight_num": float(hr_weight),
        "risk_impact": hr_impact,
        "is_primary_driver": hr_driver
    })

    # 9. Slope of Peak ST (slope)
    slope_val = int(normalized_data.get('slope', 1))
    slope_weight = feature_importance_map.get('slope', 0.0577)
    slope_names = {1: "Upsloping", 2: "Flat", 3: "Downsloping"}
    slope_name = slope_names.get(slope_val, f"Code {slope_val}")
    if slope_val == 3:
        slope_status = "High Risk (Downsloping)"
        slope_level = "danger"
        slope_finding = "Downsloping ST segment post-exercise is highly specific for severe multi-vessel CAD and myocardial ischemia."
        slope_driver = True
        slope_impact = "Severe Ischemic Indicator"
    elif slope_val == 2:
        slope_status = "Moderate Risk (Flat)"
        slope_level = "warning"
        slope_finding = "Flat (horizontal) ST segment indicates significant myocardial hypoperfusion under exertional stress."
        slope_driver = True
        slope_impact = "Ischemic ST Response"
    else:
        slope_status = "Normal (Upsloping)"
        slope_level = "success"
        slope_finding = "Upsloping ST segment is a physiological response with low risk of obstructive coronary stenosis."
        slope_driver = False
        slope_impact = "Physiological Response"

    parameters.append({
        "feature": "slope",
        "name": "Slope of Peak ST (slope)",
        "value": f"{slope_name} ({slope_val})",
        "raw_value": slope_val,
        "normal_range": "Upsloping (1)",
        "status": slope_status,
        "status_level": slope_level,
        "clinical_finding": slope_finding,
        "model_weight": f"{slope_weight * 100:.1f}%",
        "model_weight_num": float(slope_weight),
        "risk_impact": slope_impact,
        "is_primary_driver": slope_driver
    })

    # 10. Age (age)
    age_weight = feature_importance_map.get('age', 0.0635)
    if age_val >= 65:
        age_status = "High Demographic Risk (>= 65)"
        age_level = "warning"
        age_finding = f"Patient age ({age_val:.0f} yrs) falls into high cumulative atherosclerotic risk category (>= 65 years)."
        age_driver = True
        age_impact = "Advanced Age Risk"
    elif age_val >= 50:
        age_status = "Moderate Age Risk (50-64)"
        age_level = "info"
        age_finding = f"Age ({age_val:.0f} yrs) is in the intermediate demographic risk group."
        age_driver = False
        age_impact = "Moderate Baseline Risk"
    else:
        age_status = "Low Demographic Risk (< 50)"
        age_level = "success"
        age_finding = f"Younger patient age ({age_val:.0f} yrs) associated with lower baseline vascular stiffening."
        age_driver = False
        age_impact = "Low Baseline Risk"

    parameters.append({
        "feature": "age",
        "name": "Age",
        "value": f"{age_val:.0f} years",
        "raw_value": age_val,
        "normal_range": "< 50 years (Lower Risk)",
        "status": age_status,
        "status_level": age_level,
        "clinical_finding": age_finding,
        "model_weight": f"{age_weight * 100:.1f}%",
        "model_weight_num": float(age_weight),
        "risk_impact": age_impact,
        "is_primary_driver": age_driver
    })

    # 11. Biological Sex (sex)
    sex_val = int(normalized_data.get('sex', 1))
    sex_weight = feature_importance_map.get('sex', 0.1593)
    if sex_val == 1:
        sex_status = "Elevated Demographic Risk"
        sex_level = "info"
        sex_finding = "Male biological sex carries higher epidemiological incidence for early coronary artery disease."
        sex_driver = False
        sex_impact = "Male Risk Profile (Model Weight: 15.9%)"
    else:
        sex_status = "Lower Demographic Risk"
        sex_level = "success"
        sex_finding = "Female biological sex is associated with premenopausal cardioprotective vascular effects."
        sex_driver = False
        sex_impact = "Female Gender Profile"

    parameters.append({
        "feature": "sex",
        "name": "Biological Sex",
        "value": "Male (1)" if sex_val == 1 else "Female (0)",
        "raw_value": sex_val,
        "normal_range": "Demographic Baseline",
        "status": sex_status,
        "status_level": sex_level,
        "clinical_finding": sex_finding,
        "model_weight": f"{sex_weight * 100:.1f}%",
        "model_weight_num": float(sex_weight),
        "risk_impact": sex_impact,
        "is_primary_driver": sex_driver
    })

    # Sort parameters by model weight descending
    parameters.sort(key=lambda x: x["model_weight_num"], reverse=True)

    # Primary Categorization Drivers
    if prediction >= 1:
        drivers = [p for p in parameters if p["status_level"] in ["danger", "warning"]]
        if not drivers:
            drivers = [p for p in parameters if p["is_primary_driver"]]
        if not drivers:
            drivers = parameters[:3]
    else:
        drivers = [p for p in parameters if p["status_level"] in ["success", "info"]]

    primary_drivers_list = [
        f"{d['name']}: {d['value']} - {d['status']} ({d['risk_impact']})"
        for d in drivers[:5]
    ]

    feature_importance_ranking = [
        {
            "feature": p["feature"],
            "name": p["name"],
            "importance": p["model_weight_num"],
            "importance_pct": p["model_weight"]
        }
        for p in parameters
    ]

    return {
        "parameters": parameters,
        "primary_drivers": primary_drivers_list,
        "feature_importance_ranking": feature_importance_ranking
    }


# ============================================================
# PREDICTION FUNCTIONS
# ============================================================

def predict_heart_disease(patient_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Predict heart disease severity for a patient and analyze categorization parameters.
    """
    logger.info("Loading model artifacts...")

    # Load model artifacts
    artifacts = load_model_artifacts()
    model = artifacts['model']
    scaler = artifacts['scaler']
    feature_columns = artifacts['feature_columns']

    # Normalize input
    if isinstance(patient_data, dict):
        normalized_data = normalize_patient_features(patient_data)
        df = pd.DataFrame([normalized_data])
    else:
        df = patient_data.copy()
        normalized_data = dict(patient_data)

    # Expected features for scaler
    scaler_features = list(scaler.feature_names_in_)

    # Fill any missing feature with scaler mean or 0
    for feat in scaler_features:
        if feat not in df.columns or df[feat].isnull().any():
            idx = scaler_features.index(feat)
            df[feat] = scaler.mean_[idx] if hasattr(scaler, 'mean_') else 0.0

    df_for_scaling = df[scaler_features]

    # Scale features
    X_scaled = scaler.transform(df_for_scaling)
    df_scaled = pd.DataFrame(X_scaled, columns=scaler_features)
    X_for_model = df_scaled[feature_columns]

    # Make prediction
    prediction = int(model.predict(X_for_model)[0])
    prediction_proba = model.predict_proba(X_for_model)[0]
    severity_label = get_severity_label(prediction)

    # Analyze parameters on which the patient is categorized
    parameter_analysis = analyze_patient_parameters(normalized_data, prediction, severity_label)

    # Prepare result
    result = {
        'prediction': prediction,
        'severity': severity_label,
        'risk_description': get_risk_description(prediction),
        'confidence': float(prediction_proba[prediction]),
        'probabilities': {
            CLASS_LABELS.get(i, f"Class {i}"): float(prob)
            for i, prob in enumerate(prediction_proba)
        },
        'parameter_breakdown': parameter_analysis['parameters'],
        'primary_categorization_drivers': parameter_analysis['primary_drivers'],
        'feature_importance_ranking': parameter_analysis['feature_importance_ranking']
    }

    logger.info(f"Prediction: {result['severity']} (confidence: {result['confidence']:.2%})")
    return result


def batch_predict(data_file_path: str) -> pd.DataFrame:
    """
    Make predictions for multiple patients from a CSV file.
    """
    logger.info(f"Loading data from: {data_file_path}")
    df = pd.read_csv(data_file_path)

    artifacts = load_model_artifacts()
    model = artifacts['model']
    scaler = artifacts['scaler']
    feature_columns = artifacts['feature_columns']

    scaler_features = list(scaler.feature_names_in_)
    for feat in scaler_features:
        if feat not in df.columns:
            df[feat] = 0.0

    X_scaled = scaler.transform(df[scaler_features])
    df_scaled = pd.DataFrame(X_scaled, columns=scaler_features)
    X_for_model = df_scaled[feature_columns]

    predictions = model.predict(X_for_model)
    probabilities = model.predict_proba(X_for_model)

    df_results = df.copy()
    df_results['prediction'] = predictions
    df_results['severity'] = [get_severity_label(int(p)) for p in predictions]
    df_results['confidence'] = [float(probabilities[i, int(pred)]) for i, pred in enumerate(predictions)]

    return df_results


def example_prediction():
    """Example prediction runner"""
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
        'slope': 3
    }
    result = predict_heart_disease(patient)
    print("Example Prediction Result:", result)


if __name__ == "__main__":
    example_prediction()
