import pandas as pd
import json
import random

# ============================================================
# LOAD DATASET
# ============================================================

df = pd.read_csv("data/heart_disease.csv")

print("Dataset Loaded")

# ============================================================
# SAMPLE FIRST NAMES
# ============================================================

first_names = [
    "Aarav", "Vivaan", "Aditya", "Arjun", "Vihaan",
    "Sai", "Reyansh", "Krishna", "Ishaan", "Atharv",
    "Rohan", "Rahul", "Priya", "Ananya", "Sneha",
    "Kavya", "Pooja", "Neha", "Aisha", "Meera"
]

last_names = [
    "Sharma", "Patil", "Mehta", "Reddy", "Gupta",
    "Joshi", "Kapoor", "Verma", "Iyer", "Nair",
    "Kulkarni", "Desai", "Shah", "Pimprikar"
]

# ============================================================
# CONVERT TO NARRATIVES
# ============================================================

documents = []

for idx, row in df.iterrows():

    patient_id = f"P{1000 + idx}"

    patient_name = (
        random.choice(first_names)
        + " " +
        random.choice(last_names)
    )

    text = (
        f"Patient ID {patient_id}, "
        f"Name {patient_name}, "
        f"aged {row['age']} years with sex value {row['sex']}, "
        f"chest pain type {row['cp']}, "
        f"resting blood pressure {row['trestbps']} mmHg, "
        f"cholesterol level {row['chol']} mg/dL, "
        f"fasting blood sugar {row['fbs']}, "
        f"rest ECG result {row['restecg']}, "
        f"maximum heart rate {row['thalch']}, "
        f"exercise induced angina {row['exang']}, "
        f"oldpeak value {row['oldpeak']}, "
        f"slope value {row['slope']}, "
        f"ca value {row['ca']}, "
        f"thal value {row['thal']}, "
        f"and diagnosis outcome {row['num']}."
    )

    documents.append({
        "id": patient_id,
        "text": text
    })

# ============================================================
# SAVE JSON
# ============================================================

with open(
    "data/heart_narratives.json",
    "w"
) as f:

    json.dump(documents, f, indent=2)

print("Narratives Generated Successfully")