"""
Structured Patient Data Query Engine
===================================
Provides deterministic, programmatic query execution over the structured
cardiovascular patient dataset (920 records in heart_disease.csv).

Features:
1. Dynamic Schema Discovery & Column Synonym Mapping (All 16 dataset parameters)
2. Comprehensive Natural-Language Comparison Operators (>, >=, <, <=, =, between, categorical)
3. Deterministic Aggregations (Count, Average/Mean, Min, Max, Median, Sum, Percentage, Top-N Ranking, Comparison, Grouping)
4. Specific Patient Lookups (Exact ID matching, Single-Parameter isolation, All Findings, Non-existent handling)
5. Informative Error Handling with Debug Metadata Output
6. Strict Medical RAG Separation (Zero interference with general cardiology Q&A)

Author: RAGChainMed
"""

import os
import re
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
import pandas as pd
import numpy as np
from groq import Groq


# Resolve dataset path
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
DATA_DIR = BASE_DIR / "data"
CSV_PATH = DATA_DIR / "heart_disease.csv"
if not CSV_PATH.exists():
    CSV_PATH = DATA_DIR / "raw" / "heart_disease.csv"


class StructuredPatientDataEngine:
    """
    High-precision deterministic query engine for structured patient records.
    Dynamically maps all dataset columns, evaluates comparison filters, multi-conditions, and aggregations.
    """

    def __init__(self, csv_path: Optional[Path] = None):
        self.csv_path = csv_path or CSV_PATH
        self.df = self._load_and_preprocess_dataset()

        # Initialize Groq client if API key is present
        groq_key = os.getenv("GROQ_API_KEY")
        self.groq_client = Groq(api_key=groq_key) if groq_key else None

        # Build comprehensive column synonyms mapping for ALL 16 dataset columns
        self.column_synonyms: Dict[str, List[str]] = {
            "exang": [
                "exercise induced angina", "exercise-induced angina", "angina on exercise",
                "angina during exercise", "exercise angina", "induced angina", "angina with exercise", "exang"
            ],
            "oldpeak": [
                "st-segment depression", "st depression value", "exercise st depression",
                "st depression", "st-depression", "st segment", "oldpeak"
            ],
            "thalch": [
                "maximum heart rate achieved", "maximum heart rate", "max heart rate",
                "peak heart rate", "heart rate", "max hr", "thalach", "thalch", "max pulse", "pulse", "bpm"
            ],
            "trestbps": [
                "resting blood pressure", "systolic blood pressure", "blood pressure",
                "resting bp", "systolic bp", "trestbps", "hypertension", "systolic", "bp"
            ],
            "chol": [
                "serum cholesterol", "cholesterol level", "total cholesterol",
                "hypercholesterolemia", "serum chol", "cholesterol", "chol", "lipid", "lipids"
            ],
            "fbs": [
                "fasting blood sugar", "blood sugar level", "fasting sugar", "fasting glucose",
                "blood sugar", "glucose level", "glucose", "sugar level", "sugar",
                "diabetic", "diabetes", "fbs"
            ],
            "restecg": [
                "resting electrocardiographic", "resting ecg", "resting ekg",
                "electrocardiogram", "electrocardiographic", "rest ecg result", "rest ecg", "rest ekg",
                "abnormal ecg", "abnormal ekg", "normal ecg", "normal ekg",
                "lv hypertrophy", "st-t abnormality", "ecg result", "ekg result",
                "restecg", "ecg", "ekg"
            ],
            "cp": [
                "chest pain type", "chest pain", "angina type", "chest discomfort",
                "pain type", "typical angina", "atypical angina", "non-anginal pain", "non-anginal",
                "asymptomatic", "cp"
            ],
            "age": [
                "patient age", "years of age", "years old", "older than", "younger than", "aged", "age", "years"
            ],
            "sex": [
                "gender", "sex", "males", "females", "male", "female", "men", "women"
            ],
            "slope": [
                "st segment slope", "peak exercise st segment", "st slope", "slope value",
                "upsloping", "downsloping", "slope", "flat"
            ],
            "ca": [
                "number of major vessels", "major vessels colored", "major vessels",
                "vessels colored", "major vessel", "number of vessels", "fluoroscopy", "vessels", "ca"
            ],
            "thal": [
                "thallium stress test", "thallium stress", "thallium defect",
                "thalassemia", "thallium", "reversable defect", "fixed defect", "thal", "defect"
            ],
            "num": [
                "coronary artery disease", "heart disease severity", "heart disease diagnosis",
                "heart disease", "disease severity", "diagnosis outcome", "heart condition",
                "cad diagnosis", "cad outcome", "diagnosis", "target", "outcome", "num", "cad"
            ],
            "dataset": [
                "dataset origin", "dataset", "location", "hospital", "origin", "center",
                "cleveland", "hungary", "switzerland", "va long beach"
            ],
            "id": [
                "patient id", "record id", "case id", "patient number", "patient", "id"
            ]
        }

        # Build reverse lookup list sorted by synonym length descending
        self._sorted_synonym_tuples: List[Tuple[str, str]] = []
        for col, syns in self.column_synonyms.items():
            for syn in sorted(syns, key=lambda s: len(s), reverse=True):
                self._sorted_synonym_tuples.append((syn.lower(), col))
        self._sorted_synonym_tuples.sort(key=lambda t: len(t[0]), reverse=True)

    def _load_and_preprocess_dataset(self) -> pd.DataFrame:
        """Load and normalize heart_disease.csv with all 16 clinical parameters."""
        if not self.csv_path.exists():
            print(f"Warning: Patient CSV not found at {self.csv_path}")
            return pd.DataFrame()

        df = pd.read_csv(self.csv_path)

        # Standardize ID columns
        if "id" not in df.columns:
            df["id"] = list(range(1, len(df) + 1))
        else:
            df["id"] = pd.to_numeric(df["id"], errors="coerce").fillna(pd.Series(range(1, len(df) + 1))).astype(int)

        # Create narrative formatted Patient IDs: P1000, P1001... and aliases P1, P2...
        df["patient_id"] = df["id"].apply(lambda x: f"P{1000 + int(x) - 1}")
        df["patient_alias"] = df["id"].apply(lambda x: f"P{int(x)}")

        # Clean numerical columns
        num_cols = ["age", "trestbps", "chol", "thalch", "oldpeak", "ca", "num"]
        for col in num_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Standardize boolean columns
        if "fbs" in df.columns:
            df["fbs_bool"] = df["fbs"].apply(
                lambda x: True if str(x).strip().upper() in ["TRUE", "1", "1.0", "T"]
                else (False if str(x).strip().upper() in ["FALSE", "0", "0.0", "F"] else np.nan)
            )

        if "exang" in df.columns:
            df["exang_bool"] = df["exang"].apply(
                lambda x: True if str(x).strip().upper() in ["TRUE", "1", "1.0", "T"]
                else (False if str(x).strip().upper() in ["FALSE", "0", "0.0", "F"] else np.nan)
            )

        # Has heart disease indicator (num > 0)
        df["has_heart_disease"] = df["num"].apply(
            lambda x: True if (pd.notna(x) and x > 0) else (False if (pd.notna(x) and x == 0) else np.nan)
        )

        return df

    def match_column(self, term: str) -> Optional[str]:
        """
        Match a natural-language term or phrase to a canonical dataset column name.
        Uses greedy longest-synonym matching.
        """
        t = term.lower().strip()
        # Direct exact match
        if t in self.column_synonyms:
            return t

        for syn, col in self._sorted_synonym_tuples:
            if syn == t:
                return col
            # Word-bounded substring match
            pattern = r"(?:\b|^)" + re.escape(syn) + r"(?:\b|$)"
            if re.search(pattern, t):
                return col

        return None

    def get_column_display_name(self, col: str) -> str:
        """Get user-friendly display name for column."""
        display_names = {
            "age": "Age",
            "sex": "Sex / Gender",
            "dataset": "Dataset Center",
            "cp": "Chest Pain Type",
            "trestbps": "Resting Blood Pressure",
            "chol": "Serum Cholesterol",
            "fbs": "Fasting Blood Sugar",
            "restecg": "Resting ECG",
            "thalch": "Maximum Heart Rate",
            "exang": "Exercise Induced Angina",
            "oldpeak": "ST Depression (oldpeak)",
            "slope": "ST Slope",
            "ca": "Major Vessels (ca)",
            "thal": "Thallium Stress Test",
            "num": "Heart Disease Diagnosis",
            "id": "Patient ID"
        }
        return display_names.get(col, col.replace("_", " ").title())

    def get_column_unit(self, col: str) -> str:
        """Get measurement unit for column."""
        units = {
            "age": "years",
            "trestbps": "mmHg",
            "chol": "mg/dL",
            "thalch": "bpm",
            "oldpeak": "mm",
            "ca": "vessels"
        }
        return units.get(col, "")

    # ============================================================
    # 1. INTENT DETECTION & SEPARATION
    # ============================================================

    def is_general_medical_rag_query(self, query: str) -> bool:
        """
        Identify whether a query is asking for general clinical knowledge
        (e.g., 'What are the symptoms of heart disease?', 'What causes high cholesterol?',
        'Explain ST depression', 'What is angina?', 'Why is high cholesterol a risk factor?')
        which must be routed to FAISS RAG, not dataset analytics.
        """
        q = query.lower().strip()

        # Indicators of general medical concept explanations
        general_indicators = [
            "symptom", "symptoms", "cause", "causes", "what causes", "what is the cause",
            "explain", "mechanism", "pathophysiology", "risk factor", "risk factors",
            "how to treat", "treatment options", "guidelines", "prevention", "what does",
            "definition of", "how does", "what is st depression", "what is angina",
            "what is cardiovascular", "what is coronary", "why is"
        ]

        # Dataset analytical intent indicators
        dataset_indicators = [
            "how many", "count of", "number of patients", "percentage of", "average",
            "mean", "median", "maximum", "minimum", "highest", "lowest", "total patients",
            "compare", "patient p", "patient #", "show patient", "top 10", "top 5",
            "which patients", "who are the patients", "list patients"
        ]

        has_general = any(ind in q for ind in general_indicators)
        has_dataset = any(ind in q for ind in dataset_indicators)

        # If query asks for symptoms, causes, or medical explanations and NOT dataset counts/aggregations
        if has_general and not has_dataset:
            return True

        return False

    def detect_specific_patient_intent(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Check if query is asking for a specific individual patient record or parameter.
        Examples: 'Show patient P1651', 'What is P1651's cholesterol?', 'What is P1651's blood pressure?'
        """
        q = query.strip()
        q_lower = q.lower()

        # Cohort aggregation queries should NOT be treated as a single patient lookup
        cohort_keywords = ["how many", "count of", "number of patients", "percentage of", "average age", "median age", "compare", "which patients"]
        if any(w in q_lower for w in cohort_keywords):
            if not any(w in q_lower for w in ["for patient", "of patient", "patient's"]):
                return None

        # Look for explicit Patient ID patterns (e.g. "P1651", "P1005", "P125", "patient 125")
        pid_match = re.search(r"\b(p\d{1,5})\b", q_lower)
        if not pid_match:
            pid_match = re.search(r"\bpatient(?:'s)?\s*(?:id|#)?\s*(\d{1,5})\b", q_lower)

        if not pid_match:
            return None

        raw_id_str = pid_match.group(1).upper()

        # Check if asking for a specific single parameter
        specific_param = None
        # Check specific parameters in strict priority order (exang before cp, oldpeak before restecg)
        param_candidates = [
            "exang", "oldpeak", "thalch", "chol", "trestbps", "fbs",
            "restecg", "cp", "slope", "ca", "thal", "num", "age", "sex"
        ]
        for col in param_candidates:
            syns = self.column_synonyms.get(col, [])
            if any(syn in q_lower for syn in syns):
                if col == "age" and ("average age" in q_lower or "median age" in q_lower):
                    continue
                specific_param = col
                break

        # Check if the query asks for all parameters or findings
        if any(w in q_lower for w in ["all findings", "all parameters", "findings", "details", "parameters", "show patient", "profile", "record", "give me"]):
            if "what is" not in q_lower or not specific_param:
                specific_param = None

        return {
            "intent": "SPECIFIC_PATIENT_LOOKUP",
            "patient_id": raw_id_str,
            "parameter": specific_param
        }

    def detect_structured_query_intent(self, query: str) -> Optional[str]:
        """
        Classify query into structured analytical intent or None (Medical RAG / Off-topic).
        """
        q = query.lower().strip()

        # 1. Guardrail against general medical knowledge queries
        if self.is_general_medical_rag_query(query):
            return None

        # 2. Specific Patient Check
        if self.detect_specific_patient_intent(query):
            return "SPECIFIC_PATIENT_LOOKUP"

        # 3. Total Count Queries
        if re.search(r"\b(how many patients are there in total|total number of patients|how many patients are in the dataset|how many patient records|total patient count|total patients|count of all patients|how many total patients|how many patients are there|how many patients exist)\b", q):
            return "STRUCTURED_TOTAL_COUNT"

        # 4. Sorting / Ranking (Top N / Bottom N)
        if re.search(r"\b(top \d+|show \d+|show the \d+|give me \d+|first \d+|\d+ patients with the (?:highest|lowest|max|min))\b", q):
            return "STRUCTURED_RANKING"

        # 5. Percentage Queries
        if re.search(r"\b(what percentage|percentage of|% of|percent of)\b", q):
            return "STRUCTURED_PERCENTAGE"

        # 6. Comparison Queries between Cohorts
        if re.search(r"\b(compare|comparison|difference between|higher .* than|higher .* between)\b", q):
            return "STRUCTURED_COMPARISON"

        # 7. Average / Mean Queries ("What is the average cholesterol?", "What is the average age?", "What is the average maximum heart rate?")
        if re.search(r"\b(average|mean|avg)\b", q):
            return "STRUCTURED_AVERAGE"

        # 8. Median Queries
        if re.search(r"\b(median)\b", q):
            return "STRUCTURED_MEDIAN"

        # 9. List / Filter Queries ("Which patients have...", "List patients with...", "Show patients with...")
        if re.search(r"\b(which patients|who are the patients|list patients|find patients|show patients|patients who have|patients with|patients having|patients where)\b", q):
            return "STRUCTURED_FILTER"

        # 10. Conditional Count Queries ("How many patients have...", "How many patients are older than...", "How many males have...")
        if re.search(r"\b(how many patients|count of patients|number of patients|how many records|how many people|how many cases|how many males|how many females|how many men|how many women)\b", q) or (
            "how many" in q and any(w in q for w in ["patient", "patients", "cases", "people", "records", "males", "females", "men", "women", "asymptomatic", "typical", "atypical"])
        ):
            return "STRUCTURED_COUNT"

        # 11. Min / Max Queries (Asking explicitly for minimum or maximum value of a parameter)
        if re.search(r"\b(what is the highest|what is the maximum|what is the max|what is the lowest|what is the minimum|what is the min|highest .* in the dataset|maximum .* in the dataset|lowest .* in the dataset|minimum .* in the dataset)\b", q):
            return "STRUCTURED_MIN_MAX"

        # 12. Sum Queries
        if re.search(r"\b(sum of|total sum|sum)\b", q) and any(w in q for w in ["patient", "cholesterol", "age", "pressure", "records"]):
            return "STRUCTURED_SUM"

        # 13. Grouping / Distribution Queries
        if re.search(r"\b(for each|in each|grouped by|distribution of|breakdown by|each age group|each category|each heart disease)\b", q):
            return "STRUCTURED_GROUPING"

        # 14. Fallback for comparison symbols without "how many"
        if any(sym in q for sym in [">", "<", ">=", "<=", "=", "between", "above", "below", "older than", "younger than"]):
            return "STRUCTURED_COUNT"

        return None

    # ============================================================
    # 2. SPECIFIC PATIENT LOOKUP HANDLER
    # ============================================================

    def handle_specific_patient_lookup(self, patient_id_query: str, requested_param: Optional[str] = None) -> Dict[str, Any]:
        """
        Retrieve exact patient record directly from DataFrame.
        Guarantees zero hallucination and strict parameter isolation.
        """
        pid_clean = patient_id_query.strip().upper()
        numeric_part = re.sub(r"[^0-9]", "", pid_clean)

        matched_row = None

        # Strategy 1: Match narrative P-ID (e.g. P1651, P1005, P1124)
        if pid_clean.startswith("P") and pid_clean in self.df["patient_id"].values:
            matched_row = self.df[self.df["patient_id"] == pid_clean].iloc[0]

        # Strategy 2: Match exact integer CSV id (e.g. 125, 652)
        elif numeric_part and int(numeric_part) in self.df["id"].values:
            matched_row = self.df[self.df["id"] == int(numeric_part)].iloc[0]

        # Strategy 3: Match narrative P-ID format with numeric offset P(1000 + id - 1)
        elif numeric_part and int(numeric_part) >= 1000:
            target_p = f"P{numeric_part}"
            if target_p in self.df["patient_id"].values:
                matched_row = self.df[self.df["patient_id"] == target_p].iloc[0]

        # Strategy 4: Match alias P1, P2...
        elif pid_clean in self.df["patient_alias"].values:
            matched_row = self.df[self.df["patient_alias"] == pid_clean].iloc[0]

        if matched_row is None:
            return {
                "success": True,
                "answer": f"Patient {patient_id_query} was not found in the dataset.",
                "retrieved_evidence": [],
                "has_relevant_evidence": True,
                "query_type": "specific_patient_not_found",
                "patient_id": patient_id_query
            }

        # Extract details
        age = matched_row["age"]
        sex = matched_row["sex"]
        cp = matched_row["cp"]
        trestbps = matched_row["trestbps"]
        chol = matched_row["chol"]
        fbs = matched_row.get("fbs", "N/A")
        restecg = matched_row["restecg"]
        thalch = matched_row["thalch"]
        exang = matched_row["exang"]
        oldpeak = matched_row["oldpeak"]
        slope = matched_row["slope"]
        ca = matched_row["ca"]
        thal = matched_row["thal"]
        num = int(matched_row["num"]) if pd.notna(matched_row["num"]) else "N/A"

        diagnosis_labels = {
            0: "No CAD / Healthy (Outcome 0)",
            1: "Mild CAD (Class 1)",
            2: "Moderate CAD (Class 2)",
            3: "Severe CAD (Class 3)",
            4: "Very Severe CAD (Class 4)"
        }
        diag_str = diagnosis_labels.get(num, f"Class {num}") if isinstance(num, int) else str(num)
        has_hd = "Yes" if (isinstance(num, int) and num > 0) else ("No" if num == 0 else "Unknown")

        fbs_display = "Elevated (>120 mg/dL)" if fbs in [True, "TRUE", "True", 1, 1.0] else "Normal (<=120 mg/dL)"
        exang_display = "Yes (Present)" if exang in [True, "TRUE", "True", 1, 1.0] else "No (Absent)"

        # Case A: Single Parameter Requested
        if requested_param:
            param_col = self.match_column(requested_param) or requested_param
            if param_col in self.df.columns or param_col in ["fbs", "exang", "num", "has_heart_disease"]:
                val = matched_row.get(param_col)
                unit = self.get_column_unit(param_col)
                display_col = self.get_column_display_name(param_col)

                if param_col == "fbs":
                    val_str = fbs_display
                    sentence_ans = f"Patient {patient_id_query} has a fasting blood sugar level of {val_str}."
                elif param_col == "exang":
                    val_str = exang_display
                    sentence_ans = f"Patient {patient_id_query} exercise induced angina: {val_str}."
                elif param_col in ["num", "has_heart_disease"]:
                    val_str = f"{has_hd} ({diag_str})"
                    sentence_ans = f"Patient {patient_id_query} diagnosis: {val_str}."
                elif param_col == "cp":
                    val_str = str(val).title() if pd.notna(val) else "N/A"
                    sentence_ans = f"Patient {patient_id_query} has a chest pain type of {val_str}."
                elif param_col == "restecg":
                    val_str = str(val).title() if pd.notna(val) else "N/A"
                    sentence_ans = f"Patient {patient_id_query} has a resting ECG result of {val_str}."
                elif param_col == "slope":
                    val_str = str(val).title() if pd.notna(val) else "N/A"
                    sentence_ans = f"Patient {patient_id_query} has an ST slope of {val_str}."
                elif param_col == "thal":
                    val_str = str(val).title() if pd.notna(val) else "N/A"
                    sentence_ans = f"Patient {patient_id_query} thallium stress test: {val_str}."
                elif param_col == "ca":
                    val_str = f"{val} {unit}" if pd.notna(val) else "N/A"
                    sentence_ans = f"Patient {patient_id_query} has {val_str} colored by fluoroscopy."
                elif unit:
                    val_str = f"{val} {unit}" if pd.notna(val) else "N/A"
                    sentence_ans = f"Patient {patient_id_query} has a {display_col.lower()} of {val_str}."
                else:
                    val_str = f"{val}" if pd.notna(val) else "N/A"
                    sentence_ans = f"Patient {patient_id_query} {display_col.lower()}: {val_str}."

                answer_text = (
                    f"{sentence_ans}\n\n"
                    f"Patient ID: {patient_id_query}\n"
                    f"{display_col}: {val_str}"
                )
                return {
                    "success": True,
                    "answer": answer_text,
                    "retrieved_evidence": [],
                    "has_relevant_evidence": True,
                    "query_type": "specific_patient_parameter",
                    "patient_id": patient_id_query,
                    "parameter": display_col,
                    "value": val_str
                }

        # Case B: All Findings Requested
        answer_text = (
            f"Patient ID: {patient_id_query}\n\n"
            f"Findings:\n"
            f"Age: {age} years\n"
            f"Sex: {sex}\n"
            f"Chest Pain: {cp}\n"
            f"Blood Pressure: {trestbps} mmHg\n"
            f"Cholesterol: {chol} mg/dL\n"
            f"Glucose: {fbs_display}\n"
            f"Resting ECG: {restecg}\n"
            f"Heart Rate: {thalch} bpm\n"
            f"Exercise Induced Angina: {exang_display}\n"
            f"ST Depression: {oldpeak} mm\n"
            f"Slope: {slope}\n"
            f"Major Vessels: {ca}\n"
            f"Thallium Stress: {thal}\n"
            f"Diagnosis: {diag_str}"
        )

        return {
            "success": True,
            "answer": answer_text,
            "retrieved_evidence": [],
            "has_relevant_evidence": True,
            "query_type": "specific_patient_all_findings",
            "patient_id": patient_id_query
        }

    # ============================================================
    # 3. DYNAMIC OPERATOR & CONDITION PARSER
    # ============================================================

    def _parse_comparison_clause(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Parse a single condition clause into {column, operator, value, is_categorical}.
        Supports all natural language comparators and categorical column states.
        """
        t = text.strip()
        t_lower = t.lower()

        # 1. Range Check: 'between X and Y' / 'from X to Y'
        range_match = re.search(r"(\bbetween\b|\bfrom\b)\s*([0-9]+(?:\.[0-9]+)?)\s*(\band\b|\bto\b)\s*([0-9]+(?:\.[0-9]+)?)", t_lower)
        if range_match:
            low_val = float(range_match.group(2))
            high_val = float(range_match.group(4))
            col = self.match_column(t_lower) or "age"
            return {
                "column": col,
                "operator": "between",
                "value": (low_val, high_val),
                "is_categorical": False
            }

        # 2. Extract Operator and Value with Comprehensive Natural Language Patterns
        operator_patterns = [
            (r"\b(?:greater than or equal to|greater than or equal|at least|no less than|minimum of)\b", ">="),
            (r"\b(?:less than or equal to|less than or equal|at most|no more than|maximum of|up to)\b", "<="),
            (r"\b(?:greater than|more than|higher than|above|over|exceeding|strictly greater than|older than)\b", ">"),
            (r"\b(?:less than|fewer than|lower than|below|under|younger than)\b", "<"),
            (r"\b(?:equal to|equals|equal|exactly|is equal to)\b", "=="),
            (r">=", ">="),
            (r"<=", "<="),
            (r">", ">"),
            (r"<", "<"),
            (r"==", "=="),
            (r"=", "==")
        ]

        found_op = None
        op_match_span = None

        for pattern, op_symbol in operator_patterns:
            m = re.search(pattern, t_lower)
            if m:
                found_op = op_symbol
                op_match_span = m.span()
                break

        # Check for numeric threshold after/around operator
        if found_op and op_match_span:
            text_before = t_lower[:op_match_span[0]]
            text_after = t_lower[op_match_span[1]:]

            # Try finding number in text_after or text_before
            num_match = re.search(r"([0-9]+(?:\.[0-9]+)?)", text_after)
            if not num_match:
                num_match = re.search(r"([0-9]+(?:\.[0-9]+)?)", t_lower)

            if num_match:
                val = float(num_match.group(1))

                # Column resolution
                if "older" in t_lower or "younger" in t_lower or "age" in t_lower:
                    col = "age"
                elif "blood pressure" in t_lower or "bp" in t_lower or "trestbps" in t_lower or "hypertension" in t_lower:
                    col = "trestbps"
                elif "cholesterol" in t_lower or "chol" in t_lower:
                    col = "chol"
                elif "heart rate" in t_lower or "thalach" in t_lower or "thalch" in t_lower or "pulse" in t_lower:
                    col = "thalch"
                elif "oldpeak" in t_lower or "st depression" in t_lower:
                    col = "oldpeak"
                elif "vessel" in t_lower or "ca" in t_lower:
                    col = "ca"
                else:
                    col = self.match_column(text_before) or self.match_column(t_lower)

                if col:
                    if col == "fbs":
                        # Fasting blood sugar in dataset is boolean (> 120 mg/dL indicator)
                        is_elevated = (found_op in [">", ">="] and val >= 100) or (found_op in ["==", "="] and val in [1, 1.0])
                        return {
                            "column": "fbs",
                            "operator": "==",
                            "value": is_elevated,
                            "is_categorical": True
                        }
                    elif col == "exang" and val in [0, 1, 0.0, 1.0]:
                        return {
                            "column": "exang",
                            "operator": "==",
                            "value": bool(int(val)),
                            "is_categorical": True
                        }
                    elif col == "cp" and int(val) in [1, 2, 3, 4]:
                        cp_map = {1: "typical angina", 2: "atypical angina", 3: "non-anginal", 4: "asymptomatic"}
                        return {
                            "column": "cp",
                            "operator": "==",
                            "value": cp_map[int(val)],
                            "is_categorical": True
                        }
                    elif col == "num" and found_op in [">", ">="] and val in [0, 0.0]:
                        return {
                            "column": "has_heart_disease",
                            "operator": "==",
                            "value": True,
                            "is_categorical": True
                        }
                    return {
                        "column": col,
                        "operator": found_op,
                        "value": val,
                        "is_categorical": False
                    }

        # 3. Categorical Values Matching
        # Exercise Induced Angina (Check BEFORE Chest pain so "exercise induced angina" is not captured as CP!)
        if "exang" in t_lower or "exercise induced angina" in t_lower or "exercise angina" in t_lower or "angina with exercise" in t_lower:
            if any(w in t_lower for w in ["0", "false", "no", "absent", "without"]):
                return {"column": "exang", "operator": "==", "value": False, "is_categorical": True}
            else:
                return {"column": "exang", "operator": "==", "value": True, "is_categorical": True}

        # Chest Pain Type: e.g. "typical angina", "atypical angina", "non-anginal pain", "asymptomatic"
        if "typical angina" in t_lower and "atypical" not in t_lower:
            return {"column": "cp", "operator": "==", "value": "typical angina", "is_categorical": True}
        elif "atypical angina" in t_lower or "atypical" in t_lower:
            return {"column": "cp", "operator": "==", "value": "atypical angina", "is_categorical": True}
        elif "non-anginal" in t_lower or "non anginal" in t_lower:
            return {"column": "cp", "operator": "==", "value": "non-anginal", "is_categorical": True}
        elif "asymptomatic" in t_lower:
            return {"column": "cp", "operator": "==", "value": "asymptomatic", "is_categorical": True}

        if "chest pain" in t_lower or "cp" in t_lower:
            cp_num = re.search(r"(?:cp|chest pain|type)\s*([1-4])\b", t_lower)
            if cp_num:
                cp_map = {"1": "typical angina", "2": "atypical angina", "3": "non-anginal", "4": "asymptomatic"}
                return {"column": "cp", "operator": "==", "value": cp_map[cp_num.group(1)], "is_categorical": True}

        # Resting ECG: e.g. "abnormal ecg", "normal ecg", "lv hypertrophy", "st-t abnormality"
        if "ecg" in t_lower or "ekg" in t_lower or "restecg" in t_lower or "electrocardiogram" in t_lower:
            if "abnormal" in t_lower:
                return {"column": "restecg", "operator": "==", "value": "abnormal", "is_categorical": True}
            elif "normal" in t_lower:
                return {"column": "restecg", "operator": "==", "value": "normal", "is_categorical": True}
            elif "lv hypertrophy" in t_lower or "hypertrophy" in t_lower:
                return {"column": "restecg", "operator": "==", "value": "lv hypertrophy", "is_categorical": True}
            elif "st-t abnormality" in t_lower or "st-t" in t_lower or "wave abnormality" in t_lower:
                return {"column": "restecg", "operator": "==", "value": "st-t abnormality", "is_categorical": True}

        # Fasting Blood Sugar: e.g. "fasting blood sugar equal to 1", "glucose > 120", "fbs == 1", "fasting blood sugar above 120"
        if "fbs" in t_lower or "fasting blood sugar" in t_lower or "glucose" in t_lower or "blood sugar" in t_lower or "sugar" in t_lower:
            if any(w in t_lower for w in ["1", "true", "yes", "elevated", "high", "> 120", ">120", "above 120", "over 120"]):
                return {"column": "fbs", "operator": "==", "value": True, "is_categorical": True}
            elif any(w in t_lower for w in ["0", "false", "no", "normal", "<= 120", "<=120", "below 120", "under 120"]):
                return {"column": "fbs", "operator": "==", "value": False, "is_categorical": True}
            else:
                return {"column": "fbs", "operator": "==", "value": True, "is_categorical": True}

        # Sex: e.g. "male", "female", "men", "women", "males", "females"
        if "sex" in t_lower or "gender" in t_lower or "male" in t_lower or "female" in t_lower or "men" in t_lower or "women" in t_lower:
            if "female" in t_lower or "women" in t_lower or "females" in t_lower:
                return {"column": "sex", "operator": "==", "value": "Female", "is_categorical": True}
            elif "male" in t_lower or "men" in t_lower or "males" in t_lower:
                return {"column": "sex", "operator": "==", "value": "Male", "is_categorical": True}

        # Heart Disease / CAD: e.g. "heart disease", "cad", "coronary artery disease"
        if "heart disease" in t_lower or "cad" in t_lower or "coronary artery disease" in t_lower:
            if any(w in t_lower for w in ["no heart disease", "healthy", "without heart disease", "without cad", "no cad", "num 0", "num = 0", "equal to 0"]):
                return {"column": "num", "operator": "==", "value": 0, "is_categorical": True}
            elif any(w in t_lower for w in ["mild", "class 1"]):
                return {"column": "num", "operator": "==", "value": 1, "is_categorical": True}
            elif any(w in t_lower for w in ["moderate", "class 2"]):
                return {"column": "num", "operator": "==", "value": 2, "is_categorical": True}
            elif any(w in t_lower for w in ["severe", "class 3"]):
                return {"column": "num", "operator": "==", "value": 3, "is_categorical": True}
            else:
                return {"column": "has_heart_disease", "operator": "==", "value": True, "is_categorical": True}

        # ST Slope: e.g. "upsloping", "flat", "downsloping"
        if "slope" in t_lower or "upsloping" in t_lower or "flat" in t_lower or "downsloping" in t_lower:
            if "upsloping" in t_lower:
                return {"column": "slope", "operator": "==", "value": "upsloping", "is_categorical": True}
            elif "flat" in t_lower:
                return {"column": "slope", "operator": "==", "value": "flat", "is_categorical": True}
            elif "downsloping" in t_lower:
                return {"column": "slope", "operator": "==", "value": "downsloping", "is_categorical": True}

        # Thallium: e.g. "fixed defect", "reversable defect", "normal defect"
        if "thal" in t_lower or "thallium" in t_lower:
            if "fixed defect" in t_lower or "fixed" in t_lower:
                return {"column": "thal", "operator": "==", "value": "fixed defect", "is_categorical": True}
            elif "reversable" in t_lower or "reversible" in t_lower:
                return {"column": "thal", "operator": "==", "value": "reversable defect", "is_categorical": True}
            elif "normal" in t_lower:
                return {"column": "thal", "operator": "==", "value": "normal", "is_categorical": True}

        return None

    def _apply_filter_clause(self, df_target: pd.DataFrame, clause: Dict[str, Any]) -> Tuple[pd.DataFrame, str]:
        """Apply a parsed filter clause against the DataFrame and return (filtered_df, description_string)."""
        col = clause["column"]
        op = clause["operator"]
        val = clause["value"]
        is_cat = clause.get("is_categorical", False)

        disp = self.get_column_display_name(col)
        unit = self.get_column_unit(col)
        unit_str = f" {unit}" if unit else ""

        if op == "between" and isinstance(val, tuple):
            low, high = val
            mask = (df_target[col] >= low) & (df_target[col] <= high)
            cond_str = f"{disp} Between {low} and {high}{unit_str}"
            return df_target[mask], cond_str

        if is_cat:
            if col == "fbs":
                mask = df_target["fbs_bool"] == val
                cond_str = f"Fasting Blood Sugar = {'Elevated (>120 mg/dL)' if val else 'Normal (<=120 mg/dL)'}"
            elif col == "exang":
                mask = df_target["exang_bool"] == val
                cond_str = f"Exercise Induced Angina = {'Present (1)' if val else 'Absent (0)'}"
            elif col == "has_heart_disease":
                mask = df_target["has_heart_disease"] == val
                cond_str = f"Heart Disease Diagnosis = {'Present (CAD > 0)' if val else 'Absent (Healthy)'}"
            elif col == "cp" and isinstance(val, str):
                mask = df_target["cp"].astype(str).str.lower().str.contains(val.lower())
                cond_str = f"Chest Pain Type = {val.title()}"
            elif col == "restecg" and isinstance(val, str):
                if val == "abnormal":
                    mask = df_target["restecg"].astype(str).str.lower().isin(["lv hypertrophy", "st-t abnormality"])
                    cond_str = f"Resting ECG = Abnormal (LV Hypertrophy or ST-T Abnormality)"
                else:
                    mask = df_target["restecg"].astype(str).str.lower() == val.lower()
                    cond_str = f"Resting ECG = {val.title()}"
            elif col == "slope" and isinstance(val, str):
                mask = df_target["slope"].astype(str).str.lower().str.contains(val.lower())
                cond_str = f"ST Slope = {val.title()}"
            elif col == "thal" and isinstance(val, str):
                mask = df_target["thal"].astype(str).str.lower().str.contains(val.lower())
                cond_str = f"Thallium Stress Test = {val.title()}"
            elif col == "sex" and isinstance(val, str):
                mask = df_target["sex"].astype(str).str.lower() == val.lower()
                cond_str = f"Sex = {val}"
            elif col == "num":
                mask = df_target["num"] == val
                cond_str = f"Diagnosis Category = Class {val}"
            else:
                mask = df_target[col] == val
                cond_str = f"{disp} = {val}"
            return df_target[mask], cond_str

        # Numerical comparisons
        val_float = float(val)
        if op in [">", "above", "over"]:
            mask = df_target[col] > val_float
            cond_str = f"{disp} > {val_float}{unit_str}"
        elif op in [">=", "greater than or equal"]:
            mask = df_target[col] >= val_float
            cond_str = f"{disp} >= {val_float}{unit_str}"
        elif op in ["<", "below", "under"]:
            mask = df_target[col] < val_float
            cond_str = f"{disp} < {val_float}{unit_str}"
        elif op in ["<=", "less than or equal"]:
            mask = df_target[col] <= val_float
            cond_str = f"{disp} <= {val_float}{unit_str}"
        else:
            mask = df_target[col] == val_float
            cond_str = f"{disp} = {val_float}{unit_str}"

        return df_target[mask], cond_str

    # ============================================================
    # 4. STRUCTURED ANALYTICAL QUERIES EXECUTOR
    # ============================================================

    def execute_structured_query(self, query: str) -> Dict[str, Any]:
        """
        Parse and execute analytical queries directly against self.df with exact mathematical precision.
        """
        q = query.strip()
        q_lower = q.lower()

        # 1. Specific Patient Lookup Check
        patient_intent = self.detect_specific_patient_intent(q)
        if patient_intent:
            return self.handle_specific_patient_lookup(
                patient_id_query=patient_intent["patient_id"],
                requested_param=patient_intent.get("parameter")
            )

        # 2. Total Patient Count
        if (
            "total" in q_lower and any(w in q_lower for w in ["patient", "patients", "count", "dataset", "records"])
        ) or re.search(r"how many patients (?:are there )?(?:in total|in the dataset|total)", q_lower) or (
            "how many patient records are there" in q_lower
        ) or (
            "give me the total patient count" in q_lower
        ) or (
            q_lower.strip("?. ") in ["how many patients are there", "how many patients", "how many patient records", "total patients"]
        ):
            total = len(self.df)
            males = len(self.df[self.df["sex"] == "Male"])
            females = len(self.df[self.df["sex"] == "Female"])
            hd_count = len(self.df[self.df["has_heart_disease"] == True])

            answer = (
                f"Deterministic Dataset Analytics\n\n"
                f"Query:\n{query}\n\n"
                f"Operation: TOTAL PATIENT COUNT\n\n"
                f"Result:\n{total} patients\n\n"
                f"Dataset Summary:\n"
                f"• Total Records: {total}\n"
                f"• Male Patients: {males} ({(males/total)*100:.1f}%)\n"
                f"• Female Patients: {females} ({(females/total)*100:.1f}%)\n"
                f"• Patients with CAD / Heart Disease: {hd_count} ({(hd_count/total)*100:.1f}%)"
            )
            return {
                "success": True,
                "answer": answer,
                "retrieved_evidence": [],
                "has_relevant_evidence": True,
                "query_type": "structured_total_count",
                "total": total,
                "debug": {
                    "column": "all",
                    "operator": "count",
                    "value": total,
                    "operation": "total_count"
                }
            }

        # 3. Top-N Ranking / Sorting (e.g. "Show the 10 patients with the highest cholesterol")
        rank_match = re.search(r"(?:top|show|give me|first)\s*(?:the\s*)?(\d+)\s*patients\s*(?:with|by)?\s*(?:the)?\s*(highest|lowest|max|min)?\s*([a-zA-Z\s]+)", q_lower)
        if rank_match:
            n_count = int(rank_match.group(1))
            order_word = (rank_match.group(2) or "highest").lower()
            param_word = rank_match.group(3).strip()

            ascending = order_word in ["lowest", "min"]
            col = self.match_column(param_word)
            if col and col in self.df.columns:
                unit = self.get_column_unit(col)
                disp_name = self.get_column_display_name(col)
                unit_str = f" {unit}" if unit else ""

                subset = self.df.dropna(subset=[col]).sort_values(by=col, ascending=ascending).head(n_count)
                rows = []
                for rank, (_, row) in enumerate(subset.iterrows(), 1):
                    val = row[col]
                    p_id = row["id"]
                    diag = "CAD" if row["has_heart_disease"] else "No CAD"
                    rows.append(f"{rank}. Patient ID {p_id} ({row['patient_id']}): {disp_name} = {val}{unit_str} | Age: {row['age']} | Sex: {row['sex']} | Diagnosis: {diag}")

                answer = (
                    f"Deterministic Dataset Analytics\n\n"
                    f"Query:\n{query}\n\n"
                    f"Parameter: {disp_name}\n"
                    f"Operation: RANKING ({order_word.upper()} {n_count})\n\n"
                    f"Result:\nTop {len(subset)} Patients Ranked by {order_word.title()} {disp_name}:\n\n"
                    + "\n".join(rows)
                )
                return {
                    "success": True,
                    "answer": answer,
                    "retrieved_evidence": [],
                    "has_relevant_evidence": True,
                    "query_type": "structured_ranking",
                    "debug": {
                        "column": col,
                        "operator": "ranking",
                        "value": n_count,
                        "operation": f"top_{n_count}_{order_word}"
                    }
                }

        # 4. Comparison Queries between Cohorts (e.g. "Compare average glucose between patients with and without heart disease")
        if "compare" in q_lower or "difference between" in q_lower or ("higher" in q_lower and "between" in q_lower):
            target_col = None
            if "glucose" in q_lower or "sugar" in q_lower or "fbs" in q_lower:
                target_col = "fbs"
            else:
                for col in ["thalch", "chol", "trestbps", "age", "oldpeak", "ca"]:
                    syns = self.column_synonyms.get(col, [])
                    if any(syn in q_lower for syn in syns):
                        target_col = col
                        break

            if target_col == "fbs":
                hd_group = self.df[self.df["has_heart_disease"] == True]
                non_hd_group = self.df[self.df["has_heart_disease"] == False]

                fbs_hd_high = len(hd_group[hd_group["fbs_bool"] == True])
                fbs_hd_total = len(hd_group.dropna(subset=["fbs_bool"]))
                pct_hd = (fbs_hd_high / fbs_hd_total) * 100 if fbs_hd_total else 0

                fbs_non_high = len(non_hd_group[non_hd_group["fbs_bool"] == True])
                fbs_non_total = len(non_hd_group.dropna(subset=["fbs_bool"]))
                pct_non = (fbs_non_high / fbs_non_total) * 100 if fbs_non_total else 0

                answer = (
                    f"Deterministic Dataset Analytics\n\n"
                    f"Query:\n{query}\n\n"
                    f"Parameter: Fasting Blood Sugar (fbs)\n"
                    f"Operation: COMPARISON\n\n"
                    f"Comparison: Fasting Blood Sugar / Glucose (>120 mg/dL) Prevalence Between Cohorts\n\n"
                    f"1. Patients WITH Heart Disease (CAD, num > 0):\n"
                    f"   - Evaluated Records: {fbs_hd_total}\n"
                    f"   - High Glucose (>120 mg/dL): {fbs_hd_high} patients ({pct_hd:.1f}%)\n\n"
                    f"2. Patients WITHOUT Heart Disease (Healthy, num = 0):\n"
                    f"   - Evaluated Records: {fbs_non_total}\n"
                    f"   - High Glucose (>120 mg/dL): {fbs_non_high} patients ({pct_non:.1f}%)\n\n"
                    f"Summary Difference: High glucose is {pct_hd - pct_non:+.1f}% more prevalent in patients with heart disease."
                )
                return {
                    "success": True,
                    "answer": answer,
                    "retrieved_evidence": [],
                    "has_relevant_evidence": True,
                    "query_type": "structured_comparison",
                    "debug": {
                        "column": "fbs",
                        "operator": "compare",
                        "value": "cohort_diff",
                        "operation": "comparison"
                    }
                }

            elif target_col:
                unit = self.get_column_unit(target_col)
                disp_name = self.get_column_display_name(target_col)
                unit_str = f" {unit}" if unit else ""

                hd_group = self.df[self.df["has_heart_disease"] == True][target_col].dropna()
                non_hd_group = self.df[self.df["has_heart_disease"] == False][target_col].dropna()

                mean_hd = hd_group.mean()
                mean_non_hd = non_hd_group.mean()
                med_hd = hd_group.median()
                med_non_hd = non_hd_group.median()

                diff = mean_hd - mean_non_hd
                diff_str = f"+{diff:.2f}" if diff > 0 else f"{diff:.2f}"

                answer = (
                    f"Deterministic Dataset Analytics\n\n"
                    f"Query:\n{query}\n\n"
                    f"Parameter: {disp_name}\n"
                    f"Operation: COMPARISON\n\n"
                    f"Comparison: {disp_name} between Patients With vs Without Heart Disease\n\n"
                    f"1. Patients WITH Heart Disease (CAD, num > 0):\n"
                    f"   - Records: {len(hd_group)}\n"
                    f"   - Average: {mean_hd:.2f}{unit_str}\n"
                    f"   - Median: {med_hd:.2f}{unit_str}\n\n"
                    f"2. Patients WITHOUT Heart Disease (Healthy, num = 0):\n"
                    f"   - Records: {len(non_hd_group)}\n"
                    f"   - Average: {mean_non_hd:.2f}{unit_str}\n"
                    f"   - Median: {med_non_hd:.2f}{unit_str}\n\n"
                    f"Summary Difference: Patients with heart disease have an average {disp_name} that is {diff_str}{unit_str} compared to patients without heart disease."
                )
                return {
                    "success": True,
                    "answer": answer,
                    "retrieved_evidence": [],
                    "has_relevant_evidence": True,
                    "query_type": "structured_comparison",
                    "debug": {
                        "column": target_col,
                        "operator": "compare",
                        "value": diff,
                        "operation": "comparison"
                    }
                }

        # 5. Average / Mean Queries (e.g. "What is the average cholesterol?", "What is the average age?", "What is the average maximum heart rate?")
        if any(w in q_lower for w in ["average", "mean", "avg"]):
            target_col = None
            if "glucose" in q_lower or "sugar" in q_lower or "fbs" in q_lower:
                fbs_high_count = len(self.df[self.df["fbs_bool"] == True])
                fbs_total = len(self.df.dropna(subset=["fbs_bool"]))
                pct_fbs = (fbs_high_count / fbs_total) * 100 if fbs_total else 0
                return {
                    "success": True,
                    "answer": (
                        f"Deterministic Dataset Analytics\n\n"
                        f"Query:\n{query}\n\n"
                        f"Parameter: Fasting Blood Sugar / Glucose (> 120 mg/dL indicator)\n"
                        f"Operation: AVERAGE / PREVALENCE\n\n"
                        f"Result:\n{pct_fbs:.1f}% elevated glucose prevalence ({fbs_high_count} out of {fbs_total} records)"
                    ),
                    "retrieved_evidence": [],
                    "has_relevant_evidence": True,
                    "query_type": "structured_average",
                    "debug": {"column": "fbs", "operator": "average", "value": pct_fbs, "operation": "average"}
                }

            # Check columns in priority order
            for col in ["thalch", "oldpeak", "trestbps", "chol", "ca", "age", "num"]:
                syns = self.column_synonyms.get(col, [])
                if any(syn in q_lower for syn in syns):
                    target_col = col
                    break

            if target_col and target_col in self.df.columns:
                unit = self.get_column_unit(target_col)
                disp_name = self.get_column_display_name(target_col)
                unit_str = f" {unit}" if unit else ""

                if "heart disease" in q_lower or "with cad" in q_lower:
                    subset = self.df[self.df["has_heart_disease"] == True].dropna(subset=[target_col])
                    group_str = "Patients with Heart Disease (CAD, num > 0)"
                elif "without heart disease" in q_lower or "healthy" in q_lower:
                    subset = self.df[self.df["has_heart_disease"] == False].dropna(subset=[target_col])
                    group_str = "Patients without Heart Disease (Healthy, num = 0)"
                else:
                    subset = self.df.dropna(subset=[target_col])
                    group_str = "All Dataset Patients"

                avg_val = subset[target_col].mean()
                return {
                    "success": True,
                    "answer": (
                        f"Deterministic Dataset Analytics\n\n"
                        f"Query:\n{query}\n\n"
                        f"Parameter: {disp_name}\n"
                        f"Applied Cohort: {group_str}\n"
                        f"Operation: AVERAGE / MEAN\n\n"
                        f"Result:\n{avg_val:.2f}{unit_str}\n\n"
                        f"Calculated across {len(subset)} records with valid measurements."
                    ),
                    "retrieved_evidence": [],
                    "has_relevant_evidence": True,
                    "query_type": "structured_average",
                    "debug": {"column": target_col, "operator": "average", "value": avg_val, "operation": "average"}
                }

        # 6. Median Queries (e.g. "What is the median age?", "What is the median cholesterol?")
        if "median" in q_lower:
            target_col = None
            for col in ["thalch", "oldpeak", "age", "chol", "trestbps", "ca", "fbs"]:
                syns = self.column_synonyms.get(col, [])
                if any(syn in q_lower for syn in syns):
                    target_col = col
                    break

            if target_col == "fbs":
                fbs_high_count = len(self.df[self.df["fbs_bool"] == True])
                fbs_total = len(self.df.dropna(subset=["fbs_bool"]))
                pct = (fbs_high_count / fbs_total) * 100 if fbs_total else 0
                return {
                    "success": True,
                    "answer": (
                        f"Deterministic Dataset Analytics\n\n"
                        f"Query:\n{query}\n\n"
                        f"Parameter: Fasting Blood Sugar / Glucose (> 120 mg/dL indicator)\n"
                        f"Operation: MEDIAN\n\n"
                        f"Result:\nMedian Category: Normal (<= 120 mg/dL)\n"
                        f"Patients with Elevated Glucose (>120 mg/dL): {fbs_high_count} out of {fbs_total} records ({pct:.1f}%)"
                    ),
                    "retrieved_evidence": [],
                    "has_relevant_evidence": True,
                    "query_type": "structured_median",
                    "debug": {"column": "fbs", "operator": "median", "value": "Normal (<=120 mg/dL)", "operation": "median"}
                }

            if target_col and target_col in self.df.columns:
                unit = self.get_column_unit(target_col)
                disp_name = self.get_column_display_name(target_col)
                unit_str = f" {unit}" if unit else ""

                if "heart disease" in q_lower:
                    subset = self.df[self.df["has_heart_disease"] == True].dropna(subset=[target_col])
                    group_str = "Patients with Heart Disease"
                else:
                    subset = self.df.dropna(subset=[target_col])
                    group_str = "All Dataset Patients"

                med_val = subset[target_col].median()
                return {
                    "success": True,
                    "answer": (
                        f"Deterministic Dataset Analytics\n\n"
                        f"Query:\n{query}\n\n"
                        f"Parameter: {disp_name}\n"
                        f"Applied Cohort: {group_str}\n"
                        f"Operation: MEDIAN\n\n"
                        f"Result:\n{med_val:.2f}{unit_str}\n\n"
                        f"Calculated across {len(subset)} records with non-null values."
                    ),
                    "retrieved_evidence": [],
                    "has_relevant_evidence": True,
                    "query_type": "structured_median",
                    "debug": {"column": target_col, "operator": "median", "value": med_val, "operation": "median"}
                }

        # 7. Minimum / Maximum Extremum Queries (ONLY when query specifically asks for the extremum)
        # e.g. "What is the maximum cholesterol?", "What is the minimum cholesterol?"
        is_explicit_min_max_query = (
            re.search(r"\b(what is the highest|what is the maximum|what is the max|what is the lowest|what is the minimum|what is the min|highest .* in the dataset|maximum .* in the dataset|lowest .* in the dataset|minimum .* in the dataset)\b", q_lower) or
            (q_lower.startswith(("highest ", "maximum ", "max ", "lowest ", "minimum ", "min ")) and not any(sym in q_lower for sym in [">", "<", ">=", "<=", "=", "between", "above", "below"]))
        ) and not any(w in q_lower for w in ["how many", "count of", "which patients", "list patients", "show patients", "older than", "younger than", "greater than", "less than"])

        if is_explicit_min_max_query:
            is_max = any(w in q_lower for w in ["highest", "maximum", "max", "largest"])
            op_label = "MAXIMUM" if is_max else "MINIMUM"

            target_col = None
            for col in ["thalch", "oldpeak", "chol", "trestbps", "ca", "age", "num"]:
                syns = self.column_synonyms.get(col, [])
                if any(syn in q_lower for syn in syns):
                    target_col = col
                    break

            if target_col and target_col in self.df.columns:
                unit = self.get_column_unit(target_col)
                disp_name = self.get_column_display_name(target_col)
                unit_str = f" {unit}" if unit else ""

                if "heart disease" in q_lower:
                    subset = self.df[self.df["has_heart_disease"] == True].dropna(subset=[target_col])
                    group_str = "Patients with Heart Disease"
                else:
                    subset = self.df.dropna(subset=[target_col])
                    group_str = "All Dataset Patients"

                val = subset[target_col].max() if is_max else subset[target_col].min()
                rec = subset[subset[target_col] == val].iloc[0]
                p_id = rec.get("id", "N/A")
                p_name = rec.get("patient_id", f"Patient {p_id}")

                return {
                    "success": True,
                    "answer": (
                        f"Deterministic Dataset Analytics\n\n"
                        f"Query:\n{query}\n\n"
                        f"Parameter: {disp_name}\n"
                        f"Operation: {op_label}\n"
                        f"Applied Cohort: {group_str}\n\n"
                        f"Result:\n{val:.2f}{unit_str}\n\n"
                        f"Relevant Record: Patient ID {p_id} ({p_name}) | Age: {rec['age']} | Sex: {rec['sex']}"
                    ),
                    "retrieved_evidence": [],
                    "has_relevant_evidence": True,
                    "query_type": "structured_min_max",
                    "debug": {"column": target_col, "operator": "max" if is_max else "min", "value": val, "operation": op_label.lower()}
                }

        # 8. Multi-Condition Queries (e.g., "age > 60 and cholesterol > 200", "males with cholesterol > 200", "typical angina and heart disease", "cholesterol > 200 also have heart disease")
        has_multi = False
        parts = []
        if " and " in q_lower:
            parts = q_lower.split(" and ")
            has_multi = True
        elif " with " in q_lower and any(w in q_lower for w in ["heart disease", "cad", "cholesterol", "blood pressure", "glucose", "angina", "oldpeak"]):
            parts = q_lower.split(" with ")
            has_multi = True
        elif " also have " in q_lower or " also has " in q_lower:
            parts = re.split(r"\s+also\s+ha(?:ve|s)\s+", q_lower)
            has_multi = True
        elif " who have " in q_lower or " who has " in q_lower:
            parts = re.split(r"\s+who\s+ha(?:ve|s)\s+", q_lower)
            has_multi = True
        elif re.search(r"how many (?:males|females|men|women)\s+(?:have|with|where)\b", q_lower):
            m_sex = "male" if any(w in q_lower for w in ["male", "males", "men"]) else "female"
            rest_clause = re.sub(r"how many (?:males|females|men|women)\s+(?:have|with|where)\b", "", q_lower)
            parts = [m_sex, rest_clause]
            has_multi = True

        if has_multi and len(parts) >= 2:
            clauses = [self._parse_comparison_clause(p) for p in parts]
            clauses = [c for c in clauses if c is not None]

            if len(clauses) >= 2:
                filtered_df = self.df.copy()
                cond_strs = []
                for c in clauses:
                    filtered_df, c_str = self._apply_filter_clause(filtered_df, c)
                    cond_strs.append(c_str)

                count = len(filtered_df)
                total = len(self.df)
                pct = (count / total) * 100
                is_listing = any(w in q_lower for w in ["which", "list", "show", "who are"])

                if is_listing:
                    rows = []
                    for _, r in filtered_df.head(20).iterrows():
                        diag = "CAD" if r["has_heart_disease"] else "No CAD"
                        rows.append(f"{r['patient_id']} | {r['age']} | {r['sex']} | Chol: {r['chol']} mg/dL | BP: {r['trestbps']} mmHg | {diag}")

                    table_header = "Patient ID | Age | Sex | Parameters | Diagnosis\n" + "-" * 60 + "\n"
                    answer = (
                        f"Deterministic Dataset Analytics\n\n"
                        f"Query:\n{query}\n\n"
                        f"Conditions: {' AND '.join(cond_strs)}\n"
                        f"Operation: PATIENT LISTING ({count} matching patients, {pct:.1f}%)\n\n"
                        f"{table_header}"
                        + "\n".join(rows)
                    )
                    return {
                        "success": True,
                        "answer": answer,
                        "retrieved_evidence": [],
                        "has_relevant_evidence": True,
                        "query_type": "structured_filter",
                        "count": count,
                        "debug": {
                            "column": [c["column"] for c in clauses],
                            "operator": [c["operator"] for c in clauses],
                            "value": [c["value"] for c in clauses],
                            "operation": "multi_condition_listing"
                        }
                    }
                else:
                    sample_pids = filtered_df["patient_id"].head(8).tolist()
                    sample_str = f"\n\nSample Matching Patient IDs: {', '.join(sample_pids)}..." if sample_pids else ""

                    answer = (
                        f"Deterministic Dataset Analytics\n\n"
                        f"Query:\n{query}\n\n"
                        f"Conditions: {' AND '.join(cond_strs)}\n"
                        f"Operation: MULTI-CONDITION COUNT\n\n"
                        f"Result:\n{count} patients satisfy the combined criteria (out of {total} total records, {pct:.1f}%).{sample_str}"
                    )
                    return {
                        "success": True,
                        "answer": answer,
                        "retrieved_evidence": [],
                        "has_relevant_evidence": True,
                        "query_type": "structured_count",
                        "count": count,
                        "debug": {
                            "column": [c["column"] for c in clauses],
                            "operator": [c["operator"] for c in clauses],
                            "value": [c["value"] for c in clauses],
                            "operation": "multi_condition_count"
                        }
                    }

        # 9. Single Condition Count / Filter / Listing Queries (Universal Parser for ALL Columns)
        clause = self._parse_comparison_clause(q)
        if clause:
            filtered_df, cond_str = self._apply_filter_clause(self.df, clause)
            count = len(filtered_df)
            col = clause["column"]
            total_valid = len(self.df.dropna(subset=[col])) if col in self.df.columns else len(self.df)
            pct = (count / total_valid) * 100 if total_valid else 0
            disp_col = self.get_column_display_name(col)

            # Determine whether user wants a listing or count
            is_listing = any(w in q_lower for w in ["which", "show", "list", "find", "who are", "give me"]) and not any(w in q_lower for w in ["how many", "count of", "number of"])

            if is_listing:
                rows = []
                unit = self.get_column_unit(col)
                u_str = f" {unit}" if unit else ""

                for _, r in filtered_df.head(20).iterrows():
                    val = r.get(col, "N/A")
                    val_str = f"{val}{u_str}" if pd.notna(val) else "N/A"
                    diag = "CAD" if r["has_heart_disease"] else "No CAD"
                    rows.append(f"{r['patient_id']} | {r['age']} | {r['sex']} | {disp_col}: {val_str} | {diag}")

                table_header = f"Patient ID | Age | Sex | {disp_col} | Diagnosis\n" + "-" * 60 + "\n"
                answer = (
                    f"Deterministic Dataset Analytics\n\n"
                    f"Query:\n{query}\n\n"
                    f"Parameter: {disp_col}\n"
                    f"Condition: {cond_str}\n"
                    f"Operation: PATIENT LISTING ({count} matching records, {pct:.1f}%)\n\n"
                    f"{table_header}"
                    + "\n".join(rows)
                )
                return {
                    "success": True,
                    "answer": answer,
                    "retrieved_evidence": [],
                    "has_relevant_evidence": True,
                    "query_type": "structured_filter",
                    "count": count,
                    "debug": {
                        "column": col,
                        "operator": clause["operator"],
                        "value": clause["value"],
                        "operation": "filter_listing"
                    }
                }
            else:
                sample_pids = filtered_df["patient_id"].head(8).tolist()
                sample_str = f"\n\nSample Matching Patient IDs: {', '.join(sample_pids)}..." if sample_pids else ""

                answer = (
                    f"Deterministic Dataset Analytics\n\n"
                    f"Query:\n{query}\n\n"
                    f"Parameter: {disp_col}\n"
                    f"Condition: {cond_str}\n"
                    f"Operation: COUNT\n\n"
                    f"Result:\n{count} patients have {cond_str.lower()} (out of {total_valid} evaluated records, {pct:.1f}%).{sample_str}"
                )
                return {
                    "success": True,
                    "answer": answer,
                    "retrieved_evidence": [],
                    "has_relevant_evidence": True,
                    "query_type": "structured_count",
                    "count": count,
                    "debug": {
                        "column": col,
                        "operator": clause["operator"],
                        "value": clause["value"],
                        "operation": "count"
                    }
                }

        # 10. Percentage Queries fallback
        if "percentage" in q_lower or "%" in q_lower or "percent" in q_lower:
            if "heart disease" in q_lower or "cad" in q_lower:
                hd_count = len(self.df[self.df["has_heart_disease"] == True])
                total = len(self.df)
                pct = (hd_count / total) * 100
                return {
                    "success": True,
                    "answer": (
                        f"Deterministic Dataset Analytics\n\n"
                        f"Query:\n{query}\n\n"
                        f"Parameter: Heart Disease Diagnosis (Outcome > 0)\n"
                        f"Operation: PERCENTAGE\n\n"
                        f"Result:\n{pct:.1f}% of patients have heart disease ({hd_count} out of {total} total records)."
                    ),
                    "retrieved_evidence": [],
                    "has_relevant_evidence": True,
                    "query_type": "structured_percentage",
                    "debug": {"column": "num", "operator": "percentage", "value": pct, "operation": "percentage"}
                }

        # Informative Error Handling with Debug Metadata
        return {
            "success": False,
            "answer": (
                f"Deterministic Dataset Analytics\n\n"
                f"Query:\n{query}\n\n"
                f"Notice:\nUnable to execute structured calculation for the provided query parameters.\n"
                f"Supported operations: COUNT, AVERAGE, MEDIAN, MINIMUM, MAXIMUM, FILTER LISTING, SINGLE PATIENT LOOKUP.\n"
                f"Available parameters: age, sex, chest pain (cp), resting blood pressure (trestbps), cholesterol (chol), "
                f"fasting blood sugar (fbs), resting ECG (restecg), max heart rate (thalch), exercise angina (exang), "
                f"oldpeak (ST depression), slope, major vessels (ca), thal, diagnosis (num)."
            ),
            "retrieved_evidence": [],
            "has_relevant_evidence": False,
            "query_type": "structured_error",
            "debug": {
                "column": "unresolved",
                "operator": "unresolved",
                "value": "unresolved",
                "operation": "unresolved"
            }
        }


# Global singleton engine instance
_structured_engine_instance: Optional[StructuredPatientDataEngine] = None


def get_structured_engine() -> StructuredPatientDataEngine:
    """Retrieve or initialize the singleton StructuredPatientDataEngine."""
    global _structured_engine_instance
    if _structured_engine_instance is None:
        _structured_engine_instance = StructuredPatientDataEngine()
    return _structured_engine_instance
