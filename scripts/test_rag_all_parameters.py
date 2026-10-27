import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR / "backend"))

from app.clinical.structured_query_engine import get_structured_engine

engine = get_structured_engine()

test_queries = [
    # A & B: General Medical Knowledge (Must return None -> FAISS RAG)
    "What are the symptoms of heart disease?",
    "What causes high cholesterol?",
    "What is ST depression?",
    "What are the risk factors for coronary artery disease?",
    "What is angina?",
    "What does high cholesterol mean clinically?",
    "Why is high cholesterol a risk factor for heart disease?",

    # C & D: Single Patient Lookups (All 10 parameters)
    "What is P1651's cholesterol?",
    "What is P1651's blood pressure?",
    "What is P1651's maximum heart rate?",
    "What is P1651's chest pain type?",
    "What is P1651's ECG result?",
    "What is P1651's oldpeak?",
    "What is P1651's diagnosis?",
    "What is P1651's slope?",
    "What is P1651's fasting blood sugar?",
    "What is P1651's exercise induced angina?",

    # E to J & Count queries
    "How many patients have cholesterol greater than 200?",
    "How many patients have cholesterol above 240?",
    "How many patients have blood pressure greater than 140?",
    "How many patients are older than 65?",
    "How many patients have maximum heart rate below 100?",
    "How many patients have exercise induced angina?",
    "How many patients have fasting blood sugar above 120?",
    "How many patients have typical angina?",
    "How many patients have atypical angina?",
    "How many patients have non-anginal pain?",
    "How many patients are asymptomatic?",
    "How many patients have abnormal ECG?",
    "How many patients have ST depression greater than 2?",

    # K & L & Aggregations
    "What is the average cholesterol?",
    "What is the average age?",
    "What is the average resting blood pressure?",
    "What is the average maximum heart rate?",
    "What is the maximum cholesterol?",
    "What is the minimum cholesterol?",
    "What is the average oldpeak?",
    "What is the average age of patients with heart disease?",

    # M & Comparison / List queries ("Which patients have ...")
    "Which patients have cholesterol greater than 200?",
    "Which patients have blood pressure below 120?",
    "Which patients have oldpeak greater than 2?",
    "Which patients are older than 60?",
    "Which patients have maximum heart rate above 150?",
    "Which patients have exercise induced angina?",

    # Multi-condition queries
    "How many patients are older than 60 and have cholesterol greater than 200?",
    "Which patients have cholesterol above 240 and blood pressure above 140?",
    "How many males have cholesterol greater than 200?",
    "How many patients with typical angina have heart disease?",
    "How many patients with oldpeak greater than 2 have CAD?",
    "How many patients with cholesterol greater than 200 also have heart disease?"
]

print("=" * 80)
print("TESTING STRUCTURED QUERY ENGINE ON ALL PARAMETER QUERIES")
print("=" * 80)

failures = []

for idx, q in enumerate(test_queries, 1):
    intent = engine.detect_structured_query_intent(q)
    print(f"\n[{idx}] Query: \"{q}\"")
    print(f"    Intent: {intent}")
    if intent:
        res = engine.execute_structured_query(q)
        success = res.get("success", False)
        q_type = res.get("query_type")
        ans_preview = res.get("answer", "").strip().split("\n")[0]
        print(f"    Success: {success} | Type: {q_type} | Preview: {ans_preview}")
        if not success or "Unable to execute" in res.get("answer", ""):
            failures.append((q, "Execution failed / Unable to execute"))
    else:
        # General Medical RAG query
        print(f"    --> ROUTED TO GENERAL MEDICAL FAISS RAG (PASS)")

print("\n" + "=" * 80)
if failures:
    print(f"FAILED QUERIES ({len(failures)}):")
    for f_q, f_reason in failures:
        print(f"  - \"{f_q}\": {f_reason}")
else:
    print("ALL TEST QUERIES RESOLVED AND PARSED SUCCESSFULLY!")
print("=" * 80)
