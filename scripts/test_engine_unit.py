import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR / "backend"))

from app.clinical.structured_query_engine import get_structured_engine

engine = get_structured_engine()

queries = [
    # 10 Main Target Queries
    "How many patients have cholesterol greater than 120?",
    "How many patients have cholesterol greater than 200?",
    "How many patients have age greater than 50?",
    "What is the average cholesterol?",
    "What is the maximum cholesterol?",
    "How many patients have resting blood pressure greater than 140?",
    "How many patients have maximum heart rate greater than 150?",
    "How many patients have oldpeak greater than 1?",
    "How many patients have fasting blood sugar equal to 1?",
    "How many patients are there?",

    # Additional queries from prompt
    "How many patients have blood pressure below 120?",
    "How many patients have oldpeak between 1 and 2?",
    "How many patients have chest pain type 2?",
    "What is the minimum resting blood pressure?",
    "What is the median age?",

    # Medical RAG queries (Must be routed to FAISS RAG)
    "What are the symptoms of heart disease?",
    "What causes high cholesterol?"
]

print("=" * 80)
print("TESTING STRUCTURED QUERY ENGINE & RAG ROUTING")
print("=" * 80)

for idx, q in enumerate(queries, 1):
    intent = engine.detect_structured_query_intent(q)
    print(f"\n[{idx}] Query: \"{q}\"")
    print(f"    Structured Intent: {intent}")
    if intent:
        res = engine.execute_structured_query(q)
        print(f"    Success: {res.get('success')}")
        print(f"    Query Type: {res.get('query_type')}")
        print(f"    Debug: {res.get('debug')}")
        print("    Answer Preview:")
        for line in res.get("answer", "").strip().split("\n"):
            print(f"      {line}")
    else:
        print("    --> ROUTED TO GENERAL MEDICAL FAISS RAG (PASS)")
    print("-" * 60)
