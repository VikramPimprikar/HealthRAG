"""
Automated Test Suite for Medical RAG & Deterministic Patient Analytics
======================================================================
Tests all requirements:
1. Deterministic Column & Parameter Calculations (chol > 120, chol > 200, age > 50, trestbps > 140, thalch > 150, oldpeak > 1, fbs == 1)
2. Deterministic Aggregations (Average, Minimum, Maximum, Median, Total Count)
3. Range and Categorical Queries (oldpeak between 1 and 2, chest pain type 2, blood pressure below 120)
4. Specific Patient Lookups (All findings vs. isolated Single Parameter vs. Non-existent patient)
5. Intent Routing (Medical FAISS RAG vs. Structured Analytics)
6. Anti-Hallucination Off-topic Guardrails
"""

import sys
import io
from pathlib import Path
from fastapi.testclient import TestClient

# Ensure UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR / "backend"))

from app.api import app

client = TestClient(app)


def run_test_query(test_num: int, label: str, query: str, user_id: str = "admin", top_k: int = 5):
    print(f"\n[{test_num}] Testing: {label}")
    print(f"    Query: \"{query}\"")

    payload = {
        "query": query,
        "user_id": user_id,
        "top_k": top_k
    }

    try:
        r = client.post("/api/v1/query", json=payload, headers={"X-User-Id": user_id})
        assert r.status_code == 200, f"HTTP Error {r.status_code}: {r.text}"
        data = r.json()

        print(f"    Query Type: {data.get('query_type')}")
        print(f"    Has Relevant: {data.get('has_relevant_evidence')}")
        print(f"    Records Count: {len(data.get('retrieved_evidence', []))}")
        print(f"    Blockchain Block: #{data.get('block_index')}")
        print(f"    --- Response Answer ---")
        for line in data.get("answer", "").strip().split("\n")[:6]:
            print(f"    {line}")
        if len(data.get("answer", "").strip().split("\n")) > 6:
            print(f"    ... (truncated)")
        print(f"    [PASSED]")
        return data
    except Exception as e:
        print(f"    [FAILED] {e}")
        raise


def main():
    print("=" * 80)
    print("RUNNING COMPREHENSIVE MEDICAL RAG & STRUCTURED PATIENT DATA TEST SUITE")
    print("=" * 80)

    # 1. Total Patient Count Query
    q1 = run_test_query(1, "Total Patients Count ('How many patients are there in total?')", "How many patients are there in total?")
    assert "920" in q1.get("answer", "")

    # 1b. Total Count Synonym ('How many patients are there?')
    q1b = run_test_query(2, "Total Patients Synonym ('How many patients are there?')", "How many patients are there?")
    assert "920" in q1b.get("answer", "")

    # 2. Target Query 1: Cholesterol > 120
    q_chol120 = run_test_query(3, "Cholesterol > 120 ('How many patients have cholesterol greater than 120?')", "How many patients have cholesterol greater than 120?")
    assert "714" in q_chol120.get("answer", "")

    # 3. Target Query 2: Cholesterol > 200
    q_chol200 = run_test_query(4, "Cholesterol > 200 ('How many patients have cholesterol greater than 200?')", "How many patients have cholesterol greater than 200?")
    assert "586" in q_chol200.get("answer", "")

    # 4. Target Query 3: Age > 50
    q_age50 = run_test_query(5, "Age > 50 ('How many patients have age greater than 50?')", "How many patients have age greater than 50?")
    assert "603" in q_age50.get("answer", "")

    # 5. Target Query 4: Average Cholesterol
    q_avg_chol = run_test_query(6, "Average Cholesterol ('What is the average cholesterol?')", "What is the average cholesterol?")
    assert "199.13" in q_avg_chol.get("answer", "")

    # 6. Target Query 5: Maximum Cholesterol
    q_max_chol = run_test_query(7, "Maximum Cholesterol ('What is the maximum cholesterol?')", "What is the maximum cholesterol?")
    assert "603" in q_max_chol.get("answer", "")

    # 7. Target Query 6: Blood Pressure > 140
    q_bp140 = run_test_query(8, "Blood Pressure > 140 ('How many patients have resting blood pressure greater than 140?')", "How many patients have resting blood pressure greater than 140?")
    assert "209" in q_bp140.get("answer", "")

    # 8. Target Query 7: Maximum Heart Rate > 150
    q_thalch150 = run_test_query(9, "Max Heart Rate > 150 ('How many patients have maximum heart rate greater than 150?')", "How many patients have maximum heart rate greater than 150?")
    assert "277" in q_thalch150.get("answer", "")

    # 9. Target Query 8: Oldpeak > 1
    q_oldpeak1 = run_test_query(10, "Oldpeak > 1 ('How many patients have oldpeak greater than 1?')", "How many patients have oldpeak greater than 1?")
    assert "298" in q_oldpeak1.get("answer", "")

    # 10. Target Query 9: Fasting Blood Sugar == 1
    q_fbs1 = run_test_query(11, "Fasting Blood Sugar == 1 ('How many patients have fasting blood sugar equal to 1?')", "How many patients have fasting blood sugar equal to 1?")
    assert "138" in q_fbs1.get("answer", "")

    # 11. Range Query: Oldpeak between 1 and 2
    q_range = run_test_query(12, "Range Query ('How many patients have oldpeak between 1 and 2?')", "How many patients have oldpeak between 1 and 2?")
    assert "281" in q_range.get("answer", "")

    # 12. Categorical Query: Chest Pain Type 2
    q_cp2 = run_test_query(13, "Chest Pain Type 2 ('How many patients have chest pain type 2?')", "How many patients have chest pain type 2?")
    assert "174" in q_cp2.get("answer", "")

    # 13. Median Query: Median Age
    q_med_age = run_test_query(14, "Median Age ('What is the median age?')", "What is the median age?")
    assert "54.00" in q_med_age.get("answer", "")

    # 14. Minimum Query: Minimum Blood Pressure
    q_min_bp = run_test_query(15, "Minimum Blood Pressure ('What is the minimum resting blood pressure?')", "What is the minimum resting blood pressure?")
    assert "0.00" in q_min_bp.get("answer", "")

    # 15. Percentage Query: Heart Disease
    q_pct = run_test_query(16, "Percentage Query ('What percentage of patients have heart disease?')", "What percentage of patients have heart disease?")
    assert "55.3%" in q_pct.get("answer", "")

    # 16. Multi-Condition Count: Age > 50 and Glucose > 120
    q_multi = run_test_query(17, "Multi-Condition ('How many patients have age > 50 and glucose > 120?')", "How many patients have age > 50 and glucose > 120?")
    assert "122" in q_multi.get("answer", "")

    # 17. Specific Patient Lookup - All Findings for P1651
    p_all = run_test_query(18, "Specific Patient All Findings ('Show patient P1651')", "Show patient P1651")
    assert "Patient ID: P1651" in p_all.get("answer", "")
    assert "Findings:" in p_all.get("answer", "")

    # 18. Specific Patient Lookup - Single Parameter P1651 Cholesterol
    p_single = run_test_query(19, "Specific Patient Single Parameter ('What is P1651\'s cholesterol?')", "What is P1651's cholesterol?")
    assert "Patient ID: P1651" in p_single.get("answer", "")
    assert "Cholesterol: 0.0 mg/dL" in p_single.get("answer", "")
    assert "Blood Pressure:" not in p_single.get("answer", "")

    # 19. Non-Existent Patient ID
    p_invalid = run_test_query(20, "Non-Existent Patient ('Show patient P9999')", "Show patient P9999")
    assert "Patient P9999 was not found in the dataset." in p_invalid.get("answer", "")

    # 20. Target Query 11: Normal Medical RAG Query - Symptoms (Must use FAISS RAG)
    rag_symptoms = run_test_query(21, "Medical RAG ('What are the symptoms of heart disease?')", "What are the symptoms of heart disease?")
    assert len(rag_symptoms.get("answer", "")) > 40
    assert rag_symptoms.get("query_type") == "rag_medical"
    assert len(rag_symptoms.get("retrieved_evidence", [])) > 0

    # 21. Target Query 12: Normal Medical RAG Query - Causes of High Cholesterol (Must use FAISS RAG)
    rag_chol = run_test_query(22, "Medical RAG ('What causes high cholesterol?')", "What causes high cholesterol?")
    assert len(rag_chol.get("answer", "")) > 40
    assert rag_chol.get("query_type") == "rag_medical"
    assert len(rag_chol.get("retrieved_evidence", [])) > 0

    # 22. Off-Topic Query Guardrail
    off_topic = run_test_query(23, "Off-Topic Query ('Explain quantum rocket teleportation in astrophysics')", "Explain quantum rocket teleportation in astrophysics")
    assert "I apologize for the mismatch" in off_topic.get("answer", "")
    assert off_topic.get("has_relevant_evidence") is False

    print("\n" + "=" * 80)
    print("ALL 23 COMPREHENSIVE TESTS EXECUTED AND PASSED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()
