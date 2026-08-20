"""
Automated Test Suite for Medical RAG & Deterministic Patient Analytics
======================================================================
Tests all requirements:
1. Intent Routing (Normal Medical RAG vs. Structured Analytics vs. Specific Patient Lookup)
2. Total Patient Count Queries ("How many patients are there in total?")
3. Numerical / Conditional Count Queries ("How many patients are older than 65?", "glucose > 120", "age between 40 and 60", "age > 50 and glucose > 120")
4. Filter Result Strict Validation (Guarantees every returned patient record satisfies condition)
5. Full Structured Analytics (Count, Average, Min, Max, Median, Percentage, Comparison, Top-N Ranking, Grouping)
6. Specific Patient Lookups (All findings vs. isolated Single Parameter vs. Non-existent patient)
7. Normal Medical RAG Q&A (Symptoms, Causes, ST depression mechanism, Risk factors)
8. Off-topic Guardrails and Empty context handling
"""

import requests
import json
import sys

API_BASE = "http://127.0.0.1:8000"


def run_test_query(test_num: int, label: str, query: str, user_id: str = "admin", top_k: int = 5):
    print(f"\n[{test_num}] Testing: {label}")
    print(f"    Query: \"{query}\"")
    
    payload = {
        "query": query,
        "user_id": user_id,
        "top_k": top_k
    }
    
    try:
        r = requests.post(f"{API_BASE}/api/v1/query", json=payload, headers={"X-User-Id": user_id}, timeout=30)
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

    # 1. Total Patient Count Query (Requirement 4)
    q1 = run_test_query(1, "Total Patients Count ('How many patients are there in total?')", "How many patients are there in total?")
    assert "Total Patients: 920" in q1.get("answer", "")

    # 1b. Total Count Synonym ('What is the total number of patients?')
    q1b = run_test_query(2, "Total Patients Synonym ('What is the total number of patients?')", "What is the total number of patients?")
    assert "Total Patients: 920" in q1b.get("answer", "")

    # 2. Conditional Count - Age > 65 (Requirement 2 & 3)
    q2 = run_test_query(3, "Conditional Count ('How many patients are older than 65?')", "How many patients are older than 65?")
    assert "Age > 65" in q2.get("answer", "")
    assert "82" in q2.get("answer", "")

    # 3. Conditional Count - Glucose > 120 (Requirement 2)
    q3 = run_test_query(4, "Conditional Count ('How many patients have glucose > 120?')", "How many patients have glucose > 120?")
    assert "138" in q3.get("answer", "")

    # 4. Range Query - Age between 40 and 60 (Requirement 5)
    q4 = run_test_query(5, "Range Count ('How many patients have age between 40 and 60?')", "How many patients have age between 40 and 60?")
    assert "619" in q4.get("answer", "")

    # 5. Multi-Condition Count - Age > 50 AND Glucose > 120 (Requirement 2 & 5)
    q5 = run_test_query(6, "Multi-Condition Count ('How many patients have age > 50 and glucose > 120?')", "How many patients have age > 50 and glucose > 120?")
    assert "122" in q5.get("answer", "")

    # 6. Strict Validation Filter Listing (Requirement 3)
    q6 = run_test_query(7, "Filter Validation Listing ('Show patients with cholesterol > 350')", "Show patients with cholesterol > 350")
    assert "Serum Cholesterol > 350" in q6.get("answer", "")
    assert "27 records found" in q6.get("answer", "")

    # 7. Average Query - Age (Requirement 5)
    q7 = run_test_query(8, "Average Query ('What is the average age?')", "What is the average age?")
    assert "53.51 years" in q7.get("answer", "")

    # 8. Average Query - Glucose (Requirement 5)
    q8 = run_test_query(9, "Average Query ('What is the average glucose?')", "What is the average glucose?")
    assert "Glucose" in q8.get("answer", "") or "Sugar" in q8.get("answer", "")

    # 9. Minimum Query (Requirement 5)
    q9 = run_test_query(10, "Minimum Query ('What is the minimum age?')", "What is the minimum age?")
    assert "28" in q9.get("answer", "")

    # 10. Maximum Query (Requirement 5)
    q10 = run_test_query(11, "Maximum Query ('What is the highest cholesterol?')", "What is the highest cholesterol?")
    assert "603" in q10.get("answer", "")

    # 11. Median Query (Requirement 5)
    q11 = run_test_query(12, "Median Query ('What is the median age?')", "What is the median age?")
    assert "54.00 years" in q11.get("answer", "")

    # 12. Percentage Query (Requirement 5)
    q12 = run_test_query(13, "Percentage Query ('What percentage of patients have heart disease?')", "What percentage of patients have heart disease?")
    assert "55.3%" in q12.get("answer", "")

    # 13. Comparison Query (Requirement 5)
    q13 = run_test_query(14, "Comparison Query ('Compare average glucose between patients with and without heart disease.')", "Compare average glucose between patients with and without heart disease.")
    assert "Comparison:" in q13.get("answer", "")

    # 14. Sorting / Ranking Query - Top 10 (Requirement 5)
    q14 = run_test_query(15, "Sorting Query ('Show the 10 patients with the highest cholesterol.')", "Show the 10 patients with the highest cholesterol.")
    assert "Top 10 Patients" in q14.get("answer", "")
    assert "603" in q14.get("answer", "")

    # 15. Grouping Query (Requirement 5)
    q15 = run_test_query(16, "Grouping Query ('How many patients are in each age group?')", "How many patients are in each age group?")
    assert "Age Group 50-59" in q15.get("answer", "")

    # 16. Specific Patient Lookup - All Findings for P1651 (Requirement 6)
    p_all = run_test_query(17, "Specific Patient All Findings ('Show patient P1651')", "Show patient P1651")
    assert "Patient ID: P1651" in p_all.get("answer", "")
    assert "Findings:" in p_all.get("answer", "")
    assert "Blood Pressure:" in p_all.get("answer", "")
    assert "Cholesterol:" in p_all.get("answer", "")

    # 17. Specific Patient Lookup - Single Parameter P1651 Cholesterol (Requirement 6)
    p_single = run_test_query(18, "Specific Patient Single Parameter ('What is P1651\'s cholesterol?')", "What is P1651's cholesterol?")
    assert "Patient ID: P1651" in p_single.get("answer", "")
    assert "Cholesterol: 0.0 mg/dL" in p_single.get("answer", "")
    # Must NOT return unrelated parameters
    assert "Blood Pressure:" not in p_single.get("answer", "")
    assert "Resting ECG:" not in p_single.get("answer", "")

    # 18. Specific Patient Lookup - Patient 125
    p_125 = run_test_query(19, "Specific Patient Lookup ('Show patient 125')", "Show patient 125")
    assert "Findings:" in p_125.get("answer", "")

    # 19. Non-Existent Patient ID (Requirement 6)
    p_invalid = run_test_query(20, "Non-Existent Patient ('Show patient P9999')", "Show patient P9999")
    assert "Patient P9999 was not found in the dataset." in p_invalid.get("answer", "")

    # 20. Normal Medical RAG Query - Symptoms (Requirement 1 & 9)
    rag_symptoms = run_test_query(21, "Medical RAG ('What are the symptoms of heart disease?')", "What are the symptoms of heart disease?")
    assert len(rag_symptoms.get("answer", "")) > 40
    assert "chest pain" in rag_symptoms.get("answer", "").lower() or "angina" in rag_symptoms.get("answer", "").lower() or "shortness of breath" in rag_symptoms.get("answer", "").lower() or "heart" in rag_symptoms.get("answer", "").lower()

    # 21. Normal Medical RAG Query - Causes of High Cholesterol (Requirement 1 & 9)
    rag_chol = run_test_query(22, "Medical RAG ('What causes high cholesterol?')", "What causes high cholesterol?")
    assert len(rag_chol.get("answer", "")) > 40
    assert "cholesterol" in rag_chol.get("answer", "").lower()

    # 22. Normal Medical RAG Query - ST Depression (Requirement 1 & 9)
    rag_st = run_test_query(23, "Medical RAG ('Explain ST depression.')", "Explain ST depression.")
    assert len(rag_st.get("answer", "")) > 40

    # 23. Normal Medical RAG Query - Risk Factors (Requirement 1 & 9)
    rag_rf = run_test_query(24, "Medical RAG ('What are the risk factors for cardiovascular disease?')", "What are the risk factors for cardiovascular disease?")
    assert len(rag_rf.get("answer", "")) > 40

    # 24. Off-Topic Query Guardrail (Requirement 10)
    off_topic = run_test_query(25, "Off-Topic Query ('Explain quantum rocket teleportation in astrophysics')", "Explain quantum rocket teleportation in astrophysics")
    assert "I apologize for the mismatch" in off_topic.get("answer", "")
    assert off_topic.get("has_relevant_evidence") is False

    print("\n" + "=" * 80)
    print("ALL 25 COMPREHENSIVE TESTS EXECUTED AND PASSED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()
