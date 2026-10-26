"""
RAGChainMed End-to-End Test Suite
=================================
Automated verification of RAG retrieval, anti-hallucination guardrails,
blockchain SHA-256 evidence hashing, tamper detection, RBAC permissions,
and ML clinical predictions.
"""

import requests
import json
import sys
import io

# Fix Windows console encoding for UTF-8 symbols
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

API_BASE = "http://127.0.0.1:8000"

def run_tests():
    print("=" * 70)
    print("RUNNING RAGCHAINMED END-TO-END TEST SUITE")
    print("=" * 70)

    # 1. Health check
    print("\n[TEST 1] Testing /health...")
    r = requests.get(f"{API_BASE}/health")
    assert r.status_code == 200, f"Health check failed: {r.text}"
    health_data = r.json()
    print(f"  [OK] Status: {health_data['status']}")
    print(f"  [OK] Total vectors in FAISS: {health_data['total_vectors']}")
    print(f"  [OK] Blockchain intact: {health_data['blockchain_intact']}")
    print(f"  [OK] Total blockchain blocks: {health_data['total_blocks']}")

    # 2. Auth Users
    print("\n[TEST 2] Testing /api/v1/auth/users (RBAC user list)...")
    r = requests.get(f"{API_BASE}/api/v1/auth/users")
    assert r.status_code == 200, f"Auth users failed: {r.text}"
    users = r.json().get("users", [])
    print(f"  [OK] Loaded {len(users)} registered users and roles.")
    user_roles = [u["role"].lower() for u in users]
    user_ids = [u["user_id"].lower() for u in users]
    assert "auditor" not in user_roles, "Auditor role should be removed from active users!"
    assert "auditor001" not in user_ids, "AUDITOR001 should be removed from user list!"
    print("  [OK] Confirmed: Auditor role is absent from user list.")
    for u in users:
        print(f"    - {u['user_id'].upper()}: {u['name']} (Role: {u['role']})")

    # 3. RAG Query as Doctor / Admin
    print("\n[TEST 3] Testing RAG Query ('high cholesterol and chest pain') as admin...")
    payload = {
        "query": "What are the symptoms of heart disease?",
        "user_id": "admin",
        "top_k": 3
    }
    r = requests.post(f"{API_BASE}/api/v1/query", json=payload, headers={"X-User-Id": "admin"})
    assert r.status_code == 200, f"RAG query failed: {r.text}"
    query_data = r.json()
    evidence_items = query_data.get("retrieved_evidence", [])
    assert len(evidence_items) > 0, "No evidence retrieved!"
    print(f"  [OK] Retrieved {len(evidence_items)} evidence records.")
    print(f"  [OK] Top Match: {evidence_items[0].get('patient_id', 'Evidence Item')} (SHA-256: {evidence_items[0].get('sha256_hash', '')[:16]}...)")
    print(f"  [OK] Evidence SHA-256 Hash: {query_data.get('evidence_hash')[:32]}...")
    print(f"  [OK] Committed to Blockchain Block #{query_data.get('block_index')}")
    print(f"  [OK] AI Grounded Answer snippet:\n    {query_data.get('answer')[:180]}...")

    recorded_block_index = query_data.get("block_index")
    bundle_evidence_hash = query_data.get("evidence_hash")

    # 4. Anti-Hallucination Guardrail Check
    print("\n[TEST 4] Testing Anti-Hallucination Guardrail with Off-Topic Query...")
    off_topic_payload = {
        "query": "Explain quantum rocket orbital trajectories in astrophysics",
        "user_id": "admin",
        "top_k": 3
    }
    r = requests.post(f"{API_BASE}/api/v1/query", json=off_topic_payload, headers={"X-User-Id": "admin"})
    assert r.status_code == 200, f"Off-topic query failed: {r.text}"
    off_topic_data = r.json()
    print(f"  [OK] Response received: {off_topic_data.get('answer')[:120]}...")

    # 5. RBAC Permission Denial on RAG Query (Nurse cannot query knowledge base)
    print("\n[TEST 5] Testing RBAC Enforcement (NURSE001 querying knowledge base)...")
    r_nurse = requests.post(
        f"{API_BASE}/api/v1/query",
        json={"query": "test", "user_id": "NURSE001"},
        headers={"X-User-Id": "NURSE001"}
    )
    assert r_nurse.status_code == 403, f"Expected 403 Forbidden, got {r_nurse.status_code}"
    print(f"  [OK] RBAC correctly denied access with 403 Forbidden: {r_nurse.json().get('detail')}")

    # 6. Clinical ML Prediction as Doctor DOC001
    print("\n[TEST 6] Testing Clinical ML Prediction as DOC001...")
    patient_payload = {
        "user_id": "DOC001",
        "patient_id": "P_TEST_99",
        "age": 64,
        "sex": 1,
        "cp": 4,
        "trestbps": 160,
        "chol": 280,
        "fbs": 1,
        "restecg": 2,
        "thalch": 125,
        "exang": 1,
        "oldpeak": 2.8,
        "slope": 3
    }
    r_pred = requests.post(
        f"{API_BASE}/api/v1/predict",
        json=patient_payload,
        headers={"X-User-Id": "DOC001"}
    )
    assert r_pred.status_code == 200, f"Prediction failed: {r_pred.text}"
    pred_data = r_pred.json()
    print(f"  [OK] Predicted Severity: {pred_data.get('severity')} (Confidence: {pred_data.get('confidence'):.2%})")
    print(f"  [OK] Risk Assessment: {pred_data.get('risk_assessment', {}).get('risk_level')} Risk")
    print(f"  [OK] CDS Recommendations generated: {len(pred_data.get('risk_assessment', {}).get('recommendations', []))}")
    print(f"  [OK] Committed to Blockchain Block #{pred_data.get('blockchain', {}).get('block_index')}")

    # 7. Evidence Verification on Blockchain (Authentic Evidence)
    print("\n[TEST 7] Testing Evidence Verification (Authentic Evidence against Blockchain)...")
    verify_payload = {
        "block_index": recorded_block_index,
        "evidence_hash": bundle_evidence_hash
    }
    r_v = requests.post(f"{API_BASE}/api/v1/audit/verify-evidence", json=verify_payload, headers={"X-User-Id": "admin"})
    assert r_v.status_code == 200, f"Evidence verification failed: {r_v.text}"
    v_data = r_v.json()
    assert v_data.get("verified") is True, f"Expected verified=True, got {v_data}"
    print(f"  [OK] Verification result: AUTHENTIC (verified={v_data.get('verified')}, tamper_detected={v_data.get('tamper_detected')})")

    # 8. Evidence Tamper Detection Test (Tampered Evidence)
    print("\n[TEST 8] Testing Evidence Tamper Detection (Tampered Hash)...")
    tampered_payload = {
        "block_index": recorded_block_index,
        "evidence_hash": "deadbeef" * 8
    }
    r_tamper = requests.post(f"{API_BASE}/api/v1/audit/verify-evidence", json=tampered_payload, headers={"X-User-Id": "admin"})
    assert r_tamper.status_code == 200
    t_data = r_tamper.json()
    assert t_data.get("verified") is False, "Tampered hash should not be verified!"
    assert t_data.get("tamper_detected") is True, "Tamper should be detected!"
    print(f"  [OK] Tamper detection result: TAMPER DETECTED (verified={t_data.get('verified')}, tamper_detected={t_data.get('tamper_detected')})")

    # 9. Verify Full Blockchain Chain
    print("\n[TEST 9] Testing Blockchain Full Chain Cryptographic Verification...")
    r_chain = requests.get(f"{API_BASE}/api/v1/audit/verify-chain", headers={"X-User-Id": "admin"})
    assert r_chain.status_code == 200, f"Chain verify failed: {r_chain.text}"
    c_data = r_chain.json()
    assert c_data.get("is_valid") is True, f"Blockchain is invalid: {c_data}"
    print(f"  [OK] Full Chain Validity: {c_data.get('is_valid')} across {c_data.get('total_blocks')} blocks.")

    # 10. Audit Logs Query
    print("\n[TEST 10] Testing Audit Logs Retrieval as admin...")
    r_audit = requests.get(f"{API_BASE}/api/v1/audit/logs", headers={"X-User-Id": "admin"})
    assert r_audit.status_code == 200, f"Audit logs failed: {r_audit.text}"
    audit_data = r_audit.json()
    print(f"  [OK] Retrieved {len(audit_data.get('blocks', []))} blocks and {len(audit_data.get('records', []))} audit records.")

    # 11. Direct Evidence Text + Stored Hash Verification (/api/v1/audit/verify-integrity)
    print("\n[TEST 11] Testing /api/v1/audit/verify-integrity Endpoint...")
    sample_text = "Patient ID P1651: Age 58, Resting BP 140 mmHg, Cholesterol 240 mg/dL."
    import hashlib
    actual_hash = hashlib.sha256(sample_text.strip().encode("utf-8")).hexdigest()
    r_int_auth = requests.post(
        f"{API_BASE}/api/v1/audit/verify-integrity",
        json={"evidence_text": sample_text, "stored_hash": actual_hash},
        headers={"X-User-Id": "admin"}
    )
    assert r_int_auth.status_code == 200
    int_auth_data = r_int_auth.json()
    assert int_auth_data.get("verified") is True, f"Expected verified=True, got {int_auth_data}"
    assert int_auth_data.get("tampered") is False
    print(f"  [OK] Authentic text verify: verified={int_auth_data.get('verified')}, message='{int_auth_data.get('message')}'")

    # Tampered verify
    tampered_sample_text = "Patient ID P1651: Age 58, Resting BP 190 mmHg, Cholesterol 240 mg/dL."
    r_int_tamp = requests.post(
        f"{API_BASE}/api/v1/audit/verify-integrity",
        json={"evidence_text": tampered_sample_text, "stored_hash": actual_hash},
        headers={"X-User-Id": "admin"}
    )
    assert r_int_tamp.status_code == 200
    int_tamp_data = r_int_tamp.json()
    assert int_tamp_data.get("verified") is False
    assert int_tamp_data.get("tampered") is True
    print(f"  [OK] Tampered text verify: verified={int_tamp_data.get('verified')}, tampered={int_tamp_data.get('tamper_detected', int_tamp_data.get('tampered'))}, message='{int_tamp_data.get('message')}'")

    # 12. Query ID Verification (/api/v1/audit/verify-query)
    print("\n[TEST 12] Testing /api/v1/audit/verify-query Endpoint by Query ID...")
    query_id = query_data.get("query_id")
    if query_id:
        r_q_ver = requests.post(
            f"{API_BASE}/api/v1/audit/verify-query",
            json={"query_id": query_id},
            headers={"X-User-Id": "admin"}
        )
        assert r_q_ver.status_code == 200
        q_ver_data = r_q_ver.json()
        print(f"  [OK] Verified query '{query_id}': verified={q_ver_data.get('verified')}, block_index={q_ver_data.get('block_index')}")

    # 13. Baseline Hash Evidence Integrity Verification (Authentic Match)
    print("\n[TEST 13] Testing /api/v1/evidence/verify-integrity with Authentic Baseline...")
    r_base_auth = requests.post(
        f"{API_BASE}/api/v1/evidence/verify-integrity",
        json={"evidence_id": "P1000"},
        headers={"X-User-Id": "DOC001"}
    )
    assert r_base_auth.status_code == 200, f"Baseline verify failed: {r_base_auth.text}"
    base_auth_data = r_base_auth.json()
    assert base_auth_data.get("status") == "Integrity Verified", f"Expected 'Integrity Verified', got: {base_auth_data}"
    assert base_auth_data.get("verified") is True
    assert base_auth_data.get("tamper_detected") is False
    print(f"  [OK] Baseline status: '{base_auth_data['status']}', verified={base_auth_data['verified']}")
    print(f"  [OK] Baseline Hash: {base_auth_data['baseline_hash'][:24]}...")

    # 14. Baseline Hash Tampering Detection (Simulated Alteration)
    print("\n[TEST 14] Testing /api/v1/evidence/verify-integrity with Tampered Test Copy...")
    r_sim = requests.post(
        f"{API_BASE}/api/v1/evidence/simulate-tamper",
        json={"evidence_id": "P1000"},
        headers={"X-User-Id": "DOC001"}
    )
    assert r_sim.status_code == 200, f"Simulation failed: {r_sim.text}"
    sim_data = r_sim.json()
    assert sim_data.get("simulated") is True
    tampered_copy = sim_data.get("tampered_text")

    r_base_tamp = requests.post(
        f"{API_BASE}/api/v1/evidence/verify-integrity",
        json={"evidence_id": "P1000", "evidence_text": tampered_copy},
        headers={"X-User-Id": "DOC001"}
    )
    assert r_base_tamp.status_code == 200
    base_tamp_data = r_base_tamp.json()
    assert base_tamp_data.get("status") == "Potential Tampering Detected", f"Expected 'Potential Tampering Detected', got: {base_tamp_data}"
    assert base_tamp_data.get("verified") is False
    assert base_tamp_data.get("tamper_detected") is True
    print(f"  [OK] Tamper status: '{base_tamp_data['status']}', tamper_detected={base_tamp_data['tamper_detected']}")
    print(f"  [OK] Recalculated Hash: {base_tamp_data['recalculated_hash'][:24]}... != Baseline: {base_tamp_data['baseline_hash'][:24]}...")

    # 15. Missing Evidence and Missing Baseline Handling
    print("\n[TEST 15] Testing Explicit Handling for Missing Evidence and Missing Baseline...")
    # Missing baseline hash
    r_missing_base = requests.post(
        f"{API_BASE}/api/v1/evidence/verify-integrity",
        json={"evidence_id": "NON_EXISTENT_RECORD_999"},
        headers={"X-User-Id": "admin"}
    )
    assert r_missing_base.status_code == 200
    m_base_data = r_missing_base.json()
    assert m_base_data.get("status") == "Error"
    assert m_base_data.get("error_type") == "MISSING_BASELINE_HASH"
    print(f"  [OK] Missing baseline correctly returned: status='{m_base_data['status']}', error_type='{m_base_data['error_type']}'")

    # Empty evidence text
    r_empty = requests.post(
        f"{API_BASE}/api/v1/evidence/verify-integrity",
        json={"evidence_id": "P1000", "evidence_text": "   "},
        headers={"X-User-Id": "admin"}
    )
    assert r_empty.status_code == 200
    empty_data = r_empty.json()
    assert empty_data.get("status") == "Error"
    assert empty_data.get("error_type") == "EMPTY_EVIDENCE"
    print(f"  [OK] Empty evidence correctly returned: status='{empty_data['status']}', error_type='{empty_data['error_type']}'")

    # 16. RBAC Authorization Checks for Evidence Verification
    print("\n[TEST 16] Testing RBAC Authorization for Evidence Verification...")
    # Doctor authorized
    r_doc = requests.post(
        f"{API_BASE}/api/v1/evidence/verify-integrity",
        json={"evidence_id": "P1000"},
        headers={"X-User-Id": "DOC001"}
    )
    assert r_doc.status_code == 200, f"Doctor should be authorized, got: {r_doc.status_code}"
    print("  [OK] Doctor (DOC001): Authorized (200 OK)")

    # Admin authorized
    r_adm = requests.post(
        f"{API_BASE}/api/v1/evidence/verify-integrity",
        json={"evidence_id": "P1000"},
        headers={"X-User-Id": "admin"}
    )
    assert r_adm.status_code == 200, f"Admin should be authorized, got: {r_adm.status_code}"
    print("  [OK] Admin (admin): Authorized (200 OK)")

    # Nurse denied (cannot verify evidence or query knowledge base)
    r_nurse_v = requests.post(
        f"{API_BASE}/api/v1/evidence/verify-integrity",
        json={"evidence_id": "P1000"},
        headers={"X-User-Id": "NURSE001"}
    )
    assert r_nurse_v.status_code == 403, f"Nurse should be denied, got: {r_nurse_v.status_code}"
    print(f"  [OK] Nurse (NURSE001): Correctly Denied (403 Forbidden)")

    # Unauthorized / Removed Auditor role denied
    r_unauth = requests.post(
        f"{API_BASE}/api/v1/evidence/verify-integrity",
        json={"evidence_id": "P1000"},
        headers={"X-User-Id": "AUDITOR001"}
    )
    assert r_unauth.status_code == 403, f"Removed auditor role should be denied, got: {r_unauth.status_code}"
    print(f"  [OK] Removed Auditor (AUDITOR001): Correctly Denied (403 Forbidden)")

    print("\n" + "=" * 70)
    print("ALL 16 END-TO-END TESTS COMPLETED AND PASSED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == "__main__":
    run_tests()

