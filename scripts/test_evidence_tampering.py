"""
RAGChainMed Evidence Tampering Detection Test Suite
===================================================
Demonstrates and validates cryptographic evidence verification and tampering detection:

TEST 1: Original evidence -> generate hash -> verify -> PASS (VERIFIED / NO TAMPERING)
TEST 2: Modify single clinical value in evidence -> recalculate hash -> verify -> TAMPERING DETECTED
TEST 3: Modify stored hash input with authentic evidence -> verify -> TAMPERING DETECTED
TEST 4: Canonicalization invariance (CRLF vs LF, trailing whitespace) -> NO false mismatches
TEST 5: Evidence bundle list verification & single-item tampering detection
TEST 6: Structured JSON clinical object verification & tampering detection
TEST 7: Blockchain on-chain immutable hash reference verification & tampering detection
TEST 8: Blockchain verify_evidence with modified evidence text -> TAMPERING DETECTED
TEST 9: Blockchain verify_evidence with manually modified stored hash -> TAMPERING DETECTED
TEST 10: Debug diagnostics verification (Stored Hash, Recalculated Hash, Result)

Usage:
    python scripts/test_evidence_tampering.py
"""

import sys
from pathlib import Path

# Add backend directory to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = BASE_DIR / "backend"
sys.path.append(str(BACKEND_DIR))

from app.blockchain.evidence_verifier import (
    canonicalize_evidence,
    compute_canonical_hash,
    verify_evidence_integrity,
    verify_query_evidence
)
from app.blockchain.audit_log import Blockchain, BlockchainAuditLog


def run_tests():
    print("=" * 80)
    print("RAGCHAINMED: EVIDENCE INTEGRITY & TAMPERING DETECTION TEST SUITE")
    print("=" * 80)

    # -------------------------------------------------------------
    # SAMPLE CLINICAL EVIDENCE
    # -------------------------------------------------------------
    original_evidence = (
        "### Patient Record: P1651\n"
        "- Age: 58 | Sex: Male\n"
        "- Chest Pain: Typical Angina (Type 1)\n"
        "- Resting BP: 140.0 mmHg | Cholesterol: 240.0 mg/dL\n"
        "- Fasting Blood Sugar: Normal (<=120 mg/dL)\n"
        "- Resting ECG: Left Ventricular Hypertrophy (LVH) | Max Heart Rate: 160.0 bpm\n"
        "- Exercise Induced Angina: No (Absent)\n"
        "- ST Depression (oldpeak): 1.8 mm | Slope: Flat\n"
        "- Diagnosis: Mild CAD (Class 1)"
    )

    # -------------------------------------------------------------
    # TEST 1: Authentic Evidence Integrity Verification
    # -------------------------------------------------------------
    print("\n[TEST 1] Authentic Evidence -> Generate SHA-256 Hash -> Verify Integrity")
    stored_hash = compute_canonical_hash(original_evidence)
    print(f"  • Computed Original SHA-256 Hash: {stored_hash}")

    result_1 = verify_evidence_integrity(original_evidence, stored_hash)
    print(f"  • Verification Output: {result_1}")

    assert result_1["verified"] is True, "Test 1 Failed: Authentic evidence should be verified!"
    assert result_1["tampered"] is False, "Test 1 Failed: Tampered flag should be False!"
    assert result_1["stored_hash"] == stored_hash
    assert result_1["current_hash"] == stored_hash
    assert result_1["debug"]["stored_evidence_hash"] == stored_hash
    assert result_1["debug"]["recalculated_current_hash"] == stored_hash
    assert "VERIFIED" in result_1["debug"]["verification_result"]
    assert "verified" in result_1["message"].lower()
    print("  >>> [PASS] TEST 1 SUCCEEDED: Authentic evidence verified with zero tampering.\n")

    # -------------------------------------------------------------
    # TEST 2: Tampered Evidence Detection (Altering Clinical Values)
    # -------------------------------------------------------------
    print("[TEST 2] Tampered Evidence Detection (Modifying One Clinical Value)")
    # Case 2A: Modifying Resting BP from 140.0 to 185.0 mmHg
    tampered_evidence_bp = original_evidence.replace("Resting BP: 140.0 mmHg", "Resting BP: 185.0 mmHg")
    result_2a = verify_evidence_integrity(tampered_evidence_bp, stored_hash)

    print(f"  • Altered Evidence Value: Resting BP changed 140.0 -> 185.0 mmHg")
    print(f"  • Stored Hash:            {result_2a['stored_hash']}")
    print(f"  • Recalculated Hash:      {result_2a['current_hash']}")
    print(f"  • Output: {result_2a}")

    assert result_2a["verified"] is False, "Test 2A Failed: Tampered evidence must NOT be verified!"
    assert result_2a["tampered"] is True, "Test 2A Failed: Tampered flag must be True!"
    assert result_2a["current_hash"] != stored_hash, "Test 2A Failed: Recalculated hash must differ!"
    assert result_2a["debug"]["verification_result"] == "TAMPERING DETECTED"
    assert "tampering detected" in result_2a["message"].lower()
    print("  >>> [PASS] TEST 2A SUCCEEDED: Altered blood pressure value correctly flagged as TAMPERING DETECTED.")

    # Case 2B: Modifying single character in diagnosis
    tampered_evidence_diag = original_evidence.replace("Mild CAD (Class 1)", "Severe CAD (Class 3)")
    result_2b = verify_evidence_integrity(tampered_evidence_diag, stored_hash)
    assert result_2b["verified"] is False
    assert result_2b["tampered"] is True
    print("  >>> [PASS] TEST 2B SUCCEEDED: Altered diagnosis category correctly flagged as TAMPERING DETECTED.\n")

    # -------------------------------------------------------------
    # TEST 3: Manually Modified Stored Hash with Authentic Evidence
    # -------------------------------------------------------------
    print("[TEST 3] Manually Modified Stored Hash with Authentic Evidence")
    fake_stored_hash = "ffff" + stored_hash[4:]
    result_3 = verify_evidence_integrity(original_evidence, fake_stored_hash)
    print(f"  • Fake Stored Hash:       {result_3['stored_hash']}")
    print(f"  • Recalculated Hash:      {result_3['current_hash']}")
    print(f"  • Verification Output:    {result_3}")

    assert result_3["verified"] is False, "Test 3 Failed: Modified stored hash must fail verification!"
    assert result_3["tampered"] is True
    assert result_3["current_hash"] == stored_hash, "Test 3 Failed: Recalculated hash must match authentic evidence!"
    assert result_3["stored_hash"] == fake_stored_hash
    assert result_3["debug"]["verification_result"] == "TAMPERING DETECTED"
    print("  >>> [PASS] TEST 3 SUCCEEDED: Manually modified stored hash correctly flagged as TAMPERING DETECTED.\n")

    # -------------------------------------------------------------
    # TEST 4: Canonicalization Invariance (No False Mismatches)
    # -------------------------------------------------------------
    print("[TEST 4] Canonicalization Invariance (Harmless Whitespace & Line Endings)")
    # Harmless variations: Windows CRLF vs Unix LF, trailing spaces on lines, surrounding newlines
    whitespace_variant = "\r\n  " + original_evidence.replace("\n", "   \r\n") + "   \r\n\r\n"
    canonical_hash_variant = compute_canonical_hash(whitespace_variant)

    result_4 = verify_evidence_integrity(whitespace_variant, stored_hash)
    print(f"  • Original Hash:        {stored_hash}")
    print(f"  • Whitespace Var Hash:  {canonical_hash_variant}")
    print(f"  • Verification Output:  {result_4}")

    assert canonical_hash_variant == stored_hash, "Test 4 Failed: Canonicalization must produce identical hash!"
    assert result_4["verified"] is True, "Test 4 Failed: Harmless formatting must NOT cause false tampering alert!"
    print("  >>> [PASS] TEST 4 SUCCEEDED: Canonicalization prevents false mismatches from formatting differences.\n")

    # -------------------------------------------------------------
    # TEST 5: Evidence Bundle (List of Retrieved Chunks)
    # -------------------------------------------------------------
    print("[TEST 5] Evidence Bundle List Verification & Single Chunk Tampering")
    bundle_evidence = [
        "Patient ID P1001: Age 63, Male, Resting BP 145 mmHg, Cholesterol 233 mg/dL.",
        "Patient ID P1002: Age 67, Female, Resting BP 160 mmHg, Cholesterol 286 mg/dL.",
        "Verified Medical Knowledge [Angina]: Typical angina is precipitated by physical exertion."
    ]

    bundle_stored_hash = compute_canonical_hash(bundle_evidence)
    print(f"  • Bundle Stored SHA-256 Hash: {bundle_stored_hash}")

    # Authentic bundle
    result_5_auth = verify_evidence_integrity(bundle_evidence, bundle_stored_hash)
    assert result_5_auth["verified"] is True
    assert result_5_auth["tampered"] is False
    print("  • Authentic bundle check: VERIFIED (PASS)")

    # Tampered bundle (modify single number in second chunk)
    tampered_bundle = [
        bundle_evidence[0],
        "Patient ID P1002: Age 67, Female, Resting BP 199 mmHg, Cholesterol 286 mg/dL.", # changed 160 -> 199
        bundle_evidence[2]
    ]
    result_5_tamp = verify_evidence_integrity(tampered_bundle, bundle_stored_hash)
    assert result_5_tamp["verified"] is False
    assert result_5_tamp["tampered"] is True
    print(f"  • Tampered bundle recalculated hash: {result_5_tamp['current_hash']}")
    print(f"  • Tampered bundle status: {result_5_tamp['message']}")
    print("  >>> [PASS] TEST 5 SUCCEEDED: Multi-chunk evidence bundle correctly verified and tampering detected.\n")

    # -------------------------------------------------------------
    # TEST 6: Structured JSON Clinical Object Verification
    # -------------------------------------------------------------
    print("[TEST 6] Structured Clinical Object / JSON Verification")
    clinical_obj = {
        "patient_id": "P1651",
        "age": 58,
        "trestbps": 140.0,
        "chol": 240.0,
        "severity": "Mild CAD (Class 1)"
    }
    json_stored_hash = compute_canonical_hash(clinical_obj)
    
    # Authentic object check
    result_6_auth = verify_evidence_integrity(clinical_obj, json_stored_hash)
    assert result_6_auth["verified"] is True
    assert result_6_auth["tampered"] is False

    # Check JSON formatted string equivalent
    json_str = '{\n  "severity": "Mild CAD (Class 1)",\n  "chol": 240.0,\n  "trestbps": 140.0,\n  "age": 58,\n  "patient_id": "P1651"\n}'
    result_6_str = verify_evidence_integrity(json_str, json_stored_hash)
    assert result_6_str["verified"] is True

    # Tampered JSON
    tampered_json_obj = dict(clinical_obj)
    tampered_json_obj["trestbps"] = 190.0
    result_6_tamp = verify_evidence_integrity(tampered_json_obj, json_stored_hash)
    assert result_6_tamp["verified"] is False
    assert result_6_tamp["tampered"] is True
    print("  >>> [PASS] TEST 6 SUCCEEDED: Structured clinical JSON verified and altered fields caught.\n")

    # -------------------------------------------------------------
    # TEST 7: Blockchain On-Chain Immutable Hash Reference
    # -------------------------------------------------------------
    print("[TEST 7] Blockchain On-Chain Immutable Hash Reference Verification")
    test_chain = Blockchain()
    test_logger = BlockchainAuditLog(test_chain)

    # Commit a RAG query record to blockchain
    audit_rec = test_logger.add_record(
        user_id="DOC001",
        action="RAG_QUERY",
        data_type="Healthcare Records",
        details={
            "query": "Patients with angina and high blood pressure",
            "records_retrieved": 3,
            "retrieved_evidence_texts": bundle_evidence
        },
        patient_id="P1001",
        status="success",
        query_text="Patients with angina and high blood pressure",
        evidence_bundle=bundle_evidence,
        query_id="QRY-TEST-2026"
    )

    block_idx = audit_rec.block_index
    on_chain_hash = audit_rec.evidence_hash
    print(f"  • Committed to Blockchain Block #{block_idx} with On-Chain Evidence Hash: {on_chain_hash}")

    # Verify authentic evidence against on-chain block index
    chain_ver_auth = test_logger.verify_query_evidence(
        identifier=block_idx,
        current_evidence=bundle_evidence
    )
    assert chain_ver_auth["verified"] is True
    assert chain_ver_auth["tampered"] is False

    # Verify tampered evidence against on-chain block index
    chain_ver_tamp = test_logger.verify_query_evidence(
        identifier=block_idx,
        current_evidence=tampered_bundle
    )
    assert chain_ver_tamp["verified"] is False
    assert chain_ver_tamp["tampered"] is True

    # Verify query by query_id ("QRY-TEST-2026")
    qid_ver = test_logger.verify_query_evidence(
        identifier="QRY-TEST-2026",
        current_evidence=bundle_evidence
    )
    assert qid_ver["verified"] is True
    assert qid_ver["query_id"] == "QRY-TEST-2026"
    print("  >>> [PASS] TEST 7 SUCCEEDED: Blockchain immutable reference successfully verifies authentic evidence and catches tampering.\n")

    # -------------------------------------------------------------
    # TEST 8: Blockchain verify_evidence with Modified Evidence Text
    # -------------------------------------------------------------
    print("[TEST 8] Blockchain verify_evidence with Modified Evidence Text")
    # Call verify_evidence on blockchain instance directly (simulating API / UI call)
    block_ver_tamp = test_chain.verify_evidence(
        block_index=block_idx,
        evidence_text=tampered_bundle,
        stored_hash=on_chain_hash
    )
    print(f"  • Block Verify (Tampered Evidence Text): verified={block_ver_tamp['verified']}, current_hash={block_ver_tamp['current_hash']}")
    assert block_ver_tamp["verified"] is False, "Test 8 Failed: Altered evidence text must trigger TAMPERING DETECTED!"
    assert block_ver_tamp["tampered"] is True
    assert block_ver_tamp["current_hash"] != on_chain_hash
    assert block_ver_tamp["debug"]["verification_result"] == "TAMPERING DETECTED"

    # Call verify_evidence with authentic evidence text
    block_ver_auth = test_chain.verify_evidence(
        block_index=block_idx,
        evidence_text=bundle_evidence,
        stored_hash=on_chain_hash
    )
    assert block_ver_auth["verified"] is True
    assert block_ver_auth["tampered"] is False
    assert block_ver_auth["current_hash"] == on_chain_hash
    print("  >>> [PASS] TEST 8 SUCCEEDED: Blockchain verify_evidence correctly recalculates hash from evidence text.\n")

    # -------------------------------------------------------------
    # TEST 9: Blockchain verify_evidence with Altered Stored Hash
    # -------------------------------------------------------------
    print("[TEST 9] Blockchain verify_evidence with Manually Altered Stored Hash")
    altered_stored_hash = "deadbeef" + on_chain_hash[8:]
    block_ver_altered_hash = test_chain.verify_evidence(
        block_index=block_idx,
        evidence_text=bundle_evidence,
        stored_hash=altered_stored_hash
    )
    print(f"  • Block Verify (Altered Stored Hash): verified={block_ver_altered_hash['verified']}, stored_hash={block_ver_altered_hash['stored_hash']}")
    assert block_ver_altered_hash["verified"] is False
    assert block_ver_altered_hash["tampered"] is True
    assert block_ver_altered_hash["debug"]["verification_result"] == "TAMPERING DETECTED"
    print("  >>> [PASS] TEST 9 SUCCEEDED: Blockchain verify_evidence catches manual stored hash tampering.\n")

    # -------------------------------------------------------------
    # TEST 10: Debug Diagnostics Verification
    # -------------------------------------------------------------
    print("[TEST 10] Debug Diagnostics Verification")
    res_debug = verify_evidence_integrity(original_evidence, stored_hash)
    assert "debug" in res_debug
    assert "stored_evidence_hash" in res_debug["debug"]
    assert "recalculated_current_hash" in res_debug["debug"]
    assert "verification_result" in res_debug["debug"]
    assert res_debug["debug"]["stored_evidence_hash"] == stored_hash
    assert res_debug["debug"]["recalculated_current_hash"] == stored_hash
    assert res_debug["debug"]["verification_result"] == "VERIFIED / NO TAMPERING"
    print(f"  • Debug Output Dictionary: {res_debug['debug']}")
    print("  >>> [PASS] TEST 10 SUCCEEDED: Complete debug diagnostics present in verification output.\n")

    print("=" * 80)
    print("ALL 10 EVIDENCE TAMPERING DETECTION TESTS PASSED (100% SUCCESS RATE)!")
    print("=" * 80)


if __name__ == "__main__":
    run_tests()
