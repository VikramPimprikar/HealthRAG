"""
Unit & Integration Tests: Evidence Baseline Integrity & Tampering Detection
==========================================================================
Tests:
1. Baseline hash generation and separate storage in evidence_baseline_hashes.json
2. Protection against silent overwrite of baseline hashes
3. Matching SHA-256 hash comparison -> Status: 'Integrity Verified'
4. Modified evidence text comparison -> Status: 'Potential Tampering Detected'
5. Tamper simulation on in-memory test copy (verifying original file remains intact)
6. Explicit handling of missing evidence, empty evidence, and missing baseline hashes
7. Audit log generation recording integrity check without sensitive medical text
8. RBAC authorization enforcement (Doctors/Admins allowed, Nurses denied)
"""

import sys
import json
import pytest
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR / "backend"))

from app.blockchain.evidence_verifier import (
    compute_canonical_hash,
    initialize_baseline_hashes,
    load_baseline_hashes,
    get_baseline_hash,
    register_evidence_baseline,
    verify_evidence_against_baseline,
    simulate_tampered_evidence,
    get_baseline_hash_file,
    get_data_dir
)
from app.blockchain.access_control import access_control_manager, Permission, UserRole
from app.blockchain.audit_log import audit_logger


def test_baseline_storage_separate():
    """Verify baseline hashes are stored separately from editable evidence"""
    baseline_file = get_baseline_hash_file()
    assert baseline_file.name == "evidence_baseline_hashes.json"
    assert baseline_file.name != "heart_narratives.json"
    assert baseline_file.name != "medical_knowledge.json"

    store = load_baseline_hashes()
    assert len(store) >= 920, f"Expected at least 920 baseline records, got {len(store)}"
    assert "P1000" in store
    assert "KB_001" in store


def test_baseline_silent_overwrite_prevention():
    """Verify that normal registration flow cannot silently overwrite an existing baseline hash"""
    p1000_original = get_baseline_hash("P1000")
    assert p1000_original is not None

    # Attempt to overwrite with new text without force flag
    res = register_evidence_baseline(
        evidence_id="P1000",
        text="Modified patient text that should not overwrite",
        force_overwrite=False
    )
    assert res["status"] == "existing"
    assert res["overwritten"] is False
    assert res["baseline_hash"] == p1000_original

    # Verify stored hash was preserved
    assert get_baseline_hash("P1000") == p1000_original


def test_authentic_evidence_verification():
    """Verify authentic evidence matches baseline and displays 'Integrity Verified'"""
    res = verify_evidence_against_baseline("P1000", user_id="DOC001")
    assert res["status"] == "Integrity Verified"
    assert res["verified"] is True
    assert res["tamper_detected"] is False
    assert res["evidence_id"] == "P1000"
    assert res["baseline_hash"] == res["recalculated_hash"]


def test_tampered_evidence_verification():
    """Verify modified evidence text fails verification and displays 'Potential Tampering Detected'"""
    sim = simulate_tampered_evidence("P1000")
    tampered_text = sim["tampered_text"]
    assert tampered_text != sim["original_text"]

    res = verify_evidence_against_baseline("P1000", current_evidence_text=tampered_text, user_id="DOC001")
    assert res["status"] == "Potential Tampering Detected"
    assert res["verified"] is False
    assert res["tamper_detected"] is True
    assert res["baseline_hash"] != res["recalculated_hash"]


def test_simulation_does_not_corrupt_dataset():
    """Verify that simulating tampering modifies only a test copy, leaving dataset intact"""
    data_dir = get_data_dir()
    narratives_file = data_dir / "heart_narratives.json"
    with open(narratives_file, "r", encoding="utf-8") as f:
        narratives = json.load(f)
    p1000_in_file = next(item["text"] for item in narratives if item.get("id") == "P1000")

    # Run simulation
    sim = simulate_tampered_evidence("P1000")
    assert sim["simulated"] is True
    assert "TAMPER" in sim["tampered_text"] or "195.0" in sim["tampered_text"] or "ALTERED" in sim["tampered_text"]

    # Reload dataset file and confirm it was NOT altered
    with open(narratives_file, "r", encoding="utf-8") as f:
        narratives_after = json.load(f)
    p1000_after = next(item["text"] for item in narratives_after if item.get("id") == "P1000")
    assert p1000_in_file == p1000_after, "Dataset file must NOT be modified by simulation!"


def test_missing_evidence_and_missing_baseline_handling():
    """Verify explicit error handling for missing evidence, empty evidence, and missing baseline"""
    # 1. Missing baseline
    res_no_base = verify_evidence_against_baseline("NON_EXISTENT_ID_XYZ", user_id="admin")
    assert res_no_base["status"] == "Error"
    assert res_no_base["error_type"] == "MISSING_BASELINE_HASH"
    assert "Missing baseline hash" in res_no_base["message"]

    # 2. Empty evidence ID
    res_no_id = verify_evidence_against_baseline("", user_id="admin")
    assert res_no_id["status"] == "Error"
    assert res_no_id["error_type"] == "MISSING_EVIDENCE_ID"

    # 3. Empty text
    res_empty_text = verify_evidence_against_baseline("P1000", current_evidence_text="   ", user_id="admin")
    assert res_empty_text["status"] == "Error"
    assert res_empty_text["error_type"] == "EMPTY_EVIDENCE"


def test_audit_logging_integrity_check():
    """Verify integrity check is logged in audit trail without sensitive medical text"""
    initial_count = len(audit_logger.records)
    res = verify_evidence_against_baseline("KB_001", user_id="DOC001")
    assert res["status"] == "Integrity Verified"

    assert len(audit_logger.records) == initial_count + 1
    last_record = audit_logger.records[-1]
    assert last_record.action == "EVIDENCE_INTEGRITY_CHECK"
    assert last_record.user_id == "DOC001"
    assert last_record.details.get("status") == "Integrity Verified"
    assert "..." in last_record.details.get("baseline_hash", "")
    # Ensure raw clinical narrative is NOT stored in details
    assert "Common symptoms of coronary artery disease" not in str(last_record.details)


def test_rbac_roles_doctor_nurse():
    """Verify RBAC permissions for Doctor, Nurse, and Auditor removal"""
    users = access_control_manager.list_users()
    user_roles = [u["role"] for u in users]
    user_ids = [u["user_id"] for u in users]

    # Auditor must not be present in default users
    assert "auditor" not in user_roles
    assert "auditor001" not in user_ids

    # Doctor must have QUERY_KNOWLEDGE_BASE
    assert access_control_manager.check_permission("DOC001", Permission.QUERY_KNOWLEDGE_BASE) is True
    assert access_control_manager.check_permission("D101", Permission.QUERY_KNOWLEDGE_BASE) is True

    # Nurse must NOT have QUERY_KNOWLEDGE_BASE or VIEW_AUDIT_LOGS
    assert access_control_manager.check_permission("NURSE001", Permission.QUERY_KNOWLEDGE_BASE) is False
    assert access_control_manager.check_permission("NURSE001", Permission.VIEW_AUDIT_LOGS) is False


if __name__ == "__main__":
    test_baseline_storage_separate()
    print("[PASS] test_baseline_storage_separate")
    test_baseline_silent_overwrite_prevention()
    print("[PASS] test_baseline_silent_overwrite_prevention")
    test_authentic_evidence_verification()
    print("[PASS] test_authentic_evidence_verification")
    test_tampered_evidence_verification()
    print("[PASS] test_tampered_evidence_verification")
    test_simulation_does_not_corrupt_dataset()
    print("[PASS] test_simulation_does_not_corrupt_dataset")
    test_missing_evidence_and_missing_baseline_handling()
    print("[PASS] test_missing_evidence_and_missing_baseline_handling")
    test_audit_logging_integrity_check()
    print("[PASS] test_audit_logging_integrity_check")
    test_rbac_roles_doctor_nurse()
    print("[PASS] test_rbac_roles_doctor_nurse")
    print("\nALL UNIT TESTS PASSED SUCCESSFULLY!")
