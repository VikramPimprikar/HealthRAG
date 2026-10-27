"""
TASK 3 — BLOCKCHAIN & EVIDENCE INTEGRITY 5-STEP DEMONSTRATION TEST SUITE
========================================================================
Validates the 5 required demonstration tests:
TEST 1: Original evidence + blockchain hash -> VERIFIED / NO TAMPERING
TEST 2: Change one evidence value           -> EVIDENCE TAMPERING DETECTED
TEST 3: Restore evidence                    -> VERIFIED / NO TAMPERING
TEST 4: Modify a blockchain block           -> BLOCKCHAIN TAMPERING DETECTED
TEST 5: Normal untouched blockchain         -> ALL BLOCKS VERIFIED
"""

import sys
import io
from pathlib import Path

# Fix Windows console encoding for UTF-8 checkmarks and symbols
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR / "backend"))

from app.blockchain.audit_log import Blockchain, BlockchainAuditLog, sha256_hash
from app.blockchain.evidence_verifier import compute_canonical_hash, verify_evidence_integrity


def main():
    print("=" * 80)
    print("DEMONSTRATION TEST SUITE: BLOCKCHAIN & EVIDENCE INTEGRITY VERIFICATION")
    print("=" * 80)

    # Setup isolated test blockchain and audit log instance
    test_chain = Blockchain()
    test_audit = BlockchainAuditLog(test_chain)

    # Create a realistic RAG Query Evidence Bundle and commit it to Blockchain
    query_text = "What is the cardiovascular risk profile for patient P1651?"
    original_evidence = (
        "Patient ID: P1651\n"
        "Age: 55 years\n"
        "Sex: Male\n"
        "Resting Blood Pressure: 140 mmHg\n"
        "Cholesterol: 250 mg/dL\n"
        "Chest Pain Type: Typical Angina\n"
        "Diagnosis: Moderate CAD (Class 2)"
    )
    original_evidence_hash = compute_canonical_hash(original_evidence)

    # Commit RAG query & evidence hash to Blockchain (Block #1)
    record = test_audit.add_record(
        user_id="dr_alexander",
        action="RAG_QUERY",
        data_type="clinical_evidence_bundle",
        details={
            "query": query_text,
            "patient_id": "P1651",
            "records_retrieved": 1,
            "retrieved_evidence_texts": [original_evidence]
        },
        patient_id="P1651",
        query_text=query_text,
        evidence_bundle=original_evidence
    )

    block_1 = test_chain.get_latest_block()
    print(f"\n[SETUP] Initialized Blockchain with 2 blocks (Genesis Block #0 + RAG Query Block #1)")
    print(f"  • Block #1 Hash:          {block_1.hash}")
    print(f"  • Stored Evidence Hash:    {block_1.evidence_hash}")
    print(f"  • Provenance Metadata:     {block_1.provenance}")

    # =========================================================================
    # TEST 1: Original evidence + blockchain hash -> VERIFIED
    # =========================================================================
    print("\n" + "-" * 80)
    print("TEST 1: Original evidence + blockchain stored hash")
    print("-" * 80)
    current_evidence_1 = original_evidence
    result_1 = test_chain.verify_evidence(block_index=1, evidence_text=current_evidence_1)

    print(f"  • Current Evidence:\n      {current_evidence_1.replace(chr(10), chr(10) + '      ')}")
    print(f"  • Stored Blockchain Hash:   {result_1['stored_hash']}")
    print(f"  • Recalculated SHA-256:     {result_1['current_hash']}")
    print(f"  • Result Status:            {result_1['debug']['verification_result']}")
    print(f"  • Verified:                 {result_1['verified']}")
    assert result_1["verified"] is True
    assert result_1["tampered"] is False
    print("  >>> [PASS] TEST 1 SUCCEEDED: ✅ Evidence Integrity Verified (No Tampering)")

    # =========================================================================
    # TEST 2: Change one evidence value -> EVIDENCE TAMPERING DETECTED
    # =========================================================================
    print("\n" + "-" * 80)
    print("TEST 2: Change one evidence value (Cholesterol: 250 -> 150 mg/dL)")
    print("-" * 80)
    tampered_evidence = original_evidence.replace("Cholesterol: 250 mg/dL", "Cholesterol: 150 mg/dL")
    result_2 = test_chain.verify_evidence(block_index=1, evidence_text=tampered_evidence)

    print(f"  • Altered Evidence:\n      {tampered_evidence.replace(chr(10), chr(10) + '      ')}")
    print(f"  • Stored Blockchain Hash:   {result_2['stored_hash']}")
    print(f"  • Recalculated SHA-256:     {result_2['current_hash']}")
    print(f"  • Result Status:            {result_2['debug']['verification_result']}")
    print(f"  • Tampered:                 {result_2['tampered']}")
    assert result_2["verified"] is False
    assert result_2["tampered"] is True
    print("  >>> [PASS] TEST 2 SUCCEEDED: ⚠️ Evidence Tampering Detected (Hash Mismatch)")

    # =========================================================================
    # TEST 3: Restore evidence -> VERIFIED
    # =========================================================================
    print("\n" + "-" * 80)
    print("TEST 3: Restore evidence to original (Cholesterol: 150 -> 250 mg/dL)")
    print("-" * 80)
    restored_evidence = original_evidence
    result_3 = test_chain.verify_evidence(block_index=1, evidence_text=restored_evidence)

    print(f"  • Restored Evidence:\n      {restored_evidence.replace(chr(10), chr(10) + '      ')}")
    print(f"  • Stored Blockchain Hash:   {result_3['stored_hash']}")
    print(f"  • Recalculated SHA-256:     {result_3['current_hash']}")
    print(f"  • Result Status:            {result_3['debug']['verification_result']}")
    print(f"  • Verified:                 {result_3['verified']}")
    assert result_3["verified"] is True
    assert result_3["tampered"] is False
    print("  >>> [PASS] TEST 3 SUCCEEDED: ✅ Evidence Restored & Verified (Authentic)")

    # =========================================================================
    # TEST 4: Modify a blockchain block -> BLOCKCHAIN TAMPERING DETECTED
    # =========================================================================
    print("\n" + "-" * 80)
    print("TEST 4: Modify a blockchain block (Simulate unauthorized block mutation)")
    print("-" * 80)
    # Deliberately mutate Block #1 action or timestamp without re-mining/re-hashing
    test_chain.simulate_block_tampering(block_index=1, field="action", new_value="MUTATED_ACTION")

    # A) Verify individual Block #1
    block_verify_4 = test_chain.verify_block(block_index=1)
    print(f"  • Block #1 Recalculated Hash: {block_verify_4['recalculated_block_hash']}")
    print(f"  • Block #1 Stored Hash:       {block_verify_4['stored_block_hash']}")
    print(f"  • Block Verification Message: {block_verify_4['message']}")
    assert block_verify_4["verified"] is False
    assert block_verify_4["tampered"] is True

    # B) Verify entire chain
    chain_verify_4 = test_chain.verify_chain()
    print(f"  • Chain Validity:             {chain_verify_4['is_valid']}")
    print(f"  • Chain Status Message:       {chain_verify_4.get('message')}")
    assert chain_verify_4["is_valid"] is False
    assert chain_verify_4["error_at_index"] == 1
    print("  >>> [PASS] TEST 4 SUCCEEDED: ⚠️ Blockchain Tampering Detected (Block #1 Compromised)")

    # =========================================================================
    # TEST 5: Normal untouched blockchain -> ALL BLOCKS VERIFIED
    # =========================================================================
    print("\n" + "-" * 80)
    print("TEST 5: Normal untouched blockchain (Restored authentic block state)")
    print("-" * 80)
    test_chain.restore_block(block_index=1, original_action="RAG_QUERY")

    # A) Verify individual Block #1
    block_verify_5 = test_chain.verify_block(block_index=1)
    print(f"  • Block #1 Recalculated Hash: {block_verify_5['recalculated_block_hash']}")
    print(f"  • Block #1 Stored Hash:       {block_verify_5['stored_block_hash']}")
    print(f"  • Block Verification Message: {block_verify_5['message']}")
    assert block_verify_5["verified"] is True
    assert block_verify_5["tampered"] is False

    # B) Verify entire chain
    chain_verify_5 = test_chain.verify_chain()
    print(f"  • Chain Validity:             {chain_verify_5['is_valid']}")
    print(f"  • Total Blocks Verified:      {chain_verify_5['total_blocks']}")
    print(f"  • Chain Status Message:       {chain_verify_5.get('message')}")
    assert chain_verify_5["is_valid"] is True
    assert chain_verify_5["total_blocks"] == 2
    print("  >>> [PASS] TEST 5 SUCCEEDED: ✅ Blockchain Integrity Verified (All Blocks Valid & Linked)")

    print("\n" + "=" * 80)
    print("ALL 5 DEMONSTRATION TESTS EXECUTED AND PASSED WITH 100% SUCCESS!")
    print("=" * 80)


if __name__ == "__main__":
    main()
