"""
RAGChainMed: Evidence Tampering & Integrity Verification Demonstration
======================================================================
This standalone demonstration script proves:
1. Baseline SHA-256 hash generation for a sample clinical evidence document.
2. Verification of unchanged evidence -> Status: "Integrity Verified"
3. Verification of a temporary copy with exactly ONE word modified -> Status: "Potential Tampering Detected"
4. Reuses the project's existing verification engine (app.blockchain.evidence_verifier).
5. Strictly protects the original dataset, vector database, and baseline hashes.

Usage:
    python scripts/demo_evidence_tampering.py
"""

import sys
import os
import json
import hashlib
import tempfile
from pathlib import Path

# Fix Windows console UTF-8 output if necessary
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# Add backend directory to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR / "backend"))

from app.blockchain.evidence_verifier import (
    compute_canonical_hash,
    verify_evidence_against_baseline,
    get_baseline_hash,
    get_evidence_text_from_dataset
)


def compute_file_sha256(filepath: Path) -> str:
    """Compute SHA-256 checksum of a file on disk for immutability checking"""
    if not filepath.exists():
        return "NOT_FOUND"
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def run_demo():
    print("=" * 78)
    print("      RAGCHAINMED EVIDENCE TAMPERING & INTEGRITY VERIFICATION DEMO")
    print("=" * 78)

    # -------------------------------------------------------------
    # IMMUTABILITY SAFETY CHECK: Record pre-test checksums
    # -------------------------------------------------------------
    narratives_path = BASE_DIR / "data" / "heart_narratives.json"
    baselines_path = BASE_DIR / "data" / "evidence_baseline_hashes.json"
    medical_kb_path = BASE_DIR / "data" / "medical_knowledge.json"

    pre_narratives_hash = compute_file_sha256(narratives_path)
    pre_baselines_hash = compute_file_sha256(baselines_path)
    pre_medical_kb_hash = compute_file_sha256(medical_kb_path)

    print("\n[SAFETY CHECK] Pre-test Checksums of Production Files:")
    print(f"  • heart_narratives.json:         {pre_narratives_hash[:20]}...")
    print(f"  • evidence_baseline_hashes.json: {pre_baselines_hash[:20]}...")
    print(f"  • medical_knowledge.json:        {pre_medical_kb_hash[:20]}...")

    # -------------------------------------------------------------
    # 1. SAMPLE EVIDENCE DOCUMENT SELECTION & BASELINE HASH
    # -------------------------------------------------------------
    evidence_id = "P1651"
    original_text = get_evidence_text_from_dataset(evidence_id)
    if not original_text:
        # Fallback sample clinical narrative
        original_text = (
            "Patient ID P1651, Name Ananya Patil, aged 55 years with sex value Male, "
            "chest pain type asymptomatic, resting blood pressure 140.0 mmHg, "
            "cholesterol level 0.0 mg/dL, fasting blood sugar False, rest ECG result normal, "
            "maximum heart rate 83.0, exercise induced angina False, oldpeak value 0.0, "
            "slope value flat, ca value nan, thal value reversable defect, and diagnosis outcome 2."
        )

    # Step 1: Use SHA-256 to generate / retrieve baseline hash
    print("\n" + "-" * 78)
    print(f"STEP 1: Generate SHA-256 Baseline Hash for Sample Evidence ({evidence_id})")
    print("-" * 78)
    baseline_hash = compute_canonical_hash(original_text)

    print(f"Sample Evidence ID: {evidence_id}")
    print(f"Original Content:\n  \"{original_text}\"")
    print(f"\nGenerated SHA-256 Baseline Hash:\n  {baseline_hash}")

    # -------------------------------------------------------------
    # 2. VERIFY UNCHANGED EVIDENCE -> "Integrity Verified"
    # -------------------------------------------------------------
    print("\n" + "-" * 78)
    print("STEP 2: Verify Unchanged Evidence (Reusing Existing Verification Engine)")
    print("-" * 78)

    # Reusing existing verify_evidence_against_baseline function
    unchanged_result = verify_evidence_against_baseline(
        evidence_id=evidence_id,
        current_evidence_text=original_text,
        user_id="DOC001"
    )

    recalculated_unchanged_hash = unchanged_result.get("recalculated_hash")
    status_1 = unchanged_result.get("status")
    verified_1 = unchanged_result.get("verified")

    print(f"Baseline Hash:     {baseline_hash}")
    print(f"Recalculated Hash: {recalculated_unchanged_hash}")
    print(f"Hashes Match:      {baseline_hash == recalculated_unchanged_hash}")
    print(f"Verification Message: {unchanged_result.get('message')}")
    print(f"\n>>> RESULT FOR UNCHANGED EVIDENCE: \"{status_1}\" (verified={verified_1})")

    assert status_1 == "Integrity Verified", f"Expected 'Integrity Verified', got '{status_1}'"
    assert verified_1 is True, "Unchanged evidence should verify as True"

    # -------------------------------------------------------------
    # 3. MODIFY ONE WORD IN TEMPORARY COPY -> "Potential Tampering Detected"
    # -------------------------------------------------------------
    print("\n" + "-" * 78)
    print("STEP 3: Modify ONE Word in Separate Temporary Copy & Re-Verify")
    print("-" * 78)

    # Create temporary file to ensure total isolation from production data
    with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".txt", encoding="utf-8") as temp_file:
        temp_filepath = Path(temp_file.name)

        # Modify exactly one word: 'asymptomatic' -> 'symptomatic'
        modified_text = original_text.replace("asymptomatic", "symptomatic", 1)
        temp_file.write(modified_text)

    try:
        # Read back from temporary file
        with open(temp_filepath, "r", encoding="utf-8") as f:
            temp_copy_content = f.read()

        print(f"Temporary Test File Created: {temp_filepath.name}")
        print("One-Word Modification Applied:")
        print("  • Before: \"... chest pain type [asymptomatic], resting blood pressure ...\"")
        print("  • After:  \"... chest pain type [symptomatic], resting blood pressure ...\"")

        # Verify modified temporary copy against the SAME original baseline hash
        tampered_result = verify_evidence_against_baseline(
            evidence_id=evidence_id,
            current_evidence_text=temp_copy_content,
            user_id="DOC001"
        )

        recalculated_tampered_hash = tampered_result.get("recalculated_hash")
        status_2 = tampered_result.get("status")
        verified_2 = tampered_result.get("verified")
        tamper_detected_2 = tampered_result.get("tamper_detected")

        print(f"\nOriginal Baseline Hash: {baseline_hash}")
        print(f"Recalculated Hash:      {recalculated_tampered_hash}")
        print(f"Hashes Match:           {baseline_hash == recalculated_tampered_hash}")
        print(f"Verification Message:   {tampered_result.get('message')}")
        print(f"\n>>> RESULT FOR MODIFIED COPY: \"{status_2}\" (tamper_detected={tamper_detected_2})")

        assert status_2 == "Potential Tampering Detected", f"Expected 'Potential Tampering Detected', got '{status_2}'"
        assert verified_2 is False, "Modified evidence must not be verified"
        assert tamper_detected_2 is True, "Tamper flag must be True"

    finally:
        # Clean up temporary file
        if temp_filepath.exists():
            temp_filepath.unlink()

    # -------------------------------------------------------------
    # 4. IMMUTABILITY VERIFICATION: Confirm original data untouched
    # -------------------------------------------------------------
    print("\n" + "-" * 78)
    print("STEP 4: Verify Original Dataset & Baseline Hashes are 100% Untouched")
    print("-" * 78)

    post_narratives_hash = compute_file_sha256(narratives_path)
    post_baselines_hash = compute_file_sha256(baselines_path)
    post_medical_kb_hash = compute_file_sha256(medical_kb_path)

    narratives_intact = (pre_narratives_hash == post_narratives_hash)
    baselines_intact = (pre_baselines_hash == post_baselines_hash)
    medical_kb_intact = (pre_medical_kb_hash == post_medical_kb_hash)

    print(f"  • heart_narratives.json:         {'UNTOUCHED / INTACT' if narratives_intact else 'MODIFIED'}")
    print(f"  • evidence_baseline_hashes.json: {'UNTOUCHED / INTACT' if baselines_intact else 'MODIFIED'}")
    print(f"  • medical_knowledge.json:        {'UNTOUCHED / INTACT' if medical_kb_intact else 'MODIFIED'}")

    assert narratives_intact, "CRITICAL: heart_narratives.json was modified!"
    assert baselines_intact, "CRITICAL: evidence_baseline_hashes.json was modified!"
    assert medical_kb_intact, "CRITICAL: medical_knowledge.json was modified!"

    print("\n" + "=" * 78)
    print("DEMO SUMMARY:")
    print(f"  1. Baseline Hash Generated:       {baseline_hash}")
    print(f"  2. Unchanged Verification:        \"{status_1}\"")
    print(f"  3. One-Word Modified Copy:        \"{status_2}\"")
    print("  4. Original Data & Hashes:        100% UNTOUCHED (Cryptographically Verified)")
    print("=" * 78)


if __name__ == "__main__":
    run_demo()
