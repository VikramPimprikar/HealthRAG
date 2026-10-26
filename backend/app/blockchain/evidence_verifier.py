"""
Evidence Integrity & Tampering Detection Engine
================================================
Cryptographic verification module for RAGChainMed.
Guarantees deterministic canonicalization of medical evidence, recalculates SHA-256
hashes, and detects any data corruption or tampering against the immutable blockchain audit log.

Verification Flow:
CURRENT EVIDENCE -> CANONICALIZATION -> SHA-256 -> CURRENT HASH
                                                       ↓
                                     compare with STORED ON-CHAIN HASH
                                                       ↓
                                       VERIFIED (MATCH) or TAMPER DETECTED (MISMATCH)

Author: RAGChainMed
"""

import hashlib
import json
import unicodedata
from typing import Any, Dict, List, Optional, Union


def canonicalize_evidence(evidence: Any) -> str:
    """
    Deterministically canonicalize evidence content before hashing.
    Normalizes:
    - Line endings (CRLF -> LF)
    - Unicode representations (NFC)
    - Trailing whitespace on individual lines
    - Leading/trailing document whitespace
    - Structured JSON keys and compact separators
    - Lists of evidence text chunks (joined with standard separator '\n---\n')
    """
    if evidence is None:
        return ""

    if isinstance(evidence, str):
        # Normalize Unicode
        text = unicodedata.normalize("NFC", evidence)
        # Normalize line endings
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # Check if the string is formatted JSON (e.g. dict or list)
        stripped = text.strip()
        if (stripped.startswith("{") and stripped.endswith("}")) or (stripped.startswith("[") and stripped.endswith("]")):
            try:
                parsed = json.loads(stripped)
                if isinstance(parsed, (dict, list)):
                    return canonicalize_evidence(parsed)
            except Exception:
                pass

        # Strip trailing whitespace on each line
        lines = [line.rstrip() for line in text.split("\n")]
        # Rejoin and strip outer leading/trailing whitespace
        return "\n".join(lines).strip()

    if isinstance(evidence, list):
        if not evidence:
            return ""
        # If list of strings
        if all(isinstance(item, str) for item in evidence):
            canonical_items = [canonicalize_evidence(item) for item in evidence]
            return "\n---\n".join(canonical_items)
        # If list of dictionaries (e.g. retrieved evidence objects with "text" or "content")
        if all(isinstance(item, dict) for item in evidence):
            texts = []
            for item in evidence:
                if "text" in item:
                    texts.append(canonicalize_evidence(item["text"]))
                elif "content" in item:
                    texts.append(canonicalize_evidence(item["content"]))
                else:
                    texts.append(json.dumps(item, sort_keys=True, separators=(",", ":"), ensure_ascii=False))
            return "\n---\n".join(texts)
        # General list serialization
        return json.dumps(evidence, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

    if isinstance(evidence, dict):
        return json.dumps(evidence, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

    return str(evidence).strip()


def compute_canonical_hash(evidence: Any) -> str:
    """
    Compute cryptographic SHA-256 hash across canonicalized evidence text.
    """
    canonical_text = canonicalize_evidence(evidence)
    if not canonical_text:
        return ""
    return hashlib.sha256(canonical_text.encode("utf-8")).hexdigest()


def verify_evidence_integrity(
    evidence: Any,
    stored_hash: str
) -> Dict[str, Any]:
    """
    Core integrity verification function:
    1. Canonicalizes current evidence.
    2. Recalculates SHA-256 hash from CURRENT evidence.
    3. Compares newly calculated hash with originally stored hash.
    4. Returns verified / tampered status with detailed debug output.
    """
    clean_stored = (stored_hash or "").strip().lower()
    current_hash = compute_canonical_hash(evidence)

    if not clean_stored:
        return {
            "verified": False,
            "tampered": True,
            "stored_hash": "",
            "current_hash": current_hash,
            "stored_evidence_hash": "",
            "computed_evidence_hash": current_hash,
            "message": "Evidence tampering detected: No stored hash provided for comparison.",
            "debug": {
                "stored_evidence_hash": "",
                "recalculated_current_hash": current_hash,
                "verification_result": "TAMPERING DETECTED (Missing Stored Hash)"
            }
        }

    if not current_hash:
        return {
            "verified": False,
            "tampered": True,
            "stored_hash": clean_stored,
            "current_hash": "",
            "stored_evidence_hash": clean_stored,
            "computed_evidence_hash": "",
            "message": "Evidence verification failed: Current evidence text is empty.",
            "debug": {
                "stored_evidence_hash": clean_stored,
                "recalculated_current_hash": "",
                "verification_result": "TAMPERING DETECTED (Empty Evidence)"
            }
        }

    is_verified = (current_hash.lower() == clean_stored) and bool(current_hash) and bool(clean_stored)
    result_status = "VERIFIED / NO TAMPERING" if is_verified else "TAMPERING DETECTED"

    return {
        "verified": is_verified,
        "tampered": not is_verified,
        "stored_hash": clean_stored,
        "current_hash": current_hash,
        "stored_evidence_hash": clean_stored,
        "computed_evidence_hash": current_hash,
        "message": (
            "Evidence integrity verified. No tampering detected."
            if is_verified
            else "Evidence tampering detected. Hash mismatch."
        ),
        "debug": {
            "stored_evidence_hash": clean_stored,
            "recalculated_current_hash": current_hash,
            "verification_result": result_status
        }
    }


def verify_query_evidence(
    identifier: Union[int, str],
    current_evidence: Optional[Any] = None,
    blockchain_instance: Optional[Any] = None,
    audit_logger_instance: Optional[Any] = None,
    stored_hash_override: Optional[str] = None
) -> Dict[str, Any]:
    """
    Verify evidence associated with a specific query ID or blockchain block index.
    Retrieves the immutable recorded hash from the blockchain and verifies current evidence against it.
    """
    from app.blockchain.audit_log import blockchain as default_blockchain, audit_logger as default_logger

    chain = blockchain_instance or default_blockchain
    logger = audit_logger_instance or default_logger

    target_block = None
    target_record = None

    # Try numeric block index
    try:
        idx = int(identifier)
        if 0 <= idx < len(chain.chain):
            target_block = chain.chain[idx]
    except (ValueError, TypeError):
        pass

    # Try searching audit records by query_id or patient_id
    id_str = str(identifier).strip()
    if logger and logger.records:
        for r in logger.records:
            rec_qid = getattr(r, "query_id", "") or (r.details.get("query_id", "") if isinstance(r.details, dict) else "")
            if rec_qid and rec_qid.lower() == id_str.lower():
                target_record = r
                if r.block_index is not None and 0 <= r.block_index < len(chain.chain):
                    target_block = chain.chain[r.block_index]
                break

    # If block not found yet, check block provenance for query_id
    if not target_block:
        for b in chain.chain:
            b_qid = b.provenance.get("query_id", "") if isinstance(b.provenance, dict) else ""
            if b_qid and b_qid.lower() == id_str.lower():
                target_block = b
                break

    if not target_block and not target_record:
        return {
            "verified": False,
            "tampered": True,
            "stored_hash": "",
            "current_hash": "",
            "stored_evidence_hash": "",
            "computed_evidence_hash": "",
            "message": f"Verification failed: No audit block or query record found for '{identifier}'.",
            "debug": {
                "stored_evidence_hash": "",
                "recalculated_current_hash": "",
                "verification_result": "FAILED (Record Not Found)"
            }
        }

    # Extract stored evidence hash from immutable block (or record)
    on_chain_hash = (target_block.evidence_hash if target_block else target_record.evidence_hash) or ""
    on_chain_hash = on_chain_hash.strip().lower()

    target_stored_hash = (stored_hash_override or on_chain_hash).strip().lower()

    if not target_stored_hash:
        return {
            "verified": False,
            "tampered": True,
            "block_index": target_block.index if target_block else None,
            "stored_hash": "",
            "current_hash": "",
            "stored_evidence_hash": "",
            "computed_evidence_hash": "",
            "message": "Block does not contain an evidence hash.",
            "debug": {
                "stored_evidence_hash": "",
                "recalculated_current_hash": "",
                "verification_result": "FAILED (No Stored Hash)"
            }
        }

    # Determine evidence content to evaluate
    if current_evidence is not None and (not isinstance(current_evidence, str) or current_evidence.strip() != ""):
        eval_evidence = current_evidence
    elif target_record and isinstance(target_record.details, dict) and "retrieved_evidence_texts" in target_record.details:
        eval_evidence = target_record.details["retrieved_evidence_texts"]
    elif target_record and isinstance(target_record.details, dict) and "retrieved_texts" in target_record.details:
        eval_evidence = target_record.details["retrieved_texts"]
    elif target_block and isinstance(target_block.provenance, dict) and "evidence_text" in target_block.provenance:
        eval_evidence = target_block.provenance["evidence_text"]
    else:
        eval_evidence = None

    if eval_evidence is not None:
        integrity_res = verify_evidence_integrity(eval_evidence, target_stored_hash)
        current_hash = integrity_res["current_hash"]
        is_verified = integrity_res["verified"]
        # If user specified a stored_hash_override that does not match on_chain_hash, mark as tampered
        if on_chain_hash and target_stored_hash != on_chain_hash:
            is_verified = False
    else:
        # No evidence content supplied; cannot recalculate without content
        current_hash = ""
        is_verified = False

    chain_status = chain.verify_chain()
    result_status = "VERIFIED / NO TAMPERING" if is_verified else "TAMPERING DETECTED"

    result = {
        "verified": is_verified,
        "tampered": not is_verified,
        "stored_hash": target_stored_hash,
        "current_hash": current_hash,
        "stored_evidence_hash": target_stored_hash,
        "computed_evidence_hash": current_hash,
        "on_chain_evidence_hash": on_chain_hash,
        "message": (
            "Evidence integrity verified. No tampering detected."
            if is_verified
            else "Evidence tampering detected. Hash mismatch."
        ),
        "debug": {
            "stored_evidence_hash": target_stored_hash,
            "recalculated_current_hash": current_hash,
            "verification_result": result_status
        },
        "block_index": target_block.index if target_block else (target_record.block_index if target_record else None),
        "query_id": (
            target_block.provenance.get("query_id")
            if (target_block and isinstance(target_block.provenance, dict))
            else (getattr(target_record, "query_id", "") if target_record else "")
        ),
        "chain_valid": chain_status.get("is_valid", True),
        "block_timestamp": target_block.timestamp if target_block else getattr(target_record, "timestamp", ""),
        "user_id": target_block.user_id if target_block else getattr(target_record, "user_id", ""),
        "action": target_block.action if target_block else getattr(target_record, "action", ""),
        "status": target_block.status if target_block else getattr(target_record, "status", "")
    }

    return result


# ============================================================
# BASELINE HASH REPOSITORY & EVIDENCE INTEGRITY VERIFICATION
# ============================================================

import datetime
import re
from pathlib import Path

_baseline_cache: Optional[Dict[str, Any]] = None


def get_data_dir() -> Path:
    """Resolve data directory containing medical narratives and baseline hashes"""
    possible = [
        Path(__file__).resolve().parent.parent.parent.parent / "data",
        Path(__file__).resolve().parent.parent.parent / "data",
        Path.cwd() / "data",
        Path.cwd().parent / "data"
    ]
    for p in possible:
        if p.exists() and p.is_dir():
            return p
    fallback = Path(__file__).resolve().parent.parent.parent.parent / "data"
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback


def get_baseline_hash_file() -> Path:
    """Return path to baseline hash storage file (separate from editable evidence)"""
    return get_data_dir() / "evidence_baseline_hashes.json"


def save_baseline_hashes(store: Dict[str, Any]) -> None:
    """Persist baseline hashes separately to disk"""
    global _baseline_cache
    _baseline_cache = store
    target = get_baseline_hash_file()
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as f:
        json.dump(store, f, indent=2)


def initialize_baseline_hashes(force: bool = False) -> Dict[str, Any]:
    """
    Generate and store a baseline SHA-256 hash when evidence is first registered or ingested.
    The baseline hashes are stored separately from editable evidence content.
    If baseline hashes already exist on disk, force=False prevents accidental overwriting.
    """
    global _baseline_cache
    baseline_path = get_baseline_hash_file()

    if baseline_path.exists() and not force:
        try:
            with open(baseline_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                if loaded and isinstance(loaded, dict):
                    _baseline_cache = loaded
                    return loaded
        except Exception as e:
            print(f"Warning: Failed to read existing baseline hashes: {e}")

    store: Dict[str, Any] = {}
    data_dir = get_data_dir()
    now_iso = datetime.datetime.now().isoformat()

    # 1. Register Clinical Patient Narratives
    narratives_path = data_dir / "heart_narratives.json"
    if narratives_path.exists():
        try:
            with open(narratives_path, "r", encoding="utf-8") as f:
                narratives = json.load(f)
            for idx, item in enumerate(narratives):
                p_id = str(item.get("id", f"P{idx}")).strip().upper()
                text = item.get("text", "")
                h = compute_canonical_hash(text)
                store[p_id] = {
                    "evidence_id": p_id,
                    "baseline_hash": h,
                    "doc_type": "patient_record",
                    "registered_at": now_iso
                }
        except Exception as e:
            print(f"Error reading narratives for baseline hashing: {e}")

    # 2. Register Medical Reference Knowledge Base Articles
    kb_path = data_dir / "medical_knowledge.json"
    if kb_path.exists():
        try:
            with open(kb_path, "r", encoding="utf-8") as f:
                kb_data = json.load(f)
            for idx, item in enumerate(kb_data):
                kb_id = str(item.get("id", f"KB_{idx+1:03d}")).strip().upper()
                # Document in FAISS combines title and text: f"{title}: {text}"
                full_text = f"{item.get('title', '')}: {item.get('text', '')}" if item.get("title") else item.get("text", "")
                h = compute_canonical_hash(full_text)
                store[kb_id] = {
                    "evidence_id": kb_id,
                    "baseline_hash": h,
                    "doc_type": "medical_knowledge",
                    "title": item.get("title", ""),
                    "registered_at": now_iso
                }
        except Exception as e:
            print(f"Error reading medical knowledge for baseline hashing: {e}")

    save_baseline_hashes(store)
    return store


def load_baseline_hashes(force_reload: bool = False) -> Dict[str, Any]:
    """Load baseline hashes from separate store, initializing on first access if needed"""
    global _baseline_cache
    if _baseline_cache is not None and not force_reload:
        return _baseline_cache

    baseline_path = get_baseline_hash_file()
    if baseline_path.exists():
        try:
            with open(baseline_path, "r", encoding="utf-8") as f:
                _baseline_cache = json.load(f)
                return _baseline_cache
        except Exception:
            pass

    _baseline_cache = initialize_baseline_hashes(force=False)
    return _baseline_cache


def get_baseline_hash(evidence_id: str) -> Optional[str]:
    """Retrieve stored baseline SHA-256 hash for an evidence ID"""
    if not evidence_id:
        return None
    store = load_baseline_hashes()
    clean_id = evidence_id.strip().upper()
    entry = store.get(clean_id) or store.get(evidence_id.strip())
    if isinstance(entry, dict):
        return entry.get("baseline_hash")
    elif isinstance(entry, str):
        return entry
    return None


def register_evidence_baseline(
    evidence_id: str,
    text: str,
    doc_type: str = "patient_record",
    force_overwrite: bool = False
) -> Dict[str, Any]:
    """
    Register evidence content and compute baseline hash.
    Preserves existing baseline hash from silent overwrites unless force_overwrite=True.
    """
    if not evidence_id or not evidence_id.strip():
        raise ValueError("Evidence ID cannot be empty.")
    if not text or not str(text).strip():
        raise ValueError("Evidence text cannot be empty.")

    clean_id = evidence_id.strip().upper()
    store = load_baseline_hashes()

    # Guard: prevent normal editing flow from overwriting registered baseline hash
    if clean_id in store and not force_overwrite:
        return {
            "status": "existing",
            "evidence_id": clean_id,
            "baseline_hash": store[clean_id]["baseline_hash"],
            "overwritten": False,
            "message": f"Baseline hash for '{clean_id}' already exists and was preserved."
        }

    new_hash = compute_canonical_hash(text)
    store[clean_id] = {
        "evidence_id": clean_id,
        "baseline_hash": new_hash,
        "doc_type": doc_type,
        "registered_at": datetime.datetime.now().isoformat()
    }
    save_baseline_hashes(store)
    return {
        "status": "registered",
        "evidence_id": clean_id,
        "baseline_hash": new_hash,
        "overwritten": force_overwrite,
        "message": f"Baseline hash for '{clean_id}' registered successfully."
    }


def get_evidence_text_from_dataset(evidence_id: str) -> Optional[str]:
    """Look up original evidence text from active clinical dataset"""
    if not evidence_id:
        return None
    clean_id = evidence_id.strip().upper()
    data_dir = get_data_dir()

    # 1. Patient Narratives
    narratives_path = data_dir / "heart_narratives.json"
    if narratives_path.exists():
        try:
            with open(narratives_path, "r", encoding="utf-8") as f:
                for item in json.load(f):
                    if str(item.get("id", "")).strip().upper() == clean_id:
                        return item.get("text", "")
        except Exception:
            pass

    # 2. Medical Knowledge
    kb_path = data_dir / "medical_knowledge.json"
    if kb_path.exists():
        try:
            with open(kb_path, "r", encoding="utf-8") as f:
                for item in json.load(f):
                    if str(item.get("id", "")).strip().upper() == clean_id:
                        title = item.get("title", "")
                        text = item.get("text", "")
                        return f"{title}: {text}" if title else text
        except Exception:
            pass

    return None


def verify_evidence_against_baseline(
    evidence_id: str,
    current_evidence_text: Optional[str] = None,
    user_id: str = "admin"
) -> Dict[str, Any]:
    """
    Recalculates SHA-256 hash on evidence text and compares with stored baseline.
    Returns:
    - Status: 'Integrity Verified' when hashes match
    - Status: 'Potential Tampering Detected' when they differ
    Handles missing evidence, missing baseline hashes, and verification errors explicitly.
    Logs result to audit trail without unnecessary sensitive medical details.
    """
    clean_id = (evidence_id or "").strip().upper()
    now_iso = datetime.datetime.now().isoformat()

    # Handle missing evidence ID explicitly
    if not clean_id:
        return {
            "status": "Error",
            "verified": False,
            "tamper_detected": False,
            "error_type": "MISSING_EVIDENCE_ID",
            "message": "Verification error: Evidence ID must be provided.",
            "evidence_id": "",
            "baseline_hash": None,
            "recalculated_hash": None,
            "timestamp": now_iso
        }

    # Handle missing baseline hash explicitly
    baseline = get_baseline_hash(clean_id)
    if not baseline:
        return {
            "status": "Error",
            "verified": False,
            "tamper_detected": False,
            "error_type": "MISSING_BASELINE_HASH",
            "message": f"Verification error: Missing baseline hash for evidence ID '{clean_id}'.",
            "evidence_id": clean_id,
            "baseline_hash": None,
            "recalculated_hash": None,
            "timestamp": now_iso
        }

    # Resolve target evidence text
    target_text = current_evidence_text
    if target_text is None:
        target_text = get_evidence_text_from_dataset(clean_id)

    # Handle missing evidence content explicitly
    if target_text is None:
        return {
            "status": "Error",
            "verified": False,
            "tamper_detected": False,
            "error_type": "MISSING_EVIDENCE",
            "message": f"Verification error: Evidence content not found for ID '{clean_id}'.",
            "evidence_id": clean_id,
            "baseline_hash": baseline,
            "recalculated_hash": None,
            "timestamp": now_iso
        }

    if not str(target_text).strip():
        return {
            "status": "Error",
            "verified": False,
            "tamper_detected": True,
            "error_type": "EMPTY_EVIDENCE",
            "message": f"Verification error: Evidence content for '{clean_id}' is empty.",
            "evidence_id": clean_id,
            "baseline_hash": baseline,
            "recalculated_hash": "",
            "timestamp": now_iso
        }

    # Recalculate SHA-256 on current evidence text
    recalculated_hash = compute_canonical_hash(target_text)
    matches = (recalculated_hash.lower() == baseline.lower())

    status_str = "Integrity Verified" if matches else "Potential Tampering Detected"
    verified = matches
    tamper_detected = not matches
    msg = (
        f"Evidence '{clean_id}' integrity verified against baseline hash."
        if matches
        else f"Potential tampering detected in evidence '{clean_id}': recalculated SHA-256 does not match baseline hash."
    )

    # Log integrity check result to audit trail (avoid logging sensitive medical text)
    try:
        from app.blockchain.audit_log import audit_logger
        audit_logger.add_record(
            user_id=user_id,
            action="EVIDENCE_INTEGRITY_CHECK",
            data_type="Evidence Document Integrity",
            details={
                "evidence_id": clean_id,
                "status": status_str,
                "verified": verified,
                "tamper_detected": tamper_detected,
                "baseline_hash": f"{baseline[:16]}..." if baseline else "",
                "recalculated_hash": f"{recalculated_hash[:16]}..." if recalculated_hash else "",
            },
            patient_id=clean_id if clean_id.startswith("P") else None,
            status="success" if verified else "tamper_detected",
            query_text=f"Integrity check for evidence ID {clean_id}"
        )
    except Exception as log_err:
        print(f"Warning: Could not record integrity check in audit logger: {log_err}")

    return {
        "status": status_str,
        "verified": verified,
        "tamper_detected": tamper_detected,
        "evidence_id": clean_id,
        "baseline_hash": baseline,
        "recalculated_hash": recalculated_hash,
        "message": msg,
        "timestamp": now_iso
    }


def simulate_tampered_evidence(
    evidence_id: str,
    current_text: Optional[str] = None
) -> Dict[str, Any]:
    """
    Creates an in-memory tampered test copy of an evidence document for demonstration.
    Does NOT modify the underlying medical dataset files on disk.
    """
    clean_id = (evidence_id or "").strip().upper()
    text = current_text if current_text is not None else get_evidence_text_from_dataset(clean_id)
    if not text:
        text = f"Patient ID {clean_id}: Age 58, Resting BP 140.0 mmHg, Cholesterol 240.0 mg/dL."

    baseline = get_baseline_hash(clean_id)

    # Modify clinical biomarker values in the test copy
    tampered = text
    if "resting blood pressure" in tampered.lower():
        tampered = re.sub(
            r"resting blood pressure\s*[\d\.]+",
            "resting blood pressure 195.0",
            tampered,
            flags=re.IGNORECASE
        )
    elif "cholesterol level" in tampered.lower():
        tampered = re.sub(
            r"cholesterol level\s*[\d\.]+",
            "cholesterol level 390.0",
            tampered,
            flags=re.IGNORECASE
        )
    elif "symptoms" in tampered.lower():
        tampered = tampered.replace("Common symptoms", "Unusual tampered symptoms (altered for demonstration)")
    else:
        tampered = tampered + " [ALTERED_TEST_COPY]"

    if "[ALTERED TEST COPY]" not in tampered:
        tampered = tampered.rstrip() + " [ALTERED TEST COPY: FOR INTEGRITY DEMO]"

    tampered_hash = compute_canonical_hash(tampered)

    return {
        "evidence_id": clean_id,
        "original_text": text,
        "tampered_text": tampered,
        "baseline_hash": baseline,
        "tampered_hash": tampered_hash,
        "simulated": True,
        "message": "Simulated tampering copy created for demonstration (original dataset untouched)."
    }

