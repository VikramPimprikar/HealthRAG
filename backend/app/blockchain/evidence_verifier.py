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
