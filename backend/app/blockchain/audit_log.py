"""
Blockchain-Based Audit Logging for RAGChainMed
===============================================
Implements an immutable, cryptographically-verifiable audit log using SHA-256.
Maintains tamper-evident records of queries, predictions, access control decisions,
and retrieved evidence hashes without storing sensitive raw medical records on-chain.

Author: RAGChainMed
"""

import hashlib
import json
import uuid
import datetime
from typing import List, Dict, Optional, Any, Union

from app.blockchain.evidence_verifier import (
    canonicalize_evidence,
    compute_canonical_hash,
    verify_evidence_integrity as _verify_evidence_integrity,
    verify_query_evidence as _verify_query_evidence
)


def sha256_hash(content: Any) -> str:
    """Compute SHA-256 hash using deterministic canonicalization"""
    if content is None:
        return ""
    return compute_canonical_hash(content)


# ============================================================
# BLOCK CLASS
# ============================================================

class Block:
    """Represents a single immutable block in the audit blockchain"""

    def __init__(
        self,
        index: int,
        timestamp: str,
        user_id: str,
        action: str,
        status: str,
        previous_hash: str,
        query_hash: str = "",
        evidence_hash: str = "",
        provenance: Optional[Dict[str, Any]] = None
    ):
        self.index = index
        self.timestamp = timestamp
        self.user_id = user_id
        self.action = action
        self.status = status
        self.previous_hash = previous_hash
        self.query_hash = query_hash or ""
        self.evidence_hash = evidence_hash or ""
        self.provenance = provenance or {}
        self.hash = self.calculate_hash()

    def calculate_hash(self) -> str:
        """Compute SHA-256 hash across all block fields"""
        provenance_str = json.dumps(self.provenance, sort_keys=True, separators=(",", ":"))
        block_data = (
            f"{self.index}|"
            f"{self.timestamp}|"
            f"{self.user_id}|"
            f"{self.action}|"
            f"{self.status}|"
            f"{self.query_hash}|"
            f"{self.evidence_hash}|"
            f"{provenance_str}|"
            f"{self.previous_hash}"
        )
        return hashlib.sha256(block_data.encode("utf-8")).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        """Convert block to dictionary representation"""
        return {
            "index": self.index,
            "timestamp": self.timestamp,
            "user_id": self.user_id,
            "action": self.action,
            "status": self.status,
            "query_hash": self.query_hash,
            "evidence_hash": self.evidence_hash,
            "provenance": self.provenance,
            "previous_hash": self.previous_hash,
            "hash": self.hash
        }


# ============================================================
# BLOCKCHAIN CLASS
# ============================================================

class Blockchain:
    """Manages the append-only audit chain with cryptographic integrity checks"""

    def __init__(self):
        self.chain: List[Block] = []
        self.create_genesis_block()

    def create_genesis_block(self):
        """Create the genesis (first) block in the blockchain"""
        genesis_block = Block(
            index=0,
            timestamp=datetime.datetime.now().isoformat(),
            user_id="SYSTEM",
            action="GENESIS_BLOCK",
            status="SUCCESS",
            previous_hash="0" * 64,
            query_hash="",
            evidence_hash="",
            provenance={"system": "RAGChainMed Healthcare Audit Subsystem", "version": "2.0"}
        )
        self.chain.append(genesis_block)

    def get_latest_block(self) -> Block:
        """Return the most recent block in the chain"""
        return self.chain[-1]

    def add_block(
        self,
        user_id: str,
        action: str,
        status: str,
        query_hash: str = "",
        evidence_hash: str = "",
        provenance: Optional[Dict[str, Any]] = None
    ) -> Block:
        """Append a new block with computed cryptographic hashes"""
        latest_block = self.get_latest_block()

        new_block = Block(
            index=latest_block.index + 1,
            timestamp=datetime.datetime.now().isoformat(),
            user_id=user_id,
            action=action,
            status=status,
            previous_hash=latest_block.hash,
            query_hash=query_hash,
            evidence_hash=evidence_hash,
            provenance=provenance or {}
        )

        self.chain.append(new_block)
        return new_block

    def verify_chain(self) -> Dict[str, Any]:
        """Verify complete blockchain mathematical integrity"""
        for i in range(1, len(self.chain)):
            current_block = self.chain[i]
            previous_block = self.chain[i - 1]

            # 1. Verify that current block's hash matches its calculated content
            if current_block.hash != current_block.calculate_hash():
                return {
                    "is_valid": False,
                    "error_at_index": current_block.index,
                    "reason": f"⚠️ Blockchain integrity compromised. Block #{current_block.index} hash mismatch.",
                    "message": f"⚠️ Blockchain integrity compromised. Block #{current_block.index} has been modified."
                }

            # 2. Verify that previous_hash matches previous block's hash
            if current_block.previous_hash != previous_block.hash:
                return {
                    "is_valid": False,
                    "error_at_index": current_block.index,
                    "reason": f"⚠️ Blockchain integrity compromised. Chain linkage broken at block #{current_block.index}.",
                    "message": f"⚠️ Blockchain integrity compromised. Block #{current_block.index} linkage broken."
                }

        return {
            "is_valid": True,
            "total_blocks": len(self.chain),
            "latest_block_hash": self.chain[-1].hash,
            "message": "✅ Blockchain integrity verified. All blocks are valid and correctly linked."
        }

    def verify_block(self, block_index: int) -> Dict[str, Any]:
        """
        Verify cryptographic integrity of an individual block.
        Reconstructs the block data, calculates its SHA-256 hash again,
        and compares with the stored block.hash.
        """
        if block_index < 0 or block_index >= len(self.chain):
            return {
                "verified": False,
                "tampered": True,
                "block_index": block_index,
                "message": f"⚠️ Block index {block_index} out of range.",
                "status": "BLOCKCHAIN TAMPERING DETECTED",
                "debug": {
                    "stored_block_hash": "",
                    "recalculated_block_hash": "",
                    "verification_result": "⚠️ BLOCKCHAIN TAMPERING DETECTED"
                }
            }

        block = self.chain[block_index]
        recalculated_hash = block.calculate_hash()
        stored_hash = block.hash

        hash_valid = (recalculated_hash == stored_hash)
        prev_valid = True
        if block_index > 0:
            prev_block = self.chain[block_index - 1]
            prev_valid = (block.previous_hash == prev_block.hash)

        is_verified = hash_valid and prev_valid
        message = "✅ BLOCK INTEGRITY VERIFIED" if is_verified else "⚠️ BLOCKCHAIN TAMPERING DETECTED"
        status = "BLOCK INTEGRITY VERIFIED" if is_verified else "BLOCKCHAIN TAMPERING DETECTED"

        return {
            "verified": is_verified,
            "tampered": not is_verified,
            "block_index": block.index,
            "stored_block_hash": stored_hash,
            "recalculated_block_hash": recalculated_hash,
            "previous_hash": block.previous_hash,
            "previous_hash_valid": prev_valid,
            "hash_matches": hash_valid,
            "message": message,
            "status": status,
            "debug": {
                "block_index": block.index,
                "stored_block_hash": stored_hash,
                "recalculated_block_hash": recalculated_hash,
                "verification_result": message
            }
        }

    def simulate_block_tampering(self, block_index: int, field: str = "action", new_value: str = "TAMPERED_ACTION") -> Dict[str, Any]:
        """For demonstration only: deliberately modify a block's field without updating its cryptographic hash."""
        if 0 <= block_index < len(self.chain):
            setattr(self.chain[block_index], field, new_value)
            return {"success": True, "block_index": block_index, "tampered_field": field, "new_value": new_value}
        return {"success": False, "error": "Invalid block index"}

    def restore_block(self, block_index: int, original_action: str = "RAG_QUERY"):
        """Restore original block state after demonstration."""
        if 0 <= block_index < len(self.chain):
            self.chain[block_index].action = original_action
            self.chain[block_index].hash = self.chain[block_index].calculate_hash()

    def verify_evidence(
        self,
        block_index: int,
        evidence_text: Optional[Any] = None,
        stored_hash: Optional[str] = None,
        evidence_hash: Optional[str] = None,
        audit_logger_ref: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Verify that evidence text matches the on-chain recorded hash for a specific block.
        Always recalculates the SHA-256 hash strictly from the provided evidence_text (or from the audit record if omitted).
        Compares: recalculated_current_evidence_hash == stored_evidence_hash
        """
        if block_index < 0 or block_index >= len(self.chain):
            return {
                "verified": False,
                "tampered": True,
                "error": f"Block index {block_index} out of range (chain length: {len(self.chain)})",
                "stored_hash": "",
                "current_hash": "",
                "stored_evidence_hash": "",
                "computed_evidence_hash": "",
                "message": f"Block index {block_index} out of range.",
                "debug": {
                    "stored_evidence_hash": "",
                    "recalculated_current_hash": "",
                    "verification_result": "FAILED (Block Index Out Of Range)"
                }
            }

        block = self.chain[block_index]
        on_chain_hash = (block.evidence_hash or "").strip().lower()
        target_stored = (stored_hash or evidence_hash or on_chain_hash).strip().lower()

        if not target_stored:
            return {
                "verified": False,
                "tampered": True,
                "error": f"Block {block_index} does not contain an evidence hash.",
                "stored_hash": "",
                "current_hash": "",
                "stored_evidence_hash": "",
                "computed_evidence_hash": "",
                "message": f"Block {block_index} does not contain an evidence hash.",
                "debug": {
                    "stored_evidence_hash": "",
                    "recalculated_current_hash": "",
                    "verification_result": "FAILED (No Stored Evidence Hash)"
                }
            }

        # 1. Determine the actual evidence content to hash
        eval_evidence = None
        if evidence_text is not None and (not isinstance(evidence_text, str) or evidence_text.strip() != ""):
            eval_evidence = evidence_text
        else:
            # Look up recorded evidence in audit logger
            logger = audit_logger_ref or globals().get("audit_logger")
            if logger and hasattr(logger, "records"):
                for r in logger.records:
                    if r.block_index == block_index:
                        if isinstance(r.details, dict) and "retrieved_evidence_texts" in r.details:
                            eval_evidence = r.details["retrieved_evidence_texts"]
                        elif isinstance(r.details, dict) and "retrieved_texts" in r.details:
                            eval_evidence = r.details["retrieved_texts"]
                        elif r.details:
                            eval_evidence = r.details
                        break

        # 2. Recalculate hash strictly from evidence content
        if eval_evidence is not None:
            computed_hash = compute_canonical_hash(eval_evidence)
        else:
            computed_hash = ""

        # 3. Compare recalculated hash with stored hash
        matches = (computed_hash.lower() == target_stored) and bool(computed_hash) and bool(target_stored)

        # If user passed a stored hash that differs from on-chain hash, mark as tampered
        if on_chain_hash and target_stored != on_chain_hash:
            matches = False

        chain_status = self.verify_chain()
        result_status = "VERIFIED / NO TAMPERING" if matches else "TAMPERING DETECTED"

        return {
            "verified": matches,
            "tampered": not matches,
            "stored_hash": target_stored,
            "current_hash": computed_hash,
            "stored_evidence_hash": target_stored,
            "computed_evidence_hash": computed_hash,
            "on_chain_evidence_hash": on_chain_hash,
            "message": (
                "Evidence integrity verified. No tampering detected."
                if matches
                else "Evidence tampering detected. Hash mismatch."
            ),
            "debug": {
                "stored_evidence_hash": target_stored,
                "recalculated_current_hash": computed_hash,
                "verification_result": result_status
            },
            "block_index": block.index,
            "chain_valid": chain_status["is_valid"],
            "block_timestamp": block.timestamp,
            "user_id": block.user_id,
            "action": block.action,
            "status": block.status,
            "provenance": block.provenance,
            "tamper_detected": not matches
        }

    def get_blocks(self) -> List[Dict[str, Any]]:
        """Return list of all blocks as dicts"""
        return [b.to_dict() for b in self.chain]


# Global blockchain instance
blockchain = Blockchain()


# ============================================================
# AUDIT RECORD CLASS
# ============================================================

class AuditRecord:
    """Represents an audit log record with detailed clinical context"""

    def __init__(
        self,
        user_id: str,
        action: str,
        data_type: str,
        details: Any,
        patient_id: Optional[str] = None,
        status: str = "success",
        error_message: Optional[str] = None,
        query_hash: str = "",
        evidence_hash: str = "",
        block_index: Optional[int] = None,
        block_hash: Optional[str] = None,
        query_id: Optional[str] = None
    ):
        self.query_id = query_id or f"QRY-{uuid.uuid4().hex[:8].upper()}"
        self.user_id = user_id
        self.action = action
        self.data_type = data_type
        self.details = details
        self.patient_id = patient_id
        self.status = status
        self.error_message = error_message
        self.query_hash = query_hash
        self.evidence_hash = evidence_hash
        self.block_index = block_index
        self.block_hash = block_hash
        self.timestamp = datetime.datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response"""
        return {
            "query_id": self.query_id,
            "user_id": self.user_id,
            "action": self.action,
            "data_type": self.data_type,
            "details": self.details,
            "patient_id": self.patient_id,
            "status": self.status,
            "error_message": self.error_message,
            "query_hash": self.query_hash,
            "evidence_hash": self.evidence_hash,
            "block_index": self.block_index,
            "block_hash": self.block_hash,
            "timestamp": self.timestamp
        }


# ============================================================
# BLOCKCHAIN AUDIT LOG CLASS
# ============================================================

class BlockchainAuditLog:
    """Manages audit logging using blockchain"""

    def __init__(self, chain_instance: Optional[Blockchain] = None):
        self.blockchain = chain_instance or blockchain
        self.records: List[AuditRecord] = []

    def add_record(
        self,
        user_id: str,
        action: str,
        data_type: str,
        details: Any,
        patient_id: Optional[str] = None,
        status: str = "success",
        error_message: Optional[str] = None,
        query_text: Optional[str] = None,
        evidence_bundle: Optional[Any] = None,
        query_id: Optional[str] = None
    ) -> AuditRecord:
        """Add an audit record and create corresponding blockchain block"""
        q_id = query_id or f"QRY-{uuid.uuid4().hex[:8].upper()}"
        q_hash = compute_canonical_hash(query_text) if query_text else ""
        e_hash = compute_canonical_hash(evidence_bundle) if evidence_bundle is not None else ""

        # Provenance metadata (no raw full patient text stored in block)
        provenance = {
            "query_id": q_id,
            "data_type": data_type,
            "patient_id": patient_id,
            "has_evidence": bool(e_hash)
        }
        if isinstance(details, dict):
            # Extract safe summary keys
            provenance["record_count"] = details.get("records_retrieved", details.get("evidence_count", 0))
            if "patient_ids" in details:
                provenance["retrieved_patient_ids"] = details["patient_ids"]

        # Add block to blockchain
        new_block = self.blockchain.add_block(
            user_id=user_id,
            action=action,
            status=status.upper(),
            query_hash=q_hash,
            evidence_hash=e_hash,
            provenance=provenance
        )

        record = AuditRecord(
            user_id=user_id,
            action=action,
            data_type=data_type,
            details=details,
            patient_id=patient_id,
            status=status,
            error_message=error_message,
            query_hash=q_hash,
            evidence_hash=e_hash,
            block_index=new_block.index,
            block_hash=new_block.hash,
            query_id=q_id
        )
        self.records.append(record)
        return record

    def get_recent_records(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent audit records"""
        records_dict = [r.to_dict() for r in self.records]
        return records_dict[-limit:]

    def get_records_by_patient(self, patient_id: str) -> List[Dict[str, Any]]:
        """Get records for a specific patient"""
        return [r.to_dict() for r in self.records if r.patient_id == patient_id]

    def get_records_by_action(self, action: str) -> List[Dict[str, Any]]:
        """Get records for a specific action"""
        return [r.to_dict() for r in self.records if r.action == action]

    def verify_chain(self) -> Dict[str, Any]:
        """Verify blockchain integrity"""
        return self.blockchain.verify_chain()

    def verify_block(self, block_index: int) -> Dict[str, Any]:
        """Verify individual block integrity"""
        return self.blockchain.verify_block(block_index)

    def verify_evidence(
        self,
        block_index: int,
        evidence_text: Optional[Any] = None,
        stored_hash: Optional[str] = None,
        evidence_hash: Optional[str] = None
    ) -> Dict[str, Any]:
        """Verify evidence integrity against blockchain"""
        return self.blockchain.verify_evidence(
            block_index=block_index,
            evidence_text=evidence_text,
            stored_hash=stored_hash,
            evidence_hash=evidence_hash,
            audit_logger_ref=self
        )

    def verify_evidence_integrity(
        self,
        evidence: Any,
        stored_hash: str
    ) -> Dict[str, Any]:
        """Verify evidence integrity given evidence content and stored hash"""
        return _verify_evidence_integrity(evidence, stored_hash)

    def verify_query_evidence(
        self,
        identifier: Union[int, str],
        current_evidence: Optional[Any] = None,
        stored_hash_override: Optional[str] = None
    ) -> Dict[str, Any]:
        """Verify query evidence against recorded blockchain hash"""
        return _verify_query_evidence(
            identifier=identifier,
            current_evidence=current_evidence,
            blockchain_instance=self.blockchain,
            audit_logger_instance=self,
            stored_hash_override=stored_hash_override
        )

    def generate_audit_report(self, patient_id: Optional[str] = None) -> Dict[str, Any]:
        """Generate audit report"""
        if patient_id:
            report_records = self.get_records_by_patient(patient_id)
        else:
            report_records = [r.to_dict() for r in self.records]

        chain_verification = self.verify_chain()

        return {
            "total_records": len(report_records),
            "total_blocks": len(self.blockchain.chain),
            "chain_integrity_verified": chain_verification["is_valid"],
            "latest_block_hash": self.blockchain.get_latest_block().hash,
            "records": report_records,
            "blocks": self.blockchain.get_blocks(),
            "generated_at": datetime.datetime.now().isoformat()
        }

    def __len__(self) -> int:
        return len(self.records)


# Global audit log manager instance
audit_logger = BlockchainAuditLog(blockchain)


# ============================================================
# CONVENIENCE / COMPATIBILITY FUNCTIONS
# ============================================================

def add_block(
    user_id: str,
    action: str,
    status: str,
    query_hash: str = "",
    evidence_hash: str = "",
    provenance: Optional[Dict[str, Any]] = None
) -> Block:
    """Convenience function to add a block to the global chain"""
    return blockchain.add_block(user_id, action, status, query_hash, evidence_hash, provenance)


def verify_evidence_integrity(evidence: Any, stored_hash: str) -> Dict[str, Any]:
    """Recalculate SHA-256 and compare with stored_hash"""
    return _verify_evidence_integrity(evidence, stored_hash)


def verify_query_evidence(
    identifier: Union[int, str],
    current_evidence: Optional[Any] = None,
    stored_hash_override: Optional[str] = None
) -> Dict[str, Any]:
    """Retrieve stored evidence/hash for a query and verify against current evidence"""
    return _verify_query_evidence(
        identifier=identifier,
        current_evidence=current_evidence,
        blockchain_instance=blockchain,
        audit_logger_instance=audit_logger,
        stored_hash_override=stored_hash_override
    )
