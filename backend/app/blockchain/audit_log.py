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
import datetime
from typing import List, Dict, Optional, Any


def sha256_hash(content: Any) -> str:
    """Compute SHA-256 hash of string or JSON-serializable content"""
    if content is None:
        return ""
    if isinstance(content, (dict, list)):
        text = json.dumps(content, sort_keys=True)
    else:
        text = str(content)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


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
        provenance_str = json.dumps(self.provenance, sort_keys=True)
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
            provenance={"system": "RAGChainMed Healthcare Audit Subsystem", "version": "1.0"}
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
                    "reason": "Block hash mismatch - data may have been tampered with."
                }

            # 2. Verify that previous_hash matches previous block's hash
            if current_block.previous_hash != previous_block.hash:
                return {
                    "is_valid": False,
                    "error_at_index": current_block.index,
                    "reason": "Chain linkage broken - previous hash mismatch."
                }

        return {
            "is_valid": True,
            "total_blocks": len(self.chain),
            "latest_block_hash": self.chain[-1].hash
        }

    def verify_evidence(
        self,
        block_index: int,
        evidence_text: Optional[str] = None,
        evidence_hash: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Verify that evidence text or evidence hash matches the on-chain recorded hash
        for a specific block.
        """
        if block_index < 0 or block_index >= len(self.chain):
            return {
                "verified": False,
                "error": f"Block index {block_index} out of range (chain length: {len(self.chain)})"
            }

        block = self.chain[block_index]

        if not block.evidence_hash:
            return {
                "verified": False,
                "error": f"Block {block_index} does not contain an evidence hash."
            }

        # Determine target hash
        if evidence_hash:
            computed_hash = evidence_hash.strip().lower()
        elif evidence_text is not None:
            computed_hash = hashlib.sha256(evidence_text.encode("utf-8")).hexdigest()
        else:
            return {
                "verified": False,
                "error": "Neither evidence_text nor evidence_hash was provided."
            }

        stored_hash = block.evidence_hash.strip().lower()
        matches = (computed_hash == stored_hash)

        # Also ensure chain is valid
        chain_status = self.verify_chain()

        return {
            "verified": matches,
            "block_index": block.index,
            "stored_evidence_hash": stored_hash,
            "computed_evidence_hash": computed_hash,
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
        block_hash: Optional[str] = None
    ):
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
        evidence_bundle: Optional[Any] = None
    ) -> AuditRecord:
        """Add an audit record and create corresponding blockchain block"""
        q_hash = sha256_hash(query_text) if query_text else ""
        e_hash = sha256_hash(evidence_bundle) if evidence_bundle is not None else ""

        # Provenance metadata (no raw full patient text stored in block)
        provenance = {
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
            block_hash=new_block.hash
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

    def verify_evidence(
        self,
        block_index: int,
        evidence_text: Optional[str] = None,
        evidence_hash: Optional[str] = None
    ) -> Dict[str, Any]:
        """Verify evidence integrity against blockchain"""
        return self.blockchain.verify_evidence(block_index, evidence_text, evidence_hash)

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
# COMPATIBILITY FUNCTIONS
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
