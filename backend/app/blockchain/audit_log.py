import hashlib
import datetime


# ============================================================
# BLOCK CLASS
# ============================================================

class Block:

    def __init__(
        self,
        index,
        timestamp,
        user_id,
        action,
        status,
        previous_hash
    ):

        self.index = index

        self.timestamp = timestamp

        self.user_id = user_id

        self.action = action

        self.status = status

        self.previous_hash = previous_hash

        self.hash = self.calculate_hash()

    # --------------------------------------------------------

    def calculate_hash(self):

        block_data = (
            str(self.index)
            + str(self.timestamp)
            + str(self.user_id)
            + str(self.action)
            + str(self.status)
            + str(self.previous_hash)
        )

        return hashlib.sha256(
            block_data.encode()
        ).hexdigest()


# ============================================================
# BLOCKCHAIN CLASS
# ============================================================

class Blockchain:

    def __init__(self):

        self.chain = []

        self.create_genesis_block()

    # --------------------------------------------------------

    def create_genesis_block(self):

        genesis_block = Block(
            0,
            str(datetime.datetime.now()),
            "SYSTEM",
            "Genesis Block",
            "SUCCESS",
            "0"
        )

        self.chain.append(genesis_block)

    # --------------------------------------------------------

    def get_latest_block(self):

        return self.chain[-1]

    # --------------------------------------------------------

    def add_block(
        self,
        user_id,
        action,
        status
    ):

        latest_block = self.get_latest_block()

        new_block = Block(
            index=latest_block.index + 1,
            timestamp=str(datetime.datetime.now()),
            user_id=user_id,
            action=action,
            status=status,
            previous_hash=latest_block.hash
        )

        self.chain.append(new_block)

    # --------------------------------------------------------

    def verify_chain(self):

        for i in range(1, len(self.chain)):

            current_block = self.chain[i]

            previous_block = self.chain[i - 1]

            # Verify hash

            if (
                current_block.hash
                != current_block.calculate_hash()
            ):
                return False

            # Verify previous hash link

            if (
                current_block.previous_hash
                != previous_block.hash
            ):
                return False

        return True


# ============================================================
# CREATE BLOCKCHAIN OBJECT
# ============================================================

blockchain = Blockchain()


# ============================================================
# ADD BLOCK FUNCTION
# ============================================================

def add_block(
    user_id,
    action,
    status
):

    blockchain.add_block(
        user_id,
        action,
        status
    )


# ============================================================
# AUDIT RECORD CLASS
# ============================================================

class AuditRecord:
    """Represents an audit log record"""
    
    def __init__(self, user_id, action, data_type, details, patient_id=None, 
                 status="success", error_message=None):
        self.user_id = user_id
        self.action = action
        self.data_type = data_type
        self.details = details
        self.patient_id = patient_id
        self.status = status
        self.error_message = error_message
        self.timestamp = str(datetime.datetime.now())
    
    def to_dict(self):
        """Convert to dictionary for API response"""
        return {
            "user_id": self.user_id,
            "action": self.action,
            "data_type": self.data_type,
            "details": self.details,
            "patient_id": self.patient_id,
            "status": self.status,
            "error_message": self.error_message,
            "timestamp": self.timestamp
        }


# ============================================================
# BLOCKCHAIN AUDIT LOG CLASS
# ============================================================

class BlockchainAuditLog:
    """Manages audit logging using blockchain"""
    
    def __init__(self):
        self.blockchain = Blockchain()
        self.records = []
    
    def add_record(self, user_id, action, data_type, details, patient_id=None, 
                   status="success", error_message=None):
        """Add an audit record and create blockchain entry"""
        record = AuditRecord(
            user_id=user_id,
            action=action,
            data_type=data_type,
            details=details,
            patient_id=patient_id,
            status=status,
            error_message=error_message
        )
        self.records.append(record)
        
        # Add block to blockchain
        self.blockchain.add_block(
            user_id=user_id,
            action=action,
            status=status
        )
    
    def get_recent_records(self, limit=100):
        """Get recent audit records"""
        records_dict = [r.to_dict() for r in self.records]
        return records_dict[-limit:]
    
    def get_records_by_patient(self, patient_id):
        """Get records for a specific patient"""
        return [r.to_dict() for r in self.records if r.patient_id == patient_id]
    
    def get_records_by_action(self, action):
        """Get records for a specific action"""
        return [r.to_dict() for r in self.records if r.action == action]
    
    def verify_chain(self):
        """Verify blockchain integrity"""
        return self.blockchain.verify_chain()
    
    def generate_audit_report(self, patient_id=None):
        """Generate audit report"""
        if patient_id:
            report_records = self.get_records_by_patient(patient_id)
        else:
            report_records = [r.to_dict() for r in self.records]
        
        return {
            "total_records": len(report_records),
            "chain_integrity_verified": self.verify_chain(),
            "records": report_records,
            "generated_at": str(datetime.datetime.now())
        }
    
    def __len__(self):
        """Return number of records"""
        return len(self.records)
# ============================================================
# AUDIT RECORD CLASS
# ============================================================

class AuditRecord:
    """Represents an audit log record"""
    
    def __init__(self, user_id, action, data_type, details, patient_id=None, 
                 status="success", error_message=None):
        self.user_id = user_id
        self.action = action
        self.data_type = data_type
        self.details = details
        self.patient_id = patient_id
        self.status = status
        self.error_message = error_message
        self.timestamp = str(datetime.datetime.now())
    
    def to_dict(self):
        """Convert to dictionary for API response"""
        return {
            "user_id": self.user_id,
            "action": self.action,
            "data_type": self.data_type,
            "details": self.details,
            "patient_id": self.patient_id,
            "status": self.status,
            "error_message": self.error_message,
            "timestamp": self.timestamp
        }


# ============================================================
# BLOCKCHAIN AUDIT LOG CLASS
# ============================================================

class BlockchainAuditLog:
    """Manages audit logging using blockchain"""
    
    def __init__(self):
        self.blockchain = Blockchain()
        self.records = []
    
    def add_record(self, user_id, action, data_type, details, patient_id=None, 
                   status="success", error_message=None):
        """Add an audit record and create blockchain entry"""
        record = AuditRecord(
            user_id=user_id,
            action=action,
            data_type=data_type,
            details=details,
            patient_id=patient_id,
            status=status,
            error_message=error_message
        )
        self.records.append(record)
        
        # Add block to blockchain
        self.blockchain.add_block(
            user_id=user_id,
            action=action,
            status=status
        )
    
    def get_recent_records(self, limit=100):
        """Get recent audit records"""
        records_dict = [r.to_dict() for r in self.records]
        return records_dict[-limit:]
    
    def get_records_by_patient(self, patient_id):
        """Get records for a specific patient"""
        return [r.to_dict() for r in self.records if r.patient_id == patient_id]
    
    def get_records_by_action(self, action):
        """Get records for a specific action"""
        return [r.to_dict() for r in self.records if r.action == action]
    
    def verify_chain(self):
        """Verify blockchain integrity"""
        return self.blockchain.verify_chain()
    
    def generate_audit_report(self, patient_id=None):
        """Generate audit report"""
        if patient_id:
            report_records = self.get_records_by_patient(patient_id)
        else:
            report_records = [r.to_dict() for r in self.records]
        
        return {
            "total_records": len(report_records),
            "chain_integrity_verified": self.verify_chain(),
            "records": report_records,
            "generated_at": str(datetime.datetime.now())
        }
    
    def __len__(self):
        """Return number of records"""
        return len(self.records)
