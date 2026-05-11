# ============================================================
# ROLE BASED ACCESS CONTROL (RBAC)
# ============================================================

from enum import Enum
from typing import Optional, Set, Dict


# ============================================================
# ENUMS
# ============================================================

class UserRole(Enum):
    """User roles in the system"""
    ADMIN = "admin"
    DOCTOR = "doctor"
    NURSE = "nurse"
    PATIENT = "patient"
    AUDITOR = "auditor"


class Permission(Enum):
    """Available permissions"""
    VIEW_PATIENT_DATA = "view_patient_data"
    EDIT_PATIENT_DATA = "edit_patient_data"
    REQUEST_PREDICTION = "request_prediction"
    VIEW_AUDIT_LOGS = "view_audit_logs"
    QUERY_KNOWLEDGE_BASE = "query_knowledge_base"
    MANAGE_USERS = "manage_users"


# ============================================================
# USER CLASS
# ============================================================

class User:
    """Represents a system user"""
    
    def __init__(self, user_id: str, name: str, role: UserRole, permissions: Optional[Set[Permission]] = None):
        self.user_id = user_id
        self.name = name
        self.role = role
        self.permissions = permissions or set()
    
    def has_permission(self, permission: Permission) -> bool:
        """Check if user has a specific permission"""
        return permission in self.permissions
    
    def add_permission(self, permission: Permission):
        """Add permission to user"""
        self.permissions.add(permission)
    
    def remove_permission(self, permission: Permission):
        """Remove permission from user"""
        self.permissions.discard(permission)
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            "user_id": self.user_id,
            "name": self.name,
            "role": self.role.value,
            "permissions": [p.value for p in self.permissions]
        }


# ============================================================
# DEFAULT USERS AND PERMISSIONS
# ============================================================

DEFAULT_USERS = {
    "admin": User(
        user_id="admin",
        name="Administrator",
        role=UserRole.ADMIN,
        permissions={
            Permission.VIEW_PATIENT_DATA,
            Permission.EDIT_PATIENT_DATA,
            Permission.REQUEST_PREDICTION,
            Permission.VIEW_AUDIT_LOGS,
            Permission.QUERY_KNOWLEDGE_BASE,
            Permission.MANAGE_USERS
        }
    ),
    "DOC001": User(
        user_id="DOC001",
        name="Doctor One",
        role=UserRole.DOCTOR,
        permissions={
            Permission.VIEW_PATIENT_DATA,
            Permission.EDIT_PATIENT_DATA,
            Permission.REQUEST_PREDICTION,
            Permission.QUERY_KNOWLEDGE_BASE
        }
    ),
    "D101": User(
        user_id="D101",
        name="Doctor One",
        role=UserRole.DOCTOR,
        permissions={
            Permission.VIEW_PATIENT_DATA,
            Permission.EDIT_PATIENT_DATA,
            Permission.REQUEST_PREDICTION,
            Permission.QUERY_KNOWLEDGE_BASE
        }
    ),
    "NURSE001": User(
        user_id="NURSE001",
        name="Nurse One",
        role=UserRole.NURSE,
        permissions={
            Permission.VIEW_PATIENT_DATA,
            Permission.EDIT_PATIENT_DATA
        }
    ),
}


# ============================================================
# ACCESS CONTROL MANAGER CLASS
# ============================================================

class AccessControlManager:
    """Manages access control for the system"""
    
    def __init__(self):
        self.users: Dict[str, User] = DEFAULT_USERS.copy()
    
    def add_user(self, user: User):
        """Add a new user"""
        self.users[user.user_id.lower()] = user
    
    def remove_user(self, user_id: str):
        """Remove a user"""
        user_id_lower = user_id.lower()
        if user_id_lower in self.users:
            del self.users[user_id_lower]
    
    def get_user(self, user_id: str) -> Optional[User]:
        """Get user by ID"""
        user_id_lower = user_id.lower()
        return self.users.get(user_id_lower)
    
    def check_permission(self, user_id: str, permission: Permission) -> bool:
        """Check if user has required permission"""
        user = self.get_user(user_id)
        
        if not user:
            print(f"⚠️  User '{user_id}' not found in access control")
            return False
        
        has_perm = user.has_permission(permission)
        
        if not has_perm:
            print(f"❌ User '{user_id}' denied permission: {permission.value}")
        else:
            print(f"✓ User '{user_id}' granted permission: {permission.value}")
        
        return has_perm
    
    def grant_permission(self, user_id: str, permission: Permission) -> bool:
        """Grant permission to user"""
        user = self.get_user(user_id)
        
        if not user:
            return False
        
        user.add_permission(permission)
        return True
    
    def revoke_permission(self, user_id: str, permission: Permission) -> bool:
        """Revoke permission from user"""
        user = self.get_user(user_id)
        
        if not user:
            return False
        
        user.remove_permission(permission)
        return True
    
    def list_users(self):
        """List all users with their roles and permissions"""
        return [user.to_dict() for user in self.users.values()]


# ============================================================
# LEGACY FUNCTIONS FOR BACKWARD COMPATIBILITY
# ============================================================

def check_access(user_id: str) -> bool:
    """Legacy function - check if user is authorized"""
    user_id = user_id.strip().upper()
    
    authorized_users = ["D101", "DOC001", "ADMIN"]
    
    return user_id in authorized_users