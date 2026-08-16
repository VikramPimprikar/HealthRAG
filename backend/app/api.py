"""
RAGChainMed FastAPI Backend
===========================
Main REST API service providing:
- Secure grounded RAG query search with Groq LLM & FAISS
- Immutable Blockchain Audit Logging with SHA-256 Evidence Hashing
- Role-Based Access Control (RBAC) across all endpoints
- Machine Learning Heart Disease Severity Prediction & Clinical Decision Support
- Cryptographic Blockchain & Evidence Integrity Verification

Author: RAGChainMed
"""

import os
import datetime
from typing import Optional, List, Dict, Any
from pathlib import Path
from dotenv import load_dotenv

from fastapi import FastAPI, Header, Query, HTTPException, status, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Load environment
load_dotenv()

# Import internal modules
import sys
BASE_DIR = Path(__file__).resolve().parent.parent.parent
BACKEND_DIR = BASE_DIR / "backend"
sys.path.append(str(BACKEND_DIR))

from app.blockchain.access_control import (
    access_control_manager,
    Permission,
    UserRole
)
from app.blockchain.audit_log import (
    audit_logger,
    blockchain,
    sha256_hash
)
from app.rag.enhanced_rag_pipeline import get_rag_service
from app.predict import predict_heart_disease
from app.clinical.clinical_decision_support import (
    ClinicalDecisionSupportEngine
)


# ============================================================
# FASTAPI APP INITIALIZATION
# ============================================================

app = FastAPI(
    title="RAGChainMed API",
    description="Secure Healthcare Retrieval & Clinical Decision Support System with Blockchain Verification & RBAC",
    version="2.0.0"
)

# Enable CORS for React frontend (localhost:3000, 3001, etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize CDS engine
cds_engine = ClinicalDecisionSupportEngine()


# ============================================================
# REQUEST & RESPONSE SCHEMAS
# ============================================================

class QueryRequest(BaseModel):
    query: str = Field(..., description="Clinical question or patient search term")
    user_id: Optional[str] = Field("admin", description="User ID making the query")
    top_k: Optional[int] = Field(5, description="Number of top evidence records to retrieve")


class PredictionRequest(BaseModel):
    user_id: Optional[str] = Field("DOC001", description="User ID requesting prediction")
    patient_id: Optional[str] = Field("P_NEW", description="Patient ID")
    age: float = Field(..., description="Age in years")
    sex: Any = Field(..., description="Sex (1=Male, 0=Female)")
    cp: Any = Field(..., description="Chest pain type (1-4)")
    trestbps: float = Field(120.0, description="Resting blood pressure in mmHg")
    chol: float = Field(200.0, description="Serum cholesterol in mg/dL")
    fbs: Any = Field(0, description="Fasting blood sugar > 120 mg/dL (1=True, 0=False)")
    restecg: Any = Field(0, description="Resting ECG results (0-2)")
    thalch: Optional[float] = Field(150.0, description="Maximum heart rate achieved")
    thalach: Optional[float] = Field(None, description="Alias for thalch")
    exang: Any = Field(0, description="Exercise induced angina (1=Yes, 0=No)")
    oldpeak: float = Field(0.0, description="ST depression induced by exercise")
    slope: Any = Field(1, description="Slope of peak exercise ST segment (1-3)")


class VerifyEvidenceRequest(BaseModel):
    block_index: int = Field(..., description="Blockchain block index to verify")
    evidence_text: Optional[str] = Field(None, description="Retrieved evidence string")
    evidence_hash: Optional[str] = Field(None, description="SHA-256 evidence hash")


# ============================================================
# RBAC DEPENDENCY HELPER
# ============================================================

def get_authorized_user(
    x_user_id: Optional[str] = Header(None, alias="X-User-Id"),
    user_id: Optional[str] = Query(None)
) -> str:
    """Resolve and return active user ID from header or query param"""
    resolved_id = x_user_id or user_id or "admin"
    return resolved_id.strip()


def require_permission(permission: Permission, user_id: str, action_name: str) -> bool:
    """Check permission and log to blockchain if denied"""
    has_perm = access_control_manager.check_permission(user_id, permission)
    if not has_perm:
        # Log denied attempt in blockchain
        audit_logger.add_record(
            user_id=user_id,
            action=action_name,
            data_type="Access Control",
            details={"required_permission": permission.value, "reason": "Insufficient permissions"},
            status="denied",
            error_message=f"Access denied for user '{user_id}': missing permission '{permission.value}'"
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Access denied: User '{user_id}' does not have the required '{permission.value}' permission."
        )
    return True


# ============================================================
# ROOT & SYSTEM HEALTH
# ============================================================

@app.get("/")
def home():
    """System information"""
    rag = get_rag_service()
    total_vectors = rag.vectorstore.index.ntotal if rag.vectorstore else 0
    return {
        "system": "RAGChainMed API",
        "status": "online",
        "version": "2.0.0",
        "blockchain_blocks": len(blockchain.chain),
        "total_vectors": total_vectors,
        "timestamp": datetime.datetime.now().isoformat()
    }


@app.get("/health")
def health():
    """Detailed health check"""
    rag = get_rag_service()
    total_vectors = rag.vectorstore.index.ntotal if rag.vectorstore else 0
    chain_status = blockchain.verify_chain()

    return {
        "status": "healthy",
        "rag_vectorstore_ready": bool(rag.vectorstore),
        "total_vectors": total_vectors,
        "llm_client_configured": bool(rag.groq_client),
        "blockchain_intact": chain_status["is_valid"],
        "total_blocks": len(blockchain.chain),
        "total_audit_records": len(audit_logger.records),
        "timestamp": datetime.datetime.now().isoformat()
    }


# ============================================================
# AUTHENTICATION & RBAC ENDPOINTS
# ============================================================

@app.get("/api/v1/auth/users")
def get_users():
    """Get list of users and roles for frontend switcher and management"""
    return {
        "users": access_control_manager.list_users()
    }


@app.get("/api/v1/auth/me")
def get_current_user_profile(user_id: str = Depends(get_authorized_user)):
    """Get active user profile and permissions"""
    user = access_control_manager.get_user(user_id)
    if not user:
        # Create temporary guest/unauthorized profile
        return {
            "user_id": user_id,
            "name": f"User ({user_id})",
            "role": "unauthorized",
            "permissions": [],
            "authorized": False
        }
    profile = user.to_dict()
    profile["authorized"] = True
    return profile


# ============================================================
# RAG RETRIEVAL ENDPOINTS
# ============================================================

@app.get("/query")
def query_rag_get(
    query: str,
    user_id: str = Depends(get_authorized_user),
    top_k: int = Query(5, ge=1, le=20)
):
    """GET query endpoint for backward compatibility & easy browser access"""
    return perform_rag_query(query=query, user_id=user_id, top_k=top_k)


@app.post("/api/v1/query")
def query_rag_post(
    req: QueryRequest,
    user_id: str = Depends(get_authorized_user)
):
    """POST query endpoint for structured client requests"""
    actual_user = req.user_id if req.user_id else user_id
    return perform_rag_query(query=req.query, user_id=actual_user, top_k=req.top_k or 5)


def perform_rag_query(query: str, user_id: str, top_k: int = 5):
    """Internal handler for executing RBAC check, RAG search, LLM generation, and blockchain logging"""
    # 1. Enforce RBAC
    require_permission(Permission.QUERY_KNOWLEDGE_BASE, user_id, "QUERY_KNOWLEDGE_BASE")

    # 2. Run RAG Pipeline
    rag_service = get_rag_service()
    rag_result = rag_service.answer_query(query=query, user_id=user_id, top_k=top_k)

    retrieved_records = rag_result["retrieved_evidence"]
    retrieved_texts = [r["text"] for r in retrieved_records]
    patient_ids = [r["patient_id"] for r in retrieved_records]

    # 3. Log to Blockchain with SHA-256 evidence and query hashes
    record = audit_logger.add_record(
        user_id=user_id,
        action="RAG_QUERY",
        data_type="Healthcare Records",
        details={
            "query": query,
            "records_retrieved": len(retrieved_records),
            "patient_ids": patient_ids,
            "has_relevant_evidence": rag_result.get("has_relevant_evidence", False)
        },
        status="success",
        query_text=query,
        evidence_bundle=retrieved_texts
    )

    return {
        "query": query,
        "user_id": user_id,
        "answer": rag_result["answer"],
        "retrieved_records": retrieved_texts,
        "retrieved_evidence": retrieved_records,
        "has_relevant_evidence": rag_result.get("has_relevant_evidence", False),
        "evidence_hash": record.evidence_hash,
        "query_hash": record.query_hash,
        "block_index": record.block_index,
        "block_hash": record.block_hash,
        "timestamp": record.timestamp
    }


# ============================================================
# CLINICAL PREDICTION (ML + CDS) ENDPOINTS
# ============================================================

@app.post("/api/v1/predict")
def predict_risk(
    req: PredictionRequest,
    user_id: str = Depends(get_authorized_user)
):
    """Predict heart disease severity and generate clinical decision support recommendations"""
    actual_user = req.user_id if req.user_id else user_id

    # 1. Enforce RBAC
    require_permission(Permission.REQUEST_PREDICTION, actual_user, "REQUEST_PREDICTION")

    # 2. Extract clinical data dict
    patient_data = {
        "age": req.age,
        "sex": req.sex,
        "cp": req.cp,
        "trestbps": req.trestbps,
        "chol": req.chol,
        "fbs": req.fbs,
        "restecg": req.restecg,
        "thalch": req.thalch if req.thalch is not None else (req.thalach or 150.0),
        "exang": req.exang,
        "oldpeak": req.oldpeak,
        "slope": req.slope
    }

    # 3. Run ML model
    try:
        prediction_result = predict_heart_disease(patient_data)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Model prediction failed: {str(e)}"
        )

    # 4. Generate Clinical Decision Support (CDS) Assessment
    cds_assessment = cds_engine.assess_risk(
        patient_id=req.patient_id or "P_NEW",
        patient_data=patient_data,
        model_prediction=prediction_result["severity"]
    )

    recommendations_list = [
        {
            "type": rec.type.value,
            "description": rec.description,
            "priority": rec.priority,
            "evidence_sources": rec.evidence_sources
        }
        for rec in cds_assessment.recommendations
    ]

    # 5. Log prediction into Blockchain
    record = audit_logger.add_record(
        user_id=actual_user,
        action="CLINICAL_PREDICTION",
        data_type="Cardiovascular Risk Assessment",
        details={
            "patient_id": req.patient_id,
            "prediction": prediction_result["severity"],
            "confidence": round(prediction_result["confidence"], 4),
            "risk_level": cds_assessment.risk_level.value,
            "risk_score": round(cds_assessment.risk_score, 4)
        },
        patient_id=req.patient_id,
        status="success",
        query_text=str(patient_data),
        evidence_bundle=prediction_result
    )

    return {
        "patient_id": req.patient_id,
        "user_id": actual_user,
        "prediction": prediction_result["prediction"],
        "severity": prediction_result["severity"],
        "risk_description": prediction_result["risk_description"],
        "confidence": prediction_result["confidence"],
        "probabilities": prediction_result["probabilities"],
        "parameter_breakdown": prediction_result.get("parameter_breakdown", []),
        "primary_categorization_drivers": prediction_result.get("primary_categorization_drivers", []),
        "feature_importance_ranking": prediction_result.get("feature_importance_ranking", []),
        "risk_assessment": {
            "risk_level": cds_assessment.risk_level.value,
            "risk_score": cds_assessment.risk_score,
            "contributing_factors": cds_assessment.contributing_factors,
            "recommendations": recommendations_list
        },
        "blockchain": {
            "block_index": record.block_index,
            "block_hash": record.block_hash,
            "evidence_hash": record.evidence_hash
        }
    }


# ============================================================
# BLOCKCHAIN AUDIT LOGS & VERIFICATION ENDPOINTS
# ============================================================

@app.get("/api/v1/audit/logs")
def get_audit_logs(
    user_id: str = Depends(get_authorized_user),
    limit: int = Query(100, ge=1, le=500)
):
    """Retrieve blockchain audit logs (requires VIEW_AUDIT_LOGS permission)"""
    require_permission(Permission.VIEW_AUDIT_LOGS, user_id, "VIEW_AUDIT_LOGS")

    report = audit_logger.generate_audit_report()
    return report


@app.get("/api/v1/audit/verify-chain")
def verify_blockchain_chain(
    user_id: str = Depends(get_authorized_user)
):
    """Verify cryptographic integrity of the entire audit blockchain"""
    require_permission(Permission.VIEW_AUDIT_LOGS, user_id, "VERIFY_BLOCKCHAIN")
    return blockchain.verify_chain()


@app.post("/api/v1/audit/verify-evidence")
def verify_evidence(
    req: VerifyEvidenceRequest,
    user_id: str = Depends(get_authorized_user)
):
    """
    Verify whether the provided evidence text or SHA-256 hash matches
    the authentic blockchain record at the specified block index.
    """
    # Verification can be performed by authorized users (Doctors, Admins, Auditors)
    if not (
        access_control_manager.check_permission(user_id, Permission.VIEW_AUDIT_LOGS) or
        access_control_manager.check_permission(user_id, Permission.QUERY_KNOWLEDGE_BASE)
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"User '{user_id}' is not authorized to verify evidence."
        )

    verification_result = blockchain.verify_evidence(
        block_index=req.block_index,
        evidence_text=req.evidence_text,
        evidence_hash=req.evidence_hash
    )

    return verification_result