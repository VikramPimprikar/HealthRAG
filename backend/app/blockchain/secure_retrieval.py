from app.blockchain.access_control import access_control_manager, Permission
from app.blockchain.audit_log import audit_logger, sha256_hash


# ============================================================
# SECURE QUERY FUNCTION
# ============================================================

def secure_query(user_id: str, query: str, retriever):
    """
    Executes a query with RBAC permission check and blockchain logging.
    """
    # Check RBAC
    allowed = access_control_manager.check_permission(user_id, Permission.QUERY_KNOWLEDGE_BASE)

    if not allowed:
        audit_logger.add_record(
            user_id=user_id,
            action="QUERY_KNOWLEDGE_BASE",
            data_type="Healthcare Records",
            details={"query": query, "reason": "Unauthorized role"},
            status="denied",
            error_message="User is not authorized for knowledge base queries",
            query_text=query
        )

        return {
            "status": "denied",
            "message": f"User '{user_id}' is not authorized to query the healthcare knowledge base.",
            "results": [],
            "evidence": []
        }

    # Vector search
    try:
        docs = retriever.invoke(query) if hasattr(retriever, "invoke") else retriever.get_relevant_documents(query)
    except Exception as e:
        docs = []

    results = [doc.page_content for doc in docs]
    evidence_items = []
    for doc in docs:
        item = {
            "content": doc.page_content,
            "metadata": getattr(doc, "metadata", {}),
            "hash": sha256_hash(doc.page_content)
        }
        evidence_items.append(item)

    # Store blockchain log
    record = audit_logger.add_record(
        user_id=user_id,
        action="QUERY_KNOWLEDGE_BASE",
        data_type="Healthcare Records",
        details={
            "query": query,
            "records_retrieved": len(results),
            "patient_ids": [getattr(doc, "metadata", {}).get("id") for doc in docs if getattr(doc, "metadata", {}).get("id")]
        },
        status="success",
        query_text=query,
        evidence_bundle=results
    )

    return {
        "status": "success",
        "results": results,
        "evidence": evidence_items,
        "block_index": record.block_index,
        "block_hash": record.block_hash,
        "evidence_hash": record.evidence_hash
    }


# ============================================================
# GET AUDIT LOGS
# ============================================================

def get_audit_logs():
    return {
        "records": audit_logger.get_recent_records()
    }