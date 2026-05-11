from app.blockchain.access_control import check_access
from app.blockchain.audit_log import add_block, blockchain


# ============================================================
# SECURE QUERY FUNCTION
# ============================================================

def secure_query(user_id, query, retriever):

    # ========================================================
    # RBAC CHECK
    # ========================================================

    allowed = check_access(user_id)

    # ========================================================
    # ACCESS DENIED
    # ========================================================

    if not allowed:

        add_block(
            user_id=user_id,
            action=query,
            status="ACCESS DENIED"
        )

        return {
            "status": "denied",
            "message": "User is not authorized",
            "results": []
        }

    # ========================================================
    # VECTOR SEARCH
    # ========================================================

    docs = retriever.invoke(query)

    results = []

    for doc in docs:

        results.append(doc.page_content)

    # ========================================================
    # STORE BLOCKCHAIN LOG
    # ========================================================

    add_block(
        user_id=user_id,
        action=query,
        status="ACCESS GRANTED"
    )

    # ========================================================
    # RETURN RESULTS
    # ========================================================

    return {
        "status": "success",
        "results": results
    }


# ============================================================
# GET AUDIT LOGS
# ============================================================

def get_audit_logs():

    return {
        "records": blockchain
    }