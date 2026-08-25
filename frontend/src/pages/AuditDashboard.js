import React, { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";

function AuditDashboard({ activeUser, apiBase }) {
  const navigate = useNavigate();
  const [report, setReport] = useState(null);
  const [chainVerification, setChainVerification] = useState(null);
  const [loading, setLoading] = useState(true);
  const [verifying, setVerifying] = useState(false);
  const [error, setError] = useState(null);
  const [viewMode, setViewMode] = useState("blocks"); // "blocks" or "records"
  const [filterAction, setFilterAction] = useState("ALL");
  const [filterStatus, setFilterStatus] = useState("ALL");
  const [blockVerifyResult, setBlockVerifyResult] = useState({});

  const loadAuditLogs = async () => {
    setLoading(true);
    setError(null);

    try {
      const response = await fetch(`${apiBase}/api/v1/audit/logs`, {
        headers: {
          "X-User-Id": activeUser
        }
      });

      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        throw new Error(errData.detail || `HTTP Error ${response.status}: Failed to load audit logs`);
      }

      const data = await response.json();
      setReport(data);
    } catch (err) {
      console.error("Failed to load audit trail:", err);
      setError(err.message);
      setReport(null);
    } finally {
      setLoading(false);
    }
  };

  const handleVerifyChain = async () => {
    setVerifying(true);
    try {
      const response = await fetch(`${apiBase}/api/v1/audit/verify-chain`, {
        headers: {
          "X-User-Id": activeUser
        }
      });

      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        throw new Error(errData.detail || "Chain verification request failed");
      }

      const result = await response.json();
      setChainVerification(result);
    } catch (err) {
      alert(`Verification error: ${err.message}`);
    } finally {
      setVerifying(false);
    }
  };

  const handleVerifyBlockIntegrity = async (blockIndex) => {
    try {
      const response = await fetch(`${apiBase}/api/v1/audit/verify-block`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-User-Id": activeUser
        },
        body: JSON.stringify({ block_index: blockIndex })
      });

      if (response.ok) {
        const res = await response.json();
        setBlockVerifyResult((prev) => ({
          ...prev,
          [blockIndex]: res
        }));
      }
    } catch (e) {
      console.error("Block integrity verify error:", e);
    }
  };

  useEffect(() => {
    loadAuditLogs();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeUser]);

  const blocks = report?.blocks || [];
  const records = report?.records || [];

  const filteredBlocks = blocks.filter((b) => {
    if (filterAction !== "ALL" && b.action !== filterAction) return false;
    if (filterStatus !== "ALL" && b.status !== filterStatus) return false;
    return true;
  });

  const filteredRecords = records.filter((r) => {
    if (filterAction !== "ALL" && r.action !== filterAction) return false;
    if (filterStatus !== "ALL" && r.status !== filterStatus.toLowerCase()) return false;
    return true;
  });

  return (
    <div className="fade-in" style={{ display: "flex", flexDirection: "column", gap: "24px" }}>
      {/* HEADER & SUMMARY METRICS */}
      <div className="glass-panel" style={{ background: "linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.85) 100%)" }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", flexWrap: "wrap", gap: "16px" }}>
          <div>
            <h1 style={{ fontSize: "24px", fontWeight: "700", color: "#fff", marginBottom: "6px" }}>
              ⛓️ Immutable Blockchain Audit Trail &amp; Evidence Hashes
            </h1>
            <p style={{ color: "var(--text-secondary)", fontSize: "14px", maxWidth: "800px" }}>
              Every clinical query, ML prediction, and access decision is cryptographically hashed with SHA-256 and committed to an append-only verifiable blockchain.
            </p>
          </div>

          <div style={{ display: "flex", gap: "10px" }}>
            <button
              className="btn-secondary"
              onClick={loadAuditLogs}
              disabled={loading}
              style={{ fontSize: "13px" }}
            >
              🔄 Refresh Logs
            </button>

            <button
              className="btn-success"
              onClick={handleVerifyChain}
              disabled={verifying}
              style={{ fontSize: "13px" }}
            >
              {verifying ? "Verifying..." : "🛡️ Verify Blockchain Integrity"}
            </button>
          </div>
        </div>

        {/* METRICS ROW */}
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))", gap: "16px", marginTop: "24px" }}>
          <div style={{ background: "rgba(15, 23, 42, 0.7)", padding: "16px", borderRadius: "10px", border: "1px solid var(--border-color)" }}>
            <span style={{ fontSize: "12px", color: "var(--text-muted)", display: "block", marginBottom: "4px" }}>
              Total Blocks in Chain
            </span>
            <span style={{ fontSize: "28px", fontWeight: "800", color: "#fff" }}>
              {report?.total_blocks || 0}
            </span>
          </div>

          <div style={{ background: "rgba(15, 23, 42, 0.7)", padding: "16px", borderRadius: "10px", border: "1px solid var(--border-color)" }}>
            <span style={{ fontSize: "12px", color: "var(--text-muted)", display: "block", marginBottom: "4px" }}>
              Cryptographic Integrity
            </span>
            <div style={{ display: "flex", alignItems: "center", gap: "8px", marginTop: "4px" }}>
              <span
                style={{
                  width: "10px",
                  height: "10px",
                  borderRadius: "50%",
                  backgroundColor: report?.chain_integrity_verified ? "#10b981" : "#f43f5e",
                  boxShadow: report?.chain_integrity_verified ? "0 0 10px #10b981" : "0 0 10px #f43f5e"
                }}
              />
              <span style={{ fontSize: "16px", fontWeight: "700", color: report?.chain_integrity_verified ? "#34d399" : "#fb7185" }}>
                {report?.chain_integrity_verified ? "VALID & UNTAMPERED" : "INTEGRITY ALERT"}
              </span>
            </div>
          </div>

          <div style={{ background: "rgba(15, 23, 42, 0.7)", padding: "16px", borderRadius: "10px", border: "1px solid var(--border-color)" }}>
            <span style={{ fontSize: "12px", color: "var(--text-muted)", display: "block", marginBottom: "4px" }}>
              Audit Events Recorded
            </span>
            <span style={{ fontSize: "28px", fontWeight: "800", color: "#818cf8" }}>
              {report?.total_records || 0}
            </span>
          </div>
        </div>
      </div>

      {/* VERIFICATION MODAL / BANNER */}
      {chainVerification && (
        <div
          className="fade-in"
          style={{
            background: chainVerification.is_valid ? "rgba(16, 185, 129, 0.15)" : "rgba(244, 63, 94, 0.15)",
            border: `1px solid ${chainVerification.is_valid ? "#10b981" : "#f43f5e"}`,
            padding: "18px 24px",
            borderRadius: "var(--radius-md)",
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center"
          }}
        >
          <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
            <span style={{ fontSize: "24px" }}>{chainVerification.is_valid ? "✅" : "⚠️"}</span>
            <div>
              <strong style={{ display: "block", fontSize: "15px", color: chainVerification.is_valid ? "#34d399" : "#fb7185" }}>
                {chainVerification.is_valid ? "✅ Blockchain Integrity Verified" : "⚠️ Blockchain Integrity Compromised"}
              </strong>
              <span style={{ fontSize: "13px", color: "#cbd5e1" }}>
                {chainVerification.is_valid
                  ? `All ${chainVerification.total_blocks} blocks are valid and correctly linked.`
                  : (chainVerification.reason || chainVerification.message || `Block #${chainVerification.error_at_index} has been modified.`)}
              </span>
            </div>
          </div>

          <button
            className="btn-secondary"
            style={{ fontSize: "12px" }}
            onClick={() => setChainVerification(null)}
          >
            Dismiss
          </button>
        </div>
      )}

      {/* ERROR / ACCESS DENIED BANNER */}
      {error && (
        <div
          className="fade-in"
          style={{
            background: "var(--danger-bg)",
            border: "1px solid var(--danger-border)",
            color: "var(--danger-text)",
            padding: "18px 24px",
            borderRadius: "var(--radius-md)",
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between"
          }}
        >
          <span>{error}</span>
          <button className="btn-secondary" onClick={() => setError(null)}>Dismiss</button>
        </div>
      )}

      {/* VIEW SELECTOR & FILTERS */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: "16px" }}>
        <div className="tab-nav" style={{ margin: 0 }}>
          <button
            className={`tab-btn ${viewMode === "blocks" ? "active" : ""}`}
            onClick={() => setViewMode("blocks")}
          >
            ⛓️ Blockchain Blocks ({filteredBlocks.length})
          </button>
          <button
            className={`tab-btn ${viewMode === "records" ? "active" : ""}`}
            onClick={() => setViewMode("records")}
          >
            📋 Clinical Audit Records ({filteredRecords.length})
          </button>
        </div>

        <div style={{ display: "flex", gap: "10px" }}>
          <select
            className="input-field"
            style={{ width: "160px", padding: "8px 12px", fontSize: "13px" }}
            value={filterAction}
            onChange={(e) => setFilterAction(e.target.value)}
          >
            <option value="ALL">All Actions</option>
            <option value="RAG_QUERY">RAG_QUERY</option>
            <option value="ML_PREDICTION">ML_PREDICTION</option>
            <option value="ACCESS_DECISION">ACCESS_DECISION</option>
            <option value="GENESIS_BLOCK">GENESIS_BLOCK</option>
          </select>

          <select
            className="input-field"
            style={{ width: "140px", padding: "8px 12px", fontSize: "13px" }}
            value={filterStatus}
            onChange={(e) => setFilterStatus(e.target.value)}
          >
            <option value="ALL">All Statuses</option>
            <option value="SUCCESS">SUCCESS</option>
            <option value="DENIED">DENIED</option>
            <option value="FAILED">FAILED</option>
          </select>
        </div>
      </div>

      {/* BLOCKS LIST VIEW */}
      {viewMode === "blocks" && (
        <div style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
          {filteredBlocks.map((block) => (
            <div
              key={block.index}
              className="glass-panel fade-in"
              style={{
                borderLeft: `4px solid ${block.status === "SUCCESS" ? "#10b981" : "#f43f5e"}`
              }}
            >
              {/* Block Header */}
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "14px", flexWrap: "wrap", gap: "10px" }}>
                <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
                  <span className="badge badge-purple" style={{ fontSize: "13px", padding: "4px 10px" }}>
                    Block #{block.index}
                  </span>
                  <span className={`badge ${block.status === "SUCCESS" ? "badge-success" : "badge-danger"}`}>
                    {block.status}
                  </span>
                  <span className="badge badge-secondary">{block.action}</span>
                  {block.provenance?.query_id && (
                    <span className="badge" style={{ background: "rgba(56, 189, 248, 0.2)", color: "var(--accent-cyan)", border: "1px solid rgba(56, 189, 248, 0.4)" }}>
                      {block.provenance.query_id}
                    </span>
                  )}
                </div>

                <span style={{ fontSize: "12px", color: "var(--text-muted)" }}>
                  {new Date(block.timestamp).toLocaleString()} · User: <strong style={{ color: "#fff" }}>{block.user_id}</strong>
                </span>
              </div>

              {/* Hashes Grid */}
              <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(300px, 1fr))", gap: "10px", fontSize: "12px", marginBottom: "12px" }}>
                <div style={{ background: "#080c14", padding: "10px 14px", borderRadius: "8px" }}>
                  <span style={{ color: "var(--text-muted)", display: "block", marginBottom: "4px" }}>Block Hash (SHA-256):</span>
                  <span className="hash-code">{block.hash}</span>
                </div>

                <div style={{ background: "#080c14", padding: "10px 14px", borderRadius: "8px" }}>
                  <span style={{ color: "var(--text-muted)", display: "block", marginBottom: "4px" }}>Previous Block Hash:</span>
                  <span className="hash-code">{block.previous_hash}</span>
                </div>
              </div>

              {/* Provenance Details */}
              {block.provenance && Object.keys(block.provenance).length > 0 && (
                <div style={{ background: "rgba(0, 0, 0, 0.2)", padding: "10px 14px", borderRadius: "8px", fontSize: "12px", marginBottom: "12px" }}>
                  <span style={{ color: "var(--text-muted)", display: "block", marginBottom: "4px" }}>Provenance Metadata (No raw patient text stored on-chain):</span>
                  <pre style={{ color: "#94a3b8", fontSize: "11px", margin: 0, overflow: "auto" }}>
                    {JSON.stringify(block.provenance, null, 2)}
                  </pre>
                </div>
              )}

              {/* Block Integrity Verification Status Panel */}
              {blockVerifyResult[block.index] && (
                <div
                  style={{
                    background: blockVerifyResult[block.index].verified ? "rgba(16, 185, 129, 0.12)" : "rgba(244, 63, 94, 0.12)",
                    border: `1px solid ${blockVerifyResult[block.index].verified ? "#10b981" : "#f43f5e"}`,
                    padding: "12px 16px",
                    borderRadius: "8px",
                    marginBottom: "12px",
                    fontSize: "13px"
                  }}
                >
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "6px" }}>
                    <strong style={{ color: blockVerifyResult[block.index].verified ? "#34d399" : "#fb7185" }}>
                      {blockVerifyResult[block.index].message}
                    </strong>
                    <span style={{ fontSize: "11px", color: "var(--text-muted)" }}>
                      Block #{block.index} Verification
                    </span>
                  </div>
                  <div style={{ fontSize: "11px", color: "#cbd5e1" }}>
                    <div>• Stored Block Hash: <span className="hash-code">{blockVerifyResult[block.index].stored_block_hash}</span></div>
                    <div>• Recalculated Hash: <span className="hash-code">{blockVerifyResult[block.index].recalculated_block_hash}</span></div>
                    {block.index > 0 && (
                      <div>• Previous Hash Linkage: {blockVerifyResult[block.index].previous_hash_valid ? "✅ Valid" : "❌ Broken Link"}</div>
                    )}
                  </div>
                </div>
              )}

              {/* Action Buttons */}
              <div style={{ display: "flex", gap: "10px", alignItems: "center", flexWrap: "wrap" }}>
                <button
                  className="btn-success"
                  style={{ fontSize: "12px", padding: "6px 14px" }}
                  onClick={() => handleVerifyBlockIntegrity(block.index)}
                >
                  🛡️ Verify Block Integrity
                </button>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* RECORDS LIST VIEW */}
      {viewMode === "records" && (
        <div style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
          {filteredRecords.map((rec, rIdx) => (
            <div
              key={rIdx}
              className="glass-panel fade-in"
              style={{
                borderLeft: `4px solid ${rec.status === "success" ? "#10b981" : "#f43f5e"}`
              }}
            >
              <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "10px", flexWrap: "wrap", gap: "10px" }}>
                <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                  <span className={`badge ${rec.status === "success" ? "badge-success" : "badge-danger"}`}>
                    {rec.status?.toUpperCase()}
                  </span>
                  <strong style={{ fontSize: "14px", color: "#fff" }}>{rec.action}</strong>
                  {rec.query_id && (
                    <span className="badge badge-purple" style={{ fontSize: "11px" }}>
                      {rec.query_id}
                    </span>
                  )}
                  <span className="badge badge-secondary" style={{ fontSize: "11px" }}>
                    {rec.data_type}
                  </span>
                </div>

                <span style={{ fontSize: "12px", color: "var(--text-muted)" }}>
                  {new Date(rec.timestamp).toLocaleString()} · User: <strong style={{ color: "#fff" }}>{rec.user_id}</strong>
                </span>
              </div>

              {rec.patient_id && (
                <p style={{ fontSize: "13px", color: "var(--accent-cyan)", marginBottom: "8px" }}>
                  Target Patient: {rec.patient_id}
                </p>
              )}

              {rec.error_message && (
                <p style={{ fontSize: "13px", color: "#fb7185", marginBottom: "8px" }}>
                  Error: {rec.error_message}
                </p>
              )}

              <div style={{ background: "#080c14", padding: "10px", borderRadius: "8px", marginBottom: "10px" }}>
                <pre style={{ fontSize: "11px", color: "#cbd5e1", margin: 0, overflow: "auto" }}>
                  {JSON.stringify(rec.details, null, 2)}
                </pre>
              </div>

              {rec.evidence_hash && (
                <div style={{ display: "flex", gap: "10px", alignItems: "center" }}>
                  <span style={{ fontSize: "12px", color: "var(--text-muted)" }}>
                    Evidence SHA-256 Hash: <span className="hash-code">{rec.evidence_hash}</span>
                  </span>
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export default AuditDashboard;