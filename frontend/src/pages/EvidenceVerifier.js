import React, { useState, useEffect } from "react";
import { useLocation } from "react-router-dom";

function EvidenceVerifier({ activeUser, apiBase }) {
  const location = useLocation();

  const [blockIndex, setBlockIndex] = useState(1);
  const [evidenceHash, setEvidenceHash] = useState("");
  const [evidenceText, setEvidenceText] = useState("");
  const [verificationResult, setVerificationResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Pre-fill if routed from Home or AuditDashboard
  useEffect(() => {
    if (location.state) {
      if (location.state.blockIndex !== undefined && location.state.blockIndex !== null) {
        setBlockIndex(location.state.blockIndex);
      }
      if (location.state.evidenceHash) {
        setEvidenceHash(location.state.evidenceHash);
      }
      if (location.state.evidenceText) {
        setEvidenceText(location.state.evidenceText);
      }
    }
  }, [location.state]);

  const handleVerify = async (e) => {
    if (e) e.preventDefault();
    if (!evidenceHash && !evidenceText) {
      alert("Please provide either the Evidence SHA-256 Hash or the Evidence Text.");
      return;
    }

    setLoading(true);
    setError(null);
    setVerificationResult(null);

    try {
      const payload = {
        block_index: Number(blockIndex)
      };

      if (evidenceHash.trim()) {
        payload.evidence_hash = evidenceHash.trim();
      }
      if (evidenceText.trim()) {
        payload.evidence_text = evidenceText.trim();
      }

      const response = await fetch(`${apiBase}/api/v1/audit/verify-evidence`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-User-Id": activeUser
        },
        body: JSON.stringify(payload)
      });

      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        throw new Error(errData.detail || `HTTP Error ${response.status}: Verification request failed`);
      }

      const data = await response.json();
      setVerificationResult(data);
    } catch (err) {
      console.error("Verification error:", err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const simulateTampering = () => {
    if (evidenceText) {
      // Modify single character
      setEvidenceText(evidenceText + " [TAMPERED_MODIFIED]");
    } else if (evidenceHash) {
      setEvidenceHash("0000000000000000000000000000000000000000000000000000000000000000");
    } else {
      setEvidenceText("Patient ID P1000 with fabricated high blood pressure 220 mmHg.");
    }
    setVerificationResult(null);
  };

  return (
    <div className="fade-in" style={{ display: "flex", flexDirection: "column", gap: "24px" }}>
      {/* HEADER */}
      <div className="glass-panel" style={{ background: "linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.85) 100%)" }}>
        <h1 style={{ fontSize: "24px", fontWeight: "700", color: "#fff", marginBottom: "6px" }}>
          🛡️ Cryptographic Evidence Integrity Verifier
        </h1>
        <p style={{ color: "var(--text-secondary)", fontSize: "14px", maxWidth: "800px" }}>
          Verify the authenticity of any clinical record or retrieved evidence against the immutable blockchain. If even a single byte or digit was altered, the cryptographic SHA-256 proof will fail and flag data tampering.
        </p>
      </div>

      {/* ERROR BANNER */}
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
            gap: "14px"
          }}
        >
          <span style={{ fontSize: "24px" }}>⛔</span>
          <div>
            <strong style={{ display: "block", fontSize: "15px" }}>Verification Failed</strong>
            <span style={{ fontSize: "13px" }}>{error}</span>
          </div>
        </div>
      )}

      {/* VERIFIER FORM */}
      <div className="glass-panel">
        <form onSubmit={handleVerify} style={{ display: "flex", flexDirection: "column", gap: "18px" }}>
          <div>
            <label style={{ display: "block", fontSize: "13px", fontWeight: "600", color: "#fff", marginBottom: "6px" }}>
              Target Blockchain Block Index:
            </label>
            <input
              type="number"
              value={blockIndex}
              onChange={(e) => setBlockIndex(e.target.value)}
              min="0"
              style={{ width: "200px" }}
              required
            />
            <span style={{ fontSize: "11px", color: "var(--text-muted)", display: "block", marginTop: "4px" }}>
              Specify the block number where the query or prediction was recorded (e.g. Block #1, #2).
            </span>
          </div>

          <div>
            <label style={{ display: "block", fontSize: "13px", fontWeight: "600", color: "#fff", marginBottom: "6px" }}>
              Evidence SHA-256 Hash:
            </label>
            <input
              type="text"
              placeholder="e.g. ebd2063d34a9617651a0d4cf9ca384be896f64efb6ea63e52f5597793d98d249"
              value={evidenceHash}
              onChange={(e) => setEvidenceHash(e.target.value)}
            />
          </div>

          <div style={{ textAlign: "center", color: "var(--text-muted)", fontSize: "12px", fontWeight: "600" }}>
            — OR PASTE EVIDENCE TEXT TO RE-COMPUTE HASH —
          </div>

          <div>
            <label style={{ display: "block", fontSize: "13px", fontWeight: "600", color: "#fff", marginBottom: "6px" }}>
              Raw Evidence Text:
            </label>
            <textarea
              rows="4"
              placeholder="Paste patient narrative or evidence chunk here to verify..."
              value={evidenceText}
              onChange={(e) => setEvidenceText(e.target.value)}
            />
          </div>

          <div style={{ display: "flex", gap: "12px", alignItems: "center", flexWrap: "wrap" }}>
            <button
              type="submit"
              className="btn-primary"
              disabled={loading}
              style={{ padding: "12px 24px", fontSize: "14px" }}
            >
              {loading ? "Verifying On Blockchain..." : "🛡️ Verify Against Blockchain"}
            </button>

            <button
              type="button"
              className="btn-secondary"
              onClick={simulateTampering}
              style={{ padding: "12px 20px", fontSize: "14px", color: "#fb7185" }}
            >
              ⚠️ Simulate Tampered Data (Test Detection)
            </button>
          </div>
        </form>
      </div>

      {/* VERIFICATION RESULT CARD */}
      {verificationResult && (
        <div
          className="glass-panel fade-in"
          style={{
            borderLeft: `5px solid ${verificationResult.verified ? "#10b981" : "#f43f5e"}`
          }}
        >
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "16px" }}>
            <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
              <span style={{ fontSize: "28px" }}>
                {verificationResult.verified ? "✅" : "❌"}
              </span>
              <div>
                <h2
                  style={{
                    fontSize: "20px",
                    fontWeight: "800",
                    color: verificationResult.verified ? "#34d399" : "#fb7185"
                  }}
                >
                  {verificationResult.verified
                    ? "CRYPTOGRAPHICALLY AUTHENTIC & VERIFIED"
                    : "TAMPER DETECTED / EVIDENCE MISMATCH"}
                </h2>
                <span style={{ fontSize: "13px", color: "var(--text-secondary)" }}>
                  {verificationResult.verified
                    ? "The supplied evidence matches the on-chain SHA-256 record exactly. Zero data corruption or unauthorized changes."
                    : "The computed hash does NOT match the immutable blockchain record. The data has been modified or originates from a different block."}
                </span>
              </div>
            </div>

            <span className={`badge ${verificationResult.chain_valid ? "badge-success" : "badge-danger"}`}>
              {verificationResult.chain_valid ? "Chain Verified" : "Chain Alert"}
            </span>
          </div>

          {/* Comparison Table */}
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "16px", marginBottom: "16px" }}>
            <div style={{ background: "#080c14", padding: "14px", borderRadius: "8px" }}>
              <span style={{ fontSize: "12px", color: "var(--text-muted)", display: "block", marginBottom: "6px" }}>
                Stored On-Chain Evidence Hash (Block #{verificationResult.block_index}):
              </span>
              <span className="hash-code">{verificationResult.stored_evidence_hash}</span>
            </div>

            <div style={{ background: "#080c14", padding: "14px", borderRadius: "8px" }}>
              <span style={{ fontSize: "12px", color: "var(--text-muted)", display: "block", marginBottom: "6px" }}>
                Computed Evidence Hash:
              </span>
              <span
                className="hash-code"
                style={{
                  color: verificationResult.verified ? "#34d399" : "#fb7185"
                }}
              >
                {verificationResult.computed_evidence_hash}
              </span>
            </div>
          </div>

          {/* Block Provenance Info */}
          <div style={{ display: "flex", gap: "20px", fontSize: "12px", color: "var(--text-muted)", flexWrap: "wrap" }}>
            <span>Block Timestamp: <strong style={{ color: "#fff" }}>{verificationResult.block_timestamp}</strong></span>
            <span>Recorded Subject: <strong style={{ color: "#fff" }}>{verificationResult.user_id}</strong></span>
            <span>Action: <strong style={{ color: "#fff" }}>{verificationResult.action}</strong></span>
            <span>Status: <strong style={{ color: "#fff" }}>{verificationResult.status}</strong></span>
          </div>
        </div>
      )}
    </div>
  );
}

export default EvidenceVerifier;
