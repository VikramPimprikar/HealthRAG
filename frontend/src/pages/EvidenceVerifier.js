import React, { useState, useEffect } from "react";
import { useLocation } from "react-router-dom";

function EvidenceVerifier({ activeUser, apiBase }) {
  const location = useLocation();

  const [verifyMode, setVerifyMode] = useState("direct"); // "direct" (text + hash) or "blockchain" (block / query id)
  const [blockIndex, setBlockIndex] = useState(1);
  const [queryId, setQueryId] = useState("");
  const [storedHash, setStoredHash] = useState("");
  const [originalStoredHash, setOriginalStoredHash] = useState("");
  const [evidenceText, setEvidenceText] = useState("");
  const [originalEvidenceText, setOriginalEvidenceText] = useState("");
  const [verificationResult, setVerificationResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Pre-fill if routed from Home or AuditDashboard
  useEffect(() => {
    if (location.state) {
      if (location.state.blockIndex !== undefined && location.state.blockIndex !== null) {
        setBlockIndex(location.state.blockIndex);
        setVerifyMode("blockchain");
      }
      if (location.state.queryId) {
        setQueryId(location.state.queryId);
        setVerifyMode("blockchain");
      }
      if (location.state.evidenceHash || location.state.storedHash) {
        const h = location.state.evidenceHash || location.state.storedHash;
        setStoredHash(h);
        setOriginalStoredHash(h);
      }
      if (location.state.evidenceText) {
        setEvidenceText(location.state.evidenceText);
        setOriginalEvidenceText(location.state.evidenceText);
      }
    }
  }, [location.state]);

  const handleVerify = async (e) => {
    if (e) e.preventDefault();
    setLoading(true);
    setError(null);
    setVerificationResult(null);

    try {
      let endpoint = "";
      let payload = {};

      if (verifyMode === "direct") {
        if (!evidenceText.trim()) {
          throw new Error("Please enter evidence text to verify.");
        }
        if (!storedHash.trim()) {
          throw new Error("Please enter the expected / stored SHA-256 hash.");
        }
        endpoint = `${apiBase}/api/v1/audit/verify-integrity`;
        payload = {
          evidence_text: evidenceText,
          stored_hash: storedHash.trim()
        };
      } else {
        // Blockchain reference mode
        endpoint = `${apiBase}/api/v1/audit/verify-evidence`;
        payload = {
          block_index: Number(blockIndex),
          evidence_text: evidenceText,
          stored_hash: storedHash.trim() ? storedHash.trim() : undefined
        };
      }

      const response = await fetch(endpoint, {
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

  // Tamper Simulation Helpers
  const simulateTamperingValue = () => {
    if (!originalEvidenceText && evidenceText) {
      setOriginalEvidenceText(evidenceText);
    }
    const current = evidenceText || "Patient ID P1651: Resting BP: 140 mmHg, Cholesterol: 250 mg/dL, Chest Pain: Typical Angina.";
    if (!originalEvidenceText) {
      setOriginalEvidenceText(current);
    }

    // Tamper single clinical biomarker value (e.g. cholesterol 250 -> 150)
    let modified = current;
    if (modified.includes("250")) {
      modified = modified.replace("250", "150 (TAMPERED)");
    } else if (modified.includes("140")) {
      modified = modified.replace("140", "185 (TAMPERED)");
    } else if (modified.includes("240")) {
      modified = modified.replace("240", "390 (TAMPERED)");
    } else {
      // Find first number and alter it
      const numMatch = modified.match(/\b\d+(\.\d+)?\b/);
      if (numMatch) {
        const origVal = numMatch[0];
        const alteredVal = (parseFloat(origVal) + 50).toString() + " (TAMPERED)";
        modified = modified.replace(origVal, alteredVal);
      } else {
        modified = modified + " [TAMPERED_MODIFICATION]";
      }
    }

    setEvidenceText(modified);
    setVerificationResult(null);
  };

  const simulateTamperingHash = () => {
    if (!originalStoredHash && storedHash) {
      setOriginalStoredHash(storedHash);
    }
    const current = storedHash || "ebd2063d34a9617651a0d4cf9ca384be896f64efb6ea63e52f5597793d98d249";
    if (!originalStoredHash) {
      setOriginalStoredHash(current);
    }

    // Flip leading hex characters
    const modifiedHash = current.startsWith("ffff")
      ? "0000" + current.slice(4)
      : "ffff" + current.slice(4);

    setStoredHash(modifiedHash);
    setVerificationResult(null);
  };

  const restoreOriginal = () => {
    if (originalEvidenceText) {
      setEvidenceText(originalEvidenceText);
    } else {
      setEvidenceText("Patient ID P1651: Resting BP: 140 mmHg, Cholesterol: 250 mg/dL, Chest Pain: Typical Angina.");
    }
    if (originalStoredHash) {
      setStoredHash(originalStoredHash);
    }
    setVerificationResult(null);
  };

  return (
    <div className="fade-in" style={{ display: "flex", flexDirection: "column", gap: "24px" }}>
      {/* HEADER */}
      <div className="glass-panel" style={{ background: "linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.85) 100%)" }}>
        <h1 style={{ fontSize: "24px", fontWeight: "700", color: "#fff", marginBottom: "6px" }}>
          🛡️ Evidence Tampering Detection &amp; Cryptographic Verifier
        </h1>
        <p style={{ color: "var(--text-secondary)", fontSize: "14px", maxWidth: "880px" }}>
          Recalculates SHA-256 over current medical evidence text and compares it against the original stored hash (from the immutable blockchain ledger). If even a single byte or biomarker is modified, tampering is immediately flagged.
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
            padding: "16px 20px",
            borderRadius: "var(--radius-md)",
            display: "flex",
            alignItems: "center",
            gap: "12px"
          }}
        >
          <span style={{ fontSize: "22px" }}>⛔</span>
          <div>
            <strong style={{ display: "block", fontSize: "14px" }}>Verification Request Error</strong>
            <span style={{ fontSize: "13px" }}>{error}</span>
          </div>
        </div>
      )}

      {/* VERIFIER FORM */}
      <div className="glass-panel">
        {/* Mode Selector */}
        <div style={{ display: "flex", gap: "10px", marginBottom: "20px" }}>
          <button
            type="button"
            className={verifyMode === "direct" ? "btn-primary" : "btn-secondary"}
            onClick={() => { setVerifyMode("direct"); setVerificationResult(null); }}
            style={{ fontSize: "13px", padding: "8px 16px" }}
          >
            🔬 Direct Evidence Text + Stored Hash Verification
          </button>
          <button
            type="button"
            className={verifyMode === "blockchain" ? "btn-primary" : "btn-secondary"}
            onClick={() => { setVerifyMode("blockchain"); setVerificationResult(null); }}
            style={{ fontSize: "13px", padding: "8px 16px" }}
          >
            ⛓️ Blockchain Block / Query ID Reference
          </button>
        </div>

        <form onSubmit={handleVerify} style={{ display: "flex", flexDirection: "column", gap: "18px" }}>
          {verifyMode === "blockchain" && (
            <div>
              <label style={{ display: "block", fontSize: "13px", fontWeight: "600", color: "#fff", marginBottom: "6px" }}>
                Target Blockchain Block Index:
              </label>
              <input
                type="number"
                value={blockIndex}
                onChange={(e) => setBlockIndex(e.target.value)}
                min="0"
                style={{ width: "220px" }}
                required
              />
              <span style={{ fontSize: "11px", color: "var(--text-muted)", display: "block", marginTop: "4px" }}>
                Specify the block index in the append-only ledger (e.g. Block #1, #2).
              </span>
            </div>
          )}

          <div>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "6px" }}>
              <label style={{ fontSize: "13px", fontWeight: "600", color: "#fff" }}>
                Originally Stored SHA-256 Hash {verifyMode === "blockchain" ? "(Optional - fetched from block if empty)" : "(Required)"}:
              </label>
              <button
                type="button"
                onClick={simulateTamperingHash}
                style={{
                  fontSize: "11px",
                  background: "rgba(245, 158, 11, 0.15)",
                  border: "1px solid #f59e0b",
                  color: "#fbbf24",
                  padding: "2px 8px",
                  borderRadius: "6px",
                  cursor: "pointer"
                }}
              >
                🔀 Simulate Altered Stored Hash (Test Hash Tampering)
              </button>
            </div>
            <input
              type="text"
              placeholder="e.g. ebd2063d34a9617651a0d4cf9ca384be896f64efb6ea63e52f5597793d98d249"
              value={storedHash}
              onChange={(e) => setStoredHash(e.target.value)}
              required={verifyMode === "direct"}
            />
          </div>

          <div>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "6px" }}>
              <label style={{ fontSize: "13px", fontWeight: "600", color: "#fff" }}>
                Current Evidence Text to Recalculate &amp; Verify:
              </label>
              <div style={{ display: "flex", gap: "8px" }}>
                <button
                  type="button"
                  onClick={simulateTamperingValue}
                  style={{
                    fontSize: "11px",
                    background: "rgba(244, 63, 94, 0.15)",
                    border: "1px solid #f43f5e",
                    color: "#fb7185",
                    padding: "3px 8px",
                    borderRadius: "6px",
                    cursor: "pointer"
                  }}
                >
                  ⚠️ Simulate Altered Value (Test Evidence Tampering)
                </button>
                <button
                  type="button"
                  onClick={restoreOriginal}
                  style={{
                    fontSize: "11px",
                    background: "rgba(99, 102, 241, 0.15)",
                    border: "1px solid #6366f1",
                    color: "#a5b4fc",
                    padding: "3px 8px",
                    borderRadius: "6px",
                    cursor: "pointer"
                  }}
                >
                  🔄 Restore Original
                </button>
              </div>
            </div>
            <textarea
              rows="6"
              placeholder="Paste evidence text or patient narrative here..."
              value={evidenceText}
              onChange={(e) => setEvidenceText(e.target.value)}
            />
          </div>

          <div style={{ display: "flex", gap: "12px", alignItems: "center", flexWrap: "wrap", marginTop: "6px" }}>
            <button
              type="submit"
              className="btn-primary"
              disabled={loading}
              style={{ padding: "12px 28px", fontSize: "14px", fontWeight: "600" }}
            >
              {loading ? "Recalculating & Verifying..." : "🛡️ Recalculate SHA-256 & Verify Integrity"}
            </button>
          </div>
        </form>
      </div>

      {/* VERIFICATION RESULT CARD */}
      {verificationResult && (
        <div
          className="glass-panel fade-in"
          style={{
            borderLeft: `6px solid ${verificationResult.verified ? "#10b981" : "#f43f5e"}`,
            background: verificationResult.verified ? "rgba(16, 185, 129, 0.05)" : "rgba(244, 63, 94, 0.05)"
          }}
        >
          {/* Main Status Header */}
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "18px", flexWrap: "wrap", gap: "12px" }}>
            <div style={{ display: "flex", alignItems: "center", gap: "14px" }}>
              <span style={{ fontSize: "36px" }}>
                {verificationResult.verified ? "✅" : "⚠️"}
              </span>
              <div>
                <h2
                  style={{
                    fontSize: "20px",
                    fontWeight: "800",
                    color: verificationResult.verified ? "#34d399" : "#fb7185",
                    margin: 0
                  }}
                >
                  {verificationResult.verified
                    ? "✅ Evidence Integrity Verified — No Tampering Detected"
                    : "⚠️ Evidence Tampering Detected — Hash Mismatch"}
                </h2>
                <p style={{ fontSize: "13px", color: "var(--text-secondary)", margin: "4px 0 0 0" }}>
                  {verificationResult.message || (
                    verificationResult.verified
                      ? "Evidence integrity verified. No tampering detected."
                      : "Evidence tampering detected. Hash mismatch."
                  )}
                </p>
              </div>
            </div>

            <span
              className={`badge ${verificationResult.verified ? "badge-success" : "badge-danger"}`}
              style={{ fontSize: "13px", padding: "6px 14px", fontWeight: "700" }}
            >
              {verificationResult.verified ? "VERIFIED / AUTHENTIC" : "TAMPERING DETECTED"}
            </span>
          </div>

          {/* Hash Comparison Table */}
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "16px", marginBottom: "16px" }}>
            <div style={{ background: "#080c14", padding: "14px", borderRadius: "8px", border: "1px solid var(--border-color)" }}>
              <span style={{ fontSize: "12px", color: "var(--text-muted)", display: "block", marginBottom: "6px" }}>
                Original / Stored SHA-256 Hash:
              </span>
              <span className="hash-code" style={{ fontSize: "12px" }}>
                {verificationResult.stored_hash || verificationResult.stored_evidence_hash || "N/A"}
              </span>
            </div>

            <div style={{ background: "#080c14", padding: "14px", borderRadius: "8px", border: `1px solid ${verificationResult.verified ? "#10b981" : "#f43f5e"}` }}>
              <span style={{ fontSize: "12px", color: "var(--text-muted)", display: "block", marginBottom: "6px" }}>
                Recalculated Current SHA-256 Hash:
              </span>
              <span
                className="hash-code"
                style={{
                  fontSize: "12px",
                  color: verificationResult.verified ? "#34d399" : "#fb7185",
                  fontWeight: "bold"
                }}
              >
                {verificationResult.current_hash || verificationResult.computed_evidence_hash || "N/A"}
              </span>
            </div>
          </div>

          {/* EXPLICIT DEBUG OUTPUT PANEL */}
          <div
            style={{
              background: "#080d1a",
              border: "1px solid rgba(99, 102, 241, 0.25)",
              borderRadius: "8px",
              padding: "14px 18px",
              marginBottom: "14px",
              fontSize: "12px"
            }}
          >
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "8px" }}>
              <strong style={{ color: "#a5b4fc", fontSize: "13px" }}>
                🔍 Cryptographic Verification Debug Output
              </strong>
              <span
                style={{
                  fontWeight: "700",
                  color: verificationResult.verified ? "#34d399" : "#fb7185"
                }}
              >
                Status: {verificationResult.debug?.verification_result || (verificationResult.verified ? "VERIFIED / NO TAMPERING" : "TAMPERING DETECTED")}
              </span>
            </div>

            <div style={{ display: "grid", gridTemplateColumns: "160px 1fr", gap: "6px 12px", color: "#cbd5e1", fontFamily: "monospace" }}>
              <span style={{ color: "var(--text-muted)" }}>Stored Evidence Hash:</span>
              <span style={{ color: "#93c5fd", wordBreak: "break-all" }}>
                {verificationResult.debug?.stored_evidence_hash || verificationResult.stored_hash || "N/A"}
              </span>

              <span style={{ color: "var(--text-muted)" }}>Recalculated Current Hash:</span>
              <span style={{ color: verificationResult.verified ? "#34d399" : "#fb7185", wordBreak: "break-all" }}>
                {verificationResult.debug?.recalculated_current_hash || verificationResult.current_hash || "N/A"}
              </span>

              <span style={{ color: "var(--text-muted)" }}>Verification Result:</span>
              <span style={{ fontWeight: "700", color: verificationResult.verified ? "#34d399" : "#fb7185" }}>
                {verificationResult.debug?.verification_result || (verificationResult.verified ? "VERIFIED / NO TAMPERING" : "TAMPERING DETECTED")}
              </span>
            </div>
          </div>

          {/* Additional Provenance Metadata */}
          {verificationResult.block_index !== undefined && verificationResult.block_index !== null && (
            <div style={{ display: "flex", gap: "20px", fontSize: "12px", color: "var(--text-muted)", flexWrap: "wrap", paddingTop: "8px", borderTop: "1px solid rgba(255, 255, 255, 0.05)" }}>
              <span>Block Index: <strong style={{ color: "#fff" }}>#{verificationResult.block_index}</strong></span>
              {verificationResult.block_timestamp && <span>Timestamp: <strong style={{ color: "#fff" }}>{verificationResult.block_timestamp}</strong></span>}
              {verificationResult.user_id && <span>User: <strong style={{ color: "#fff" }}>{verificationResult.user_id}</strong></span>}
              {verificationResult.chain_valid !== undefined && (
                <span>Blockchain Ledger: <strong style={{ color: verificationResult.chain_valid ? "#34d399" : "#fb7185" }}>{verificationResult.chain_valid ? "Valid & Intact" : "Chain Alert"}</strong></span>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default EvidenceVerifier;
