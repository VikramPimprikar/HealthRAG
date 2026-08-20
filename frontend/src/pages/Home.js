import React, { useState } from "react";
import { useNavigate } from "react-router-dom";

function Home({ activeUser, apiBase }) {
  const [query, setQuery] = useState("");
  const [topK, setTopK] = useState(5);
  const [results, setResults] = useState([]);
  const [answer, setAnswer] = useState("");
  const [evidenceHash, setEvidenceHash] = useState("");
  const [blockIndex, setBlockIndex] = useState(null);
  const [blockHash, setBlockHash] = useState("");
  const [hasRelevantEvidence, setHasRelevantEvidence] = useState(true);
  const [queryType, setQueryType] = useState("rag");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [copiedHash, setCopiedHash] = useState(null);

  const sampleQueries = [
    { label: "👤 Patient P1651 Findings", q: "Show patient P1651" },
    { label: "🩺 P1651's Cholesterol", q: "What is P1651's cholesterol?" },
    { label: "📊 Total Patients Count", q: "How many patients are there in total?" },
    { label: "🎂 Older than 65", q: "How many patients are older than 65?" },
    { label: "🍬 Glucose > 120", q: "How many patients have glucose > 120?" },
    { label: "📈 Average Age", q: "What is the average age?" },
    { label: "🔝 Highest Cholesterol", q: "What is the highest cholesterol?" },
    { label: "📊 Heart Disease %", q: "What percentage of patients have heart disease?" },
    { label: "🔬 Symptoms of Heart Disease", q: "What are the symptoms of heart disease?" },
    { label: "🩺 Causes of High Cholesterol", q: "What causes high cholesterol?" },
    { label: "⚡ Explain ST Depression", q: "Explain ST depression." },
    { label: "🚫 Off-Topic Guardrail", q: "Explain quantum computer quantum teleportation protocols in rocket engines" }
  ];

  // Helper to determine if query is structured
  const isStructuredQuery = (qType, qText) => {
    if (qType && (qType.startsWith("structured") || qType.startsWith("specific_patient"))) {
      return true;
    }
    const lower = (qText || "").toLowerCase();
    return (
      lower.includes("how many") ||
      lower.includes("total patient") ||
      lower.includes("percentage of") ||
      lower.includes("average") ||
      lower.includes("median") ||
      lower.includes("highest") ||
      lower.includes("lowest") ||
      lower.includes("minimum") ||
      lower.includes("maximum") ||
      lower.includes("show patient") ||
      lower.includes("what is p") ||
      lower.includes("patient p") ||
      lower.includes("older than") ||
      lower.includes("younger than")
    );
  };

  const handleSearch = async (searchQuery = query) => {
    const q = searchQuery.trim();
    if (!q) {
      alert("Please enter a medical question, patient query, or clinical term.");
      return;
    }

    setLoading(true);
    setError(null);
    setResults([]);
    setAnswer("");
    setEvidenceHash("");
    setBlockIndex(null);
    setBlockHash("");

    try {
      const response = await fetch(`${apiBase}/api/v1/query`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-User-Id": activeUser
        },
        body: JSON.stringify({
          query: q,
          user_id: activeUser,
          top_k: Number(topK)
        })
      });

      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        throw new Error(errData.detail || `HTTP Error ${response.status}: Search request failed`);
      }

      const data = await response.json();
      const isMismatch =
        data.is_mismatch === true ||
        data.has_relevant_evidence === false ||
        (data.answer && data.answer.includes("I apologize for the mismatch in the query"));

      const currentQueryType = data.query_type || (isMismatch ? "off_topic_mismatch" : "rag_medical");
      setQueryType(currentQueryType);

      if (isMismatch) {
        setResults([]);
        setHasRelevantEvidence(false);
      } else if (currentQueryType.startsWith("structured") || currentQueryType.startsWith("specific_patient")) {
        // Structured queries do NOT use or display Top-K vector results
        setResults([]);
        setHasRelevantEvidence(true);
      } else {
        setResults(data.retrieved_evidence || []);
        setHasRelevantEvidence(true);
      }

      setAnswer(data.answer || "");
      setEvidenceHash(isMismatch ? "" : (data.evidence_hash || ""));
      setBlockIndex(data.block_index);
      setBlockHash(data.block_hash || "");
    } catch (err) {
      console.error("Search failed:", err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const copyToClipboard = (text, key) => {
    navigator.clipboard.writeText(text);
    setCopiedHash(key);
    setTimeout(() => setCopiedHash(null), 2000);
  };

  const structuredActive = isStructuredQuery(queryType, query);

  return (
    <div className="fade-in" style={{ display: "flex", flexDirection: "column", gap: "24px" }}>
      {/* HEADER SECTION */}
      <div className="glass-panel" style={{ background: "linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.85) 100%)" }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", flexWrap: "wrap", gap: "16px" }}>
          <div>
            <h1 style={{ fontSize: "24px", fontWeight: "700", color: "#fff", marginBottom: "6px" }}>
              Clinical Evidence Retrieval &amp; Deterministic Patient Analytics
            </h1>
            <p style={{ color: "var(--text-secondary)", fontSize: "14px", maxWidth: "820px" }}>
              Separates grounded medical literature Q&amp;A from exact structured dataset analytics (920 patient records). Every transaction is cryptographically hashed and verified in the blockchain audit trail.
            </p>
          </div>

          <div style={{ display: "flex", gap: "10px" }}>
            <span className="badge badge-info">Dataset: 920 Structured Patient Records</span>
            <span className="badge badge-success">Grounded Medical RAG</span>
          </div>
        </div>

        {/* QUICK PRESET BUTTONS */}
        <div style={{ marginTop: "18px", display: "flex", flexWrap: "wrap", gap: "8px", alignItems: "center" }}>
          <span style={{ fontSize: "12px", color: "var(--text-muted)", fontWeight: "500" }}>Quick Queries:</span>
          {sampleQueries.map((item, idx) => (
            <button
              key={idx}
              type="button"
              className="btn-secondary"
              style={{ fontSize: "12px", padding: "5px 12px", borderRadius: "20px" }}
              onClick={() => {
                setQuery(item.q);
                handleSearch(item.q);
              }}
            >
              {item.label}
            </button>
          ))}
        </div>

        {/* SEARCH INPUT BAR */}
        <div style={{ marginTop: "20px", display: "flex", gap: "12px", alignItems: "center" }}>
          <div style={{ flex: 1, position: "relative" }}>
            <input
              type="text"
              placeholder="Ask normal medical questions (e.g. 'Symptoms of heart disease') or structured patient queries ('Show patient P1651', 'How many patients are older than 65?')..."
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleSearch()}
              style={{
                padding: "14px 18px",
                fontSize: "15px",
                borderRadius: "12px",
                background: "#080d1a",
                border: "1px solid rgba(99, 102, 241, 0.3)"
              }}
            />
          </div>

          {/* Top-K Selector is ONLY shown for general RAG retrieval, hidden/disabled for structured queries (Requirement 7) */}
          {!structuredActive && (
            <div style={{ width: "135px" }}>
              <select
                value={topK}
                onChange={(e) => setTopK(e.target.value)}
                style={{ padding: "14px", borderRadius: "12px", background: "#080d1a" }}
                title="Top-K context chunks retrieved for grounded medical RAG"
              >
                <option value="3">Top-3 Records</option>
                <option value="5">Top-5 Records</option>
                <option value="10">Top-10 Records</option>
              </select>
            </div>
          )}

          <button
            className="btn-primary"
            onClick={() => handleSearch()}
            disabled={loading}
            style={{ padding: "14px 28px", borderRadius: "12px", fontSize: "15px", minWidth: "140px" }}
          >
            {loading ? "Processing..." : "🔍 Execute Query"}
          </button>
        </div>
      </div>

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
            gap: "14px"
          }}
        >
          <span style={{ fontSize: "24px" }}>⛔</span>
          <div>
            <strong style={{ display: "block", fontSize: "15px" }}>RBAC Access Control / Request Error</strong>
            <span style={{ fontSize: "13px" }}>{error}</span>
          </div>
        </div>
      )}

      {/* RESPONSE / ANSWER SECTION */}
      {answer && (
        <div
          className="glass-panel fade-in"
          style={{
            border: hasRelevantEvidence
              ? (queryType.startsWith("specific_patient")
                  ? "1px solid rgba(147, 51, 234, 0.5)"
                  : queryType.startsWith("structured")
                  ? "1px solid rgba(59, 130, 246, 0.5)"
                  : "1px solid rgba(16, 185, 129, 0.4)")
              : "1px solid rgba(245, 158, 11, 0.4)"
          }}
        >
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "16px", flexWrap: "wrap", gap: "10px" }}>
            <div style={{ display: "flex", alignItems: "center", gap: "10px", flexWrap: "wrap" }}>
              <span style={{ fontSize: "20px" }}>
                {queryType.startsWith("specific_patient")
                  ? "👤"
                  : queryType.startsWith("structured")
                  ? "📊"
                  : "🩺"}
              </span>
              <h2 style={{ fontSize: "18px", fontWeight: "700", color: "#fff" }}>
                {queryType.startsWith("specific_patient")
                  ? "Structured Patient Record"
                  : queryType.startsWith("structured")
                  ? "Deterministic Dataset Analytics"
                  : "Grounded Medical Response"}
              </h2>
              {hasRelevantEvidence ? (
                queryType.startsWith("specific_patient") ? (
                  <span className="badge badge-purple">👤 Exact Dataset Record</span>
                ) : queryType.startsWith("structured") ? (
                  <span className="badge badge-info">📊 Calculated from 920 Records (No Vector Estimation)</span>
                ) : (
                  <span className="badge badge-success">✓ Grounded in Verified Medical Literature</span>
                )
              ) : (
                <span className="badge badge-warning">⚠️ Insufficient Context / Out of Scope</span>
              )}
            </div>

            {blockIndex !== null && (
              <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                <span className="badge badge-purple" title={`Transaction Hash: ${blockHash}`}>
                  ⛓️ Blockchain Block #{blockIndex}
                </span>
                <button
                  className="btn-secondary"
                  style={{ fontSize: "11px", padding: "4px 10px" }}
                  onClick={() => navigate("/audit")}
                >
                  View Block
                </button>
              </div>
            )}
          </div>

          <div
            style={{
              background: "rgba(11, 15, 25, 0.85)",
              padding: "18px 22px",
              borderRadius: "10px",
              fontSize: "14px",
              lineHeight: "1.75",
              color: "#e2e8f0",
              whiteSpace: "pre-wrap",
              border: hasRelevantEvidence ? "1px solid var(--border-color)" : "1px solid rgba(245, 158, 11, 0.3)"
            }}
          >
            {answer}
          </div>

          {/* Evidence Bundle Hash Bar */}
          {evidenceHash && hasRelevantEvidence && (
            <div
              style={{
                marginTop: "16px",
                display: "flex",
                alignItems: "center",
                justifyContent: "space-between",
                background: "rgba(0, 0, 0, 0.3)",
                padding: "10px 14px",
                borderRadius: "8px",
                fontSize: "12px",
                flexWrap: "wrap",
                gap: "8px"
              }}
            >
              <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                <span style={{ color: "var(--text-muted)" }}>Cryptographic SHA-256 Provenance Hash:</span>
                <span className="hash-code">{evidenceHash}</span>
              </div>

              <button
                className="btn-secondary"
                style={{ fontSize: "11px", padding: "3px 8px" }}
                onClick={() => copyToClipboard(evidenceHash, "bundle")}
              >
                {copiedHash === "bundle" ? "✓ Copied" : "Copy Hash"}
              </button>
            </div>
          )}
        </div>
      )}

      {/* RETRIEVED EVIDENCE SECTION (ONLY FOR GENERAL MEDICAL RAG QUERIES) */}
      {!queryType.startsWith("structured") && !queryType.startsWith("specific_patient") && (
        <div style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <h2 style={{ fontSize: "18px", fontWeight: "700", color: "#fff" }}>
              {`Retrieved Medical Evidence Cards (${hasRelevantEvidence ? results.length : 0})`}
            </h2>
          </div>

          {answer && !hasRelevantEvidence && (
            <div
              className="glass-panel fade-in"
              style={{ textAlign: "center", padding: "36px 20px", color: "var(--text-muted)" }}
            >
              <span style={{ fontSize: "36px", display: "block", marginBottom: "10px" }}>🛡️</span>
              <p style={{ fontSize: "15px", fontWeight: "600", color: "#f59e0b" }}>
                Evidence Withheld / Out of Scope
              </p>
              <p style={{ fontSize: "13px", marginTop: "4px" }}>
                Retrieved records are hidden when a query is out of scope or lacks sufficient context to prevent hallucination.
              </p>
            </div>
          )}

          {results.length === 0 && !answer && !loading && !error && (
            <div
              className="glass-panel"
              style={{ textAlign: "center", padding: "40px 20px", color: "var(--text-muted)" }}
            >
              <span style={{ fontSize: "36px", display: "block", marginBottom: "12px" }}>📋</span>
              <p style={{ fontSize: "15px" }}>No query executed yet.</p>
              <p style={{ fontSize: "13px", marginTop: "4px" }}>
                Select a quick query above or enter structured / clinical terms to query patient records or medical knowledge.
              </p>
            </div>
          )}

          {/* Evidence Cards - WITHOUT ANY SIMILARITY SCORES (Requirement 8) */}
          {hasRelevantEvidence && results.map((item, index) => {
            const meta = item.metadata || {};
            const isMedicalKb = meta.doc_type === "medical_knowledge" || (meta.id && meta.id.startsWith("KB"));

            return (
              <div
                key={index}
                className="glass-panel fade-in"
                style={{
                  borderLeft: isMedicalKb ? "4px solid #10b981" : "4px solid #6366f1",
                  display: "flex",
                  flexDirection: "column",
                  gap: "14px"
                }}
              >
                {/* Evidence Card Header */}
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: "10px" }}>
                  <div style={{ display: "flex", alignItems: "center", gap: "10px", flexWrap: "wrap" }}>
                    <span className="badge badge-purple" style={{ fontWeight: "700" }}>
                      #{item.rank || index + 1}
                    </span>
                    <span style={{ fontSize: "15px", fontWeight: "700", color: "#fff" }}>
                      {isMedicalKb ? (meta.title || "Clinical Knowledge Article") : `Patient ID: ${meta.id || item.patient_id}`}
                    </span>
                    {meta.category && (
                      <span className="badge badge-info" style={{ fontSize: "11px" }}>
                        {meta.category}
                      </span>
                    )}
                    {meta.age && (
                      <span className="badge badge-secondary" style={{ fontSize: "11px" }}>
                        Age: {meta.age} | Sex: {meta.sex}
                      </span>
                    )}
                  </div>

                  <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
                    <button
                      className="btn-primary"
                      style={{ fontSize: "12px", padding: "6px 14px" }}
                      onClick={() => navigate("/audit")}
                      title="View in Blockchain Audit Trail"
                    >
                      ⛓️ View in Audit Trail
                    </button>
                  </div>
                </div>

                {/* Biomarkers / Attributes Tag Bar (No similarity scores) */}
                {!isMedicalKb && (
                  <div style={{ display: "flex", flexWrap: "wrap", gap: "8px" }}>
                    {meta.chest_pain && (
                      <span className="badge badge-info" style={{ fontSize: "11px" }}>
                        Chest Pain: {meta.chest_pain}
                      </span>
                    )}
                    {meta.resting_bp && (
                      <span className="badge badge-secondary" style={{ fontSize: "11px" }}>
                        BP: {meta.resting_bp} mmHg
                      </span>
                    )}
                    {meta.cholesterol && (
                      <span className="badge badge-warning" style={{ fontSize: "11px" }}>
                        Cholesterol: {meta.cholesterol} mg/dL
                      </span>
                    )}
                    {meta.max_heart_rate && (
                      <span className="badge badge-secondary" style={{ fontSize: "11px" }}>
                        Max HR: {meta.max_heart_rate} bpm
                      </span>
                    )}
                    {meta.diagnosis_label && (
                      <span
                        className={`badge ${meta.diagnosis_outcome === 0 ? "badge-success" : "badge-danger"}`}
                        style={{ fontSize: "11px" }}
                      >
                        Diagnosis: {meta.diagnosis_label}
                      </span>
                    )}
                  </div>
                )}

                {/* Evidence Text */}
                <div
                  style={{
                    background: "#080c14",
                    padding: "14px",
                    borderRadius: "8px",
                    fontSize: "13px",
                    color: "#cbd5e1",
                    lineHeight: "1.6"
                  }}
                >
                  {item.text}
                </div>

                {/* Hash Footer */}
                <div
                  style={{
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "space-between",
                    fontSize: "11px",
                    color: "var(--text-muted)",
                    borderTop: "1px solid rgba(255, 255, 255, 0.05)",
                    paddingTop: "10px"
                  }}
                >
                  <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
                    <span>Chunk SHA-256:</span>
                    <span className="hash-code">{item.sha256_hash}</span>
                  </div>

                  <button
                    className="btn-secondary"
                    style={{ fontSize: "10px", padding: "2px 8px" }}
                    onClick={() => copyToClipboard(item.sha256_hash, `chunk-${index}`)}
                  >
                    {copiedHash === `chunk-${index}` ? "✓ Copied" : "Copy"}
                  </button>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

export default Home;