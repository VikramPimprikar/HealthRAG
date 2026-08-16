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
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [copiedHash, setCopiedHash] = useState(null);

  const navigate = useNavigate();

  const sampleQueries = [
    { label: "High Cholesterol & Angina", q: "Patients with high cholesterol level and exercise induced angina" },
    { label: "Patient P1005 Record", q: "What is the diagnosis outcome and clinical indicators for patient P1005?" },
    { label: "Atypical Chest Pain & High HR", q: "Patients with chest pain type atypical angina and maximum heart rate above 150" },
    { label: "Severe CAD Outcomes", q: "Find patient records with diagnosis outcome severe or critical heart disease" },
    { label: "🚫 Off-Topic (Hallucination Test)", q: "Explain quantum computer quantum teleportation protocols in rocket engines" }
  ];

  const handleSearch = async (searchQuery = query) => {
    const q = searchQuery.trim();
    if (!q) {
      alert("Please enter a medical query or patient search term.");
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
      setResults(data.retrieved_evidence || []);
      setAnswer(data.answer || "");
      setEvidenceHash(data.evidence_hash || "");
      setBlockIndex(data.block_index);
      setBlockHash(data.block_hash || "");
      setHasRelevantEvidence(data.has_relevant_evidence !== false);
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

  return (
    <div className="fade-in" style={{ display: "flex", flexDirection: "column", gap: "24px" }}>
      {/* HEADER SECTION */}
      <div className="glass-panel" style={{ background: "linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.85) 100%)" }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", flexWrap: "wrap", gap: "16px" }}>
          <div>
            <h1 style={{ fontSize: "24px", fontWeight: "700", color: "#fff", marginBottom: "6px" }}>
              Medical Evidence Retrieval &amp; Grounded Clinical Q&amp;A
            </h1>
            <p style={{ color: "var(--text-secondary)", fontSize: "14px", maxWidth: "800px" }}>
              Semantic vector search across clinical narratives and medical records. Every retrieved chunk is hashed with SHA-256 and committed to the immutable blockchain audit trail.
            </p>
          </div>

          <div style={{ display: "flex", gap: "10px" }}>
            <span className="badge badge-info">FAISS Index: 920 Vectors</span>
            <span className="badge badge-success">Groq Llama-3.1 Grounded</span>
          </div>
        </div>

        {/* QUICK PRESET BUTTONS */}
        <div style={{ marginTop: "18px", display: "flex", flexWrap: "wrap", gap: "8px", alignItems: "center" }}>
          <span style={{ fontSize: "12px", color: "var(--text-muted)", fontWeight: "500" }}>Preset Queries:</span>
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
              placeholder="Search patient records, symptoms, biomarkers, diagnoses..."
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

          <div style={{ width: "130px" }}>
            <select
              value={topK}
              onChange={(e) => setTopK(e.target.value)}
              style={{ padding: "14px", borderRadius: "12px", background: "#080d1a" }}
              title="Top-K evidence records to retrieve"
            >
              <option value="3">Top-3 Records</option>
              <option value="5">Top-5 Records</option>
              <option value="10">Top-10 Records</option>
            </select>
          </div>

          <button
            className="btn-primary"
            onClick={() => handleSearch()}
            disabled={loading}
            style={{ padding: "14px 28px", borderRadius: "12px", fontSize: "15px", minWidth: "130px" }}
          >
            {loading ? "Retrieving..." : "🔍 Search RAG"}
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
            <strong style={{ display: "block", fontSize: "15px" }}>RBAC Access Control Violation</strong>
            <span style={{ fontSize: "13px" }}>{error}</span>
          </div>
        </div>
      )}

      {/* AI GROUNDED ANSWER SECTION */}
      {answer && (
        <div className="glass-panel fade-in" style={{ border: hasRelevantEvidence ? "1px solid rgba(16, 185, 129, 0.4)" : "1px solid rgba(245, 158, 11, 0.4)" }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "16px", flexWrap: "wrap", gap: "10px" }}>
            <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
              <span style={{ fontSize: "20px" }}>🤖</span>
              <h2 style={{ fontSize: "18px", fontWeight: "700", color: "#fff" }}>
                AI Grounded Clinical Response
              </h2>
              {hasRelevantEvidence ? (
                <span className="badge badge-success">✓ Grounded in Verified Records</span>
              ) : (
                <span className="badge badge-warning">⚠️ No Grounding Evidence Found</span>
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
              background: "rgba(11, 15, 25, 0.8)",
              padding: "18px",
              borderRadius: "10px",
              fontSize: "14px",
              lineHeight: "1.7",
              color: "#e2e8f0",
              whiteSpace: "pre-wrap",
              border: "1px solid var(--border-color)"
            }}
          >
            {answer}
          </div>

          {/* Evidence Bundle Hash Bar */}
          {evidenceHash && (
            <div
              style={{
                marginTop: "16px",
                display: "flex",
                alignItems: "center",
                justifyContent: "space-between",
                background: "rgba(0, 0, 0, 0.25)",
                padding: "10px 14px",
                borderRadius: "8px",
                fontSize: "12px",
                flexWrap: "wrap",
                gap: "8px"
              }}
            >
              <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                <span style={{ color: "var(--text-muted)" }}>Evidence Bundle SHA-256 Hash:</span>
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

      {/* RETRIEVED EVIDENCE CARDS */}
      <div style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <h2 style={{ fontSize: "18px", fontWeight: "700", color: "#fff" }}>
            Retrieved Evidence Cards ({results.length})
          </h2>
          {results.length > 0 && (
            <span style={{ fontSize: "12px", color: "var(--text-muted)" }}>
              Top-{results.length} ranked by Cosine/L2 Similarity
            </span>
          )}
        </div>

        {results.length === 0 && !loading && !error && (
          <div
            className="glass-panel"
            style={{ textAlign: "center", padding: "40px 20px", color: "var(--text-muted)" }}
          >
            <span style={{ fontSize: "36px", display: "block", marginBottom: "12px" }}>📋</span>
            <p style={{ fontSize: "15px" }}>No evidence retrieved yet.</p>
            <p style={{ fontSize: "13px", marginTop: "4px" }}>
              Select a preset query above or enter clinical terms to perform semantic RAG retrieval.
            </p>
          </div>
        )}

        {results.map((item, index) => {
          const meta = item.metadata || {};
          const similarityPct = Math.round((item.similarity_score || 0) * 100);

          return (
            <div
              key={index}
              className="glass-panel fade-in"
              style={{
                borderLeft: `4px solid ${similarityPct > 70 ? "#10b981" : similarityPct > 50 ? "#6366f1" : "#f59e0b"}`,
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
                    Patient ID: {meta.id || item.patient_id}
                  </span>
                  {meta.age && (
                    <span className="badge badge-secondary" style={{ fontSize: "11px" }}>
                      Age: {meta.age} | Sex: {meta.sex}
                    </span>
                  )}
                </div>

                <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
                  <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                    <span style={{ fontSize: "12px", color: "var(--text-muted)" }}>Similarity:</span>
                    <span
                      style={{
                        fontWeight: "700",
                        fontSize: "13px",
                        color: similarityPct > 70 ? "#34d399" : similarityPct > 50 ? "#818cf8" : "#fbbf24"
                      }}
                    >
                      {similarityPct}%
                    </span>
                  </div>

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

              {/* Biomarkers / Attributes Tag Bar */}
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
    </div>
  );
}

export default Home;