import React, { useState, useEffect } from "react";
import { BrowserRouter, Routes, Route, NavLink } from "react-router-dom";

import Home from "./pages/Home";
import ClinicalPrediction from "./pages/ClinicalPrediction";
import AuditDashboard from "./pages/AuditDashboard";

const API_BASE = "http://127.0.0.1:8000";

function App() {
  const [currentUser, setCurrentUser] = useState("admin");
  const [usersList, setUsersList] = useState([]);
  const [userProfile, setUserProfile] = useState(null);
  const [systemHealth, setSystemHealth] = useState(null);

  // Fetch available users / roles
  useEffect(() => {
    fetch(`${API_BASE}/api/v1/auth/users`)
      .then((res) => res.json())
      .then((data) => {
        if (data.users && Array.isArray(data.users)) {
          setUsersList(data.users);
        }
      })
      .catch((err) => console.error("Failed to load users:", err));
  }, []);

  // Fetch current user profile when user changes
  useEffect(() => {
    fetch(`${API_BASE}/api/v1/auth/me`, {
      headers: { "X-User-Id": currentUser }
    })
      .then((res) => res.json())
      .then((data) => setUserProfile(data))
      .catch((err) => console.error("Failed to load user profile:", err));
  }, [currentUser]);

  // Fetch system health
  useEffect(() => {
    const checkHealth = () => {
      fetch(`${API_BASE}/health`)
        .then((res) => res.json())
        .then((data) => setSystemHealth(data))
        .catch((err) => console.error("Health check error:", err));
    };

    checkHealth();
    const interval = setInterval(checkHealth, 15000);
    return () => clearInterval(interval);
  }, []);

  return (
    <BrowserRouter>
      <div style={{ display: "flex", flexDirection: "column", minHeight: "100vh" }}>
        {/* TOP HEADER */}
        <header
          style={{
            background: "linear-gradient(180deg, #111827 0%, rgba(17, 24, 39, 0.95) 100%)",
            borderBottom: "1px solid var(--border-color)",
            padding: "14px 28px",
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            position: "sticky",
            top: 0,
            zIndex: 100,
            backdropFilter: "blur(12px)"
          }}
        >
          {/* Logo & Branding */}
          <div style={{ display: "flex", alignItems: "center", gap: "14px" }}>
            <div
              style={{
                width: "36px",
                height: "36px",
                borderRadius: "10px",
                background: "linear-gradient(135deg, #6366f1 0%, #06b6d4 100%)",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                fontWeight: "bold",
                color: "white",
                fontSize: "18px",
                boxShadow: "0 0 16px rgba(99, 102, 241, 0.4)"
              }}
            >
              +
            </div>
            <div>
              <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                <span style={{ fontSize: "19px", fontWeight: "700", letterSpacing: "-0.02em", color: "#fff" }}>
                  RAGChain<span style={{ color: "#06b6d4" }}>Med</span>
                </span>
                <span className="badge badge-purple" style={{ fontSize: "10px", padding: "2px 8px" }}>
                  v2.0 Blockchain
                </span>
              </div>
              <p style={{ fontSize: "11px", color: "var(--text-muted)", margin: 0 }}>
                Clinical RAG &amp; Cryptographic Evidence Audit
              </p>
            </div>
          </div>

          {/* Navigation Links */}
          <nav style={{ display: "flex", gap: "6px" }}>
            <NavLink
              to="/"
              className={({ isActive }) =>
                `nav-link ${isActive ? "active" : ""}`
              }
              style={({ isActive }) => ({
                padding: "8px 14px",
                borderRadius: "8px",
                textDecoration: "none",
                fontSize: "13px",
                fontWeight: "500",
                color: isActive ? "#fff" : "var(--text-secondary)",
                background: isActive ? "rgba(99, 102, 241, 0.2)" : "transparent",
                border: isActive ? "1px solid rgba(99, 102, 241, 0.4)" : "1px solid transparent",
                transition: "all 0.2s"
              })}
            >
              🔍 RAG Search &amp; Evidence
            </NavLink>

            <NavLink
              to="/predict"
              className={({ isActive }) =>
                `nav-link ${isActive ? "active" : ""}`
              }
              style={({ isActive }) => ({
                padding: "8px 14px",
                borderRadius: "8px",
                textDecoration: "none",
                fontSize: "13px",
                fontWeight: "500",
                color: isActive ? "#fff" : "var(--text-secondary)",
                background: isActive ? "rgba(99, 102, 241, 0.2)" : "transparent",
                border: isActive ? "1px solid rgba(99, 102, 241, 0.4)" : "1px solid transparent",
                transition: "all 0.2s"
              })}
            >
              🫀 Clinical ML Prediction
            </NavLink>

            <NavLink
              to="/audit"
              className={({ isActive }) =>
                `nav-link ${isActive ? "active" : ""}`
              }
              style={({ isActive }) => ({
                padding: "8px 14px",
                borderRadius: "8px",
                textDecoration: "none",
                fontSize: "13px",
                fontWeight: "500",
                color: isActive ? "#fff" : "var(--text-secondary)",
                background: isActive ? "rgba(99, 102, 241, 0.2)" : "transparent",
                border: isActive ? "1px solid rgba(99, 102, 241, 0.4)" : "1px solid transparent",
                transition: "all 0.2s"
              })}
            >
              ⛓️ Blockchain Audit Trail
            </NavLink>
          </nav>

          {/* RBAC Role Switcher & System Status */}
          <div style={{ display: "flex", alignItems: "center", gap: "16px" }}>
            {/* Blockchain Health Indicator */}
            {systemHealth && (
              <div
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: "6px",
                  fontSize: "12px",
                  color: systemHealth.blockchain_intact ? "var(--success-text)" : "var(--danger-text)"
                }}
                title={`Vector DB: ${systemHealth.total_vectors} vectors | Blockchain: ${systemHealth.total_blocks} blocks`}
              >
                <span
                  style={{
                    width: "8px",
                    height: "8px",
                    borderRadius: "50%",
                    backgroundColor: systemHealth.blockchain_intact ? "#10b981" : "#f43f5e",
                    boxShadow: systemHealth.blockchain_intact
                      ? "0 0 8px #10b981"
                      : "0 0 8px #f43f5e"
                  }}
                />
                <span style={{ fontWeight: "600" }}>
                  {systemHealth.blockchain_intact ? "Chain Verified" : "Chain Alert"}
                </span>
              </div>
            )}

            {/* Role Switcher */}
            <div
              style={{
                display: "flex",
                alignItems: "center",
                gap: "8px",
                background: "rgba(15, 23, 42, 0.8)",
                border: "1px solid var(--border-color)",
                padding: "4px 10px",
                borderRadius: "10px"
              }}
            >
              <span style={{ fontSize: "12px", color: "var(--text-muted)" }}>Role:</span>
              <select
                value={currentUser}
                onChange={(e) => setCurrentUser(e.target.value)}
                style={{
                  background: "transparent",
                  border: "none",
                  color: "#fff",
                  fontSize: "13px",
                  fontWeight: "600",
                  cursor: "pointer",
                  padding: "4px 8px",
                  width: "auto"
                }}
              >
                {usersList.length > 0 ? (
                  usersList.map((u) => (
                    <option key={u.user_id} value={u.user_id} style={{ background: "#111827", color: "#fff" }}>
                      {u.user_id.toUpperCase()} ({u.role.toUpperCase()})
                    </option>
                  ))
                ) : (
                  <>
                    <option value="admin" style={{ background: "#111827" }}>ADMIN (ADMIN)</option>
                    <option value="DOC001" style={{ background: "#111827" }}>DOC001 (DOCTOR)</option>
                    <option value="D101" style={{ background: "#111827" }}>D101 (DOCTOR)</option>
                    <option value="NURSE001" style={{ background: "#111827" }}>NURSE001 (NURSE)</option>
                    <option value="AUDITOR001" style={{ background: "#111827" }}>AUDITOR001 (AUDITOR)</option>
                    <option value="PATIENT001" style={{ background: "#111827" }}>PATIENT001 (PATIENT)</option>
                  </>
                )}
                <option value="UNAUTHORIZED_USER" style={{ background: "#111827", color: "#fb7185" }}>
                  ⛔ Unauthorized User (Test RBAC Denials)
                </option>
              </select>
            </div>
          </div>
        </header>

        {/* ACTIVE ROLE PERMISSION BAR */}
        {userProfile && (
          <div
            style={{
              background: "rgba(15, 23, 42, 0.9)",
              borderBottom: "1px solid rgba(255, 255, 255, 0.04)",
              padding: "6px 28px",
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
              fontSize: "12px",
              color: "var(--text-secondary)"
            }}
          >
            <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
              <span>Active Role: <strong style={{ color: "#fff" }}>{userProfile.name || userProfile.role?.toUpperCase()}</strong></span>
              <span className={`badge ${userProfile.role === "admin" ? "badge-purple" : userProfile.role === "doctor" ? "badge-info" : userProfile.role === "nurse" ? "badge-warning" : "badge-secondary"}`}>
                {userProfile.role?.toUpperCase()}
              </span>
            </div>

            <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
              <span style={{ color: "var(--text-muted)" }}>Granted RBAC Permissions:</span>
              {userProfile.permissions && userProfile.permissions.length > 0 ? (
                userProfile.permissions.map((p) => (
                  <span
                    key={p}
                    style={{
                      fontSize: "10px",
                      background: "rgba(255, 255, 255, 0.05)",
                      border: "1px solid rgba(255, 255, 255, 0.08)",
                      padding: "2px 6px",
                      borderRadius: "4px",
                      color: "#94a3b8"
                    }}
                  >
                    {p}
                  </span>
                ))
              ) : (
                <span className="badge badge-danger" style={{ fontSize: "10px", padding: "2px 6px" }}>
                  No Permissions (Access Denied Mode)
                </span>
              )}
            </div>
          </div>
        )}

        {/* MAIN BODY ROUTING */}
        <main style={{ flex: 1, padding: "24px 28px", maxWidth: "1400px", width: "100%", margin: "0 auto" }}>
          <Routes>
            <Route path="/" element={<Home activeUser={currentUser} apiBase={API_BASE} />} />
            <Route path="/predict" element={<ClinicalPrediction activeUser={currentUser} apiBase={API_BASE} />} />
            <Route path="/audit" element={<AuditDashboard activeUser={currentUser} apiBase={API_BASE} />} />
          </Routes>
        </main>

        {/* FOOTER */}
        <footer
          style={{
            borderTop: "1px solid var(--border-color)",
            padding: "16px 28px",
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            fontSize: "12px",
            color: "var(--text-muted)",
            background: "#080c14"
          }}
        >
          <div>
            <strong>RAGChainMed</strong> — Secure Clinical Decision Support System with FAISS, Groq LLM, &amp; Blockchain Audit Integrity
          </div>
          <div>
            Cardiovascular AI &amp; Cryptographic Audit Ledger
          </div>
        </footer>
      </div>
    </BrowserRouter>
  );
}

export default App;