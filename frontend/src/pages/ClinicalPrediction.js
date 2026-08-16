import React, { useState } from "react";

function ClinicalPrediction({ activeUser, apiBase }) {
  const [formData, setFormData] = useState({
    patient_id: "P_NEW_101",
    age: 63,
    sex: 1,
    cp: 3,
    trestbps: 145,
    chol: 233,
    fbs: 1,
    restecg: 2,
    thalch: 150,
    exang: 0,
    oldpeak: 2.3,
    slope: 2
  });

  const [predictionResult, setPredictionResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const presets = [
    {
      name: "Healthy Patient Profile",
      data: {
        patient_id: "P_HEALTHY_01",
        age: 45,
        sex: 0,
        cp: 1,
        trestbps: 118,
        chol: 185,
        fbs: 0,
        restecg: 0,
        thalch: 172,
        exang: 0,
        oldpeak: 0.0,
        slope: 1
      }
    },
    {
      name: "Moderate Risk CAD Profile",
      data: {
        patient_id: "P_MODERATE_02",
        age: 58,
        sex: 1,
        cp: 2,
        trestbps: 142,
        chol: 245,
        fbs: 0,
        restecg: 1,
        thalch: 140,
        exang: 1,
        oldpeak: 1.6,
        slope: 2
      }
    },
    {
      name: "Severe Critical CAD Profile",
      data: {
        patient_id: "P_CRITICAL_03",
        age: 67,
        sex: 1,
        cp: 4,
        trestbps: 165,
        chol: 298,
        fbs: 1,
        restecg: 2,
        thalch: 115,
        exang: 1,
        oldpeak: 3.2,
        slope: 3
      }
    }
  ];

  const handleInputChange = (field, value) => {
    setFormData((prev) => ({
      ...prev,
      [field]: Number(value) || value
    }));
  };

  const applyPreset = (presetData) => {
    setFormData(presetData);
    setPredictionResult(null);
    setError(null);
  };

  const handleSubmit = async (e) => {
    if (e) e.preventDefault();
    setLoading(true);
    setError(null);

    try {
      const response = await fetch(`${apiBase}/api/v1/predict`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-User-Id": activeUser
        },
        body: JSON.stringify({
          ...formData,
          user_id: activeUser
        })
      });

      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        throw new Error(errData.detail || `HTTP Error ${response.status}: Prediction request failed`);
      }

      const data = await response.json();
      setPredictionResult(data);
    } catch (err) {
      console.error("Prediction error:", err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="fade-in" style={{ display: "flex", flexDirection: "column", gap: "24px" }}>
      {/* HEADER */}
      <div className="glass-panel" style={{ background: "linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.85) 100%)" }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", flexWrap: "wrap", gap: "16px" }}>
          <div>
            <h1 style={{ fontSize: "24px", fontWeight: "700", color: "#fff", marginBottom: "6px" }}>
              🫀 Machine Learning Risk Prediction &amp; Clinical Decision Support
            </h1>
            <p style={{ color: "var(--text-secondary)", fontSize: "14px", maxWidth: "800px" }}>
              Trained XGBoost model evaluates cardiovascular biomarkers, predicts coronary artery disease severity (5 Classes), and generates evidence-based clinical recommendations with blockchain provenance.
            </p>
          </div>

          <div style={{ display: "flex", gap: "10px" }}>
            <span className="badge badge-purple">XGBoost Multi-Class</span>
            <span className="badge badge-info">RBAC: Doctor / Admin</span>
          </div>
        </div>

        {/* PRESET CASE SELECTOR */}
        <div style={{ marginTop: "18px", display: "flex", flexWrap: "wrap", gap: "8px", alignItems: "center" }}>
          <span style={{ fontSize: "12px", color: "var(--text-muted)", fontWeight: "500" }}>Load Clinical Presets:</span>
          {presets.map((preset, idx) => (
            <button
              key={idx}
              type="button"
              className="btn-secondary"
              style={{ fontSize: "12px", padding: "5px 12px", borderRadius: "20px" }}
              onClick={() => applyPreset(preset.data)}
            >
              {preset.name}
            </button>
          ))}
        </div>
      </div>

      {/* ERROR / RBAC DENIAL BANNER */}
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
            <strong style={{ display: "block", fontSize: "15px" }}>RBAC Authorization Error</strong>
            <span style={{ fontSize: "13px" }}>{error}</span>
          </div>
        </div>
      )}

      {/* FORM AND RESULTS GRID */}
      <div style={{ display: "grid", gridTemplateColumns: predictionResult ? "1fr 1fr" : "1fr", gap: "24px" }}>
        {/* INPUT FORM PANEL */}
        <div className="glass-panel">
          <h2 style={{ fontSize: "18px", fontWeight: "700", color: "#fff", marginBottom: "16px" }}>
            Patient Clinical Measurements
          </h2>

          <form onSubmit={handleSubmit} style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "16px" }}>
            <div>
              <label style={{ display: "block", fontSize: "12px", color: "var(--text-secondary)", marginBottom: "6px" }}>
                Patient ID:
              </label>
              <input
                type="text"
                value={formData.patient_id}
                onChange={(e) => setFormData((prev) => ({ ...prev, patient_id: e.target.value }))}
                required
              />
            </div>

            <div>
              <label style={{ display: "block", fontSize: "12px", color: "var(--text-secondary)", marginBottom: "6px" }}>
                Age (years):
              </label>
              <input
                type="number"
                value={formData.age}
                onChange={(e) => handleInputChange("age", e.target.value)}
                min="1"
                max="120"
                required
              />
            </div>

            <div>
              <label style={{ display: "block", fontSize: "12px", color: "var(--text-secondary)", marginBottom: "6px" }}>
                Sex:
              </label>
              <select
                value={formData.sex}
                onChange={(e) => handleInputChange("sex", e.target.value)}
              >
                <option value="1">1 - Male</option>
                <option value="0">0 - Female</option>
              </select>
            </div>

            <div>
              <label style={{ display: "block", fontSize: "12px", color: "var(--text-secondary)", marginBottom: "6px" }}>
                Chest Pain Type (cp):
              </label>
              <select
                value={formData.cp}
                onChange={(e) => handleInputChange("cp", e.target.value)}
              >
                <option value="1">1 - Typical Angina</option>
                <option value="2">2 - Atypical Angina</option>
                <option value="3">3 - Non-Anginal Pain</option>
                <option value="4">4 - Asymptomatic</option>
              </select>
            </div>

            <div>
              <label style={{ display: "block", fontSize: "12px", color: "var(--text-secondary)", marginBottom: "6px" }}>
                Resting Blood Pressure (mm Hg):
              </label>
              <input
                type="number"
                value={formData.trestbps}
                onChange={(e) => handleInputChange("trestbps", e.target.value)}
                min="60"
                max="250"
                required
              />
            </div>

            <div>
              <label style={{ display: "block", fontSize: "12px", color: "var(--text-secondary)", marginBottom: "6px" }}>
                Serum Cholesterol (mg/dL):
              </label>
              <input
                type="number"
                value={formData.chol}
                onChange={(e) => handleInputChange("chol", e.target.value)}
                min="50"
                max="600"
                required
              />
            </div>

            <div>
              <label style={{ display: "block", fontSize: "12px", color: "var(--text-secondary)", marginBottom: "6px" }}>
                Fasting Blood Sugar &gt; 120 mg/dL:
              </label>
              <select
                value={formData.fbs}
                onChange={(e) => handleInputChange("fbs", e.target.value)}
              >
                <option value="0">0 - False (&le; 120 mg/dL)</option>
                <option value="1">1 - True (&gt; 120 mg/dL)</option>
              </select>
            </div>

            <div>
              <label style={{ display: "block", fontSize: "12px", color: "var(--text-secondary)", marginBottom: "6px" }}>
                Resting ECG (restecg):
              </label>
              <select
                value={formData.restecg}
                onChange={(e) => handleInputChange("restecg", e.target.value)}
              >
                <option value="0">0 - Normal</option>
                <option value="1">1 - ST-T Wave Abnormality</option>
                <option value="2">2 - Left Ventricular Hypertrophy</option>
              </select>
            </div>

            <div>
              <label style={{ display: "block", fontSize: "12px", color: "var(--text-secondary)", marginBottom: "6px" }}>
                Max Heart Rate (bpm):
              </label>
              <input
                type="number"
                value={formData.thalch}
                onChange={(e) => handleInputChange("thalch", e.target.value)}
                min="50"
                max="240"
                required
              />
            </div>

            <div>
              <label style={{ display: "block", fontSize: "12px", color: "var(--text-secondary)", marginBottom: "6px" }}>
                Exercise Induced Angina (exang):
              </label>
              <select
                value={formData.exang}
                onChange={(e) => handleInputChange("exang", e.target.value)}
              >
                <option value="0">0 - No</option>
                <option value="1">1 - Yes</option>
              </select>
            </div>

            <div>
              <label style={{ display: "block", fontSize: "12px", color: "var(--text-secondary)", marginBottom: "6px" }}>
                ST Depression (oldpeak):
              </label>
              <input
                type="number"
                step="0.1"
                value={formData.oldpeak}
                onChange={(e) => handleInputChange("oldpeak", e.target.value)}
                min="0"
                max="8"
                required
              />
            </div>

            <div>
              <label style={{ display: "block", fontSize: "12px", color: "var(--text-secondary)", marginBottom: "6px" }}>
                Slope of Peak ST:
              </label>
              <select
                value={formData.slope}
                onChange={(e) => handleInputChange("slope", e.target.value)}
              >
                <option value="1">1 - Upsloping</option>
                <option value="2">2 - Flat</option>
                <option value="3">3 - Downsloping</option>
              </select>
            </div>

            <div style={{ gridColumn: "1 / -1", marginTop: "8px" }}>
              <button
                type="submit"
                className="btn-primary"
                disabled={loading}
                style={{ width: "100%", padding: "14px", fontSize: "15px" }}
              >
                {loading ? "Analyzing Biomarkers..." : "🫀 Run ML Risk Prediction"}
              </button>
            </div>
          </form>
        </div>

        {/* RESULTS PANEL */}
        {predictionResult && (
          <div className="fade-in" style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
            {/* Severity Card */}
            <div
              className="glass-panel"
              style={{
                borderTop: `4px solid ${
                  predictionResult.prediction === 0
                    ? "#10b981"
                    : predictionResult.prediction === 1
                    ? "#38bdf8"
                    : predictionResult.prediction === 2
                    ? "#f59e0b"
                    : "#f43f5e"
                }`
              }}
            >
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "12px" }}>
                <span style={{ fontSize: "13px", color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.05em" }}>
                  Cardiovascular Risk Prediction
                </span>
                <span className="badge badge-purple">
                  ⛓️ Block #{predictionResult.blockchain?.block_index}
                </span>
              </div>

              <div style={{ display: "flex", alignItems: "baseline", gap: "12px", marginBottom: "8px" }}>
                <h3
                  style={{
                    fontSize: "26px",
                    fontWeight: "800",
                    color:
                      predictionResult.prediction === 0
                        ? "#34d399"
                        : predictionResult.prediction === 1
                        ? "#38bdf8"
                        : predictionResult.prediction === 2
                        ? "#fbbf24"
                        : "#fb7185"
                  }}
                >
                  {predictionResult.severity}
                </h3>
                <span style={{ fontSize: "14px", color: "var(--text-secondary)" }}>
                  (Confidence: {Math.round((predictionResult.confidence || 0) * 100)}%)
                </span>
              </div>

              <p style={{ fontSize: "14px", color: "#cbd5e1", marginBottom: "16px" }}>
                {predictionResult.risk_description}
              </p>

              {/* Primary Categorization Drivers Highlight */}
              {predictionResult.primary_categorization_drivers && predictionResult.primary_categorization_drivers.length > 0 && (
                <div
                  style={{
                    background: "rgba(15, 23, 42, 0.9)",
                    border: "1px solid rgba(99, 102, 241, 0.3)",
                    borderRadius: "var(--radius-md)",
                    padding: "16px",
                    marginTop: "16px"
                  }}
                >
                  <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "10px" }}>
                    <span style={{ fontSize: "16px" }}>🎯</span>
                    <strong style={{ fontSize: "13px", color: "#e2e8f0", textTransform: "uppercase", letterSpacing: "0.04em" }}>
                      Key Parameters Categorizing Patient as {predictionResult.severity}:
                    </strong>
                  </div>
                  <div style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
                    {predictionResult.primary_categorization_drivers.map((driver, dIdx) => (
                      <div
                        key={dIdx}
                        style={{
                          fontSize: "12px",
                          color: "#cbd5e1",
                          display: "flex",
                          alignItems: "flex-start",
                          gap: "8px",
                          background: "#090d16",
                          padding: "8px 12px",
                          borderRadius: "6px",
                          borderLeft: `3px solid ${
                            predictionResult.prediction >= 3
                              ? "#f43f5e"
                              : predictionResult.prediction >= 2
                              ? "#f59e0b"
                              : predictionResult.prediction >= 1
                              ? "#38bdf8"
                              : "#10b981"
                          }`
                        }}
                      >
                        <span style={{ color: "var(--accent-cyan)", fontWeight: "bold" }}>•</span>
                        <span>{driver}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Probability Breakdown */}
              <div style={{ display: "flex", flexDirection: "column", gap: "8px", marginTop: "16px" }}>
                <span style={{ fontSize: "12px", fontWeight: "600", color: "var(--text-muted)" }}>
                  Multi-Class Probability Distribution:
                </span>
                {predictionResult.probabilities &&
                  Object.entries(predictionResult.probabilities).map(([cls, prob]) => {
                    const pct = Math.round(prob * 100);
                    return (
                      <div key={cls} style={{ fontSize: "12px" }}>
                        <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "3px" }}>
                          <span>{cls}</span>
                          <span style={{ fontWeight: "600" }}>{pct}%</span>
                        </div>
                        <div style={{ height: "6px", background: "#1e293b", borderRadius: "3px", overflow: "hidden" }}>
                          <div
                            style={{
                              width: `${pct}%`,
                              height: "100%",
                              background:
                                cls.includes("Healthy")
                                  ? "#10b981"
                                  : cls.includes("Mild")
                                  ? "#38bdf8"
                                  : cls.includes("Moderate")
                                  ? "#f59e0b"
                                  : "#f43f5e",
                              borderRadius: "3px",
                              transition: "width 0.4s ease"
                            }}
                          />
                        </div>
                      </div>
                    );
                  })}
              </div>
            </div>

            {/* Detailed Parameter Categorization & Biomarkers Evaluation Breakdown */}
            {predictionResult.parameter_breakdown && predictionResult.parameter_breakdown.length > 0 && (
              <div className="glass-panel">
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "14px", flexWrap: "wrap", gap: "8px" }}>
                  <div>
                    <h3 style={{ fontSize: "16px", fontWeight: "700", color: "#fff", margin: 0 }}>
                      🔬 Clinical Biomarkers &amp; Parameter Evaluation
                    </h3>
                    <p style={{ fontSize: "12px", color: "var(--text-secondary)", margin: "4px 0 0 0" }}>
                      Detailed evaluation of all 11 patient parameters against clinical reference ranges and XGBoost model weights.
                    </p>
                  </div>
                  <span className="badge badge-purple" style={{ fontSize: "11px" }}>
                    11 Biomarkers Evaluated
                  </span>
                </div>

                <div style={{ display: "flex", flexDirection: "column", gap: "10px" }}>
                  {predictionResult.parameter_breakdown.map((param, pIdx) => {
                    const isDanger = param.status_level === "danger";
                    const isWarning = param.status_level === "warning";
                    const isSuccess = param.status_level === "success";

                    return (
                      <div
                        key={pIdx}
                        style={{
                          background: "#090d16",
                          border: `1px solid ${
                            isDanger
                              ? "rgba(244, 63, 94, 0.3)"
                              : isWarning
                              ? "rgba(245, 158, 11, 0.3)"
                              : "rgba(255, 255, 255, 0.07)"
                          }`,
                          borderRadius: "var(--radius-md)",
                          padding: "14px",
                          transition: "all 0.2s ease"
                        }}
                      >
                        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", flexWrap: "wrap", gap: "8px", marginBottom: "6px" }}>
                          <div>
                            <span style={{ fontSize: "14px", fontWeight: "700", color: "#f8fafc" }}>
                              {param.name}
                            </span>
                            <span style={{ fontSize: "12px", color: "var(--accent-cyan)", marginLeft: "8px", fontWeight: "600" }}>
                              Value: {param.value}
                            </span>
                          </div>

                          <div style={{ display: "flex", gap: "6px", alignItems: "center" }}>
                            <span
                              className={`badge ${
                                isDanger
                                  ? "badge-danger"
                                  : isWarning
                                  ? "badge-warning"
                                  : isSuccess
                                  ? "badge-success"
                                  : "badge-info"
                              }`}
                              style={{ fontSize: "11px", padding: "3px 8px" }}
                            >
                              {param.status}
                            </span>
                            <span
                              style={{
                                fontSize: "11px",
                                color: "var(--text-muted)",
                                background: "rgba(255, 255, 255, 0.05)",
                                padding: "3px 8px",
                                borderRadius: "12px"
                              }}
                            >
                              Weight: {param.model_weight}
                            </span>
                          </div>
                        </div>

                        <div style={{ display: "grid", gridTemplateColumns: "1fr", gap: "4px", fontSize: "12px" }}>
                          <div style={{ color: "var(--text-secondary)" }}>
                            <strong style={{ color: "var(--text-muted)" }}>Normal Reference:</strong>{" "}
                            <span style={{ color: "#94a3b8" }}>{param.normal_range}</span>
                          </div>
                          <div style={{ color: "#cbd5e1", marginTop: "2px" }}>
                            <strong style={{ color: "var(--text-muted)" }}>Clinical Finding:</strong>{" "}
                            {param.clinical_finding}
                          </div>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}

            {/* Global XGBoost Feature Importance Bar Ranking */}
            {predictionResult.feature_importance_ranking && (
              <div className="glass-panel">
                <h3 style={{ fontSize: "15px", fontWeight: "700", color: "#fff", marginBottom: "8px" }}>
                  📊 Global XGBoost Model Feature Importance Weights
                </h3>
                <p style={{ fontSize: "12px", color: "var(--text-secondary)", marginBottom: "12px" }}>
                  Relative influence of each clinical feature within the trained machine learning ensemble.
                </p>

                <div style={{ display: "flex", flexDirection: "column", gap: "6px" }}>
                  {predictionResult.feature_importance_ranking.map((feat, fIdx) => (
                    <div key={fIdx} style={{ fontSize: "11px" }}>
                      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "2px" }}>
                        <span style={{ color: "#cbd5e1" }}>{feat.name}</span>
                        <span style={{ color: "var(--accent-cyan)", fontWeight: "600" }}>{feat.importance_pct}</span>
                      </div>
                      <div style={{ height: "4px", background: "#1e293b", borderRadius: "2px", overflow: "hidden" }}>
                        <div
                          style={{
                            width: `${(feat.importance / 0.26) * 100}%`,
                            height: "100%",
                            background: "linear-gradient(90deg, #6366f1, #06b6d4)",
                            borderRadius: "2px"
                          }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Clinical Decision Support Recommendations */}
            {predictionResult.risk_assessment?.recommendations && (
              <div className="glass-panel">
                <h3 style={{ fontSize: "16px", fontWeight: "700", color: "#fff", marginBottom: "12px" }}>
                  📋 Evidence-Based Clinical Recommendations
                </h3>

                <div style={{ display: "flex", flexDirection: "column", gap: "10px" }}>
                  {predictionResult.risk_assessment.recommendations.map((rec, rIdx) => (
                    <div
                      key={rIdx}
                      style={{
                        background: "#0f172a",
                        padding: "12px",
                        borderRadius: "8px",
                        borderLeft: `3px solid ${rec.priority >= 5 ? "#f43f5e" : rec.priority >= 4 ? "#f59e0b" : "#6366f1"}`
                      }}
                    >
                      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "4px" }}>
                        <span className="badge badge-info" style={{ fontSize: "10px" }}>
                          {rec.type?.toUpperCase()}
                        </span>
                        <span style={{ fontSize: "11px", color: "var(--text-muted)" }}>
                          Priority: {rec.priority}/5
                        </span>
                      </div>
                      <p style={{ fontSize: "13px", color: "#e2e8f0", margin: 0 }}>
                        {rec.description}
                      </p>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

export default ClinicalPrediction;
