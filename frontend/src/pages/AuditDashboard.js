import React, { useEffect, useState } from "react";

function AuditDashboard() {

  const [logs, setLogs] = useState([]);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);

  // ============================================================
  // LOAD AUDIT LOGS
  // ============================================================

  const loadLogs = async () => {

    setLoading(true);
    setError(null);

    try {

      console.log("📋 Fetching audit logs...");

      const response = await fetch(
        "http://127.0.0.1:8000/api/v1/audit/logs",
        {
          method: "GET",
          headers: {
            "user_id": "admin",
            "Content-Type": "application/json"
          }
        }
      );

      console.log("Response status:", response.status);

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(`API Error: ${response.status} - ${errorData.detail || response.statusText}`);
      }

      const data = await response.json();

      console.log("✓ Data received:", data);

      // IMPORTANT
      const recordsArray = Array.isArray(data.records) ? data.records : [];
      setLogs(recordsArray);

      if (recordsArray.length === 0) {
        setError("No audit logs found in the system");
      }

    } catch (error) {

      console.error("❌ Error loading audit logs:", error);
      setError(error.message);
      setLogs([]);

    } finally {
      setLoading(false);
    }
  };

  // ============================================================
  // INITIAL LOAD
  // ============================================================

  useEffect(() => {

    loadLogs();

  }, []);

  return (

    <div style={{ padding: "40px", fontFamily: "Arial" }}>

      <h1>Blockchain Audit Dashboard</h1>

      <h3>Secure Healthcare Access Logs</h3>

      {loading && (
        <p>⏳ Loading audit logs...</p>
      )}

      {error && (
        <div style={{
          backgroundColor: "#ffcccc",
          color: "#cc0000",
          padding: "15px",
          borderRadius: "5px",
          marginBottom: "20px",
          border: "1px solid #cc0000"
        }}>
          <b>⚠️ Error:</b> {error}
        </div>
      )}

      {!loading && logs.length === 0 && !error && (
        <p style={{color: "#666"}}>📭 No audit logs found</p>
      )}

      {logs.map((log, index) => (

        <div
          key={index}
          style={{
            border: "2px solid black",
            padding: "20px",
            marginBottom: "20px",
            borderRadius: "10px",
            backgroundColor: "#f5f5f5"
          }}
        >

          <p><b>User ID:</b> {log.user_id}</p>

          <p><b>Action:</b> {log.action}</p>

          <p><b>Data Type:</b> {log.data_type}</p>

          <p><b>Status:</b> <span style={{
            color: log.status === "success" ? "green" : "red"
          }}>{log.status}</span></p>

          <p><b>Timestamp:</b> {log.timestamp}</p>

          {log.patient_id && <p><b>Patient ID:</b> {log.patient_id}</p>}

          {log.error_message && <p><b>Error:</b> <span style={{color: "red"}}>{log.error_message}</span></p>}

          <p><b>Details:</b></p>

          <pre style={{backgroundColor: "#fff", padding: "10px", overflow: "auto"}}>
            {JSON.stringify(log.details, null, 2)}
          </pre>

        </div>

      ))}

    </div>
  );
}

export default AuditDashboard;