import React from "react";

import {
  BrowserRouter,
  Routes,
  Route,
  Link
} from "react-router-dom";

import Home from "./pages/Home";
import AuditDashboard from "./pages/AuditDashboard";

function App() {

  return (

    <BrowserRouter>

      <div
        style={{
          padding: "20px",
          backgroundColor: "#222",
          color: "white"
        }}
      >

        <Link
          to="/"
          style={{
            color: "white",
            marginRight: "20px",
            textDecoration: "none",
            fontSize: "18px"
          }}
        >
          Home
        </Link>

        <Link
          to="/audit"
          style={{
            color: "white",
            textDecoration: "none",
            fontSize: "18px"
          }}
        >
          Audit Dashboard
        </Link>

      </div>

      <Routes>

        <Route path="/" element={<Home />} />

        <Route
          path="/audit"
          element={<AuditDashboard />}
        />

      </Routes>

    </BrowserRouter>

  );
}

export default App;