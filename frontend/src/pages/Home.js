import React, { useState } from "react";

function Home() {

  const [query, setQuery] = useState("");

  const [results, setResults] = useState([]);

  const [answer, setAnswer] = useState("");

  const [loading, setLoading] = useState(false);

  const [userId, setUserId] = useState("");

  // ============================================================
  // HANDLE SEARCH
  // ============================================================

  const handleSubmit = async () => {

    if (!query || !userId) {

      alert("Enter User ID and Query");

      return;
    }

    setLoading(true);

    try {

      const response = await fetch(

        "http://127.0.0.1:8000/query?query="
        + encodeURIComponent(query)
        + "&user_id="
        + encodeURIComponent(userId)

      );

      const data = await response.json();

      console.log(data);

      // ========================================================
      // SET RESULTS
      // ========================================================

      setResults(data.retrieved_records || []);

      // ========================================================
      // SET AI ANSWER
      // ========================================================

      setAnswer(data.answer || "");

    } catch (error) {

      console.log(error);

      alert("Backend Error");

    }

    setLoading(false);
  };

  return (

    <div
      style={{
        padding: "40px",
        fontFamily: "Arial"
      }}
    >

      <h1>RAGChainMed</h1>

      <h3>Secure Healthcare Retrieval</h3>

      {/* USER INPUT */}

      <input
        type="text"
        placeholder="Enter User ID"
        value={userId}
        onChange={(e) => setUserId(e.target.value)}
        style={{
          width: "300px",
          padding: "12px",
          marginBottom: "20px",
          display: "block"
        }}
      />

      {/* QUERY INPUT */}

      <input
        type="text"
        placeholder="Enter medical query..."
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        style={{
          width: "70%",
          padding: "12px",
          fontSize: "16px"
        }}
      />

      <button
        onClick={handleSubmit}
        style={{
          padding: "12px 20px",
          marginLeft: "10px",
          cursor: "pointer"
        }}
      >
        Search
      </button>

      {loading && <p>Loading...</p>}

      {/* AI ANSWER */}

      {answer && (

        <div
          style={{
            marginTop: "30px",
            border: "2px solid green",
            padding: "20px",
            borderRadius: "10px",
            backgroundColor: "#f0fff0"
          }}
        >

          <h2>AI Answer</h2>

          <p>{answer}</p>

        </div>

      )}

      {/* RESULTS */}

      <div style={{ marginTop: "40px" }}>

        <h2>Retrieved Records</h2>

        {results.length === 0 && !loading && (

          <p>No records retrieved.</p>

        )}

        {results.map((result, index) => (

          <div
            key={index}
            style={{
              border: "1px solid gray",
              padding: "15px",
              marginBottom: "15px",
              borderRadius: "10px"
            }}
          >

            <h4>Record {index + 1}</h4>

            <p>{result}</p>

          </div>

        ))}

      </div>

    </div>
  );
}

export default Home;