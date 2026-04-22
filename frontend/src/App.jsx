import { useState } from "react";
import axios from "axios";

function App() {
  const [query, setQuery] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleSearch = async () => {
    if (!query) return;

    setLoading(true);
    try {
      const res = await axios.post("http://127.0.0.1:8000/api/query", {
        query: query,
      });
      setResult(res.data);
    } catch (err) {
      console.error(err);
      alert("Error fetching response");
    }
    setLoading(false);
  };

  return (
    <div style={{ padding: "30px", fontFamily: "Arial" }}>
      <h1>🚀 Domain-Adaptive RAG Chatbot</h1>

      <input
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        placeholder="Ask something..."
        style={{ width: "60%", padding: "10px" }}
      />

      <button onClick={handleSearch} style={{ marginLeft: "10px" }}>
        Search
      </button>

      {loading && <p>⏳ Processing...</p>}

      {result && (
        <div style={{ marginTop: "20px" }}>
          <h3>🧠 Answer:</h3>
          <p>{result.answer}</p>

          <h4>📊 Confidence:</h4>
          <p>{result.confidence}</p>

          <h4>📂 Detected Domain:</h4>
          <p>{result.meta?.detected_domain}</p>

          <h4>📚 Sources:</h4>
          {result.sources.map((s, i) => (
            <div key={i} style={{ marginBottom: "10px" }}>
              <b>{s.domain}</b> | {s.source}
              <p>{s.text.substring(0, 150)}...</p>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export default App;
