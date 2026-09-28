import axios from "axios";

const API_BASE_URL = (
  import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000" || ""
).replace(/\/$/, "");

const API_URL = `${API_BASE_URL}/api/query`;
const HEALTH_URL = `${API_BASE_URL}/api/health`;

export const sendQuery = async (
  query,
  { topKDocuments = 5, maxSentences = 3 } = {},
) => {
  const res = await axios.post(API_URL, {
    query,
    top_k_documents: topKDocuments,
    max_sentences: maxSentences,
  });
  return res.data;
};

export const getApiHealth = async () => {
  const res = await axios.get(HEALTH_URL);
  return res.data;
};
