import axios from "axios";

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || "").replace(
  /\/$/,
  "",
);

const API_URL = `${API_BASE_URL}/api/query`;
const HEALTH_URL = `${API_BASE_URL}/api/health`;

// Configure global headers to bypass ngrok warning on all requests
const config = {
  headers: {
    "ngrok-skip-browser-warning": "true",
  },
};

export const sendQuery = async (
  query,
  { topKDocuments = 5, maxSentences = 3, conversationHistory = [] } = {},
) => {
  const res = await axios.post(
    API_URL,
    {
      query,
      top_k_documents: topKDocuments,
      max_sentences: maxSentences,
      conversation_history: conversationHistory,
    },
    config,
  );
  return res.data;
};

export const getApiHealth = async () => {
  const res = await axios.get(HEALTH_URL, config);
  return res.data;
};
