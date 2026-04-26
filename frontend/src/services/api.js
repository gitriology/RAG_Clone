import axios from "axios";

const API_URL = "http://127.0.0.1:8000/api/query";

export const sendQuery = async (query) => {
  const res = await axios.post(API_URL, { query });
  return res.data;
};
