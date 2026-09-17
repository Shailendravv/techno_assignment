import axios from 'axios';

// Create an axios instance with base configuration
const api = axios.create({
  // It uses the URL from .env file, or fallbacks to localhost for development
  baseURL: import.meta.env.VITE_API_URL || 'http://localhost:8000',
  headers: {
    'Content-Type': 'application/json',
  },
});

/**
 * Ask the runbook agent a question.
 * Mirrors the backend's `POST /ask` contract (see backend/app/schemas.py:
 * AskRequest/AskResponse) — a plain JSON question in, `{answer,
 * cited_doc_ids, confidence, trace, llm_calls}` out. `confidence: "no_match"`
 * is a normal, successful response, not an error.
 */
export const askQuestion = async (question, { explain = false } = {}) => {
  try {
    const response = await api.post('/ask', { question, explain });
    return response.data;
  } catch (error) {
    console.error('API Error:', error);
    throw new Error(error.response?.data?.detail || 'Failed to get an answer', { cause: error });
  }
};

export default api;
