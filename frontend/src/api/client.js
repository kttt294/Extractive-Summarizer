import axios from 'axios';

// API Base URL - relative URL in production (Nginx proxy) or http://localhost:8000 for dev
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL !== undefined
  ? import.meta.env.VITE_API_BASE_URL
  : (import.meta.env.DEV ? 'http://localhost:8000' : '');

const client = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 60000, // 60 seconds timeout to allow model loading
});

export const summarizeTextApi = async (text, lang = 'auto', length = 'medium') => {
  const response = await client.post('/api/v1/summarize/text', { text, lang, length });
  return response.data;
};

export const summarizeUrlApi = async (url, lang = 'auto', length = 'medium') => {
  const response = await client.post('/api/v1/summarize/url', { url, lang, length });
  return response.data;
};

export const summarizeTopicApi = async (topic, lang = 'vi', length = 'medium') => {
  const response = await client.post('/api/v1/summarize/topic', { topic, lang, length });
  return response.data;
};

export default client;
