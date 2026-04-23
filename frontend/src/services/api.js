/**
 * API client for the Fraud Detection backend.
 * All calls go through /api/v1 — proxied to FastAPI in dev, via nginx in prod.
 */

import axios from 'axios'

const BASE_URL = import.meta.env.VITE_API_BASE_URL || ''

const api = axios.create({
  baseURL: BASE_URL,
  timeout: 10000,
  headers: { 'Content-Type': 'application/json' },
})

// ── Request / response interceptors ────────────────────────────────────────
api.interceptors.response.use(
  (response) => response,
  (error) => {
    const message =
      error.response?.data?.detail ||
      error.message ||
      'An unexpected error occurred'
    return Promise.reject(new Error(message))
  }
)

// ── API methods ─────────────────────────────────────────────────────────────

/**
 * Submit a single transaction for fraud prediction.
 * @param {Object} transaction - TransactionRequest fields (V1-V28, Amount, Time)
 * @returns {Promise<PredictionResponse>}
 */
export async function predictFraud(transaction) {
  const { data } = await api.post('/api/v1/predict', transaction)
  return data
}

/**
 * Submit a batch of transactions.
 * @param {Array} transactions - Array of TransactionRequest objects
 * @returns {Promise<BatchPredictionResponse>}
 */
export async function predictBatch(transactions) {
  const { data } = await api.post('/api/v1/predict/batch', { transactions })
  return data
}

/**
 * Fetch API health status.
 * @returns {Promise<HealthResponse>}
 */
export async function getHealth() {
  const { data } = await api.get('/health')
  return data
}

/**
 * Fetch readiness status.
 */
export async function getReady() {
  const { data } = await api.get('/ready')
  return data
}

/**
 * Trigger hot model reload from MLflow registry.
 */
export async function reloadModel() {
  const { data } = await api.post('/api/v1/reload-model')
  return data
}
