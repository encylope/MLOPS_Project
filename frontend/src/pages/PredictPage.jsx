import { useState } from 'react'
import { predictFraud } from '../services/api'

// Sample transactions for demo (from Kaggle dataset)
const SAMPLE_LEGIT = {
  V1: -0.6, V2: 1.1, V3: 1.3, V4: 0.9, V5: -0.2,
  V6: 0.4, V7: 0.3, V8: 0.1, V9: 0.5, V10: 0.2,
  V11: -0.4, V12: -0.1, V13: -0.3, V14: -0.2, V15: 0.9,
  V16: -0.2, V17: 0.1, V18: 0.0, V19: 0.3, V20: 0.1,
  V21: -0.01, V22: 0.1, V23: -0.05, V24: 0.05, V25: 0.09,
  V26: -0.15, V27: 0.08, V28: -0.01, Amount: 42.50, Time: 5000,
}

const SAMPLE_FRAUD = {
  V1: -3.04, V2: -3.16, V3: 1.08, V4: 2.29, V5: -4.76,
  V6: -2.83, V7: -3.44, V8: 0.74, V9: -0.88, V10: -3.54,
  V11: 1.67, V12: -3.39, V13: 0.62, V14: -3.56, V15: 0.66,
  V16: -0.79, V17: -2.35, V18: -2.95, V19: 0.34, V20: 0.08,
  V21: 0.34, V22: -0.36, V23: -0.14, V24: -0.08, V25: 0.04,
  V26: -0.07, V27: -0.17, V28: -0.07, Amount: 1.00, Time: 406,
}

const FEATURES = Array.from({ length: 28 }, (_, i) => `V${i + 1}`)

const initialForm = () => Object.fromEntries(
  [...FEATURES, 'Amount', 'Time'].map(k => [k, ''])
)

function RiskMeter({ probability }) {
  const pct = Math.round(probability * 100)
  const color = pct < 30 ? '#22d3a5' : pct < 70 ? '#f59e0b' : '#ff4d6d'
  return (
    <div style={{ marginTop: 16 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 6, fontSize: 13 }}>
        <span style={{ color: '#9299b8' }}>Fraud probability</span>
        <span style={{ color, fontWeight: 700 }}>{pct}%</span>
      </div>
      <div style={{ height: 8, background: '#22263a', borderRadius: 4, overflow: 'hidden' }}>
        <div style={{
          height: '100%', width: `${pct}%`, background: color,
          borderRadius: 4, transition: 'width 0.6s ease',
        }} />
      </div>
    </div>
  )
}

function ResultCard({ result }) {
  const riskClass = `badge badge-${result.risk_level.toLowerCase()}`
  return (
    <div className="card" style={{ borderColor: result.is_fraud ? 'rgba(255,77,109,0.4)' : 'rgba(34,211,165,0.3)' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
        <h3>Result</h3>
        <span className={result.is_fraud ? 'badge badge-fraud' : 'badge badge-legit'}>
          {result.is_fraud ? '⚠ FRAUD' : '✓ LEGITIMATE'}
        </span>
      </div>

      <RiskMeter probability={result.fraud_probability} />

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, marginTop: 16 }}>
        {[
          ['Risk level', <span className={riskClass}>{result.risk_level}</span>],
          ['Amount', `$${result.amount.toFixed(2)}`],
          ['Model version', `v${result.model_version}`],
          ['Latency', `${result.inference_time_ms.toFixed(1)}ms`],
          ['Transaction ID', <span style={{ fontSize: 11, color: '#9299b8', wordBreak: 'break-all' }}>{result.transaction_id}</span>],
        ].map(([label, value]) => (
          <div key={label} style={{ background: '#22263a', borderRadius: 8, padding: '10px 14px' }}>
            <div style={{ fontSize: 11, color: '#9299b8', marginBottom: 4 }}>{label}</div>
            <div style={{ fontSize: 14, fontWeight: 600 }}>{value}</div>
          </div>
        ))}
      </div>
    </div>
  )
}

export default function PredictPage() {
  const [form, setForm] = useState(initialForm)
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const fill = (sample) => {
    setForm(Object.fromEntries(Object.entries(sample).map(([k, v]) => [k, String(v)])))
    setResult(null)
    setError(null)
  }

  const handleChange = (key, value) => {
    setForm(f => ({ ...f, [key]: value }))
  }

  const handleSubmit = async () => {
    setError(null)
    setLoading(true)
    try {
      const payload = Object.fromEntries(
        Object.entries(form).map(([k, v]) => [k, parseFloat(v)])
      )
      const res = await predictFraud(payload)
      setResult(res)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const allFilled = Object.values(form).every(v => v !== '')

  return (
    <div>
      <h1>Transaction Analysis</h1>
      <p className="subtitle">Submit transaction features to receive an instant fraud risk assessment.</p>

      <div className="row">
        {/* Input panel */}
        <div className="col">
          <div className="card">
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
              <h3>Transaction Features</h3>
              <div style={{ display: 'flex', gap: 8 }}>
                <button className="btn btn-secondary" style={{ padding: '6px 12px', fontSize: 12 }} onClick={() => fill(SAMPLE_LEGIT)}>
                  Load legitimate
                </button>
                <button className="btn btn-secondary" style={{ padding: '6px 12px', fontSize: 12 }} onClick={() => fill(SAMPLE_FRAUD)}>
                  Load fraud
                </button>
              </div>
            </div>

            <div className="form-grid">
              {FEATURES.map(key => (
                <div className="field" key={key}>
                  <label>{key}</label>
                  <input
                    type="number"
                    step="any"
                    value={form[key]}
                    onChange={e => handleChange(key, e.target.value)}
                    placeholder="0.000"
                  />
                </div>
              ))}
              <div className="field">
                <label>Amount (USD)</label>
                <input type="number" step="any" value={form.Amount} onChange={e => handleChange('Amount', e.target.value)} placeholder="0.00" />
              </div>
              <div className="field">
                <label>Time (seconds)</label>
                <input type="number" step="any" value={form.Time} onChange={e => handleChange('Time', e.target.value)} placeholder="0" />
              </div>
            </div>

            <div style={{ marginTop: 16, display: 'flex', gap: 10 }}>
              <button
                className="btn btn-primary"
                onClick={handleSubmit}
                disabled={loading || !allFilled}
              >
                {loading ? <><span className="spinner" style={{ marginRight: 8 }} />Analyzing...</> : 'Analyze Transaction'}
              </button>
              <button className="btn btn-secondary" onClick={() => { setForm(initialForm()); setResult(null); setError(null) }}>
                Clear
              </button>
            </div>

            {error && <div className="error-msg mt1">{error}</div>}
          </div>
        </div>

        {/* Result panel */}
        <div className="col" style={{ maxWidth: 380 }}>
          {result ? (
            <ResultCard result={result} />
          ) : (
            <div className="card" style={{ textAlign: 'center', padding: '3rem 1.5rem', color: '#9299b8' }}>
              <div style={{ fontSize: 48, marginBottom: 12 }}>🔍</div>
              <div style={{ fontSize: 14 }}>Fill in the transaction features and click <strong>Analyze</strong> to get a fraud risk assessment.</div>
              <div style={{ marginTop: 12, fontSize: 12 }}>Or use <em>Load legitimate</em> / <em>Load fraud</em> to try sample transactions.</div>
            </div>
          )}

          {/* User guidance */}
          <div className="card mt1" style={{ fontSize: 13, color: '#9299b8' }}>
            <strong style={{ color: '#e8eaf6', display: 'block', marginBottom: 8 }}>How to use</strong>
            <ul style={{ paddingLeft: 16, lineHeight: 2 }}>
              <li>V1–V28 are PCA-transformed card features</li>
              <li>Amount is the transaction value in USD</li>
              <li>Time is seconds since the first transaction</li>
              <li>Results show probability, risk tier, and model version</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  )
}
