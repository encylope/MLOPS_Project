import { useEffect, useState } from 'react'
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from 'recharts'
import axios from 'axios'

// Simulated recent predictions for the chart (in production, poll your backend)
function useMetrics() {
  const [data, setData] = useState(() =>
    Array.from({ length: 12 }, (_, i) => ({
      time: `${i * 5}m`,
      requests: Math.floor(Math.random() * 40 + 10),
      fraud: Math.floor(Math.random() * 5),
      latency: +(Math.random() * 30 + 5).toFixed(1),
    }))
  )

  useEffect(() => {
    const id = setInterval(() => {
      setData(prev => [
        ...prev.slice(1),
        {
          time: 'now',
          requests: Math.floor(Math.random() * 40 + 10),
          fraud: Math.floor(Math.random() * 5),
          latency: +(Math.random() * 30 + 5).toFixed(1),
        },
      ])
    }, 5000)
    return () => clearInterval(id)
  }, [])

  return data
}

function MetricTile({ label, value, unit, color }) {
  return (
    <div className="card" style={{ textAlign: 'center' }}>
      <div style={{ fontSize: 28, fontWeight: 700, color: color || '#4f7cff' }}>{value}<span style={{ fontSize: 14, fontWeight: 400, marginLeft: 4 }}>{unit}</span></div>
      <div style={{ fontSize: 12, color: '#9299b8', marginTop: 4 }}>{label}</div>
    </div>
  )
}

export default function DashboardPage() {
  const metrics = useMetrics()
  const [health, setHealth] = useState(null)

  useEffect(() => {
    axios.get('/health').then(r => setHealth(r.data)).catch(() => {})
  }, [])

  const totals = metrics.reduce((a, b) => ({
    requests: a.requests + b.requests,
    fraud: a.fraud + b.fraud,
  }), { requests: 0, fraud: 0 })

  const fraudRate = totals.requests ? ((totals.fraud / totals.requests) * 100).toFixed(1) : 0
  const avgLatency = (metrics.reduce((a, b) => a + b.latency, 0) / metrics.length).toFixed(1)

  return (
    <div>
      <h1>Monitoring Dashboard</h1>
      <p className="subtitle">Live API metrics from Prometheus. For full dashboards, open Grafana.</p>

      {/* KPI tiles */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(160px, 1fr))', gap: 12, marginBottom: 24 }}>
        <MetricTile label="Total requests (last hour)" value={totals.requests} />
        <MetricTile label="Fraud detections" value={totals.fraud} color="#ff4d6d" />
        <MetricTile label="Fraud rate" value={fraudRate} unit="%" color={parseFloat(fraudRate) > 5 ? '#ff4d6d' : '#22d3a5'} />
        <MetricTile label="Avg latency" value={avgLatency} unit="ms" color={parseFloat(avgLatency) > 200 ? '#f59e0b' : '#22d3a5'} />
        <MetricTile label="Model loaded" value={health?.model_loaded ? 'Yes' : '—'} color="#22d3a5" />
        <MetricTile label="Model version" value={health?.model_version ? `v${health.model_version}` : '—'} />
      </div>

      {/* Charts */}
      <div className="row">
        <div className="col">
          <div className="card">
            <h3 style={{ marginBottom: 16 }}>Requests & fraud detections</h3>
            <ResponsiveContainer width="100%" height={200}>
              <LineChart data={metrics}>
                <CartesianGrid strokeDasharray="3 3" stroke="#2e3248" />
                <XAxis dataKey="time" tick={{ fill: '#9299b8', fontSize: 11 }} />
                <YAxis tick={{ fill: '#9299b8', fontSize: 11 }} />
                <Tooltip contentStyle={{ background: '#1a1d27', border: '1px solid #2e3248', color: '#e8eaf6', fontSize: 12 }} />
                <Line type="monotone" dataKey="requests" stroke="#4f7cff" strokeWidth={2} dot={false} name="Requests" />
                <Line type="monotone" dataKey="fraud" stroke="#ff4d6d" strokeWidth={2} dot={false} name="Fraud" />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
        <div className="col">
          <div className="card">
            <h3 style={{ marginBottom: 16 }}>Inference latency (ms)</h3>
            <ResponsiveContainer width="100%" height={200}>
              <LineChart data={metrics}>
                <CartesianGrid strokeDasharray="3 3" stroke="#2e3248" />
                <XAxis dataKey="time" tick={{ fill: '#9299b8', fontSize: 11 }} />
                <YAxis tick={{ fill: '#9299b8', fontSize: 11 }} />
                <Tooltip contentStyle={{ background: '#1a1d27', border: '1px solid #2e3248', color: '#e8eaf6', fontSize: 12 }} />
                <Line type="monotone" dataKey="latency" stroke="#22d3a5" strokeWidth={2} dot={false} name="Latency (ms)" />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* External tool links */}
      <div className="card mt2">
        <h3 style={{ marginBottom: 12 }}>External monitoring tools</h3>
        <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap' }}>
          {[
            { label: 'Grafana Dashboards', url: 'http://localhost:3001', color: '#f59e0b', desc: 'NRT metrics visualization' },
            { label: 'Prometheus', url: 'http://localhost:9090', color: '#e05d44', desc: 'Raw metrics + alerting' },
            { label: 'MLflow UI', url: 'http://localhost:5000', color: '#4f7cff', desc: 'Experiment tracking' },
            { label: 'Airflow UI', url: 'http://localhost:8080', color: '#22d3a5', desc: 'Pipeline management' },
          ].map(tool => (
            <a key={tool.label} href={tool.url} target="_blank" rel="noreferrer"
              style={{
                flex: '1 1 180px', padding: '14px 16px',
                background: '#22263a', borderRadius: 10,
                border: `1px solid ${tool.color}33`,
                textDecoration: 'none', transition: 'border-color 0.15s',
              }}
              onMouseEnter={e => e.currentTarget.style.borderColor = tool.color}
              onMouseLeave={e => e.currentTarget.style.borderColor = `${tool.color}33`}
            >
              <div style={{ fontWeight: 600, color: tool.color, marginBottom: 4 }}>{tool.label} ↗</div>
              <div style={{ fontSize: 12, color: '#9299b8' }}>{tool.desc}</div>
            </a>
          ))}
        </div>
      </div>

      <div style={{ marginTop: 16, fontSize: 12, color: '#9299b8' }}>
        Charts above show simulated data for demo purposes. Connect to your Prometheus endpoint for real metrics.
        The backend exposes <code style={{ background: '#22263a', padding: '2px 6px', borderRadius: 4 }}>/metrics</code> in Prometheus format.
      </div>
    </div>
  )
}
