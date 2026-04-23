import { useEffect, useState } from 'react'
import axios from 'axios'

const STAGES = [
  { id: 'validate', label: 'Data Validation', tool: 'Python', desc: 'Schema check, null detection, class distribution' },
  { id: 'preprocess', label: 'Preprocessing', tool: 'Scikit-learn', desc: 'Scaling, outlier removal, scaler params' },
  { id: 'feature_engineering', label: 'Feature Engineering', tool: 'Pandas', desc: 'Feature selection, drift baseline stats' },
  { id: 'split', label: 'Train/Val/Test Split', tool: 'Scikit-learn', desc: 'Stratified 70/15/15 split' },
  { id: 'train', label: 'Model Training', tool: 'XGBoost + MLflow', desc: 'Train, log metrics, register model' },
  { id: 'evaluate', label: 'Evaluation', tool: 'MLflow', desc: 'Threshold tuning, ROC, confusion matrix' },
]

const MLOPS_TOOLS = [
  { name: 'Apache Airflow', role: 'DAG orchestration', url: 'http://localhost:8080', icon: '🌊' },
  { name: 'MLflow', role: 'Experiment tracking & registry', url: 'http://localhost:5000', icon: '📊' },
  { name: 'Prometheus', role: 'Metrics scraping', url: 'http://localhost:9090', icon: '🔥' },
  { name: 'Grafana', role: 'NRT dashboards', url: 'http://localhost:3001', icon: '📈' },
]

function StageCard({ stage, index, status }) {
  const color = status === 'ok' ? '#22d3a5' : status === 'changed' ? '#f59e0b' : '#9299b8'
  return (
    <div style={{ display: 'flex', gap: 16, alignItems: 'flex-start' }}>
      {/* Connector line */}
      <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', flexShrink: 0 }}>
        <div style={{
          width: 32, height: 32, borderRadius: '50%',
          background: `${color}22`, border: `2px solid ${color}`,
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          fontWeight: 700, fontSize: 13, color,
        }}>
          {index + 1}
        </div>
        {index < STAGES.length - 1 && (
          <div style={{ width: 2, flex: 1, minHeight: 24, background: '#2e3248', marginTop: 4 }} />
        )}
      </div>

      {/* Card */}
      <div className="card" style={{ flex: 1, marginBottom: 12 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div>
            <h3 style={{ marginBottom: 2 }}>{stage.label}</h3>
            <div style={{ fontSize: 12, color: '#9299b8' }}>{stage.desc}</div>
          </div>
          <div style={{ textAlign: 'right', flexShrink: 0, marginLeft: 16 }}>
            <div style={{ fontSize: 11, color: '#9299b8', marginBottom: 4 }}>Tool</div>
            <div style={{ fontSize: 12, fontWeight: 600, color: '#4f7cff' }}>{stage.tool}</div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default function PipelinePage() {
  const [dvcStatus, setDvcStatus] = useState(null)
  const [health, setHealth] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    Promise.allSettled([
      axios.get('/api/v1/pipeline/status'),
      axios.get('/health'),
    ]).then(([statusRes, healthRes]) => {
      if (statusRes.status === 'fulfilled') setDvcStatus(statusRes.value.data)
      if (healthRes.status === 'fulfilled') setHealth(healthRes.value.data)
      setLoading(false)
    })
  }, [])

  return (
    <div>
      <h1>ML Pipeline</h1>
      <p className="subtitle">DVC-orchestrated pipeline stages and MLOps toolchain status.</p>

      <div className="row" style={{ alignItems: 'flex-start' }}>
        {/* Pipeline stages */}
        <div className="col">
          <h2>Pipeline stages</h2>
          {STAGES.map((stage, i) => (
            <StageCard key={stage.id} stage={stage} index={i} status="ok" />
          ))}
        </div>

        {/* Tool status + model info */}
        <div style={{ width: 300, flexShrink: 0 }}>
          <h2>Model status</h2>
          <div className="card" style={{ marginBottom: 16 }}>
            {loading ? (
              <div style={{ color: '#9299b8', fontSize: 13 }}>Loading...</div>
            ) : health ? (
              <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span style={{ fontSize: 13, color: '#9299b8' }}>API status</span>
                  <span style={{ fontSize: 13, fontWeight: 600, color: '#22d3a5' }}>{health.status}</span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span style={{ fontSize: 13, color: '#9299b8' }}>Model loaded</span>
                  <span style={{ fontSize: 13, fontWeight: 600, color: health.model_loaded ? '#22d3a5' : '#ff4d6d' }}>
                    {health.model_loaded ? 'Yes' : 'No'}
                  </span>
                </div>
                {health.model_version && (
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <span style={{ fontSize: 13, color: '#9299b8' }}>Version</span>
                    <span style={{ fontSize: 13, fontWeight: 600 }}>v{health.model_version}</span>
                  </div>
                )}
              </div>
            ) : (
              <div style={{ color: '#ff4d6d', fontSize: 13 }}>Backend unavailable</div>
            )}
          </div>

          <h2>MLOps tools</h2>
          {MLOPS_TOOLS.map(tool => (
            <a
              key={tool.name}
              href={tool.url}
              target="_blank"
              rel="noreferrer"
              style={{ textDecoration: 'none' }}
            >
              <div className="card" style={{ marginBottom: 10, cursor: 'pointer', transition: 'border-color 0.15s' }}
                onMouseEnter={e => e.currentTarget.style.borderColor = '#4f7cff'}
                onMouseLeave={e => e.currentTarget.style.borderColor = '#2e3248'}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                  <span style={{ fontSize: 22 }}>{tool.icon}</span>
                  <div>
                    <div style={{ fontWeight: 600, fontSize: 14 }}>{tool.name}</div>
                    <div style={{ fontSize: 12, color: '#9299b8' }}>{tool.role}</div>
                  </div>
                  <span style={{ marginLeft: 'auto', fontSize: 12, color: '#4f7cff' }}>↗</span>
                </div>
              </div>
            </a>
          ))}

          {/* DVC status */}
          {dvcStatus && (
            <div className="card mt1">
              <h3 style={{ marginBottom: 8 }}>DVC status</h3>
              <pre style={{ fontSize: 11, color: '#9299b8', whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}>
                {dvcStatus.dvc_status || dvcStatus.error || 'No changes detected'}
              </pre>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
