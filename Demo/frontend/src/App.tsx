import { useEffect, useState } from 'react';
import './index.css';

interface Run {
  id: number;
  run_name: string;
  algorithm: string;
  model_name: string;
  params: any;
  dataset_meta: any;
  started_at: string;
  runtime_ms: number;
}

interface CV {
  id: number;
  external_key: string;
  payload: any;
  text_content: string;
}

interface JD {
  id: number;
  external_key: string;
  payload: any;
  text_content: string;
}

interface Match {
  cv_id: number;
  jd_id: number;
  rank: number;
  score: number;
  meta: any;
  jd: JD;
}

interface RunResult {
  cv: CV;
  matches: Match[];
}

const API_BASE = "http://localhost:8000/api";

function App() {
  const [runs, setRuns] = useState<Run[]>([]);
  const [selectedRun, setSelectedRun] = useState<Run | null>(null);
  const [results, setResults] = useState<RunResult[]>([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    fetch(`${API_BASE}/runs`)
      .then(r => r.json())
      .then(data => setRuns(data))
      .catch(console.error);
  }, []);

  const loadRun = async (run: Run) => {
    setSelectedRun(run);
    setLoading(true);
    try {
      const res = await fetch(`${API_BASE}/runs/${run.id}/results`);
      const data = await res.json();
      setResults(data);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-container">
      <nav className="top-nav">
        <div className="body-emphasis" style={{ marginRight: '32px' }}>IBM CV/JD Matcher Demo</div>
      </nav>

      <main style={{ padding: 'var(--spacing-xxl) var(--spacing-lg)' }}>
        {!selectedRun ? (
          <div>
            <h1 className="display-md" style={{ marginBottom: 'var(--spacing-xl)' }}>Test Runs</h1>
            <div className="grid grid-cols-3">
              {runs.map(run => (
                <div key={run.id} className="feature-card">
                  <h3 className="card-title" style={{ marginBottom: 'var(--spacing-xs)' }}>{run.run_name}</h3>
                  <p className="body-sm" style={{ color: 'var(--ink-muted)', marginBottom: 'var(--spacing-md)' }}>
                    Algorithm: {run.algorithm} {run.model_name ? `(${run.model_name})` : ''}
                    <br />
                    Runtime: {run.runtime_ms}ms
                  </p>
                  <button className="button-primary" onClick={() => loadRun(run)}>
                    View Results
                  </button>
                </div>
              ))}
              {runs.length === 0 && (
                <p className="body" style={{ color: 'var(--ink-muted)' }}>No test runs found in database.</p>
              )}
            </div>
          </div>
        ) : (
          <div>
            <button className="button-secondary" onClick={() => setSelectedRun(null)} style={{ marginBottom: 'var(--spacing-xl)' }}>
              &larr; Back to Runs
            </button>
            <h1 className="display-md" style={{ marginBottom: 'var(--spacing-sm)' }}>Results for {selectedRun.run_name}</h1>
            <p className="subhead" style={{ marginBottom: 'var(--spacing-xxl)' }}>Algorithm: {selectedRun.algorithm}</p>

            {loading ? (
              <p className="body">Loading results...</p>
            ) : (
              <div className="grid grid-cols-1">
                {results.map(res => (
                  <div key={res.cv.id} className="product-card">
                    <h2 className="headline" style={{ marginBottom: 'var(--spacing-sm)' }}>
                      CV: {res.cv.payload.name || res.cv.external_key}
                    </h2>
                    <p className="body" style={{ marginBottom: 'var(--spacing-lg)' }}>
                      <strong>Skills:</strong> {Array.isArray(res.cv.payload.skills) ? res.cv.payload.skills.join(', ') : res.cv.payload.skills}
                    </p>
                    
                    <h3 className="card-title" style={{ marginBottom: 'var(--spacing-md)', borderBottom: '1px solid var(--hairline)', paddingBottom: '8px' }}>
                      Top Matches
                    </h3>
                    
                    <div className="grid grid-cols-2">
                      {res.matches.map(m => (
                        <div key={m.jd_id} className="feature-card-elevated" style={{ padding: 'var(--spacing-md)' }}>
                          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                            <h4 className="body-emphasis">{m.jd?.payload.title || m.jd?.external_key}</h4>
                            <span className="badge badge-primary">Rank #{m.rank}</span>
                          </div>
                          <p className="body-sm" style={{ marginBottom: '8px' }}>Score: <strong>{m.score.toFixed(4)}</strong></p>
                          <p className="caption" style={{ color: 'var(--ink-muted)', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                            {m.jd?.payload.description || m.jd?.text_content || 'No description available'}
                          </p>
                        </div>
                      ))}
                    </div>
                  </div>
                ))}
                {results.length === 0 && (
                  <p className="body">No matching results found.</p>
                )}
              </div>
            )}
          </div>
        )}
      </main>

      <footer className="footer" style={{ marginTop: 'var(--spacing-section)' }}>
        <h4 className="body-emphasis" style={{ marginBottom: 'var(--spacing-md)' }}>IBM CV/JD Matching Demo</h4>
        <p className="body-sm">Designed with Carbon Design System.</p>
      </footer>
    </div>
  );
}

export default App;
