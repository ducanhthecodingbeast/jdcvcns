import { useEffect, useMemo, useState } from 'react';
import './index.css';

interface Run {
  id: number;
  run_name: string;
  algorithm: string;
  model_name: string | null;
  params: Record<string, unknown>;
  dataset_meta: Record<string, unknown>;
  started_at: string;
  finished_at?: string | null;
  runtime_ms?: number | null;
  cv_count?: number;
  jd_count?: number;
  match_count?: number;
  rank1_avg_score?: number | null;
  source_version?: string;
  source_label?: string;
  source_port?: string;
}

interface SourceSummary {
  version: string;
  label: string;
  port: string;
  connected: boolean;
  run_count: number;
  latest_run: Run | null;
  algorithms: string[];
  cv_count: number;
  jd_count: number;
  match_count: number;
  rank1_avg_score: number | null;
  proposal: string;
  error?: string;
}

interface BenchmarkSummary {
  sources: SourceSummary[];
  note: string;
}

interface CV {
  id: number;
  external_key: string;
  payload: Record<string, unknown>;
  text_content: string;
}

interface JD {
  id: number;
  external_key: string;
  payload: Record<string, unknown>;
  text_content: string;
}

interface Match {
  cv_id: number;
  jd_id: number;
  rank: number;
  score: number;
  meta: Record<string, unknown>;
  jd: JD | null;
}

interface RunResult {
  cv: CV;
  matches: Match[];
}

const API_BASE = import.meta.env.VITE_API_URL ?? 'http://localhost:18000/api';
const targetSources = ['3.0', '4.0', '6.0'];

function formatNumber(value: unknown): string {
  if (typeof value !== 'number' || Number.isNaN(value)) return '-';
  return Intl.NumberFormat('en-US').format(value);
}

function formatScore(value: unknown): string {
  if (typeof value !== 'number' || Number.isNaN(value)) return '-';
  return value.toFixed(4);
}

function formatRuntime(value: unknown): string {
  if (typeof value !== 'number' || Number.isNaN(value)) return '-';
  if (value < 1000) return `${value} ms`;
  return `${(value / 1000).toFixed(1)} s`;
}

function formatDate(value: string | null | undefined): string {
  if (!value) return '-';
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? '-' : date.toLocaleString();
}

function textValue(value: unknown): string {
  if (value === null || value === undefined) return '';
  if (Array.isArray(value)) return value.map(textValue).filter(Boolean).join(', ');
  if (typeof value === 'object') return JSON.stringify(value);
  return String(value);
}

function payloadValue(payload: Record<string, unknown> | null | undefined, names: string[]): string {
  if (!payload) return '';
  for (const name of names) {
    const value = textValue(payload[name]).trim();
    if (value) return value;
  }
  return '';
}

function runSource(run: Run): string {
  return run.source_version ?? '3.0';
}

async function fetchJson<T>(path: string): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`);
  if (!response.ok) {
    throw new Error(`Request failed: ${response.status}`);
  }
  return response.json() as Promise<T>;
}

function App() {
  const [runs, setRuns] = useState<Run[]>([]);
  const [summary, setSummary] = useState<BenchmarkSummary | null>(null);
  const [selectedRun, setSelectedRun] = useState<Run | null>(null);
  const [results, setResults] = useState<RunResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    async function loadDashboard() {
      setError(null);
      try {
        const [summaryData, runData] = await Promise.all([
          fetchJson<BenchmarkSummary>('/benchmark/summary'),
          fetchJson<Run[]>('/runs'),
        ]);
        if (!cancelled) {
          setSummary(summaryData);
          setRuns(runData);
        }
      } catch (err) {
        if (!cancelled) setError(err instanceof Error ? err.message : 'Could not load benchmark data.');
      }
    }

    loadDashboard();
    return () => {
      cancelled = true;
    };
  }, []);

  const filteredRuns = useMemo(() => {
    return runs.filter((run) => targetSources.includes(runSource(run)));
  }, [runs]);

  const latestBySource = useMemo(() => {
    const map = new Map<string, Run>();
    for (const run of filteredRuns) {
      const source = runSource(run);
      if (!map.has(source)) map.set(source, run);
    }
    return map;
  }, [filteredRuns]);

  const sourceSummaries = useMemo<SourceSummary[]>(() => {
    if (summary) return summary.sources;
    return targetSources.map((version) => ({
      version,
      label: `${version} results`,
      port: '-',
      connected: false,
      run_count: 0,
      latest_run: null,
      algorithms: [],
      cv_count: 0,
      jd_count: 0,
      match_count: 0,
      rank1_avg_score: null,
      proposal: 'Waiting for API data.',
    }));
  }, [summary]);

  const loadRun = async (run: Run) => {
    setSelectedRun(run);
    setLoading(true);
    setError(null);
    try {
      const source = encodeURIComponent(runSource(run));
      const data = await fetchJson<RunResult[]>(`/runs/${source}/${run.id}/results`);
      setResults(data);
    } catch (err) {
      setResults([]);
      setError(err instanceof Error ? err.message : 'Could not load run details.');
    } finally {
      setLoading(false);
    }
  };

  const visibleResults = results.slice(0, 12);

  return (
    <div className="app-container">
      <div className="utility-bar">
        <span className="caption">CV/JD Matching Result Console</span>
      </div>

      <nav className="top-nav">
        <div className="body-emphasis nav-title">Benchmark Console</div>
        <a href="#summary" className="body-sm">3/4/6 Summary</a>
        <a href="#runs" className="body-sm">Runs</a>
        <a href="#details" className="body-sm">Details</a>
      </nav>

      <main>
        <section id="summary" className="dashboard-section">
          <div className="content-shell">
            <div className="section-header">
              <div>
                <h1 className="display-md">Testing Results for 3.0, 4.0, and 6.x</h1>
                <p className="body" style={{ color: 'var(--ink-muted)', marginTop: 'var(--spacing-sm)' }}>
                  Aggregates independent result stores from project subports 15430, 15440, and 15600.
                </p>
              </div>
              <div className="summary-note">
                <span className="caption">Comparison rule</span>
                <p className="body-sm">{summary?.note ?? 'Load benchmark runs to compare result quality.'}</p>
              </div>
            </div>

            {error && <div className="alert-banner body-sm">{error}</div>}

            <div className="grid grid-cols-3">
              {sourceSummaries.map((source) => (
                <div key={source.version} className="feature-card result-source-card">
                  <div className="card-row">
                    <h2 className="card-title">{source.version}</h2>
                    <span className={source.connected ? 'badge badge-success' : 'badge'}>
                      {source.connected ? 'Connected' : 'No DB'}
                    </span>
                  </div>
                  <p className="body-sm muted">{source.label}</p>
                  <div className="metric-grid">
                    <div>
                      <span className="caption">Runs</span>
                      <strong>{formatNumber(source.run_count)}</strong>
                    </div>
                    <div>
                      <span className="caption">Matches</span>
                      <strong>{formatNumber(source.match_count)}</strong>
                    </div>
                    <div>
                      <span className="caption">CV/JD</span>
                      <strong>{formatNumber(source.cv_count)} / {formatNumber(source.jd_count)}</strong>
                    </div>
                    <div>
                      <span className="caption">Rank 1 avg</span>
                      <strong>{formatScore(source.rank1_avg_score)}</strong>
                    </div>
                  </div>
                  <p className="body-sm proposal-text">{source.proposal}</p>
                  {source.error && <p className="caption muted">Port {source.port}: {source.error}</p>}
                </div>
              ))}
            </div>
          </div>
        </section>

        <section id="runs" className="dashboard-section dashboard-section-alt">
          <div className="content-shell">
            <div className="section-header compact">
              <div>
                <h2 className="display-md">Run Comparison</h2>
                <p className="body muted">Latest stored runs from 3.0, 4.0, and 6.x.</p>
              </div>
              <div className="version-strip">
                {targetSources.map((version) => (
                  <span key={version} className="badge badge-primary">
                    {version}: {latestBySource.has(version) ? 'has runs' : 'empty'}
                  </span>
                ))}
              </div>
            </div>

            <div className="run-table">
              <div className="run-table-head">
                <span>Version</span>
                <span>Run</span>
                <span>Algorithm</span>
                <span>Rows</span>
                <span>Runtime</span>
                <span>Action</span>
              </div>
              {filteredRuns.map((run) => (
                <div key={`${runSource(run)}-${run.id}`} className="run-table-row">
                  <span><strong>{runSource(run)}</strong></span>
                  <span>
                    <strong>{run.run_name}</strong>
                    <small>{formatDate(run.started_at)}</small>
                  </span>
                  <span>
                    {run.algorithm}
                    <small>{run.model_name ?? 'No model recorded'}</small>
                  </span>
                  <span>
                    {formatNumber(run.cv_count)} CVs
                    <small>{formatNumber(run.match_count)} matches</small>
                  </span>
                  <span>{formatRuntime(run.runtime_ms)}</span>
                  <span>
                    <button className="button-tertiary" onClick={() => loadRun(run)}>Open</button>
                  </span>
                </div>
              ))}
              {filteredRuns.length === 0 && (
                <div className="empty-state">
                  <p className="body">No 3.0, 4.0, or 6.x runs found yet.</p>
                  <p className="body-sm muted">Run the combined script after preparing `Data/jd.csv` and CV data.</p>
                </div>
              )}
            </div>
          </div>
        </section>

        <section id="details" className="dashboard-section">
          <div className="content-shell">
            {!selectedRun ? (
              <div className="empty-state">
                <h2 className="display-md">Run Details</h2>
                <p className="body muted">Select a run above to inspect top ranked JD matches by CV.</p>
              </div>
            ) : (
              <div>
                <button className="button-ghost back-button" onClick={() => setSelectedRun(null)}>
                  Back to comparison
                </button>
                <div className="section-header compact">
                  <div>
                    <h2 className="display-md">{selectedRun.run_name}</h2>
                    <p className="body muted">
                      {runSource(selectedRun)} · {selectedRun.algorithm} · {selectedRun.model_name ?? 'No model recorded'}
                    </p>
                  </div>
                  <div className="metric-grid narrow">
                    <div>
                      <span className="caption">Matches</span>
                      <strong>{formatNumber(selectedRun.match_count)}</strong>
                    </div>
                    <div>
                      <span className="caption">Rank 1 avg</span>
                      <strong>{formatScore(selectedRun.rank1_avg_score)}</strong>
                    </div>
                  </div>
                </div>

                {loading ? (
                  <p className="body">Loading detailed results...</p>
                ) : (
                  <div className="grid grid-cols-1 detail-list">
                    {visibleResults.map((result) => {
                      const cvTitle = payloadValue(result.cv.payload, [
                        'Tên ứng viên',
                        'Vị trí ứng tuyển',
                        'User Name',
                        'Desired Job',
                      ]) || result.cv.external_key;
                      const cvSkills = payloadValue(result.cv.payload, ['Kỹ năng', 'skills', 'Skills']);
                      return (
                        <div key={result.cv.id} className="product-card">
                          <div className="card-row">
                            <h3 className="card-title">{cvTitle}</h3>
                            <span className="caption muted">CV ID {result.cv.id}</span>
                          </div>
                          {cvSkills && <p className="body-sm muted detail-text">Skills: {cvSkills}</p>}

                          <div className="match-grid">
                            {result.matches.map((match) => {
                              const jdTitle = payloadValue(match.jd?.payload, [
                                'Vị trí cần tuyển',
                                'Job Title',
                                'title',
                              ]) || match.jd?.external_key || `JD ${match.jd_id}`;
                              const jdText = payloadValue(match.jd?.payload, [
                                'Yêu cầu công việc',
                                'Mô tả công việc',
                                'description',
                              ]) || match.jd?.text_content || 'No description available';
                              return (
                                <div key={`${result.cv.id}-${match.jd_id}-${match.rank}`} className="feature-card-elevated match-card">
                                  <div className="card-row">
                                    <h4 className="body-emphasis">{jdTitle}</h4>
                                    <span className={match.rank === 1 ? 'badge badge-primary' : 'badge'}>#{match.rank}</span>
                                  </div>
                                  <p className="body-sm">Score: <strong>{formatScore(match.score)}</strong></p>
                                  <p className="caption muted clamp-text">{jdText}</p>
                                </div>
                              );
                            })}
                          </div>
                        </div>
                      );
                    })}
                    {results.length > visibleResults.length && (
                      <p className="body-sm muted">Showing first {visibleResults.length} CVs. Use the API for the full result set.</p>
                    )}
                    {results.length === 0 && <p className="body muted">No detailed matches found for this run.</p>}
                  </div>
                )}
              </div>
            )}
          </div>
        </section>
      </main>
    </div>
  );
}

export default App;
