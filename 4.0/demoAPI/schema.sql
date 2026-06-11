-- Results storage schema (separate from embeddings tables)
CREATE TABLE IF NOT EXISTS test_runs (
  id BIGSERIAL PRIMARY KEY,
  run_name TEXT NOT NULL,
  algorithm TEXT NOT NULL,
  model_name TEXT,
  params JSONB NOT NULL DEFAULT '{}'::jsonb,
  dataset_meta JSONB NOT NULL DEFAULT '{}'::jsonb,
  notes TEXT,
  started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  finished_at TIMESTAMPTZ,
  runtime_ms BIGINT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_test_runs_created_at ON test_runs (created_at DESC);

CREATE TABLE IF NOT EXISTS run_cvs (
  id BIGSERIAL PRIMARY KEY,
  run_id BIGINT NOT NULL REFERENCES test_runs(id) ON DELETE CASCADE,
  external_key TEXT,
  payload JSONB NOT NULL DEFAULT '{}'::jsonb,
  text_content TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE (run_id, external_key)
);

CREATE INDEX IF NOT EXISTS idx_run_cvs_run_id ON run_cvs (run_id);

CREATE TABLE IF NOT EXISTS run_jds (
  id BIGSERIAL PRIMARY KEY,
  run_id BIGINT NOT NULL REFERENCES test_runs(id) ON DELETE CASCADE,
  external_key TEXT,
  payload JSONB NOT NULL DEFAULT '{}'::jsonb,
  text_content TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE (run_id, external_key)
);

CREATE INDEX IF NOT EXISTS idx_run_jds_run_id ON run_jds (run_id);

CREATE TABLE IF NOT EXISTS run_matches (
  id BIGSERIAL PRIMARY KEY,
  run_id BIGINT NOT NULL REFERENCES test_runs(id) ON DELETE CASCADE,
  cv_id BIGINT NOT NULL REFERENCES run_cvs(id) ON DELETE CASCADE,
  jd_id BIGINT NOT NULL REFERENCES run_jds(id) ON DELETE CASCADE,
  rank INT NOT NULL,
  score DOUBLE PRECISION NOT NULL,
  meta JSONB NOT NULL DEFAULT '{}'::jsonb,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE (run_id, cv_id, rank)
);

CREATE INDEX IF NOT EXISTS idx_run_matches_run_cv_rank ON run_matches (run_id, cv_id, rank);
CREATE INDEX IF NOT EXISTS idx_run_matches_run_jd ON run_matches (run_id, jd_id);

-- Resume support: stable run key (idempotent/restartable runs)
ALTER TABLE IF EXISTS test_runs
  ADD COLUMN IF NOT EXISTS run_key TEXT;

CREATE UNIQUE INDEX IF NOT EXISTS ux_test_runs_run_key
  ON test_runs (run_key)
  WHERE run_key IS NOT NULL;
