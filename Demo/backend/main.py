import os
from contextlib import contextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import psycopg2
from psycopg2.extras import RealDictCursor

app = FastAPI(title="Demo API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

VERSION_SOURCES = [
    {
        "version": "3.0",
        "label": "3.0 BGE-M3 Qdrant",
        "port_env": "DEMO_POSTGRES_30_PORT",
        "default_port": "15430",
    },
    {
        "version": "4.0",
        "label": "4.0 BGE-M3 + ViRanker",
        "port_env": "DEMO_POSTGRES_40_PORT",
        "default_port": "15440",
    },
    {
        "version": "6.0",
        "label": "6.x JobBERT + BM25",
        "port_env": "DEMO_POSTGRES_60_PORT",
        "default_port": "15600",
    },
]


def normalize_version(version: str) -> str:
    aliases = {
        "3": "3.0",
        "3.0": "3.0",
        "4": "4.0",
        "4.0": "4.0",
        "6": "6.0",
        "6.0": "6.0",
        "6.1": "6.0",
        "6.2": "6.0",
    }
    key = str(version).strip()
    if key not in aliases:
        raise HTTPException(status_code=404, detail=f"Unknown result source: {version}")
    return aliases[key]


def source_config(version: str) -> dict[str, str]:
    normalized = normalize_version(version)
    for source in VERSION_SOURCES:
        if source["version"] == normalized:
            return source
    raise HTTPException(status_code=404, detail=f"Unknown result source: {version}")


def get_database_url(source: dict[str, str] | None = None) -> str:
    source = VERSION_SOURCES[0] if source is None else source
    host = os.environ.get("POSTGRES_HOST", "localhost")
    port = os.environ.get(source["port_env"], source["default_port"])
    user = os.environ.get("POSTGRES_USER", "jdcvcns")
    password = os.environ.get("POSTGRES_PASSWORD", "jdcvcns_dev_password")
    database = os.environ.get("POSTGRES_DB", "jdcvcns")
    return f"postgresql://{user}:{password}@{host}:{port}/{database}"


@contextmanager
def get_db_cursor(source: dict[str, str] | None = None):
    conn = psycopg2.connect(get_database_url(source))
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            yield cur
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def enrich_run(row: dict[str, Any], source: dict[str, str]) -> dict[str, Any]:
    enriched = dict(row)
    enriched["source_version"] = source["version"]
    enriched["source_label"] = source["label"]
    enriched["source_port"] = os.environ.get(source["port_env"], source["default_port"])
    return enriched


def fetch_runs_for_source(source: dict[str, str]) -> list[dict[str, Any]]:
    with get_db_cursor(source) as cur:
        cur.execute(
            """
            SELECT
              tr.*,
              COALESCE(cv_stats.cv_count, 0) AS cv_count,
              COALESCE(jd_stats.jd_count, 0) AS jd_count,
              COALESCE(match_stats.match_count, 0) AS match_count,
              match_stats.rank1_avg_score
            FROM test_runs tr
            LEFT JOIN (
              SELECT run_id, COUNT(*) AS cv_count
              FROM run_cvs
              GROUP BY run_id
            ) cv_stats ON cv_stats.run_id = tr.id
            LEFT JOIN (
              SELECT run_id, COUNT(*) AS jd_count
              FROM run_jds
              GROUP BY run_id
            ) jd_stats ON jd_stats.run_id = tr.id
            LEFT JOIN (
              SELECT
                run_id,
                COUNT(*) AS match_count,
                AVG(score) FILTER (WHERE rank = 1) AS rank1_avg_score
              FROM run_matches
              GROUP BY run_id
            ) match_stats ON match_stats.run_id = tr.id
            ORDER BY tr.created_at DESC
            """
        )
        return [enrich_run(dict(row), source) for row in cur.fetchall()]


@app.get("/api/runs")
def list_runs():
    runs: list[dict[str, Any]] = []
    for source in VERSION_SOURCES:
        try:
            runs.extend(fetch_runs_for_source(source))
        except psycopg2.Error:
            continue
    return sorted(runs, key=lambda item: item.get("created_at"), reverse=True)


@app.get("/api/runs/{source_version}/{run_id}")
def get_run_from_source(source_version: str, run_id: int):
    source = source_config(source_version)
    with get_db_cursor(source) as cur:
        cur.execute("SELECT * FROM test_runs WHERE id = %s", (run_id,))
        run = cur.fetchone()
        if not run:
            raise HTTPException(status_code=404, detail="Run not found")
        return enrich_run(dict(run), source)


@app.get("/api/runs/{run_id}/results")
def get_run_results(run_id: int):
    return get_run_results_from_source("3.0", run_id)


@app.get("/api/runs/{source_version}/{run_id}/results")
def get_run_results_from_source(source_version: str, run_id: int):
    source = source_config(source_version)
    with get_db_cursor(source) as cur:
        cur.execute("SELECT id, external_key, payload, text_content FROM run_cvs WHERE run_id = %s", (run_id,))
        cvs = {row["id"]: row for row in cur.fetchall()}

        cur.execute("SELECT id, external_key, payload, text_content FROM run_jds WHERE run_id = %s", (run_id,))
        jds = {row["id"]: row for row in cur.fetchall()}

        cur.execute("SELECT cv_id, jd_id, rank, score, meta FROM run_matches WHERE run_id = %s ORDER BY cv_id, rank", (run_id,))
        matches = cur.fetchall()

        results = []
        for cv_id, cv in cvs.items():
            cv_matches = []
            for m in matches:
                if m["cv_id"] == cv_id:
                    match_data = dict(m)
                    match_data["jd"] = jds.get(m["jd_id"])
                    cv_matches.append(match_data)

            results.append({
                "cv": cv,
                "matches": cv_matches
            })

        return results


@app.get("/api/benchmark/summary")
def benchmark_summary():
    summaries = []
    for source in VERSION_SOURCES:
        port = os.environ.get(source["port_env"], source["default_port"])
        summary: dict[str, Any] = {
            "version": source["version"],
            "label": source["label"],
            "port": port,
            "connected": False,
            "run_count": 0,
            "latest_run": None,
            "algorithms": [],
            "cv_count": 0,
            "jd_count": 0,
            "match_count": 0,
            "rank1_avg_score": None,
            "proposal": "Run this benchmark to populate results.",
        }
        try:
            runs = fetch_runs_for_source(source)
        except psycopg2.Error as exc:
            summary["error"] = str(exc).splitlines()[0]
            summaries.append(summary)
            continue

        summary["connected"] = True
        summary["run_count"] = len(runs)
        if runs:
            latest = runs[0]
            summary["latest_run"] = latest
            summary["algorithms"] = sorted({str(run.get("algorithm")) for run in runs if run.get("algorithm")})
            summary["cv_count"] = int(latest.get("cv_count") or 0)
            summary["jd_count"] = int(latest.get("jd_count") or 0)
            summary["match_count"] = int(latest.get("match_count") or 0)
            score = latest.get("rank1_avg_score")
            summary["rank1_avg_score"] = float(score) if score is not None else None
            if source["version"] == "4.0":
                summary["proposal"] = "Use as reranked quality candidate after 3.0 recall is populated."
            elif source["version"] == "6.0":
                summary["proposal"] = "Use as JobBERT/BM25 baseline suite for comparison against BGE-M3."
            else:
                summary["proposal"] = "Use as BGE-M3 retrieval baseline before cross-encoder reranking."
        summaries.append(summary)

    return {
        "sources": summaries,
        "note": "Scores are not normalized across all algorithms. Compare rank quality with labeled relevance metrics when available.",
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
