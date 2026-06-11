import os
from contextlib import contextmanager
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
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

def get_database_url() -> str:
    host = os.environ.get("POSTGRES_HOST", "localhost")
    port = os.environ.get("POSTGRES_PORT", "15430")
    user = os.environ.get("POSTGRES_USER", "jdcvcns")
    password = os.environ.get("POSTGRES_PASSWORD", "jdcvcns_dev_password")
    database = os.environ.get("POSTGRES_DB", "jdcvcns")
    return f"postgresql://{user}:{password}@{host}:{port}/{database}"

@contextmanager
def get_db_cursor():
    conn = psycopg2.connect(get_database_url())
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            yield cur
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

@app.get("/api/runs")
def list_runs():
    with get_db_cursor() as cur:
        cur.execute("SELECT * FROM test_runs ORDER BY created_at DESC")
        return cur.fetchall()

@app.get("/api/runs/{run_id}")
def get_run(run_id: int):
    with get_db_cursor() as cur:
        cur.execute("SELECT * FROM test_runs WHERE id = %s", (run_id,))
        run = cur.fetchone()
        if not run:
            raise HTTPException(status_code=404, detail="Run not found")
        return run

@app.get("/api/runs/{run_id}/results")
def get_run_results(run_id: int):
    with get_db_cursor() as cur:
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
