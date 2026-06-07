from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from .db import get_conn
from .migrate import migrate


app = FastAPI(title="JobCVmatching Results Demo")

static_dir = Path(__file__).resolve().parent.parent / "frontend_dashboard"
app.mount("/static", StaticFiles(directory=str(static_dir), html=True), name="static")
_schema_ready = False


def ensure_results_schema() -> None:
    global _schema_ready
    if not _schema_ready:
        migrate()
        _schema_ready = True


def _first_payload_value(payload: dict[str, Any] | None, keys: list[str], fallback: str = "") -> str:
    payload = payload or {}
    for key in keys:
        value = payload.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return fallback


def _cv_summary(cv_id: int, external_key: str | None, payload: dict[str, Any] | None) -> dict[str, Any]:
    return {
        "id": cv_id,
        "external_key": external_key,
        "name": _first_payload_value(
            payload,
            [
                "Tên ứng viên",
                "TÃªn ứng viên",
                "TÃªn á»©ng viÃªn",
                "TÃƒÂªn Ã¡Â»Â©ng viÃƒÂªn",
                "name",
            ],
            external_key or f"CV {cv_id}",
        ),
        "role": _first_payload_value(
            payload,
            [
                "Vị trí ứng tuyển",
                "Vá»‹ trÃ­ á»©ng tuyá»ƒn",
                "VÃ¡Â»â€¹ trÃƒÂ­ Ã¡Â»Â©ng tuyÃ¡Â»Æ’n",
                "role",
            ],
        ),
    }


def _jd_summary(jd_id: int, external_key: str | None, payload: dict[str, Any] | None) -> dict[str, Any]:
    return {
        "id": jd_id,
        "external_key": external_key,
        "title": _first_payload_value(
            payload,
            [
                "title",
                "Vị trí cần tuyển",
                "Vá»‹ trÃ­ cáº§n tuyá»ƒn",
                "VÃ¡Â»â€¹ trÃƒÂ­ cÃ¡ÂºÂ§n tuyÃ¡Â»Æ’n",
            ],
            external_key or f"JD {jd_id}",
        ),
        "company": _first_payload_value(
            payload,
            [
                "company",
                "Tên công ty",
                "TÃªn cÃ´ng ty",
                "TÃƒÂªn cÃƒÂ´ng ty",
            ],
        ),
    }


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    index_path = static_dir / "index.html"
    return index_path.read_text(encoding="utf-8")


@app.get("/api/health")
def health() -> dict[str, str]:
    ensure_results_schema()
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 1;")
            cur.fetchone()
    return {"status": "ok"}


@app.get("/api/runs")
def list_runs(limit: int = 50) -> list[dict[str, Any]]:
    ensure_results_schema()
    limit = max(1, min(limit, 200))
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, run_name, algorithm, model_name, started_at, finished_at, runtime_ms, created_at
                FROM test_runs
                ORDER BY created_at DESC
                LIMIT %s
                """,
                (limit,),
            )
            rows = cur.fetchall()
    return [
        {
            "id": rid,
            "run_name": run_name,
            "algorithm": algorithm,
            "model_name": model_name,
            "started_at": started_at.isoformat() if started_at else None,
            "finished_at": finished_at.isoformat() if finished_at else None,
            "runtime_ms": runtime_ms,
            "created_at": created_at.isoformat() if created_at else None,
        }
        for (rid, run_name, algorithm, model_name, started_at, finished_at, runtime_ms, created_at) in rows
    ]


@app.get("/api/runs/{run_id}")
def get_run(run_id: int) -> dict[str, Any]:
    ensure_results_schema()
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, run_name, algorithm, model_name, params, dataset_meta, notes,
                       started_at, finished_at, runtime_ms, created_at
                FROM test_runs
                WHERE id = %s
                """,
                (run_id,),
            )
            row = cur.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="run not found")

            (
                rid,
                run_name,
                algorithm,
                model_name,
                params,
                dataset_meta,
                notes,
                started_at,
                finished_at,
                runtime_ms,
                created_at,
            ) = row

            cur.execute("SELECT COUNT(*) FROM run_cvs WHERE run_id=%s", (run_id,))
            cv_count = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM run_jds WHERE run_id=%s", (run_id,))
            jd_count = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM run_matches WHERE run_id=%s", (run_id,))
            match_count = cur.fetchone()[0]

    return {
        "id": rid,
        "run_name": run_name,
        "algorithm": algorithm,
        "model_name": model_name,
        "params": params,
        "dataset_meta": dataset_meta,
        "notes": notes,
        "started_at": started_at.isoformat() if started_at else None,
        "finished_at": finished_at.isoformat() if finished_at else None,
        "runtime_ms": runtime_ms,
        "created_at": created_at.isoformat() if created_at else None,
        "counts": {"cvs": cv_count, "jds": jd_count, "matches": match_count},
    }


@app.get("/api/runs/{run_id}/cvs")
def list_cvs(run_id: int, search: str | None = None, limit: int = 50, offset: int = 0) -> dict[str, Any]:
    ensure_results_schema()
    limit = max(1, min(limit, 200))
    offset = max(0, offset)
    params: list[Any] = [run_id]
    where = "WHERE run_id = %s"
    if search:
        where += " AND (text_content ILIKE %s OR payload::text ILIKE %s)"
        like = f"%{search}%"
        params.extend([like, like])
    params.extend([limit, offset])

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM run_cvs {where}", tuple(params[:-2]))
            total = cur.fetchone()[0]
            cur.execute(
                f"""
                SELECT id, external_key, payload
                FROM run_cvs
                {where}
                ORDER BY id ASC
                LIMIT %s OFFSET %s
                """,
                tuple(params),
            )
            rows = cur.fetchall()

    return {
        "total": total,
        "items": [
            {"id": cid, "external_key": external_key, "payload": payload} for (cid, external_key, payload) in rows
        ],
    }


@app.get("/api/runs/{run_id}/cv_cards")
def list_cv_cards(
    run_id: int,
    search: str | None = None,
    limit: int = 10,
    offset: int = 0,
    top_k: int = 10,
) -> dict[str, Any]:
    """
    Lightweight batch endpoint for paginated CV cards and match summaries.
    """
    ensure_results_schema()
    limit = max(1, min(limit, 50))
    offset = max(0, offset)
    top_k = max(1, min(top_k, 100))

    params: list[Any] = [run_id]
    where = "WHERE run_id = %s"
    if search:
        where += " AND (text_content ILIKE %s OR payload::text ILIKE %s)"
        like = f"%{search}%"
        params.extend([like, like])

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM run_cvs {where}", tuple(params))
            total = int(cur.fetchone()[0])

            cur.execute(
                f"""
                SELECT id, external_key, payload
                FROM run_cvs
                {where}
                ORDER BY id ASC
                LIMIT %s OFFSET %s
                """,
                tuple(params + [limit, offset]),
            )
            cv_rows = cur.fetchall()

            cv_ids = [r[0] for r in cv_rows]
            matches_by_cv: dict[int, list[dict[str, Any]]] = {cid: [] for cid in cv_ids}

            if cv_ids:
                cur.execute(
                    """
                    WITH ranked AS (
                      SELECT rm.cv_id,
                             rm.rank,
                             rm.score,
                             jd.id,
                             jd.external_key,
                             jd.payload,
                             ROW_NUMBER() OVER (PARTITION BY rm.cv_id ORDER BY rm.rank ASC) AS rn
                      FROM run_matches rm
                      JOIN run_jds jd ON jd.id = rm.jd_id
                      WHERE rm.run_id = %s AND rm.cv_id = ANY(%s::bigint[])
                    )
                    SELECT cv_id, rank, score, id, external_key, payload
                    FROM ranked
                    WHERE rn <= %s
                    ORDER BY cv_id ASC, rank ASC
                    """,
                    (run_id, cv_ids, top_k),
                )
                for (cv_id, rank, score, jd_id, jd_key, jd_payload) in cur.fetchall():
                    matches_by_cv[int(cv_id)].append(
                        {
                            "rank": rank,
                            "score": score,
                            "jd": _jd_summary(jd_id, jd_key, jd_payload),
                        }
                    )

    items = []
    for (cid, external_key, payload) in cv_rows:
        items.append(
            {
                "cv": _cv_summary(cid, external_key, payload),
                "matches": matches_by_cv.get(int(cid), []),
            }
        )

    return {"total": total, "limit": limit, "offset": offset, "items": items}


@app.get("/api/runs/{run_id}/cvs/{cv_id}")
def get_cv(run_id: int, cv_id: int, top_k: int = 10) -> dict[str, Any]:
    ensure_results_schema()
    top_k = max(1, min(top_k, 100))
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, external_key, payload, text_content FROM run_cvs WHERE run_id=%s AND id=%s",
                (run_id, cv_id),
            )
            cv_row = cur.fetchone()
            if not cv_row:
                raise HTTPException(status_code=404, detail="cv not found")
            _, external_key, payload, text_content = cv_row

            cur.execute(
                """
                SELECT rm.rank, rm.score, rm.meta,
                       jd.id, jd.external_key, jd.payload
                FROM run_matches rm
                JOIN run_jds jd ON jd.id = rm.jd_id
                WHERE rm.run_id=%s AND rm.cv_id=%s
                ORDER BY rm.rank ASC
                LIMIT %s
                """,
                (run_id, cv_id, top_k),
            )
            matches = cur.fetchall()

    return {
        "cv": {"id": cv_id, "external_key": external_key, "payload": payload, "text_content": text_content},
        "matches": [
            {
                "rank": rank,
                "score": score,
                "meta": meta,
                "jd": {"id": jd_id, "external_key": jd_key, "payload": jd_payload},
            }
            for (rank, score, meta, jd_id, jd_key, jd_payload) in matches
        ],
    }


@app.get("/api/runs/{run_id}/jds/{jd_id}")
def get_jd(run_id: int, jd_id: int) -> dict[str, Any]:
    ensure_results_schema()
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, external_key, payload, text_content FROM run_jds WHERE run_id=%s AND id=%s",
                (run_id, jd_id),
            )
            row = cur.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="jd not found")
            _, external_key, payload, text_content = row
    return {"id": jd_id, "external_key": external_key, "payload": payload, "text_content": text_content}
