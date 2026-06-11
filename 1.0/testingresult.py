import json
import os
import time
from dataclasses import dataclass
from datetime import date, datetime
from hashlib import sha256
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd
import psycopg2
from tqdm import tqdm

from demoAPI.db import get_database_url

TRUTHY = {"1", "true", "yes", "y"}


def env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    return default if value is None else value.strip().lower() in TRUTHY


def _database_url() -> str:
    return get_database_url()


def _connect():
    return psycopg2.connect(_database_url())


def ensure_schema(conn) -> None:
    schema_path = os.path.join(os.path.dirname(__file__), "demoAPI", "schema.sql")
    with open(schema_path, "r", encoding="utf-8") as f:
        sql = f.read()
    with conn.cursor() as cur:
        cur.execute(sql)
    conn.commit()


def to_jsonable(value: Any) -> Any:
    if value is None:
        return None
    try:
        if not isinstance(value, (dict, list, tuple, np.ndarray)) and pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, float) and np.isnan(value):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        return [to_jsonable(v) for v in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return value


def row_payload(row: Any) -> dict[str, Any]:
    return to_jsonable(row.to_dict())


@dataclass
class RunInfo:
    run_name: str
    algorithm: str
    model_name: str | None = None
    params: dict[str, Any] | None = None
    dataset_meta: dict[str, Any] | None = None
    notes: str | None = None


def compute_run_key(info: RunInfo) -> str:
    payload = {
        "run_name": info.run_name,
        "algorithm": info.algorithm,
        "model_name": info.model_name,
        "params": to_jsonable(info.params or {}),
        "dataset_meta": to_jsonable(info.dataset_meta or {}),
        "notes": info.notes,
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return sha256(blob).hexdigest()


def get_or_create_run(conn, info: RunInfo, *, allow_new_attempt: bool = True) -> int:
    """
    Resume behavior:
    - If env RUN_ID is set, use that (no lookup).
    - Else use stable run_key derived from RunInfo.
      - If a run exists and is unfinished -> resume it.
      - If it exists and is finished:
          - if allow_new_attempt: create a new run with run_key suffixed with timestamp
          - else: reuse the finished run (read-only mode)
    """
    env_run_id = os.environ.get("RUN_ID", "").strip()
    if env_run_id:
        return int(env_run_id)

    run_key = compute_run_key(info)
    params = info.params or {}
    dataset_meta = info.dataset_meta or {}

    with conn.cursor() as cur:
        cur.execute("SELECT id, finished_at FROM test_runs WHERE run_key = %s", (run_key,))
        row = cur.fetchone()
        if row:
            run_id, finished_at = row
            if finished_at is None:
                return int(run_id)
            if not allow_new_attempt:
                return int(run_id)

            # Create a new attempt key to avoid uniqueness collisions.
            attempt_key = f"{run_key}:{int(time.time())}"
            cur.execute(
                """
                INSERT INTO test_runs (run_key, run_name, algorithm, model_name, params, dataset_meta, notes, started_at)
                VALUES (%s, %s, %s, %s, %s::jsonb, %s::jsonb, %s, now())
                RETURNING id
                """,
                (
                    attempt_key,
                    info.run_name,
                    info.algorithm,
                    info.model_name,
                    json.dumps(to_jsonable(params), ensure_ascii=False),
                    json.dumps(to_jsonable(dataset_meta), ensure_ascii=False),
                    info.notes,
                ),
            )
            run_id = cur.fetchone()[0]
            conn.commit()
            return int(run_id)

        cur.execute(
            """
            INSERT INTO test_runs (run_key, run_name, algorithm, model_name, params, dataset_meta, notes, started_at)
            VALUES (%s, %s, %s, %s, %s::jsonb, %s::jsonb, %s, now())
            RETURNING id
            """,
            (
                run_key,
                info.run_name,
                info.algorithm,
                info.model_name,
                json.dumps(to_jsonable(params), ensure_ascii=False),
                json.dumps(to_jsonable(dataset_meta), ensure_ascii=False),
                info.notes,
            ),
        )
        run_id = cur.fetchone()[0]
    conn.commit()
    return int(run_id)


def create_run(conn, info: RunInfo) -> int:
    params = info.params or {}
    dataset_meta = info.dataset_meta or {}
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO test_runs (run_name, algorithm, model_name, params, dataset_meta, notes, started_at)
            VALUES (%s, %s, %s, %s::jsonb, %s::jsonb, %s, now())
            RETURNING id
            """,
            (
                info.run_name,
                info.algorithm,
                info.model_name,
                json.dumps(to_jsonable(params), ensure_ascii=False),
                json.dumps(to_jsonable(dataset_meta), ensure_ascii=False),
                info.notes,
            ),
        )
        run_id = cur.fetchone()[0]
    conn.commit()
    return run_id


def finish_run(conn, run_id: int, started_monotonic: float) -> None:
    runtime_ms = int((time.monotonic() - started_monotonic) * 1000)
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE test_runs SET finished_at=now(), runtime_ms=%s WHERE id=%s",
            (runtime_ms, run_id),
        )
    conn.commit()


def upsert_cv(conn, run_id: int, external_key: str | None, payload: dict[str, Any], text_content: str | None) -> int:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO run_cvs (run_id, external_key, payload, text_content)
            VALUES (%s, %s, %s::jsonb, %s)
            ON CONFLICT (run_id, external_key)
            DO UPDATE SET payload = EXCLUDED.payload, text_content = EXCLUDED.text_content
            RETURNING id
            """,
            (run_id, external_key, json.dumps(to_jsonable(payload), ensure_ascii=False), text_content),
        )
        cid = cur.fetchone()[0]
    return cid


def upsert_jd(conn, run_id: int, external_key: str | None, payload: dict[str, Any], text_content: str | None) -> int:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO run_jds (run_id, external_key, payload, text_content)
            VALUES (%s, %s, %s::jsonb, %s)
            ON CONFLICT (run_id, external_key)
            DO UPDATE SET payload = EXCLUDED.payload, text_content = EXCLUDED.text_content
            RETURNING id
            """,
            (run_id, external_key, json.dumps(to_jsonable(payload), ensure_ascii=False), text_content),
        )
        jid = cur.fetchone()[0]
    return jid


def insert_matches(
    conn,
    run_id: int,
    cv_id: int,
    matches: Iterable[dict[str, Any]],
) -> None:
    with conn.cursor() as cur:
        for m in matches:
            cur.execute(
                """
                INSERT INTO run_matches (run_id, cv_id, jd_id, rank, score, meta)
                VALUES (%s, %s, %s, %s, %s, %s::jsonb)
                ON CONFLICT (run_id, cv_id, rank)
                DO UPDATE SET jd_id = EXCLUDED.jd_id, score = EXCLUDED.score, meta = EXCLUDED.meta
                """,
                (
                    run_id,
                    cv_id,
                    int(m["jd_id"]),
                    int(m["rank"]),
                    float(m["score"]),
                    json.dumps(to_jsonable(m.get("meta", {})), ensure_ascii=False),
                ),
            )
    conn.commit()


def cv_has_matches(conn, run_id: int, cv_id: int, *, expected_top_k: int) -> bool:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT COUNT(*) FROM run_matches WHERE run_id=%s AND cv_id=%s",
            (run_id, cv_id),
        )
        count = int(cur.fetchone()[0])
    return count >= int(expected_top_k)


def store_match_run(
    conn,
    info: RunInfo,
    df_cv: pd.DataFrame,
    df_jd: pd.DataFrame,
    cv_text_fn: Callable[[pd.Series], str],
    jd_text_fn: Callable[[pd.Series], str],
    ranked_matches_fn: Callable[[int], Iterable[dict[str, Any]]],
    *,
    top_k: int,
    started_monotonic: float,
    desc: str = "Writing CV matches",
) -> int:
    run_id = get_or_create_run(conn, info)

    jd_ids = []
    for j, (_, jd_row) in enumerate(df_jd.iterrows()):
        jd_ids.append(
            upsert_jd(
                conn,
                run_id,
                external_key=str(j),
                payload=row_payload(jd_row),
                text_content=jd_text_fn(jd_row),
            )
        )

    for i, (_, cv_row) in enumerate(tqdm(df_cv.iterrows(), total=len(df_cv), desc=desc)):
        cv_id = upsert_cv(
            conn,
            run_id,
            external_key=str(i),
            payload=row_payload(cv_row),
            text_content=cv_text_fn(cv_row),
        )
        if cv_has_matches(conn, run_id, cv_id, expected_top_k=top_k):
            continue

        matches = []
        for rank, match in enumerate(ranked_matches_fn(i), start=1):
            jd_idx = int(match["jd_idx"])
            match_meta = to_jsonable(match.get("meta") or {})
            meta = {"jd_idx": jd_idx, **match_meta} if isinstance(match_meta, dict) else {"jd_idx": jd_idx}
            matches.append(
                {
                    "jd_id": jd_ids[jd_idx],
                    "rank": int(match.get("rank", rank)),
                    "score": float(match["score"]),
                    "meta": meta,
                }
            )
        insert_matches(conn, run_id, cv_id, matches)

    finish_run(conn, run_id, started_monotonic)
    return run_id


def open_store():
    conn = _connect()
    ensure_schema(conn)
    return conn
