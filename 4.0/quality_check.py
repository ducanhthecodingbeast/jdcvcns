from __future__ import annotations

import os
import sys
from pathlib import Path

import psycopg2

from demoAPI.db import get_database_url
from pipeline import DATASET_DIR, is_non_empty_csv


PROJECT_ROOT = Path(__file__).resolve().parent


def fail(message: str) -> None:
    print(f"FAIL: {message}")
    raise SystemExit(1)


def pass_check(message: str) -> None:
    print(f"OK: {message}")


def check_data() -> None:
    jd_path = DATASET_DIR / "jd.csv"
    cv_candidates = [
        DATASET_DIR / "mockcv.csv",
        PROJECT_ROOT / "TestingResults" / "mockcv.csv",
        DATASET_DIR / "cv.csv",
    ]

    if not is_non_empty_csv(jd_path):
        fail(f"missing or empty JD data file: {jd_path}")
    pass_check(f"JD data file available: {jd_path}")

    cv_path = next((path for path in cv_candidates if is_non_empty_csv(path)), None)
    if cv_path is None:
        fail("missing CV input. Expected mockcv.csv or cv.csv in ../Data")
    pass_check(f"CV input available: {cv_path}")


def check_imports() -> None:
    try:
        from FlagEmbedding import BGEM3FlagModel, FlagReranker  # noqa: F401
        from qdrant_client import QdrantClient  # noqa: F401
    except ImportError as exc:
        fail(f"missing dependency: {exc}")
    pass_check("FlagEmbedding and qdrant-client imports work")


def check_config() -> None:
    positive_ints = [
        "TOP_K",
        "BGE_BATCH_SIZE",
        "VIRANKER_BATCH_SIZE",
        "VIRANKER_MAX_LENGTH",
        "PREFETCH_MULTIPLIER",
    ]
    non_negative_ints = ["VIRANKER_QUERY_MAX_LENGTH"]

    for name in positive_ints:
        value = os.environ.get(name)
        if value is None or not value.strip():
            continue
        try:
            parsed = int(value)
        except ValueError:
            fail(f"{name} must be an integer, got {value!r}")
        if parsed < 1:
            fail(f"{name} must be >= 1, got {parsed}")

    for name in non_negative_ints:
        value = os.environ.get(name)
        if value is None or not value.strip():
            continue
        try:
            parsed = int(value)
        except ValueError:
            fail(f"{name} must be an integer, got {value!r}")
        if parsed < 0:
            fail(f"{name} must be >= 0, got {parsed}")

    pass_check("runtime configuration values are valid")


def check_postgres() -> None:
    try:
        conn = psycopg2.connect(get_database_url())
        with conn.cursor() as cur:
            cur.execute("SELECT 1;")
            cur.fetchone()
        conn.close()
    except Exception as exc:
        fail(f"PostgreSQL connection failed: {exc}")
    pass_check("PostgreSQL connection works")


def check_qdrant() -> None:
    try:
        from qdrant_client import QdrantClient

        qdrant_path = os.environ.get("QDRANT_PATH")
        qdrant_url = os.environ.get("QDRANT_URL")
        if qdrant_path:
            client = QdrantClient(path=qdrant_path)
        elif qdrant_url:
            client = QdrantClient(url=qdrant_url)
        else:
            client = QdrantClient(
                host=os.environ.get("QDRANT_HOST", "localhost"),
                port=int(os.environ.get("QDRANT_PORT", "16340")),
            )
        client.get_collections()
    except Exception as exc:
        fail(f"Qdrant connection failed: {exc}")
    pass_check("Qdrant connection works")


def main() -> None:
    print("Checking 4.0 quality harness...")
    check_data()
    check_imports()
    check_config()
    check_postgres()
    check_qdrant()
    pass_check("4.0 is ready to run virankertesting4.0.py")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
