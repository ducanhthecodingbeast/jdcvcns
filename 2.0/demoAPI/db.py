import os
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import quote_plus

import psycopg2


try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - dotenv is optional at import time
    load_dotenv = None


if load_dotenv:
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")


def get_database_url() -> str:
    database_url = os.environ.get("DATABASE_URL", "").strip()
    if database_url:
        if database_url.startswith("sqlite:"):
            raise RuntimeError("DATABASE_URL must be a PostgreSQL URL; SQLite is not supported by this schema.")
        return database_url

    host = os.environ.get("POSTGRES_HOST", "localhost")
    port = os.environ.get("POSTGRES_PORT", "15420")
    user = os.environ.get("POSTGRES_USER", "jdcvcns")
    password = quote_plus(os.environ.get("POSTGRES_PASSWORD", "jdcvcns_dev_password"))
    database = os.environ.get("POSTGRES_DB", "jdcvcns")
    return f"postgresql://{user}:{password}@{host}:{port}/{database}"


@contextmanager
def get_conn():
    conn = psycopg2.connect(get_database_url())
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
