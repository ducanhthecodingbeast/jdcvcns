import os
from contextlib import contextmanager

import psycopg2


def get_database_url() -> str:
    return os.environ.get(
        "DATABASE_URL",
        "postgresql://postgres.hihyvqdpwcvuadunpjfj:_Epitwh%3FWbb6qvz@aws-1-ap-southeast-1.pooler.supabase.com:5432/postgres",
    )


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

