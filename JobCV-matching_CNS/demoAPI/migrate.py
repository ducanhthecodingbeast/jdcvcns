from pathlib import Path

try:
    from .db import get_conn
except ImportError:
    from db import get_conn


def migrate() -> None:
    schema_path = Path(__file__).with_name("schema.sql")
    sql = schema_path.read_text(encoding="utf-8")
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql)


if __name__ == "__main__":
    migrate()
    print("OK: schema applied")
