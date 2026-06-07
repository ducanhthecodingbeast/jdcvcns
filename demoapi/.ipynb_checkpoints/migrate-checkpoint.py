from pathlib import Path

try:
    from .db import get_conn
except ImportError:
    from db import get_conn


def _columns(conn, table_name: str) -> set[str]:
    rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    return {row[1] for row in rows}


def _ensure_sqlite_columns(conn) -> None:
    columns = _columns(conn, "test_runs")
    if not columns:
        return
    additions = {
        "default_page_size": "ALTER TABLE test_runs ADD COLUMN default_page_size INTEGER NOT NULL DEFAULT 10",
        "default_top_fit": "ALTER TABLE test_runs ADD COLUMN default_top_fit INTEGER NOT NULL DEFAULT 10",
    }
    for column, sql in additions.items():
        if column not in columns:
            conn.execute(sql)


def migrate() -> None:
    schema_path = Path(__file__).with_name("schema.sql")
    sql = schema_path.read_text(encoding="utf-8")
    with get_conn() as conn:
        conn.executescript(sql)
        _ensure_sqlite_columns(conn)


if __name__ == "__main__":
    migrate()
    print("OK: schema applied")
