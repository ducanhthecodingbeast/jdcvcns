import psycopg2

from demoAPI.db import get_database_url


print("Testing PostgreSQL connection...")
try:
    conn = psycopg2.connect(get_database_url())
    print("Successfully connected to PostgreSQL.")

    cur = conn.cursor()
    cur.execute("SELECT extname FROM pg_extension WHERE extname = 'vector';")
    ext = cur.fetchone()
    if ext:
        print("pgvector extension is enabled and ready to use!")
    else:
        print("pgvector extension is NOT enabled.")

    conn.close()
except Exception as e:
    print(f"Connection failed: {e}")
