import psycopg2

db_url = "postgresql://postgres:_Epitwh%3FWbb6qvz@hihyvqdpwcvuadunpjfj.pooler.supabase.com:6543/postgres"

print("Testing connection to Supabase...")
try:
    conn = psycopg2.connect(db_url)
    print("Successfully connected to Supabase!")
    
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
