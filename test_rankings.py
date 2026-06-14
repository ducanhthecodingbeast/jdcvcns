import psycopg2
import json

PORTS = [15410, 15420, 15430, 15440, 15600]

print("\n" + "="*80)
print("RANKING RESULTS (Top 9 JDs for First CV)")
print("="*80)

for port in PORTS:
    try:
        conn = psycopg2.connect(
            host="localhost",
            port=port,
            user="jdcvcns",
            password="jdcvcns_dev_password",
            dbname="jdcvcns"
        )
        cur = conn.cursor()
        cur.execute("SELECT id, run_name, algorithm FROM test_runs ORDER BY started_at DESC LIMIT 10;")
        all_runs = cur.fetchall()
        
        seen_algos = set()
        
        for run_id, run_name, algo in all_runs:
            if algo in seen_algos:
                continue
            seen_algos.add(algo)
            
            print(f"\n[Phase from DB on Port {port}] - {run_name} ({algo})")
            print("-" * 50)
            
            cur.execute("SELECT id, text_content FROM run_cvs WHERE run_id = %s ORDER BY id ASC LIMIT 1;", (run_id,))
            cv = cur.fetchone()
            if not cv:
                print("No CVs found.")
                continue
                
            cv_id, cv_text = cv
            cv_preview = (cv_text[:60] + "...") if cv_text and len(cv_text) > 60 else cv_text
            print(f"CV: {cv_preview}")
            
            cur.execute("""
                SELECT m.rank, m.score, j.text_content 
                FROM run_matches m 
                JOIN run_jds j ON m.jd_id = j.id 
                WHERE m.run_id = %s AND m.cv_id = %s 
                ORDER BY m.rank ASC 
                LIMIT 9;
            """, (run_id, cv_id))
            
            matches = cur.fetchall()
            for rank, score, jd_text in matches:
                jd_preview = (jd_text[:60].replace("\n", " ") + "...") if jd_text else "None"
                print(f"Rank {rank}: [Score: {score:.4f}] - {jd_preview}")
                
        cur.close()
        conn.close()
    except Exception as e:
        print(f"Could not connect or read from DB on port {port}. Error: {e}")
        
print("\n" + "="*80)
