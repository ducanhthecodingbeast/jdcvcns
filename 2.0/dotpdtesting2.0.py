import os
os.environ["HF_HOME"] = os.path.join(os.path.dirname(__file__), ".cache")
os.environ["SENTENCE_TRANSFORMERS_HOME"] = os.path.join(os.path.dirname(__file__), ".cache")

import json
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import torch
import psycopg2
from pgvector.psycopg2 import register_vector

def get_jd_text(row):
    """Combine relevant fields for JD representation - EXCLUDING Tên công ty"""
    parts = []
    if pd.notna(row.get('Vị trí cần tuyển')): parts.append(f"Vị trí cần tuyển: {row['Vị trí cần tuyển']}")
    if pd.notna(row.get('Giới thiệu công ty')): parts.append(f"Giới thiệu công ty: {row['Giới thiệu công ty']}")
    if pd.notna(row.get('Quy mô công ty')): parts.append(f"Quy mô công ty: {row['Quy mô công ty']}")
    if pd.notna(row.get('Địa chỉ công ty')): parts.append(f"Địa điểm: {row['Địa chỉ công ty']}")
    if pd.notna(row.get('Mô tả công việc')): parts.append(f"Mô tả công việc: {row['Mô tả công việc']}")
    if pd.notna(row.get('Yêu cầu công việc')): parts.append(f"Yêu cầu: {row['Yêu cầu công việc']}")
    if pd.notna(row.get('Quyền lợi')): parts.append(f"Quyền lợi: {row['Quyền lợi']}")
    return "\n".join(parts)

def load_and_preprocess_jd(source_dir):
    jd_path = os.path.join(source_dir, "jd.csv")
    
    if os.path.exists(jd_path):
        df_jd = pd.read_csv(jd_path)
    else:
        raw_file = os.path.join(source_dir, "JOB_DATA_FINAL.csv")
        if not os.path.exists(raw_file):
            raise FileNotFoundError(f"Missing raw data {raw_file}. Please place it in the Dataset folder.")
            
        df_jd = pd.read_csv(raw_file)
        df_jd = df_jd.drop(columns=['URL Job', 'JobID'], errors='ignore')
        if 'Job Title' in df_jd.columns:
            df_jd['Job Title'] = df_jd['Job Title'].astype(str).str.lower().str.strip()
            
        df_jd = df_jd.rename(columns={
            'Job Title': 'Vị trí cần tuyển',
            'Name Company': 'Tên công ty',
            'Company Overview': 'Giới thiệu công ty',
            'Company Size': 'Quy mô công ty',
            'Company Address': 'Địa chỉ công ty',
            'Job Description': 'Mô tả công việc',
            'Job Requirements': 'Yêu cầu công việc',
            'Benefits': 'Quyền lợi'
        })
        df_jd.to_csv(jd_path, index=False)
        
    df_jd.columns = df_jd.columns.str.strip()
    return df_jd

def main():
    source_dir = "Dataset"
    
    # Load and preprocess data
    df_jd = load_and_preprocess_jd(source_dir)
    jd_texts = [get_jd_text(row) for _, row in df_jd.iterrows()]
    
    # 1. Connect to PostgreSQL FIRST to avoid wasting time on embeddings if connection is down
    print("🔌 Connecting to PostgreSQL...")
    try:
        # Connect to your new Supabase instance!
        conn = psycopg2.connect("postgresql://postgres.hihyvqdpwcvuadunpjfj:_Epitwh%3FWbb6qvz@aws-1-ap-southeast-1.pooler.supabase.com:5432/postgres")
    except psycopg2.Error as e:
        print(f"❌ Error connecting to the database: {e}")
        print("💡 The database container maps port 15566 on localhost. Please check that 'docker-db-1' is active.")
        return

    with conn.cursor() as cur:
        # Enable pgvector and create tables
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        conn.commit()
        
        register_vector(conn)
        
        # Check if table exists and has the correct vector dimension (1024)
        cur.execute("""
            SELECT atttypmod FROM pg_attribute 
            WHERE attrelid = to_regclass('job_descriptions') AND attname = 'embedding';
        """)
        res = cur.fetchone()
        if res is not None:
            db_dim = res[0]
            if db_dim != 1024:
                print(f"⚠️ Vector dimension mismatch: DB has {db_dim}, script expects 1024.")
                print("🔄 Dropping existing 'job_descriptions' table to recreate it with 1024 dimensions...")
                cur.execute("DROP TABLE IF EXISTS job_descriptions CASCADE;")
                conn.commit()
        
        cur.execute("""
            CREATE TABLE IF NOT EXISTS job_descriptions (
                id SERIAL PRIMARY KEY,
                title TEXT,
                company TEXT,
                searchable_text TEXT,
                embedding vector(1024)
            );
        """)
        conn.commit()

        # 2. Embed JDs - Check cache first to avoid re-computing (saves 7 minutes!)
        embeddings_cache_path = os.path.join(source_dir, "jd_embeddings.npy")
        jd_embeddings = None
        
        if os.path.exists(embeddings_cache_path):
            print(f"🔄 Loading cached embeddings from {embeddings_cache_path}...")
            try:
                jd_embeddings = np.load(embeddings_cache_path)
                if len(jd_embeddings) != len(df_jd):
                    print("⚠️ Cache size mismatch. Re-embedding...")
                    jd_embeddings = None
            except Exception as e:
                print(f"⚠️ Failed to load embedding cache: {e}. Re-embedding...")
                jd_embeddings = None

        if jd_embeddings is None:
            print("🚀 Initialize SentenceTransformer and generating embeddings...")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            try:
                model = SentenceTransformer("AITeamVN/Vietnamese_Embedding_v2", device=device)
            except Exception as e:
                if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
                    print("⚠️ GPU CUDA is out of memory. Falling back to CPU for embedding...")
                    model = SentenceTransformer("AITeamVN/Vietnamese_Embedding_v2", device="cpu")
                else:
                    raise
            
            jd_embeddings = model.encode(jd_texts, show_progress_bar=True)
            np.save(embeddings_cache_path, jd_embeddings)
            print(f"💾 Saved computed embeddings to cache: {embeddings_cache_path}")

        # 3. Insert JDs with progress tracking and duplicate checking
        inserted_count = 0
        skipped_count = 0
        
        for idx in tqdm(range(len(df_jd)), total=len(df_jd), desc="Uploading to DB"):
            row = df_jd.iloc[idx]
            title = str(row.get('Vị trí cần tuyển', 'Unknown Job'))
            company = str(row.get('Tên công ty', 'Unknown Company'))
            searchable_text = jd_texts[idx]
            emb = jd_embeddings[idx].tolist()
            
            # Check if this record is already in PostgreSQL
            cur.execute("""
                SELECT 1 FROM job_descriptions 
                WHERE title = %s AND company = %s AND searchable_text = %s 
                LIMIT 1;
            """, (title, company, searchable_text))
            
            if cur.fetchone():
                skipped_count += 1
                continue
            
            cur.execute("""
                INSERT INTO job_descriptions (title, company, searchable_text, embedding)
                VALUES (%s, %s, %s, %s)
            """, (title, company, searchable_text, emb))
            inserted_count += 1
            
            # Commit periodically to keep saving progress in case of crash
            if inserted_count % 50 == 0:
                conn.commit()
                
        conn.commit()
        print(f"🎉 DB Upload completed! Inserted: {inserted_count}, Skipped (Already existed): {skipped_count}")

if __name__ == "__main__":
    main()
