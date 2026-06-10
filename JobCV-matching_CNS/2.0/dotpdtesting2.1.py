import os
os.environ["HF_HOME"] = os.path.join(os.path.dirname(__file__), ".cache")
os.environ["SENTENCE_TRANSFORMERS_HOME"] = os.path.join(os.path.dirname(__file__), ".cache")
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
import time
from pipeline import first_non_empty_csv, get_cv_text, get_jd_text
from testingresult import (
    RunInfo,
    env_flag,
    open_store,
    store_match_run,
)

def get_jd_match_text(row):
    return get_jd_text(row, include_company=False)

def main():
    started = time.monotonic()
    source_dir = "Dataset"
    target_dir = "TestingResults"
    os.makedirs(target_dir, exist_ok=True)

    store_db = env_flag("STORE_DB", True)
    top_k = int(os.environ.get("TOP_K", "10"))
    
    # 1. Load JDs and CVs
    jd_path = os.path.join(source_dir, "jd.csv")
    cv_path = first_non_empty_csv([
        os.path.join(target_dir, "mockcv.csv"),
        os.path.join(source_dir, "mockcv.csv"),
    ])
        
    if not os.path.exists(jd_path):
        # Check raw CSV
        raw_jd = os.path.join(source_dir, "JOB_DATA_FINAL.csv")
        if not os.path.exists(raw_jd):
            raise FileNotFoundError(f"Missing JD dataset in {source_dir}. Please place jd.csv or JOB_DATA_FINAL.csv there.")
        df_jd_raw = pd.read_csv(raw_jd)
        df_jd_raw = df_jd_raw.rename(columns={
            'Job Title': 'Vị trí cần tuyển',
            'Name Company': 'Tên công ty',
            'Company Overview': 'Giới thiệu công ty',
            'Company Size': 'Quy mô công ty',
            'Company Address': 'Địa chỉ công ty',
            'Job Description': 'Mô tả công việc',
            'Job Requirements': 'Yêu cầu công việc',
            'Benefits': 'Quyền lợi'
        })
        df_jd_raw.to_csv(jd_path, index=False)
        
    if cv_path is None:
        raise FileNotFoundError("Missing non-empty mockcv.csv. Run: python -m Dataset.mockcv")

    df_jd = pd.read_csv(jd_path)
    df_cv = pd.read_csv(cv_path)
    
    df_jd.columns = df_jd.columns.str.strip()
    df_cv.columns = df_cv.columns.str.strip()
    
    # Consistent deterministic sampling: 600 CVs, 100 JDs
    df_cv = df_cv.sample(n=min(600, len(df_cv)), random_state=42).reset_index(drop=True)
    df_jd = df_jd.sample(n=min(100, len(df_jd)), random_state=42).reset_index(drop=True)
    
    jd_texts = [get_jd_match_text(row) for _, row in df_jd.iterrows()]
    cv_texts = [get_cv_text(row) for _, row in df_cv.iterrows()]
    
    # 2. Resilient Embedding Loading (Cache check)
    cv_cache_path = os.path.join(source_dir, "cv_embeddings_v2.npy")
    jd_cache_path = os.path.join(source_dir, "jd_embeddings_v2.npy")
    
    cv_embeddings = None
    jd_embeddings = None
    
    if os.path.exists(cv_cache_path) and os.path.exists(jd_cache_path):
        try:
            cv_embeddings = np.load(cv_cache_path)
            jd_embeddings = np.load(jd_cache_path)
            if len(cv_embeddings) != len(df_cv) or len(jd_embeddings) != len(df_jd):
                cv_embeddings = None
                jd_embeddings = None
        except Exception as e:
            cv_embeddings = None
            jd_embeddings = None
            
    if cv_embeddings is None or jd_embeddings is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            model = SentenceTransformer("AITeamVN/Vietnamese_Embedding_v2", device=device)
        except Exception as e:
            if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
                model = SentenceTransformer("AITeamVN/Vietnamese_Embedding_v2", device="cpu")
            else:
                raise
                
        cv_embeddings = model.encode(cv_texts, show_progress_bar=True)
        jd_embeddings = model.encode(jd_texts, show_progress_bar=True)
        
        np.save(cv_cache_path, cv_embeddings)
        np.save(jd_cache_path, jd_embeddings)

    # 3. Match using Dot Product
    # Standard dot product matching
    similarities = cv_embeddings @ jd_embeddings.T # shape: (600, 100)
    
    # 4. Store results to Postgres (recommended)
    if store_db:
        conn = open_store()
        try:
            def ranked_matches(cv_idx):
                scores = similarities[cv_idx]
                top_k_idx = np.argsort(scores)[::-1][:top_k]
                return [{"jd_idx": int(jd_idx), "score": float(scores[int(jd_idx)])} for jd_idx in top_k_idx]

            store_match_run(
                conn,
                RunInfo(
                    run_name="dotpdtesting2.1",
                    algorithm="dot_product",
                    model_name="AITeamVN/Vietnamese_Embedding_v2",
                    params={"top_k": top_k, "jd_count": int(len(df_jd)), "cv_count": int(len(df_cv))},
                    dataset_meta={"source_dir": source_dir},
                ),
                df_cv,
                df_jd,
                get_cv_text,
                get_jd_match_text,
                ranked_matches,
                top_k=top_k,
                started_monotonic=started,
            )
        finally:
            conn.close()

if __name__ == "__main__":
    main()
