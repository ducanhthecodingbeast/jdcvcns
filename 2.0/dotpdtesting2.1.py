import os
os.environ["HF_HOME"] = os.path.join(os.path.dirname(__file__), ".cache")
os.environ["SENTENCE_TRANSFORMERS_HOME"] = os.path.join(os.path.dirname(__file__), ".cache")
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
EMBEDDING_CACHE_DIR = PROJECT_ROOT / ".cache" / "embeddings"

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
import time
from pipeline import DATASET_DIR, RESULTS_DIR, get_cv_text, get_jd_text, load_datasets
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
    source_dir = DATASET_DIR
    target_dir = RESULTS_DIR
    os.makedirs(target_dir, exist_ok=True)

    store_db = env_flag("STORE_DB", True)
    top_k = int(os.environ.get("TOP_K", "10"))
    
    df_cv, df_jd = load_datasets(source_dir)
    
    jd_texts = [get_jd_match_text(row) for _, row in df_jd.iterrows()]
    cv_texts = [get_cv_text(row) for _, row in df_cv.iterrows()]
    
    # 2. Resilient Embedding Loading (Cache check)
    EMBEDDING_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cv_cache_path = EMBEDDING_CACHE_DIR / "selected_cv_embeddings_v2.npy"
    jd_cache_path = EMBEDDING_CACHE_DIR / "selected_jd_embeddings_v2.npy"
    
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
        print(f"Loading AITeamVN/Vietnamese_Embedding_v2 on {device.upper()}")
        try:
            model = SentenceTransformer("AITeamVN/Vietnamese_Embedding_v2", device=device)
        except Exception as e:
            if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
                print(f"WARNING: Failed to load on {device}. Falling back to CPU. Error: {e}")
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
                    params={
                        "top_k": top_k,
                        "jd_count": int(len(df_jd)),
                        "cv_count": int(len(df_cv)),
                        "selection": "job_title_to_desired_job_dot_product",
                    },
                    dataset_meta={"source_dir": str(source_dir), "cv_path": str(source_dir / "cv.csv")},
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
