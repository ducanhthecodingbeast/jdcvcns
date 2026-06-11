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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import torch
import time

from pipeline import DATASET_DIR, RESULTS_DIR, first_non_empty_csv, get_cv_text, get_jd_text
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
    
    # 1. Load JDs and CVs
    jd_path = source_dir / "jd.csv"
    cv_path = first_non_empty_csv([
        source_dir / "mockcv.csv",
        target_dir / "mockcv.csv",
    ])
        
    if not os.path.exists(jd_path) or cv_path is None:
        raise FileNotFoundError("Missing datasets. Please make sure Data/jd.csv and a non-empty Data/mockcv.csv exist.")

    print(f"📖 Loading datasets...")
    df_jd = pd.read_csv(jd_path)
    df_cv = pd.read_csv(cv_path)
    
    df_jd.columns = df_jd.columns.str.strip()
    df_cv.columns = df_cv.columns.str.strip()
    
    # Consistent sampling
    df_cv = df_cv.sample(n=min(600, len(df_cv)), random_state=42).reset_index(drop=True)
    df_jd = df_jd.sample(n=min(100, len(df_jd)), random_state=42).reset_index(drop=True)
    
    print(f"Loaded {len(df_cv)} CVs and {len(df_jd)} JDs.")
    
    jd_texts = [get_jd_match_text(row) for _, row in df_jd.iterrows()]
    cv_texts = [get_cv_text(row) for _, row in df_cv.iterrows()]
    
    # --- LOAD EMBEDDINGS (CACHE FIRST) ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # A. Load AITeamVN/Vietnamese_Embedding_v2 Embeddings
    EMBEDDING_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cv_cache_v2 = EMBEDDING_CACHE_DIR / "cv_embeddings_v2.npy"
    jd_cache_v2 = EMBEDDING_CACHE_DIR / "jd_embeddings_v2.npy"
    cv_emb_v2, jd_emb_v2 = None, None
    
    if os.path.exists(cv_cache_v2) and os.path.exists(jd_cache_v2):
        print("🔄 Loading Vietnamese_Embedding_v2 from cache...")
        cv_emb_v2 = np.load(cv_cache_v2)
        jd_emb_v2 = np.load(jd_cache_v2)
        if len(cv_emb_v2) != len(df_cv) or len(jd_emb_v2) != len(df_jd):
            print("⚠️ V2 cache size mismatch. Re-computing...")
            cv_emb_v2, jd_emb_v2 = None, None
    if cv_emb_v2 is None or jd_emb_v2 is None:
        print("🚀 Computing dense embeddings using AITeamVN/Vietnamese_Embedding_v2...")
        try:
            model_v2 = SentenceTransformer("AITeamVN/Vietnamese_Embedding_v2", device=device)
        except Exception as e:
            model_v2 = SentenceTransformer("AITeamVN/Vietnamese_Embedding_v2", device="cpu")
        cv_emb_v2 = model_v2.encode(cv_texts, show_progress_bar=True)
        jd_emb_v2 = model_v2.encode(jd_texts, show_progress_bar=True)
        np.save(cv_cache_v2, cv_emb_v2)
        np.save(jd_cache_v2, jd_emb_v2)
        print("💾 Saved v2 embeddings to disk cache.")

    # B. Load BAAI/bge-m3 Embeddings
    cv_cache_bge = EMBEDDING_CACHE_DIR / "cv_embeddings_bge.npy"
    jd_cache_bge = EMBEDDING_CACHE_DIR / "jd_embeddings_bge.npy"
    cv_emb_bge, jd_emb_bge = None, None
    
    if os.path.exists(cv_cache_bge) and os.path.exists(jd_cache_bge):
        print("🔄 Loading BGE-M3 embeddings from cache...")
        cv_emb_bge = np.load(cv_cache_bge)
        jd_emb_bge = np.load(jd_cache_bge)
        if len(cv_emb_bge) != len(df_cv) or len(jd_emb_bge) != len(df_jd):
            print("⚠️ BGE cache size mismatch. Re-computing...")
            cv_emb_bge, jd_emb_bge = None, None
    if cv_emb_bge is None or jd_emb_bge is None:
        print("🚀 Computing dense embeddings using BAAI/bge-m3...")
        try:
            model_bge = SentenceTransformer("BAAI/bge-m3", device=device)
        except Exception as e:
            model_bge = SentenceTransformer("BAAI/bge-m3", device="cpu")
        cv_emb_bge = model_bge.encode(cv_texts, show_progress_bar=True)
        jd_emb_bge = model_bge.encode(jd_texts, show_progress_bar=True)
        np.save(cv_cache_bge, cv_emb_bge)
        np.save(jd_cache_bge, jd_emb_bge)
        print("💾 Saved BGE-M3 embeddings to disk cache.")

    # --- LEXICAL SIMILARITY (TF-IDF) ---
    print("📝 Computing TF-IDF Lexical similarity matrix...")
    vectorizer = TfidfVectorizer(token_pattern=r'(?u)\b\w+\b')
    # Fit vectorizer on all text to build vocabulary
    all_texts = jd_texts + cv_texts
    vectorizer.fit(all_texts)
    
    cv_sparse = vectorizer.transform(cv_texts)
    jd_sparse = vectorizer.transform(jd_texts)
    sparse_similarities = cosine_similarity(cv_sparse, jd_sparse) # shape: (600, 100)

    # --- TWO-PHASE RETRIEVAL & RE-RANKING ---
    print("⚡ Executing Two-Phase Retrieval (Dot Product v2 -> Hybrid BGE-M3)...")
    # Phase 1: Retrieve top 30 JDs using Vietnamese_Embedding_v2
    v2_similarities = cv_emb_v2 @ jd_emb_v2.T # shape: (600, 100)
    
    # BGE Dense similarities
    # We normalize to ensure dot product operates as cosine similarity
    cv_emb_bge_norm = cv_emb_bge / np.linalg.norm(cv_emb_bge, axis=1, keepdims=True)
    jd_emb_bge_norm = jd_emb_bge / np.linalg.norm(jd_emb_bge, axis=1, keepdims=True)
    bge_dense_similarities = cv_emb_bge_norm @ jd_emb_bge_norm.T # shape: (600, 100)
    
    if store_db:
        print("Storing results to Postgres...")
        conn = open_store()
        try:
            def ranked_matches(cv_idx):
                cv_v2_scores = v2_similarities[cv_idx]
                top_30_idx = np.argsort(cv_v2_scores)[::-1][:30]
                bge_dense_scores = bge_dense_similarities[cv_idx]
                lexical_scores = sparse_similarities[cv_idx]
                top_k_scores = [
                    (int(jd_idx), float(0.5 * bge_dense_scores[jd_idx] + 0.5 * lexical_scores[jd_idx]))
                    for jd_idx in top_30_idx
                ]
                top_k_scores.sort(key=lambda item: item[1], reverse=True)
                return [
                    {
                        "jd_idx": jd_idx,
                        "score": hybrid_score,
                        "meta": {
                            "v2": float(v2_similarities[cv_idx][jd_idx]),
                            "bge_dense": float(bge_dense_similarities[cv_idx][jd_idx]),
                            "lexical": float(sparse_similarities[cv_idx][jd_idx]),
                        },
                    }
                    for jd_idx, hybrid_score in top_k_scores[:top_k]
                ]

            run_id = store_match_run(
                conn,
                RunInfo(
                    run_name="dotpdtesting2.2",
                    algorithm="hybrid_dotproduct",
                    model_name="Vietnamese_Embedding_v2 + BGE-M3 + TFIDF",
                    params={
                        "top_k": top_k,
                        "jd_count": int(len(df_jd)),
                        "cv_count": int(len(df_cv)),
                    },
                    dataset_meta={"source_dir": str(source_dir)},
                ),
                df_cv,
                df_jd,
                get_cv_text,
                get_jd_match_text,
                ranked_matches,
                top_k=top_k,
                started_monotonic=started,
            )
            print(f"Stored run_id={run_id}")
        finally:
            conn.close()

if __name__ == "__main__":
    main()
