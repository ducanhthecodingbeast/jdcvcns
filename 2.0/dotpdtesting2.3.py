import os
os.environ["HF_HOME"] = os.path.join(os.path.dirname(__file__), ".cache")
os.environ["SENTENCE_TRANSFORMERS_HOME"] = os.path.join(os.path.dirname(__file__), ".cache")

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import torch
import time

from html_reports import build_recommendations, render_cv_to_jd_report
from results_store import (
    RunInfo,
    cv_has_matches,
    get_or_create_run,
    open_store,
    finish_run,
    row_payload,
    upsert_cv,
    upsert_jd,
    insert_matches,
)

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

def get_cv_text(row):
    """Combine relevant fields for CV representation"""
    parts = []
    if pd.notna(row.get('Tên ứng viên')): parts.append(f"Tên ứng viên: {row['Tên ứng viên']}")
    if pd.notna(row.get('Vị trí ứng tuyển')): parts.append(f"Vị trí ứng tuyển: {row['Vị trí ứng tuyển']}")
    if pd.notna(row.get('Lĩnh vực')): parts.append(f"Lĩnh vực: {row['Lĩnh vực']}")
    if pd.notna(row.get('Nơi làm việc mong muốn')): parts.append(f"Nơi làm việc: {row['Nơi làm việc mong muốn']}")
    if pd.notna(row.get('Mức lương mong muốn')): parts.append(f"Mức lương mong muốn: {row['Mức lương mong muốn']}")
    if pd.notna(row.get('Giới tính')): parts.append(f"Giới tính: {row['Giới tính']}")
    if pd.notna(row.get('Tình trạng hôn nhân')): parts.append(f"Tình trạng hôn nhân: {row['Tình trạng hôn nhân']}")
    if pd.notna(row.get('Tuổi')): parts.append(f"Tuổi: {row['Tuổi']}")
    if pd.notna(row.get('Mục tiêu nghề nghiệp')): parts.append(f"Mục tiêu: {row['Mục tiêu nghề nghiệp']}")
    if pd.notna(row.get('Kỹ năng')): parts.append(f"Kỹ năng: {row['Kỹ năng']}")
    if pd.notna(row.get('Kinh nghiệm')): parts.append(f"Kinh nghiệm: {row['Kinh nghiệm']}")
    if pd.notna(row.get('Bằng cấp')): parts.append(f"Bằng cấp: {row['Bằng cấp']}")
    return "\n".join(parts)

def main():
    started = time.monotonic()
    source_dir = "Dataset"
    target_dir = "TestingResults"
    os.makedirs(target_dir, exist_ok=True)

    # Default behavior: store results to Postgres and avoid huge HTML exports.
    store_db = os.environ.get("STORE_DB", "1").lower() in {"1", "true", "yes", "y"}
    export_html = os.environ.get("EXPORT_HTML", "0").lower() in {"1", "true", "yes", "y"}
    top_k = int(os.environ.get("TOP_K", "10"))
    
    # 1. Load JDs and CVs
    jd_path = os.path.join(source_dir, "jd.csv")
    cv_path = os.path.join(target_dir, "mockcv.csv")
    if not os.path.exists(cv_path):
        cv_path = os.path.join(source_dir, "mockcv.csv")
        
    if not os.path.exists(jd_path) or not os.path.exists(cv_path):
        raise FileNotFoundError("Missing datasets. Please make sure Dataset/jd.csv and TestingResults/mockcv.csv exist.")

    print(f"📖 Loading datasets...")
    df_jd = pd.read_csv(jd_path)
    df_cv = pd.read_csv(cv_path)
    
    df_jd.columns = df_jd.columns.str.strip()
    df_cv.columns = df_cv.columns.str.strip()
    
    # Consistent sampling
    df_cv = df_cv.sample(n=min(600, len(df_cv)), random_state=42).reset_index(drop=True)
    df_jd = df_jd.sample(n=min(100, len(df_jd)), random_state=42).reset_index(drop=True)
    
    print(f"Loaded {len(df_cv)} CVs and {len(df_jd)} JDs.")
    
    jd_texts = [get_jd_text(row) for _, row in df_jd.iterrows()]
    cv_texts = [get_cv_text(row) for _, row in df_cv.iterrows()]
    
    # --- LOAD EMBEDDINGS (CACHE FIRST) ---
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load BAAI/bge-m3 Embeddings
    cv_cache_bge = os.path.join(source_dir, "cv_embeddings_bge.npy")
    jd_cache_bge = os.path.join(source_dir, "jd_embeddings_bge.npy")
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

    # Normalize BGE dense embeddings to act as cosine similarity
    cv_emb_bge_norm = cv_emb_bge / np.linalg.norm(cv_emb_bge, axis=1, keepdims=True)
    jd_emb_bge_norm = jd_emb_bge / np.linalg.norm(jd_emb_bge, axis=1, keepdims=True)
    bge_dense_similarities = cv_emb_bge_norm @ jd_emb_bge_norm.T # shape: (600, 100)

    # --- DIRECT HYBRID SEARCH MATCHING ---
    print("⚡ Executing Direct BGE-M3 Hybrid Search Matching (no dot product pre-filter)...")
    # Hybrid score = 0.5 * Dense similarity + 0.5 * Sparse/Lexical similarity
    hybrid_similarities = 0.5 * bge_dense_similarities + 0.5 * sparse_similarities
    
    # Generate HTML Report
    # Store results to Postgres (recommended)
    if store_db:
        print("Storing results to Postgres...")
        conn = open_store()
        run_id = None
        try:
            run_id = get_or_create_run(
                conn,
                RunInfo(
                    run_name="dotpdtesting2.3",
                    algorithm="hybrid_only",
                    model_name="BGE-M3 + TFIDF",
                    params={"top_k": top_k, "jd_count": int(len(df_jd)), "cv_count": int(len(df_cv)), "hybrid_weights": [0.5, 0.5]},
                    dataset_meta={"source_dir": source_dir},
                ),
            )

            jd_ids = []
            for j, (_, jd_row) in enumerate(df_jd.iterrows()):
                jd_payload = row_payload(jd_row)
                jd_text = get_jd_text(jd_row)
                jd_id = upsert_jd(conn, run_id, external_key=str(j), payload=jd_payload, text_content=jd_text)
                jd_ids.append(jd_id)

            for i, (_, cv_row) in enumerate(tqdm(df_cv.iterrows(), total=len(df_cv), desc="Writing CV matches")):
                cv_payload = row_payload(cv_row)
                cv_text = get_cv_text(cv_row)
                cv_id = upsert_cv(conn, run_id, external_key=str(i), payload=cv_payload, text_content=cv_text)

                if cv_has_matches(conn, run_id, cv_id, expected_top_k=top_k):
                    continue

                cv_scores = hybrid_similarities[i]
                top_k_idx = np.argsort(cv_scores)[::-1][:top_k]
                matches = []
                for rank, jd_idx in enumerate(top_k_idx, start=1):
                    jd_idx = int(jd_idx)
                    matches.append(
                        {
                            "jd_id": jd_ids[jd_idx],
                            "rank": rank,
                            "score": float(cv_scores[jd_idx]),
                            "meta": {"jd_idx": jd_idx, "bge_dense": float(bge_dense_similarities[i][jd_idx]), "lexical": float(sparse_similarities[i][jd_idx])},
                        }
                    )
                insert_matches(conn, run_id, cv_id, matches)

            finish_run(conn, run_id, started)
            print(f"Stored run_id={run_id}")
        finally:
            conn.close()

    # Optional HTML export (disabled by default)
    if not export_html:
        print("Skipping HTML export (set EXPORT_HTML=1 to enable).")
        return

    html_out = os.path.join(target_dir, "results_2.3_hybrid_only.html")
    recommendations = build_recommendations(hybrid_similarities, top_k=10)
    render_cv_to_jd_report(
        html_out,
        df_cv,
        df_jd,
        recommendations,
        title="CV Matching Results - 2.3 Hybrid Only",
        subtitle="BGE-M3 dense similarity with TF-IDF lexical weighting",
    )
    print(f"Saved visual matching report to {html_out}")

if __name__ == "__main__":
    main()
