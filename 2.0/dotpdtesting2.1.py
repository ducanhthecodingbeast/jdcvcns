import os
os.environ["HF_HOME"] = os.path.join(os.path.dirname(__file__), ".cache")
os.environ["SENTENCE_TRANSFORMERS_HOME"] = os.path.join(os.path.dirname(__file__), ".cache")
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
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
        # Fallback to Dataset dir
        cv_path = os.path.join(source_dir, "mockcv.csv")
        
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
        
    if not os.path.exists(cv_path):
        raise FileNotFoundError(f"Missing {cv_path}. Please run dotpdtesting.py first to generate the mock CVs.")

    df_jd = pd.read_csv(jd_path)
    df_cv = pd.read_csv(cv_path)
    
    df_jd.columns = df_jd.columns.str.strip()
    df_cv.columns = df_cv.columns.str.strip()
    
    # Consistent deterministic sampling: 600 CVs, 100 JDs
    df_cv = df_cv.sample(n=min(600, len(df_cv)), random_state=42).reset_index(drop=True)
    df_jd = df_jd.sample(n=min(100, len(df_jd)), random_state=42).reset_index(drop=True)
    
    jd_texts = [get_jd_text(row) for _, row in df_jd.iterrows()]
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
        run_id = None
        try:
            run_id = get_or_create_run(
                conn,
                RunInfo(
                    run_name="dotpdtesting2.1",
                    algorithm="dot_product",
                    model_name="AITeamVN/Vietnamese_Embedding_v2",
                    params={"top_k": top_k, "jd_count": int(len(df_jd)), "cv_count": int(len(df_cv))},
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

                scores = similarities[i]
                top_k_idx = np.argsort(scores)[::-1][:top_k]
                matches = []
                for rank, jd_idx in enumerate(top_k_idx, start=1):
                    jd_idx = int(jd_idx)
                    matches.append({"jd_id": jd_ids[jd_idx], "rank": rank, "score": float(scores[jd_idx]), "meta": {"jd_idx": jd_idx}})
                insert_matches(conn, run_id, cv_id, matches)

            finish_run(conn, run_id, started)
        finally:
            conn.close()

    # 5. Optional: Generate Visual HTML Report (disabled by default)
    if not export_html:
        return

    html_out = os.path.join(target_dir, "results_2.1_dotproduct.html")
    recommendations = build_recommendations(similarities, top_k=10)
    render_cv_to_jd_report(
        html_out,
        df_cv,
        df_jd,
        recommendations,
        title="CV Matching Results - 2.1 Dot Product",
        subtitle="AITeamVN/Vietnamese_Embedding_v2 dense dot product retrieval",
    )

if __name__ == "__main__":
    main()
