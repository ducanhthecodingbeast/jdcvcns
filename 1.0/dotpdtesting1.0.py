import os
import time
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from Dataset.mockcv import generate_and_mix_cvs
from html_reports import render_jd_to_cv_report
from results_store import (
    RunInfo,
    cv_has_matches,
    get_or_create_run,
    finish_run,
    insert_matches,
    open_store,
    row_payload,
    upsert_cv,
    upsert_jd,
)

def get_jd_text(row):
    """Combine relevant fields for JD representation"""
    parts = []
    if pd.notna(row.get('Vị trí cần tuyển')): parts.append(f"Vị trí cần tuyển: {row['Vị trí cần tuyển']}")
    if pd.notna(row.get('Tên công ty')): parts.append(f"Tên công ty: {row['Tên công ty']}")
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

    # LLM Generated Fields
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

    store_db = os.environ.get("STORE_DB", "1").lower() in {"1", "true", "yes", "y"}
    export_html = os.environ.get("EXPORT_HTML", "0").lower() in {"1", "true", "yes", "y"}
    top_k = int(os.environ.get("TOP_K", "10"))
    
    jd_path = os.path.join(source_dir, "jd.csv")
    cv_path = os.path.join(source_dir, "cv.csv")

    if not os.path.exists(jd_path) or not os.path.exists(cv_path):
        raise FileNotFoundError(f"Missing {jd_path} or {cv_path}")

    df_jd = pd.read_csv(jd_path)
    raw_cv = pd.read_csv(cv_path)
    
    # Strip column whitespace BEFORE any processing
    df_jd.columns = df_jd.columns.str.strip()
    raw_cv.columns = raw_cv.columns.str.strip()
    
    df_cv = generate_and_mix_cvs(df_jd, raw_cv, target_dir)
    df_cv.columns = df_cv.columns.str.strip()

    title_col = 'Vị trí cần tuyển'


    top_titles = df_jd[title_col].value_counts().head(30).index
    
    sample_jds = pd.concat([
        df_jd[df_jd[title_col] == title].sample(n=2, replace=True) 
        for title in top_titles
    ]).reset_index(drop=True)




    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        model = SentenceTransformer("AITeamVN/Vietnamese_Embedding_v2", device=device)
    except Exception as e:
        if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
            model = SentenceTransformer("AITeamVN/Vietnamese_Embedding_v2", device="cpu")
        else:
            raise


    jd_texts = [get_jd_text(row) for _, row in sample_jds.iterrows()]
    cv_texts = [get_cv_text(row) for _, row in df_cv.iterrows()]


    jd_embeddings = model.encode(jd_texts, show_progress_bar=True)
    

    cv_embeddings = model.encode(cv_texts, show_progress_bar=True)


    similarities = jd_embeddings @ cv_embeddings.T

    if store_db:
        conn = open_store()
        run_id = None
        try:
            run_id = get_or_create_run(
                conn,
                RunInfo(
                    run_name="dotpdtesting1.0",
                    algorithm="jd_to_cv_dot_product",
                    model_name="AITeamVN/Vietnamese_Embedding_v2",
                    params={
                        "top_k": top_k,
                        "sample_jd_count": int(len(sample_jds)),
                        "cv_count": int(len(df_cv)),
                    },
                    dataset_meta={
                        "source_dir": source_dir,
                        "jd_path": jd_path,
                        "cv_path": cv_path,
                        "mock_cv_path": os.path.join(target_dir, "mockcv.csv"),
                    },
                ),
            )

            jd_ids = []
            for j, (_, jd_row) in enumerate(sample_jds.iterrows()):
                jd_id = upsert_jd(conn, run_id, external_key=str(j), payload=row_payload(jd_row), text_content=get_jd_text(jd_row))
                jd_ids.append(jd_id)

            cv_ids = []
            for c, (_, cv_row) in enumerate(df_cv.iterrows()):
                cv_id = upsert_cv(conn, run_id, external_key=str(c), payload=row_payload(cv_row), text_content=get_cv_text(cv_row))
                cv_ids.append(cv_id)

            cv_to_jd_scores = similarities.T
            for cv_idx, cv_scores in enumerate(tqdm(cv_to_jd_scores, total=len(cv_to_jd_scores), desc="Writing CV matches")):
                if cv_has_matches(conn, run_id, cv_ids[cv_idx], expected_top_k=top_k):
                    continue

                top_jd_idx = np.argsort(cv_scores)[::-1][:top_k]
                matches = []
                for rank, jd_idx in enumerate(top_jd_idx, start=1):
                    jd_idx = int(jd_idx)
                    matches.append(
                        {
                            "jd_id": jd_ids[jd_idx],
                            "rank": rank,
                            "score": float(cv_scores[jd_idx]),
                            "meta": {"jd_idx": jd_idx, "cv_idx": cv_idx, "source_view": "dotpdtesting1.0_jd_to_cv"},
                        }
                    )
                insert_matches(conn, run_id, cv_ids[cv_idx], matches)

            finish_run(conn, run_id, started)
        finally:
            conn.close()

    if not export_html:
        return

    html_out = os.path.join(target_dir, "results.html")
    render_jd_to_cv_report(html_out, sample_jds, df_cv, similarities, title_col)

if __name__ == "__main__":
    main()
