import os
os.environ["HF_HOME"] = os.path.join(os.path.dirname(__file__), ".cache")
os.environ["SENTENCE_TRANSFORMERS_HOME"] = os.path.join(os.path.dirname(__file__), ".cache")

import pandas as pd
import psycopg2
from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm import tqdm

def get_db_connection():
    db_url = os.environ.get("DATABASE_URL", "postgresql://postgres.hihyvqdpwcvuadunpjfj:_Epitwh%3FWbb6qvz@aws-1-ap-southeast-1.pooler.supabase.com:5432/postgres")
    return psycopg2.connect(db_url)

def setup_database():
    conn = get_db_connection()
    cur = conn.cursor()
    
    # Enable pgvector extension
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    
    # Create jobs table with Vietnamese column names
    cur.execute("""
        CREATE TABLE IF NOT EXISTS jobs (
            id SERIAL PRIMARY KEY,
            vi_tri_can_tuyen TEXT,
            ten_cong_ty TEXT,
            gioi_thieu_cong_ty TEXT,
            quy_mo_cong_ty TEXT,
            dia_chi_cong_ty TEXT,
            mo_ta_cong_viec TEXT,
            yeu_cau_cong_viec TEXT,
            quyen_loi TEXT,
            full_text TEXT,
            embedding vector(1024)
        );
    """)
    
    # Create cvs table with Vietnamese column names
    cur.execute("""
        CREATE TABLE IF NOT EXISTS cvs (
            id SERIAL PRIMARY KEY,
            ten_ung_vien TEXT,
            vi_tri_ung_tuyen TEXT,
            linh_vuc TEXT,
            noi_lam_viec TEXT,
            muc_luong TEXT,
            gioi_tinh TEXT,
            tinh_trang_hon_nhan TEXT,
            tuoi TEXT,
            muc_tieu_nghe_nghiep TEXT,
            ky_nang TEXT,
            kinh_nghiem TEXT,
            bang_cap TEXT,
            full_text TEXT,
            embedding vector(1024)
        );
    """)
    
    conn.commit()
    cur.close()
    conn.close()

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

def process_and_store():
    source_dir = "Dataset"
    mock_dir = "TestingResults"
    jd_path = os.path.join(source_dir, "jd.csv")
    cv_path = os.path.join(mock_dir, "mockcv.csv")

    if not os.path.exists(jd_path):
        raise FileNotFoundError(f"Missing {jd_path}")
    if not os.path.exists(cv_path):
        raise FileNotFoundError(f"Missing {cv_path} — run dotpdtesting.py first to generate mock CVs")

    df_jd = pd.read_csv(jd_path)
    df_cv = pd.read_csv(cv_path)

    # Strip column whitespace for both DataFrames
    df_jd.columns = df_jd.columns.str.strip()
    df_cv.columns = df_cv.columns.str.strip()

    # Sample for demonstration (remove in production)
    df_jd = df_jd.sample(n=min(50, len(df_jd)), random_state=42).reset_index(drop=True)
    df_cv = df_cv.sample(n=min(200, len(df_cv)), random_state=42).reset_index(drop=True)

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        model = SentenceTransformer("AITeamVN/Vietnamese_Embedding_v2", device=device)
    except Exception as e:
        if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
            print("⚠️ GPU CUDA is out of memory. Falling back to CPU for embedding...")
            model = SentenceTransformer("AITeamVN/Vietnamese_Embedding_v2", device="cpu")
        else:
            raise

    # Batch encode all texts at once for performance
    print("Encoding JD texts...")
    jd_texts = [get_jd_text(row) for _, row in df_jd.iterrows()]
    jd_embeddings = model.encode(jd_texts, show_progress_bar=True)

    print("Encoding CV texts...")
    cv_texts = [get_cv_text(row) for _, row in df_cv.iterrows()]
    cv_embeddings = model.encode(cv_texts, show_progress_bar=True)

    conn = get_db_connection()
    cur = conn.cursor()

    # Register pgvector type adapter so psycopg2 can serialize vector data
    from pgvector.psycopg2 import register_vector
    register_vector(conn)

    try:
        # Process JDs
        for idx in tqdm(range(len(df_jd)), desc="Inserting JDs into DB"):
            row = df_jd.iloc[idx]
            cur.execute("""
                INSERT INTO jobs (vi_tri_can_tuyen, ten_cong_ty, gioi_thieu_cong_ty, quy_mo_cong_ty, 
                                  dia_chi_cong_ty, mo_ta_cong_viec, yeu_cau_cong_viec, quyen_loi, 
                                  full_text, embedding)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                str(row.get('Vị trí cần tuyển', '')),
                str(row.get('Tên công ty', '')),
                str(row.get('Giới thiệu công ty', '')),
                str(row.get('Quy mô công ty', '')),
                str(row.get('Địa chỉ công ty', '')),
                str(row.get('Mô tả công việc', '')),
                str(row.get('Yêu cầu công việc', '')),
                str(row.get('Quyền lợi', '')),
                jd_texts[idx],
                jd_embeddings[idx].tolist()
            ))
            
        # Process CVs
        for idx in tqdm(range(len(df_cv)), desc="Inserting CVs into DB"):
            row = df_cv.iloc[idx]
            cur.execute("""
                INSERT INTO cvs (ten_ung_vien, vi_tri_ung_tuyen, linh_vuc, noi_lam_viec, muc_luong, 
                                 gioi_tinh, tinh_trang_hon_nhan, tuoi, muc_tieu_nghe_nghiep, 
                                 ky_nang, kinh_nghiem, bang_cap, full_text, embedding)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                str(row.get('Tên ứng viên', '')),
                str(row.get('Vị trí ứng tuyển', '')),
                str(row.get('Lĩnh vực', '')),
                str(row.get('Nơi làm việc mong muốn', '')),
                str(row.get('Mức lương mong muốn', '')),
                str(row.get('Giới tính', '')),
                str(row.get('Tình trạng hôn nhân', '')),
                str(row.get('Tuổi', '')),
                str(row.get('Mục tiêu nghề nghiệp', '')),
                str(row.get('Kỹ năng', '')),
                str(row.get('Kinh nghiệm', '')),
                str(row.get('Bằng cấp', '')),
                cv_texts[idx],
                cv_embeddings[idx].tolist()
            ))
            
        conn.commit()
        print(f"Successfully inserted {len(df_jd)} JDs and {len(df_cv)} CVs into database.")
    except Exception as e:
        conn.rollback()
        print(f"[DB Error] Transaction rolled back: {e}")
        raise
    finally:
        cur.close()
        conn.close()

if __name__ == "__main__":
    setup_database()
    process_and_store()
