import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_ROOT.parent
for path in (PROJECT_ROOT, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import pandas as pd
import numpy as np
from Dataset.mockcv import generate_and_mix_cvs
from pipeline import DATASET_DIR, RESULTS_DIR, get_cv_text, get_jd_text, load_sentence_transformer, read_csv_stripped
from testingresult import (
    RunInfo,
    env_flag,
    get_or_create_run,
    open_store,
    store_match_run,
)

def main():
    started = time.monotonic()
    source_dir = DATASET_DIR
    mock_target_dir = DATASET_DIR
    os.makedirs(RESULTS_DIR, exist_ok=True)

    store_db = env_flag("STORE_DB", True)
    top_k = int(os.environ.get("TOP_K", "10"))
    
    jd_path = source_dir / "jd.csv"
    cv_path = source_dir / "cv.csv"

    if not os.path.exists(jd_path) or not os.path.exists(cv_path):
        raise FileNotFoundError(f"Missing {jd_path} or {cv_path}")

    df_jd = read_csv_stripped(jd_path)
    raw_cv = read_csv_stripped(cv_path)
    
    df_cv = generate_and_mix_cvs(df_jd, raw_cv, mock_target_dir)
    df_cv.columns = df_cv.columns.str.strip()

    title_col = 'Vị trí cần tuyển'


    top_titles = df_jd[title_col].value_counts().head(30).index
    
    sample_jds = pd.concat([
        df_jd[df_jd[title_col] == title].sample(n=2, replace=True) 
        for title in top_titles
    ]).reset_index(drop=True)


    model = load_sentence_transformer("AITeamVN/Vietnamese_Embedding_v2")

    jd_texts = [get_jd_text(row) for _, row in sample_jds.iterrows()]
    cv_texts = [get_cv_text(row) for _, row in df_cv.iterrows()]


    jd_embeddings = model.encode(jd_texts, show_progress_bar=True)
    

    cv_embeddings = model.encode(cv_texts, show_progress_bar=True)


    similarities = jd_embeddings @ cv_embeddings.T

    if store_db:
        conn = open_store()
        try:
            cv_to_jd_scores = similarities.T
            def ranked_matches(cv_idx):
                cv_scores = cv_to_jd_scores[cv_idx]
                top_jd_idx = np.argsort(cv_scores)[::-1][:top_k]
                return [
                    {
                        "jd_idx": int(jd_idx),
                        "score": float(cv_scores[int(jd_idx)]),
                        "meta": {"cv_idx": cv_idx, "source_view": "dotpdtesting1.0_jd_to_cv"},
                    }
                    for jd_idx in top_jd_idx
                ]

            store_match_run(
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
                        "source_dir": str(source_dir),
                        "jd_path": str(jd_path),
                        "cv_path": str(cv_path),
                        "mock_cv_path": str(mock_target_dir / "mockcv.csv"),
                    },
                ),
                df_cv,
                sample_jds,
                get_cv_text,
                get_jd_text,
                ranked_matches,
                top_k=top_k,
                started_monotonic=started,
            )
        finally:
            conn.close()

if __name__ == "__main__":
    main()
