from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_ROOT.parent
for path in (PROJECT_ROOT, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import numpy as np
import pandas as pd

from Dataset.mockcv import generate_and_mix_cvs
from testingresult import (
    RunInfo,
    env_flag,
    open_store,
    store_match_run,
)


DATASET_DIR = REPO_ROOT / "Data"+
RESULTS_DIR = PROJECT_ROOT / "TestingResults"
MODEL_NAME = "AITeamVN/Vietnamese_Embedding_v2"
TITLE_COLUMN = "Vị trí cần tuyển"
TOP_TITLE_COUNT = 30
SAMPLES_PER_TITLE = 2


def patch_transformers_compatibility() -> None:
    try:
        import transformers
    except ImportError:
        return

    tokenizer_base = transformers.PreTrainedTokenizerBase
    if not hasattr(tokenizer_base, "prepare_for_model"):

        def prepare_for_model(
            self,
            ids,
            pair_ids=None,
            add_special_tokens=True,
            padding=False,
            truncation=False,
            max_length=None,
            **kwargs,
        ):
            if truncation == "only_second" and max_length is not None and pair_ids is not None:
                special_count = self.num_special_tokens_to_add(pair=True) if hasattr(self, "num_special_tokens_to_add") else 3
                max_pair_len = max_length - len(ids) - special_count
                pair_ids = pair_ids[:max_pair_len] if max_pair_len > 0 else []

            cls = self.cls_token_id if self.cls_token_id is not None else 0
            sep = self.sep_token_id if self.sep_token_id is not None else 2

            if pair_ids is None:
                input_ids = [cls] + ids + [sep] if add_special_tokens else ids
                token_type_ids = [0] * len(input_ids)
            elif type(self).__name__ in {
                "XLMRobertaTokenizer",
                "XLMRobertaTokenizerFast",
                "RobertaTokenizer",
                "RobertaTokenizerFast",
            }:
                input_ids = [cls] + ids + [sep, sep] + pair_ids + [sep] if add_special_tokens else ids + pair_ids
                token_type_ids = [0] * len(input_ids)
            else:
                input_ids = [cls] + ids + [sep] + pair_ids + [sep] if add_special_tokens else ids + pair_ids
                token_type_ids = (
                    [0] * (len(ids) + 2) + [1] * (len(pair_ids) + 1)
                    if add_special_tokens
                    else [0] * len(ids) + [1] * len(pair_ids)
                )

            result = {"input_ids": input_ids, "attention_mask": [1] * len(input_ids)}
            if "token_type_ids" in self.model_input_names:
                result["token_type_ids"] = token_type_ids
            return result

        tokenizer_base.prepare_for_model = prepare_for_model

    for model_class_name in ("AutoModel", "AutoModelForSequenceClassification"):
        if not hasattr(transformers, model_class_name):
            continue

        model_class = getattr(transformers, model_class_name)
        original_from_pretrained = model_class.from_pretrained

        def make_patched_from_pretrained(original):
            @classmethod
            def patched_from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
                if "dtype" in kwargs:
                    if "torch_dtype" not in kwargs:
                        kwargs["torch_dtype"] = kwargs.pop("dtype")
                    else:
                        kwargs.pop("dtype")
                return original.__func__(cls, pretrained_model_name_or_path, *model_args, **kwargs)

            return patched_from_pretrained

        model_class.from_pretrained = make_patched_from_pretrained(original_from_pretrained)


def configure_cache() -> None:
    cache_dir = PROJECT_ROOT / ".cache"
    os.environ.setdefault("HF_HOME", str(cache_dir / "huggingface"))
    os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(cache_dir / "sentence-transformers"))


def read_csv_stripped(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df


def load_sentence_transformer(model_name: str):
    patch_transformers_compatibility()
    configure_cache()

    import torch
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model {model_name} on device: {device.upper()}")
    try:
        return SentenceTransformer(model_name, device=device)
    except Exception as exc:
        message = str(exc).lower()
        if "out of memory" in message or "cuda" in message:
            print(f"WARNING: Failed to load on {device}. Falling back to CPU. Error: {exc}")
            return SentenceTransformer(model_name, device="cpu")
        raise


def row_text(row: pd.Series, fields: Iterable[tuple[str, str]]) -> str:
    parts = []
    for column, label in fields:
        value = row.get(column)
        if pd.notna(value) and str(value).strip():
            parts.append(f"{label}: {value}")
    return "\n".join(parts)


def get_jd_text(row: pd.Series, include_company: bool = True) -> str:
    fields = [
        ("Vị trí cần tuyển", "Vị trí cần tuyển"),
        ("Giới thiệu công ty", "Giới thiệu công ty"),
        ("Quy mô công ty", "Quy mô công ty"),
        ("Địa chỉ công ty", "Địa điểm"),
        ("Mô tả công việc", "Mô tả công việc"),
        ("Yêu cầu công việc", "Yêu cầu"),
        ("Quyền lợi", "Quyền lợi"),
    ]
    if include_company:
        fields.insert(1, ("Tên công ty", "Tên công ty"))
    return row_text(row, fields)


def get_cv_text(row: pd.Series) -> str:
    fields = [
        ("Tên ứng viên", "Tên ứng viên"),
        ("Vị trí ứng tuyển", "Vị trí ứng tuyển"),
        ("Lĩnh vực", "Lĩnh vực"),
        ("Nơi làm việc mong muốn", "Nơi làm việc"),
        ("Mức lương mong muốn", "Mức lương mong muốn"),
        ("Giới tính", "Giới tính"),
        ("Tình trạng hôn nhân", "Tình trạng hôn nhân"),
        ("Tuổi", "Tuổi"),
        ("Mục tiêu nghề nghiệp", "Mục tiêu"),
        ("Kỹ năng", "Kỹ năng"),
        ("Kinh nghiệm", "Kinh nghiệm"),
        ("Bằng cấp", "Bằng cấp"),
    ]
    return row_text(row, fields)


def sample_top_jds(df_jd: pd.DataFrame) -> pd.DataFrame:
    top_titles = df_jd[TITLE_COLUMN].value_counts().head(TOP_TITLE_COUNT).index
    samples = [
        df_jd[df_jd[TITLE_COLUMN] == title].sample(n=SAMPLES_PER_TITLE, replace=True)
        for title in top_titles
    ]
    return pd.concat(samples).reset_index(drop=True)


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

    sample_jds = sample_top_jds(df_jd)
    model = load_sentence_transformer(MODEL_NAME)

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
                    model_name=MODEL_NAME,
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
