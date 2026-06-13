from __future__ import annotations

import os
import re
import unicodedata
from pathlib import Path
from typing import Iterable

import pandas as pd

try:
    import transformers
    if not hasattr(transformers.PreTrainedTokenizerBase, "prepare_for_model"):
        def monkey_patch_prepare_for_model(self, ids, pair_ids=None, add_special_tokens=True, padding=False, truncation=False, max_length=None, **kwargs):
            if truncation == 'only_second' and max_length is not None and pair_ids is not None:
                num_special = self.num_special_tokens_to_add(pair=True)
                max_pair_len = max_length - len(ids) - num_special
                pair_ids = pair_ids[:max_pair_len] if max_pair_len > 0 else []
            input_ids = self.build_inputs_with_special_tokens(ids, pair_ids) if add_special_tokens else (ids + (pair_ids if pair_ids else []))
            res = {"input_ids": input_ids, "attention_mask": [1] * len(input_ids)}
            if "token_type_ids" in self.model_input_names:
                try:
                    res["token_type_ids"] = self.create_token_type_ids_from_sequences(ids, pair_ids) if add_special_tokens else ([0]*len(ids) + [1]*len(pair_ids if pair_ids else []))
                except Exception:
                    pass
            return res
        transformers.PreTrainedTokenizerBase.prepare_for_model = monkey_patch_prepare_for_model
except ImportError:
    pass



PROJECT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_ROOT.parent
DATASET_DIR = REPO_ROOT / "Data"
RESULTS_DIR = PROJECT_ROOT / "TestingResults"
DEFAULT_RANDOM_STATE = 42
WHITESPACE_RE = re.compile(r"\s+")


CV_RENAME_MAP = {
    "User Name": "Tên ứng viên",
    "Desired Job": "Vị trí ứng tuyển",
    "Industry": "Lĩnh vực",
    "Workplace Desired": "Nơi làm việc mong muốn",
    "Desired Salary": "Mức lương mong muốn",
    "Gender": "Giới tính",
    "Marriage": "Tình trạng hôn nhân",
    "Age": "Tuổi",
}


JD_RENAME_MAP = {
    "Job Title": "Vị trí cần tuyển",
    "Name Company": "Tên công ty",
    "Company Overview": "Giới thiệu công ty",
    "Company Size": "Quy mô công ty",
    "Company Address": "Địa chỉ công ty",
    "Job Description": "Mô tả công việc",
    "Job Requirements": "Yêu cầu công việc",
    "Benefits": "Quyền lợi",
}


def normalize_cv(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.drop(columns=["URL User", "UserID"], errors="ignore")
    df = df.rename(columns=CV_RENAME_MAP)
    df.columns = df.columns.str.strip()
    return df


def normalize_jd(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.drop(columns=["URL Job", "JobID"], errors="ignore")
    df = df.rename(columns=JD_RENAME_MAP)
    df.columns = df.columns.str.strip()
    return df


def _load_csv(path: Path, normalizer) -> pd.DataFrame:
    return normalizer(pd.read_csv(path))


def is_non_empty_csv(path: str | Path) -> bool:
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        return not pd.read_csv(path, nrows=1).empty
    except (pd.errors.EmptyDataError, OSError, UnicodeDecodeError):
        return False


def first_non_empty_csv(paths: Iterable[str | Path]) -> Path | None:
    return next((Path(path) for path in paths if is_non_empty_csv(path)), None)


def configure_cache() -> None:
    cache_dir = PROJECT_ROOT / ".cache"
    os.environ.setdefault("HF_HOME", str(cache_dir / "huggingface"))
    os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(cache_dir / "sentence-transformers"))


def read_csv_stripped(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df


def sample_dataframe(df: pd.DataFrame, limit: int | None, *, random_state: int = DEFAULT_RANDOM_STATE) -> pd.DataFrame:
    if limit is None:
        return df.reset_index(drop=True)
    return df.sample(n=min(limit, len(df)), random_state=random_state).reset_index(drop=True)


def load_sentence_transformer(model_name: str):
    configure_cache()
    import torch
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        return SentenceTransformer(model_name, device=device)
    except Exception as exc:
        message = str(exc).lower()
        if "out of memory" in message or "cuda" in message:
            return SentenceTransformer(model_name, device="cpu")
        raise


def load_or_encode_embeddings(
    cache_path: str | Path,
    texts: list[str],
    model_name: str,
):
    import numpy as np

    cache_path = Path(cache_path)
    if cache_path.exists():
        try:
            embeddings = np.load(cache_path)
            if len(embeddings) == len(texts):
                return embeddings
        except Exception:
            pass

    model = load_sentence_transformer(model_name)
    embeddings = model.encode(texts, show_progress_bar=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, embeddings)
    return embeddings


def load_datasets(dataset_dir: str | Path = DATASET_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    dataset_dir = Path(dataset_dir)
    if not dataset_dir.is_absolute():
        dataset_dir = PROJECT_ROOT / dataset_dir

    jd_path = dataset_dir / "jd.csv"
    raw_jd_path = dataset_dir / "JOB_DATA_FINAL.csv"
    if jd_path.exists():
        df_jd = _load_csv(jd_path, normalize_jd)
    elif raw_jd_path.exists():
        df_jd = _load_csv(raw_jd_path, normalize_jd)
    else:
        raise FileNotFoundError(f"Missing {jd_path} or {raw_jd_path}")

    cv_candidates = [
        dataset_dir / "mockcv.csv",
        RESULTS_DIR / "mockcv.csv",
        dataset_dir / "cv.csv",
        dataset_dir / "USER_DATA_FINAL.csv",
    ]
    cv_path = first_non_empty_csv(cv_candidates)
    if cv_path is None:
        expected = ", ".join(str(path) for path in cv_candidates)
        raise FileNotFoundError(f"Missing CV dataset. Expected one of: {expected}")

    df_cv = _load_csv(cv_path, normalize_cv)
    return df_cv, df_jd


def _row_text(row: pd.Series, fields: Iterable[tuple[str, str]]) -> str:
    parts: list[str] = []
    for column, label in fields:
        value = row.get(column)
        if pd.notna(value):
            text = clean_text(value)
            if text:
                parts.append(f"{label}: {text}")
    return "\n".join(parts)


def clean_text(value: object) -> str:
    text = unicodedata.normalize("NFC", str(value))
    return WHITESPACE_RE.sub(" ", text).strip()


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
    return _row_text(row, fields)


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
    return _row_text(row, fields)
