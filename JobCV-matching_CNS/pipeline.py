from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_ROOT / "Dataset"
RESULTS_DIR = PROJECT_ROOT / "TestingResults"


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
