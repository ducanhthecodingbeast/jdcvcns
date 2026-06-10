from __future__ import annotations

import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from tqdm import tqdm

from pipeline import normalize_cv, normalize_jd


KAGGLE_DATASET = "phamtheds/job-dataset-for-recommendation"
KAGGLE_DATASET_URL = f"https://www.kaggle.com/datasets/{KAGGLE_DATASET}"
HF_RESUME_DATASET = "lhoestq/resumes-raw-pdf-for-ocr"


def configure_cache() -> None:
    os.environ.setdefault("HF_HOME", ".cache/huggingface")
    os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", ".cache/sentence-transformers")
    Path(os.environ["HF_HOME"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["SENTENCE_TRANSFORMERS_HOME"]).mkdir(parents=True, exist_ok=True)


def load_resume_dataset(split: str | None = None):
    configure_cache()
    from datasets import load_dataset

    return load_dataset(HF_RESUME_DATASET, split=split)


def extract_pdf_with_docling(pdf_path: str | Path) -> str:
    from docling.document_converter import DocumentConverter

    result = DocumentConverter().convert(str(pdf_path))
    return result.document.export_to_csv()


def setup_kaggle(non_interactive: bool = True) -> bool:
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if kaggle_json.exists():
        return True

    username = os.environ.get("KAGGLE_USERNAME", "").strip()
    key = os.environ.get("KAGGLE_KEY", "").strip()
    if not username or not key:
        if non_interactive:
            raise RuntimeError("KAGGLE_USERNAME and KAGGLE_KEY are required.")
        username = input("Kaggle Username: ").strip()
        key = input("Kaggle API Key: ").strip()

    if not username or not key:
        return False

    kaggle_json.parent.mkdir(parents=True, exist_ok=True)
    kaggle_json.write_text(json.dumps({"username": username, "key": key}), encoding="utf-8")
    if os.name == "posix":
        kaggle_json.chmod(0o600)
    return True


def download_and_extract(target_dir: str | Path = "Dataset") -> None:
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    with tqdm(total=5, desc="Data Pipeline") as pbar:
        pbar.set_description("Checking Kaggle CLI")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "kaggle"], check=True)
        pbar.update(1)

        pbar.set_description("Downloading Kaggle dataset")
        subprocess.run(["kaggle", "datasets", "download", "-d", KAGGLE_DATASET, "-p", str(target_dir)], check=True)
        pbar.update(1)

        pbar.set_description("Extracting ZIP")
        for file_path in target_dir.glob("*.zip"):
            with zipfile.ZipFile(file_path, "r") as zip_ref:
                zip_ref.extractall(target_dir)
            file_path.unlink()
        pbar.update(1)

        pbar.set_description("Preprocessing CV")
        user_file = target_dir / "USER_DATA_FINAL.csv"
        if user_file.exists():
            cv = normalize_cv(pd.read_csv(user_file))
            if "Vị trí ứng tuyển" in cv.columns:
                cv["Vị trí ứng tuyển"] = cv["Vị trí ứng tuyển"].astype(str).str.lower().str.strip()
            cv.to_csv(target_dir / "cv.csv", index=False)
        pbar.update(1)

        pbar.set_description("Preprocessing JD")
        job_file = target_dir / "JOB_DATA_FINAL.csv"
        if job_file.exists():
            jd = normalize_jd(pd.read_csv(job_file))
            if "Vị trí cần tuyển" in jd.columns:
                jd["Vị trí cần tuyển"] = jd["Vị trí cần tuyển"].astype(str).str.lower().str.strip()
            jd.to_csv(target_dir / "jd.csv", index=False)
        pbar.update(1)

    for name in ("cv.csv", "jd.csv"):
        output = target_dir / name
        if output.exists():
            print(output)
            print(pd.read_csv(output).head(20))


def main() -> None:
    configure_cache()
    print(f"Kaggle dataset: {KAGGLE_DATASET_URL}")
    if setup_kaggle(non_interactive=True):
        download_and_extract()


if __name__ == "__main__":
    main()
