import argparse
import ast
import re
import subprocess
import zipfile
from pathlib import Path
from urllib.parse import unquote, urlparse

import pandas as pd


SENIORITY_WORDS = {
    "apprentice",
    "associate",
    "intern",
    "internship",
    "jr",
    "junior",
    "lead",
    "principal",
    "senior",
    "sr",
    "staff",
}

TRAILING_QUALIFIERS = {
    "contract",
    "contractor",
    "ft",
    "hybrid",
    "onsite",
    "part",
    "pt",
    "remote",
    "temp",
    "temporary",
    "time",
}


def extract_job_title_from_link(link: str) -> str:
    parsed = urlparse(str(link))
    path = unquote(parsed.path)
    marker = "/jobs/view/"

    if marker in path:
        slug = path.split(marker, 1)[1].strip("/")
    else:
        parts = [part for part in path.split("/") if part]
        slug = parts[-1] if parts else ""

    slug = slug.split("?", 1)[0].split("#", 1)[0]
    slug = re.sub(r"-\d+$", "", slug)
    slug = re.split(r"-at-", slug, maxsplit=1)[0]
    slug = re.sub(r"-[a-z]+-\d+$", "", slug)

    words = [word for word in re.split(r"[-_\s]+", slug.lower()) if word and not word.isdigit()]
    while words and words[0] in SENIORITY_WORDS:
        words.pop(0)
    while words and words[-1] in TRAILING_QUALIFIERS | SENIORITY_WORDS:
        words.pop()

    return " ".join(words).strip()


def split_skills(value) -> list[str]:
    if isinstance(value, (list, tuple, set)):
        return [clean_skill(skill) for skill in value if clean_skill(skill)]

    if pd.isna(value):
        return []

    text = str(value).strip()
    if not text:
        return []

    if text[0] in "[(" and text[-1] in "])":
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, (list, tuple, set)):
                return [clean_skill(skill) for skill in parsed if clean_skill(skill)]
        except (SyntaxError, ValueError):
            pass

    return [clean_skill(skill) for skill in re.split(r"[,;|\n]+", text) if clean_skill(skill)]


def clean_skill(value) -> str:
    return re.sub(r"\s+", " ", str(value)).strip(" '\"\t\r\n")


def find_url_column(df: pd.DataFrame) -> str:
    for column in df.columns:
        if df[column].astype(str).str.contains("https", na=False).any():
            return column
    raise ValueError("Could not find a column containing https URLs.")


def find_skills_column(df: pd.DataFrame, url_column: str) -> str:
    candidates = [column for column in df.columns if "skill" in column.lower() and column != url_column]
    if not candidates:
        raise ValueError("Could not find a skill column.")
    return candidates[0]


def prepare_jobskill_pairs(df: pd.DataFrame) -> pd.DataFrame:
    url_column = find_url_column(df)
    skills_column = find_skills_column(df, url_column)

    result = df[[url_column, skills_column]].rename(
        columns={url_column: "job_link", skills_column: "skill"}
    )
    result = result.dropna(subset=["job_link"])
    result["job title"] = result["job_link"].map(extract_job_title_from_link)
    result["skill"] = result["skill"].map(split_skills)
    result = result.explode("skill")
    result["skill"] = result["skill"].map(clean_skill)
    result = result[(result["job title"] != "") & (result["skill"] != "")]
    result = result.drop_duplicates(subset=["job title", "skill"]).reset_index(drop=True)
    result.insert(0, "id", range(1, len(result) + 1))
    return result[["id", "job title", "skill"]]


def download_dataset(raw_dir: Path) -> None:
    dataset_name = "asaniczka/1-3m-linkedin-jobs-and-skills-2024"
    raw_dir.mkdir(parents=True, exist_ok=True)

    if not list(raw_dir.glob("*.csv")):
        print("Downloading dataset...")
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", dataset_name, "-p", str(raw_dir)],
            check=True,
        )

    print("Extracting dataset...")
    for zip_path in raw_dir.glob("*.zip"):
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(raw_dir)
        zip_path.unlink()


def source_csv_from(raw_dir: Path) -> Path:
    csv_files = sorted(raw_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {raw_dir}")
    return csv_files[0]


def parse_args() -> argparse.Namespace:
    base_dir = Path(__file__).resolve().parents[2] / "Dataset" / "Data"
    parser = argparse.ArgumentParser(description="Create id/job title/skill CSV from LinkedIn jobs data.")
    parser.add_argument("--input", type=Path, help="Optional local source CSV.")
    parser.add_argument("--output", type=Path, default=base_dir / "jobskillpair.csv")
    parser.add_argument("--raw-dir", type=Path, default=base_dir / "jobskillpair_raw")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.input:
        source_csv = args.input
    else:
        download_dataset(args.raw_dir)
        source_csv = source_csv_from(args.raw_dir)

    print(f"Loading data from {source_csv}...")
    df = pd.read_csv(source_csv)

    output_df = prepare_jobskill_pairs(df)
    if output_df.empty:
        raise RuntimeError("Exported 0 rows. Could not extract any job title and skill pairs.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(args.output, index=False)
    print(f"Exported {len(output_df)} rows to {args.output}")


if __name__ == "__main__":
    main()
