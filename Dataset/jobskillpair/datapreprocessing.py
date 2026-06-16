import argparse
import ast
import re
import zipfile
from pathlib import Path
from urllib.parse import unquote, urlparse

import pandas as pd


MAX_LOCAL_DOWNLOAD_BYTES = 100 * 1024 * 1024
DATASET_NAME = "asaniczka/1-3m-linkedin-jobs-and-skills-2024"

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


def clean_skill(value) -> str:
    return re.sub(r"\s+", " ", str(value)).strip(" '\"\t\r\n")


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


def csv_has_columns(path: Path, columns: set[str]) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False

    try:
        header = set(pd.read_csv(path, nrows=0).columns)
    except Exception:
        return False
    return columns <= header


def output_is_combined(path: Path) -> bool:
    if not csv_has_columns(path, {"id", "job title", "skill"}):
        return False

    df = pd.read_csv(path, usecols=["job title", "skill"])
    if df.empty:
        return False

    df["job title"] = df["job title"].astype(str).str.strip()
    df["skill"] = df["skill"].astype(str).str.strip()
    df = df[(df["job title"] != "") & (df["skill"] != "")]
    return not df.empty and not df["job title"].duplicated().any()


def source_csv_is_usable(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False

    try:
        sample = pd.read_csv(path, nrows=100)
    except Exception:
        return False

    has_url = any(sample[column].astype(str).str.contains("https", na=False).any() for column in sample.columns)
    has_skill = any("skill" in column.lower() for column in sample.columns)
    return has_url and has_skill


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


def combine_existing_output(output_csv: Path) -> None:
    df = pd.read_csv(output_csv)
    df["job title"] = df["job title"].astype(str).str.strip()
    df["skill"] = df["skill"].map(split_skills)
    df = df.explode("skill")
    df["skill"] = df["skill"].map(clean_skill)
    df = df[(df["job title"] != "") & (df["skill"] != "")]
    df = df.drop_duplicates(subset=["job title", "skill"])
    df = (
        df.groupby("job title", as_index=False, sort=False)["skill"]
        .agg(lambda skills: ", ".join(skills))
        .reset_index(drop=True)
    )
    df.insert(0, "id", range(1, len(df) + 1))
    df[["id", "job title", "skill"]].to_csv(output_csv, index=False)


def build_combined_jobskill_csv(source_csv: Path, output_csv: Path) -> None:
    df = pd.read_csv(source_csv)
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
    result = result.drop_duplicates(subset=["job title", "skill"])
    result = (
        result.groupby("job title", as_index=False, sort=True)["skill"]
        .agg(lambda skills: ", ".join(skills))
        .reset_index(drop=True)
    )
    result.insert(0, "id", range(1, len(result) + 1))

    if result.empty:
        raise RuntimeError("Preprocessing produced 0 rows.")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    result[["id", "job title", "skill"]].to_csv(output_csv, index=False)


def find_existing_source_csv(raw_dir: Path) -> Path | None:
    for path in sorted(raw_dir.glob("*.csv")):
        if source_csv_is_usable(path):
            return path
    return None


def extract_zip_if_needed(raw_dir: Path) -> Path | None:
    source_csv = find_existing_source_csv(raw_dir)
    if source_csv:
        print(f"PASS extract: found source CSV {source_csv}")
        return source_csv

    zip_files = sorted(raw_dir.glob("*.zip"))
    if not zip_files:
        print("SKIP extract: no zip file found")
        return None

    for zip_path in zip_files:
        if zip_path.stat().st_size > MAX_LOCAL_DOWNLOAD_BYTES:
            raise RuntimeError(
                f"{zip_path} is larger than 100MB. Move preprocessing to the server or ask permission first."
            )

        print(f"RUN extract: {zip_path}")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(raw_dir)

    source_csv = find_existing_source_csv(raw_dir)
    if not source_csv:
        raise FileNotFoundError(f"Extracted files did not include a usable source CSV in {raw_dir}")
    return source_csv


def parse_args() -> argparse.Namespace:
    base_dir = Path(__file__).resolve().parents[2] / "Dataset" / "Data"
    parser = argparse.ArgumentParser(description="Checkpointed job-skill preprocessing.")
    parser.add_argument("--input", type=Path, help="Existing source CSV. No download is performed.")
    parser.add_argument("--raw-dir", type=Path, default=base_dir / "jobskillpair_raw")
    parser.add_argument("--output", type=Path, default=base_dir / "jobskillpair.csv")
    parser.add_argument("--force", action="store_true", help="Rebuild output even if it already looks valid.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.raw_dir.mkdir(parents=True, exist_ok=True)

    if not args.force and output_is_combined(args.output):
        print(f"PASS preprocess: output already exists at {args.output}")
        return

    if not args.force and csv_has_columns(args.output, {"id", "job title", "skill"}):
        print(f"RUN combine: grouping duplicate job titles in {args.output}")
        combine_existing_output(args.output)
        print(f"DONE combine: wrote {args.output}")
        return

    if args.input:
        source_csv = args.input
        print(f"PASS source: using input {source_csv}")
    else:
        source_csv = extract_zip_if_needed(args.raw_dir)

    if not source_csv:
        raise FileNotFoundError(
            "No usable source CSV found. Provide --input or place an extracted CSV in the raw directory. "
            f"This script will not download {DATASET_NAME} locally."
        )

    if not source_csv_is_usable(source_csv):
        raise ValueError(f"{source_csv} does not contain both an https URL column and a skill column.")

    print(f"RUN preprocess: {source_csv} -> {args.output}")
    build_combined_jobskill_csv(source_csv, args.output)
    print(f"DONE preprocess: wrote {args.output}")


if __name__ == "__main__":
    main()
