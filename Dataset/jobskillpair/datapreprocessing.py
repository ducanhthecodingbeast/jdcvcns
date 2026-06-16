import argparse
import ast
import json
import re
import zipfile
from collections import defaultdict
from pathlib import Path
from urllib.parse import unquote, urlparse

import pandas as pd

try:
    import spacy
except ImportError:
    spacy = None

try:
    from tqdm import tqdm
except Exception:
    def tqdm(iterable=None, **kwargs):
        return iterable

MAX_LOCAL_DOWNLOAD_BYTES = 100 * 1024 * 1024
DATASET_NAME = "asaniczka/1-3m-linkedin-jobs-and-skills-2024"
DONE_VERSION = "jobskillpair-preprocess-v2"
DEFAULT_CHUNK_SIZE = 100_000
NLP = None

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
TRAILING_TITLE_WORDS = TRAILING_QUALIFIERS | SENIORITY_WORDS

LEVEL_SUFFIXES = re.compile(r"\b(i|ii|iii|iv|senior|junior|sr\.?|jr\.?)\s*$", re.IGNORECASE)
EXPERIENCE_PREFIXES = [
    "entry-level",
    "entry level",
    "experienced",
    "assistant",
    "traveling",
    "temporary",
    "part time",
    "full time",
    "seasonal",
    "deputy",
    "senior",
    "junior",
    "travel",
    "lead",
    "temp",
]
SCHEDULE_WORDS = ["part time", "full time", "part-time", "full-time", "per diem"]
EXPERIENCE_PREFIX_PATTERNS = [
    re.compile(r"^" + re.escape(prefix) + r"\s+", re.IGNORECASE)
    for prefix in EXPERIENCE_PREFIXES
]
SCHEDULE_PATTERNS = [
    re.compile(r"\b" + re.escape(schedule) + r"\b", re.IGNORECASE)
    for schedule in SCHEDULE_WORDS
]
LOCATION_PATTERN = re.compile(r"\b(?:salem va|brea ca|rochester|gta)\b", re.IGNORECASE)
BRANDS = ["calvin klein", "apple", "amazon", "google"]
BRAND_PATTERNS = [re.compile(r"\b" + re.escape(brand) + r"\b", re.IGNORECASE) for brand in BRANDS]


def get_nlp():
    global NLP
    if NLP is not None or spacy is None:
        return NLP

    try:
        NLP = spacy.load("en_core_web_sm")
    except OSError:
        NLP = None
    return NLP


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
    while words and words[-1] in TRAILING_TITLE_WORDS:
        words.pop()

    return " ".join(words).strip()


def clean_skill(value) -> str:
    value = re.sub(r"[\x00-\x1f\x7f]+", " ", str(value))
    return re.sub(r"\s+", " ", value).strip(" '\"\t\r\n")


def clean_job_title(title: str) -> str:
    title = str(title).lower().strip()
    title = LEVEL_SUFFIXES.sub("", title).strip()

    for pattern in EXPERIENCE_PREFIX_PATTERNS:
        title = pattern.sub("", title)

    for pattern in SCHEDULE_PATTERNS:
        title = pattern.sub(" ", title)

    title = LOCATION_PATTERN.sub(" ", title)

    for pattern in BRAND_PATTERNS:
        title = pattern.sub(" ", title)

    title = re.sub(r"[^a-z0-9\s\-/+#]", " ", title)
    title = re.sub(r"\s+", " ", title).strip()
    title = remove_location_entities(title)

    return re.sub(r"\s+", " ", title).strip()


def remove_location_entities(title: str) -> str:
    nlp = get_nlp()
    if not nlp:
        return LOCATION_PATTERN.sub(" ", title).strip()

    doc = nlp(title)
    kept_tokens = []
    skip_indexes = set()

    for entity in doc.ents:
        if entity.label_ not in {"GPE", "LOC", "FAC"}:
            continue

        start = entity.start
        if start > 0 and doc[start - 1].lower_ in {"in", "at", "near"}:
            start -= 1

        skip_indexes.update(range(start, entity.end))

    for token in doc:
        if token.i in skip_indexes:
            continue
        if token.like_num and not token.text.isalpha():
            continue
        kept_tokens.append(token.text)

    return " ".join(kept_tokens).strip()


def split_skills(value) -> list[str]:
    if isinstance(value, (list, tuple, set)):
        return [clean_skill(skill) for skill in value if clean_skill(skill)]

    if value is None or pd.isna(value):
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


def done_path(output_csv: Path) -> Path:
    return output_csv.with_suffix(output_csv.suffix + ".done")


def read_csv_with_fallback(path: Path, **kwargs) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8", **kwargs)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin1", **kwargs)


def iter_csv_chunks(path: Path, chunksize: int):
    try:
        yield from pd.read_csv(path, encoding="utf-8", chunksize=chunksize)
    except UnicodeDecodeError:
        yield from pd.read_csv(path, encoding="latin1", chunksize=chunksize)


def csv_has_columns(path: Path, columns: set[str]) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False

    try:
        header = set(read_csv_with_fallback(path, nrows=0).columns)
    except Exception:
        return False
    return columns <= header


def output_is_combined(path: Path) -> bool:
    if not csv_has_columns(path, {"id", "job title", "skill"}):
        return False

    metadata_path = done_path(path)
    if not metadata_path.exists():
        return False

    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return metadata.get("version") == DONE_VERSION


def write_done(output_csv: Path) -> None:
    done_path(output_csv).write_text(json.dumps({"version": DONE_VERSION}) + "\n", encoding="utf-8")


def source_csv_is_usable(path: Path, url_col: str | None = None, skill_col: str | None = None) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False

    try:
        sample = read_csv_with_fallback(path, nrows=100)
    except Exception:
        return False

    try:
        find_url_column(sample, url_col)
        find_skills_column(sample, find_url_column(sample, url_col), skill_col)
    except ValueError:
        return False
    return True


def find_url_column(df: pd.DataFrame, url_col: str | None = None) -> str:
    if url_col:
        if url_col in df.columns:
            return url_col
        raise ValueError(f"URL column '{url_col}' not found. Available columns: {list(df.columns)}")

    for column in df.columns:
        if df[column].astype(str).str.contains("https", na=False).any():
            return column
    raise ValueError(f"Could not find a column containing https URLs. Available columns: {list(df.columns)}")


def find_skills_column(df: pd.DataFrame, url_column: str, skill_col: str | None = None) -> str:
    if skill_col:
        if skill_col in df.columns and skill_col != url_column:
            return skill_col
        raise ValueError(f"Skill column '{skill_col}' not found or matches URL column. Available columns: {list(df.columns)}")

    candidates = [column for column in df.columns if "skill" in column.lower() and column != url_column]
    if not candidates:
        raise ValueError(f"Could not find a skill column. Available columns: {list(df.columns)}")
    return candidates[0]


def add_pair(aggregated: dict[str, list[str]], seen: set[tuple[str, str]], title: str, skill: str) -> None:
    key = (title, skill)
    if title and skill and key not in seen:
        aggregated[title].append(skill)
        seen.add(key)


def combine_existing_output(output_csv: Path) -> None:
    aggregated: dict[str, list[str]] = defaultdict(list)
    seen: set[tuple[str, str]] = set()

    for chunk in tqdm(iter_csv_chunks(output_csv, DEFAULT_CHUNK_SIZE), desc="chunks", unit="chunk"):
        for row in tqdm(chunk[["job title", "skill"]].itertuples(index=False), total=len(chunk), desc="rows", unit="row"):
            title = clean_job_title(row[0])
            for skill in split_skills(row[1]):
                add_pair(aggregated, seen, title, skill)

    write_output(aggregated, output_csv)


def write_output(aggregated: dict[str, list[str]], output_csv: Path) -> None:
    rows = [
        {"job title": title, "skill": ", ".join(aggregated[title])}
        for title in sorted(aggregated)
    ]
    result = pd.DataFrame(rows)

    if result.empty:
        raise RuntimeError("Preprocessing produced 0 rows.")

    result.insert(0, "id", range(1, len(result) + 1))
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    result[["id", "job title", "skill"]].to_csv(output_csv, index=False)
    write_done(output_csv)


def build_combined_jobskill_csv(
    source_csv: Path,
    output_csv: Path,
    url_col: str | None = None,
    skill_col: str | None = None,
    chunksize: int = DEFAULT_CHUNK_SIZE,
) -> None:
    aggregated: dict[str, list[str]] = defaultdict(list)
    seen: set[tuple[str, str]] = set()

    for chunk in tqdm(iter_csv_chunks(source_csv, chunksize), desc="chunks", unit="chunk"):
        url_column = find_url_column(chunk, url_col)
        skills_column = find_skills_column(chunk, url_column, skill_col)

        for row in tqdm(chunk[[url_column, skills_column]].itertuples(index=False), total=len(chunk), desc="rows", unit="row"):
            title = clean_job_title(extract_job_title_from_link(row[0]))
            for skill in split_skills(row[1]):
                add_pair(aggregated, seen, title, skill)

    write_output(aggregated, output_csv)


def find_existing_source_csv(raw_dir: Path, url_col: str | None = None, skill_col: str | None = None) -> Path | None:
    for path in sorted(raw_dir.glob("*.csv")):
        if source_csv_is_usable(path, url_col, skill_col):
            return path
    return None


def extract_zip_if_needed(raw_dir: Path, url_col: str | None = None, skill_col: str | None = None) -> Path | None:
    source_csv = find_existing_source_csv(raw_dir, url_col, skill_col)
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

    source_csv = find_existing_source_csv(raw_dir, url_col, skill_col)
    if not source_csv:
        files = sorted(path.name for path in raw_dir.iterdir())
        raise FileNotFoundError(f"Extracted files did not include a usable source CSV in {raw_dir}. Files: {files}")
    return source_csv


def parse_args() -> argparse.Namespace:
    base_dir = Path(__file__).resolve().parents[2] / "Dataset" / "Data"
    parser = argparse.ArgumentParser(description="Checkpointed job-skill preprocessing.")
    parser.add_argument("--input", type=Path, help="Existing source CSV. No download is performed.")
    parser.add_argument("--url-col", help="Explicit source URL column name.")
    parser.add_argument("--skill-col", help="Explicit source skill column name.")
    parser.add_argument("--raw-dir", type=Path, default=base_dir / "jobskillpair_raw")
    parser.add_argument("--output", type=Path, default=base_dir / "jobskillpair.csv")
    parser.add_argument("--chunksize", type=int, default=DEFAULT_CHUNK_SIZE)
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
        source_csv = extract_zip_if_needed(args.raw_dir, args.url_col, args.skill_col)

    if not source_csv:
        raise FileNotFoundError(
            "No usable source CSV found. Provide --input or place an extracted CSV in the raw directory. "
            f"This script will not download {DATASET_NAME} locally."
        )

    print(f"RUN preprocess: {source_csv} -> {args.output}")
    build_combined_jobskill_csv(source_csv, args.output, args.url_col, args.skill_col, args.chunksize)
    print(f"DONE preprocess: wrote {args.output}")


if __name__ == "__main__":
    main()
