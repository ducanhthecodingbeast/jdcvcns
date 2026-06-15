import re
import subprocess
import zipfile
from pathlib import Path
from urllib.parse import unquote, urlparse

import pandas as pd
from tqdm import tqdm


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

def main():
    dataset_name = "asaniczka/1-3m-linkedin-jobs-and-skills-2024"
    
    # Set paths
    base_dir = Path(__file__).resolve().parents[2] / "Dataset" / "Data"
    raw_dir = base_dir / "jobskillpair_raw"
    output_csv = base_dir / "jobskillpair.csv"

    raw_dir.mkdir(parents=True, exist_ok=True)

    # 1. Download and extract
    print("Downloading dataset...")
    subprocess.run(["kaggle", "datasets", "download", "-d", dataset_name, "-p", str(raw_dir)], check=True)

    print("Extracting dataset...")
    for zip_path in raw_dir.glob("*.zip"):
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(raw_dir)
        zip_path.unlink()

    # 2. Load data with Pandas
    source_csv = list(raw_dir.glob("*.csv"))[0]
    print(f"Loading data from {source_csv}...")
    df = pd.read_csv(source_csv)

    # Find URL column by checking for 'https'
    link_col = next((c for c in df.columns if df[c].astype(str).str.contains("https", na=False).sum() > 1), None)
    if not link_col:
        raise ValueError("Could not find a column containing URLs.")
        
    # Find Skills column (keep it simple)
    skills_col = [c for c in df.columns if "skill" in c.lower() and c != link_col][0]

    df = df[[link_col, skills_col]].rename(columns={link_col: "job_link", skills_col: "skills"})
    df = df.dropna(subset=["job_link"]).drop_duplicates(subset=["job_link"]).reset_index(drop=True)

    print("Extracting job titles from LinkedIn URLs...")

    # 4. Format and export continuously
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=["id", "job_title", "skills"]).to_csv(output_csv, index=False)
    
    current_id = 1
    skipped_rows = 0
    for index, row in tqdm(df.iterrows(), total=len(df)):
        link = row["job_link"]
        skills = row["skills"]
        title = extract_job_title_from_link(link)

        if not title:
            skipped_rows += 1
            continue

        temp_df = pd.DataFrame([{"id": current_id, "job_title": title, "skills": skills}])
        temp_df.to_csv(output_csv, mode='a', header=False, index=False)
        current_id += 1

    if current_id == 1:
        raise RuntimeError("Exported 0 rows. Could not extract job titles from the job_link column.")

    print(f"Exported {current_id - 1} rows to {output_csv}")
    if skipped_rows:
        print(f"Skipped {skipped_rows} rows without extractable job titles.")

if __name__ == "__main__":
    main()
