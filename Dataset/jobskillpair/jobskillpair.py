from pathlib import Path
import subprocess
import zipfile
import pandas as pd
from tqdm import tqdm
from Dataset.localllm import generate_with_local_llm

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
    source_csv = list(raw_dir.glob(" *.csv"))[0]
    print(f"Loading data from {source_csv}...")
    df = pd.read_csv(source_csv)

    # Find URL column by checking for 'https'
    link_col = next((c for c indf.columns if df[c].astype(str).str.contains("https", na=False).sum() > 1), None)
    if not link_col:
        raise ValueError("Could not find a column containing URLs.")
        
    # Find Skills column (keep it simple)
    skills_col = [c for c in df.columns if "skill" in c.lower() and c != link_col][0]

    df = df[[link_col, skills_col]].rename(columns={link_col: "job_link", skills_col: "skills"})
    df = df.dropna(subset=["job_link"]).drop_duplicates(subset=["job_link"]).reset_index(drop=True)

    # 3. Extract Job Titles using LLM
    print("Extracting job titles using local LLM...")
    job_titles = []
    for link in tqdm(df["job_link"]):
        prompt = f"Analyze this LinkedIn job URL and return only the job title.\n\nURL:\n{link}"
        try:
            title = generate_with_local_llm(prompt)
            # Simple clean up
            title = title.replace("```text", "").replace("```", "").strip()
            job_titles.append(title)
        except Exception:
            job_titles.append("")

    df["job_title"] = job_titles
    df = df[df["job_title"] != ""].reset_index(drop=True)

    # 4. Format and export
    df.insert(0, "id", range(1, len(df) + 1))
    final_df = df[["id", "job_title", "skills"]]
    
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(output_csv, index=False)
    print(f"Exported {len(final_df)} rows to {output_csv}")

if __name__ == "__main__":
    main()
