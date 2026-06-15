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

    # 3. Extract Job Titles using LLM
    import os
    OLLAMA_GENERATE_URL = os.environ.get("OLLAMA_GENERATE_URL", "http://localhost:16434/api/generate")
    DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3.5:9b")
    DEFAULT_TIMEOUT_SECONDS = int(os.environ.get("OLLAMA_TIMEOUT_SECONDS", "120"))

    print("Extracting job titles using local LLM...")
    job_titles = []
    
    prompt_template = """Analyze this LinkedIn job URL and return only the extracted job title.
Do not include any other text or explanation.

Examples:
URL: https://www.linkedin.com/jobs/view/housekeeper-1-pt-at-jacksonville-state-university-3802280436
Title: housekeeper

URL: https://www.linkedin.com/jobs/view/assistant-general-manager-huntington-4131-at-ruby-tuesday-3575032
Title: assistant general manager

URL: https://www.linkedin.com/jobs/view/school-based-behavior-analyst-at-ccres-educational-and-behavioral
Title: school based behavior analyst

URL: https://www.linkedin.com/jobs/view/electrical-assembly-lead-at-sanmina-3704300377
Title: electrical assembly

URL: https://www.linkedin.com/jobs/view/senior-lead-technician-programmer-at-security-101-3785441848
Title: technician programmer

note: in this part we only need job title, for instance with 
URL: https://www.linkedin.com/jobs/view/senior-lead-technician-programmer-at-security-101-3785441848
Title: technician programmer, we have the job title is technician programmer, we don't need senir lead as this is just level of employees.

URL: {link}
Title: """

    # 4. Format and export continuously
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=["id", "job_title", "skills"]).to_csv(output_csv, index=False)
    
    current_id = 1
    for index, row in tqdm(df.iterrows(), total=len(df)):
        link = row["job_link"]
        skills = row["skills"]
        prompt = prompt_template.format(link=link)
        try:
            title = generate_with_local_llm(
                prompt,
                model=DEFAULT_MODEL,
                url=OLLAMA_GENERATE_URL,
                timeout=DEFAULT_TIMEOUT_SECONDS
            )
            # Simple clean up
            title = title.replace("```text", "").replace("```", "").strip()
        except Exception:
            title = ""

        if title:
            temp_df = pd.DataFrame([{"id": current_id, "job title": title, "skills": skills}])
            temp_df.to_csv(output_csv, mode='a', header=False, index=False)
            current_id += 1

    print(f"Exported {current_id - 1} rows to {output_csv}")

if __name__ == "__main__":
    main()
