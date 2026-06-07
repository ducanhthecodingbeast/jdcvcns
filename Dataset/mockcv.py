
import argparse
import json
import os
import re
from pathlib import Path

import pandas as pd
from tqdm import tqdm

try:
    from .localllm import (
        build_enrich_cv_prompt,
        build_mock_cv_prompt,
        generate_with_local_llm,
    )
except ImportError:
    from localllm import (
        build_enrich_cv_prompt,
        build_mock_cv_prompt,
        generate_with_local_llm,
    )


def clean_llm_json(raw_text: str) -> str:
    data = raw_text.strip()
    data = re.sub(r"<think>.*?</think>", "", data, flags=re.DOTALL).strip()
    if data.startswith("```json"):
        return data.split("```json", 1)[1].split("```", 1)[0].strip()
    if data.startswith("```"):
        return data.split("```", 1)[1].split("```", 1)[0].strip()
    return data


def parse_cv_payload(raw_text: str):
    data = clean_llm_json(raw_text)
    try:
        return json.loads(data)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        for index, char in enumerate(data):
            if char not in "[{":
                continue
            try:
                payload, _ = decoder.raw_decode(data[index:])
                return payload
            except json.JSONDecodeError:
                continue
        raise


def extend_generated_cvs(generated_cvs: list, cv_payload) -> None:
    if isinstance(cv_payload, list):
        generated_cvs.extend(cv_payload)
        return

    if isinstance(cv_payload, dict):
        for value in cv_payload.values():
            if isinstance(value, list):
                generated_cvs.extend(value)
                return
        generated_cvs.append(cv_payload)
        return

    generated_cvs.append(cv_payload)


def build_jd_context(sample_jd: pd.Series) -> str:
    return f"""
THÔNG TIN CÔNG VIỆC THỰC TẾ:
- Công ty: {sample_jd.get('Tên công ty', '')}
- Giới thiệu công ty: {sample_jd.get('Giới thiệu công ty', '')}
- Mô tả công việc: {sample_jd.get('Mô tả công việc', '')}
- Yêu cầu công việc: {sample_jd.get('Yêu cầu công việc', '')}
- Quyền lợi: {sample_jd.get('Quyền lợi', '')}
"""


def generate_mock_cvs_for_titles(
    df_jd: pd.DataFrame,
    title_col: str = "Vị trí cần tuyển",
    title_count: int = 30,
) -> pd.DataFrame:
    top_titles = df_jd[title_col].value_counts().head(title_count).index
    generated_cvs = []

    for title in tqdm(top_titles, desc="Generating Mock CVs via LLM"):
        sample_jd = df_jd[df_jd[title_col] == title].iloc[0]
        prompt = build_mock_cv_prompt(title, build_jd_context(sample_jd))

        try:
            data = generate_with_local_llm(prompt)
            extend_generated_cvs(generated_cvs, parse_cv_payload(data))
        except Exception as exc:
            tqdm.write(f"[LLM Error] Failed for '{title}': {exc}")

    return pd.DataFrame(generated_cvs)


def enrich_real_cvs(df_cv: pd.DataFrame, sample_size: int = 100) -> pd.DataFrame:
    df_real = df_cv.sample(n=min(sample_size, len(df_cv)), replace=False)
    enriched_real_cvs = []

    for _, row in tqdm(df_real.iterrows(), total=len(df_real), desc="Enriching Real CVs via LLM"):
        name = row.get("Tên ứng viên", "")
        prompt = build_enrich_cv_prompt(
            name=name,
            job=row.get("Vị trí ứng tuyển", ""),
            industry=row.get("Lĩnh vực", ""),
            workplace=row.get("Nơi làm việc mong muốn", ""),
            marriage=row.get("Tình trạng hôn nhân", ""),
            salary=row.get("Mức lương mong muốn", ""),
            gender=row.get("Giới tính", ""),
            age=row.get("Tuổi", ""),
        )

        try:
            data = generate_with_local_llm(prompt)
            enriched_real_cvs.append(parse_cv_payload(data))
        except Exception as exc:
            tqdm.write(f"[LLM Error] Failed enriching CV '{name}': {exc}")

    return pd.DataFrame(enriched_real_cvs)


def generate_and_mix_cvs(
    df_jd: pd.DataFrame,
    df_cv: pd.DataFrame,
    target_dir: str,
    force: bool = False,
    title_count: int = 30,
    sample_size: int = 100,
) -> pd.DataFrame:
    mock_cv_path = os.path.join(target_dir, "mockcv.csv")
    if os.path.exists(mock_cv_path) and not force:
        return pd.read_csv(mock_cv_path)

    os.makedirs(target_dir, exist_ok=True)
    df_generated = generate_mock_cvs_for_titles(df_jd, title_count=title_count)
    df_real_enriched = enrich_real_cvs(df_cv, sample_size=sample_size)

    final_cvs = pd.concat([df_real_enriched, df_generated], ignore_index=True)
    final_cvs.to_csv(mock_cv_path, index=False)
    return final_cvs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate mockcv.csv with the local LLM.")
    parser.add_argument("--jd-path", default="Dataset/jd.csv", help="Path to normalized JD CSV.")
    parser.add_argument("--cv-path", default="Dataset/cv.csv", help="Path to normalized CV CSV.")
    parser.add_argument("--target-dir", default="TestingResults", help="Directory to write mockcv.csv.")
    parser.add_argument("--title-count", type=int, default=30, help="Number of distinct JD titles to generate mock CVs for.")
    parser.add_argument("--sample-size", type=int, default=100, help="Number of real CVs to enrich.")
    parser.add_argument("--force", action="store_true", help="Regenerate even if mockcv.csv already exists.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    jd_path = Path(args.jd_path)
    cv_path = Path(args.cv_path)

    if not jd_path.exists() or not cv_path.exists():
        raise FileNotFoundError(f"Missing {jd_path} or {cv_path}")

    df_jd = pd.read_csv(jd_path)
    df_cv = pd.read_csv(cv_path)
    df_jd.columns = df_jd.columns.str.strip()
    df_cv.columns = df_cv.columns.str.strip()

    final_cvs = generate_and_mix_cvs(
        df_jd,
        df_cv,
        args.target_dir,
        force=args.force,
        title_count=args.title_count,
        sample_size=args.sample_size,
    )
    output_path = Path(args.target_dir) / "mockcv.csv"
    print(f"Wrote {len(final_cvs)} CVs to {output_path}")

if __name__ == "__main__":
    main()
