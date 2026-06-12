from __future__ import annotations

# ruff: noqa: E402

import argparse
import hashlib
import math
import os
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".cache" / "huggingface"))
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(PROJECT_ROOT / ".cache" / "sentence-transformers"))

import numpy as np
import pandas as pd
from tqdm import tqdm

from pipeline import DATASET_DIR, RESULTS_DIR, get_cv_text, get_jd_text, load_datasets, sample_dataframe
from testingresult import RunInfo, env_flag, open_store, store_match_run


JOBBERT_MODEL = os.environ.get("JOBBERT_MODEL", "TechWolf/JobBERT-v2")
TOKEN_RE = re.compile(r"(?u)\b\w+\b")
DENSE_ALGORITHMS = {"cosine", "dot_product"}


@dataclass(frozen=True)
class Config:
    run_name: str
    algorithm: str
    dataset_dir: Path
    results_dir: Path
    model_name: str
    top_k: int
    batch_size: int
    score_batch_size: int
    max_length: int
    cv_limit: int | None
    jd_limit: int | None
    random_state: int
    bm25_k1: float
    bm25_b: float
    regex_tokenizer: bool
    store_db: bool
    write_results: bool


@dataclass(frozen=True)
class BM25Index:
    postings: dict[str, list[tuple[int, int]]]
    idf: dict[str, float]
    doc_lens: np.ndarray
    avg_doc_len: float
    k1: float
    b: float


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name, "").strip()
    return default if not value else int(value)


def env_float(name: str, default: float) -> float:
    value = os.environ.get(name, "").strip()
    return default if not value else float(value)


def env_optional_int(name: str) -> int | None:
    value = os.environ.get(name, "").strip()
    return None if not value else int(value)


def configure_torch_runtime() -> None:
    try:
        import torch
    except ImportError:
        return

    cpu_threads = os.environ.get("CPU_THREADS") or os.environ.get("OMP_NUM_THREADS")
    if cpu_threads:
        try:
            torch.set_num_threads(max(1, int(cpu_threads)))
        except (RuntimeError, ValueError):
            pass

    try:
        torch.set_float32_matmul_precision("high")
    except (AttributeError, RuntimeError):
        pass

    if not torch.cuda.is_available():
        return

    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except AttributeError:
        pass

    fraction = os.environ.get("GPU_MEMORY_FRACTION", "").strip()
    if fraction:
        try:
            torch.cuda.set_per_process_memory_fraction(min(1.0, max(0.01, float(fraction))))
        except (RuntimeError, ValueError):
            pass


def parse_args(run_name: str, algorithm: str) -> Config:
    parser = argparse.ArgumentParser(
        description=f"Run {run_name} CV/JD matching with algorithm={algorithm}."
    )
    parser.add_argument("--dataset-dir", default=str(DATASET_DIR))
    parser.add_argument("--results-dir", default=str(RESULTS_DIR))
    parser.add_argument("--model-name", default=JOBBERT_MODEL)
    parser.add_argument("--top-k", type=int, default=env_int("TOP_K", 10))
    parser.add_argument("--batch-size", type=int, default=env_int("JOBBERT_BATCH_SIZE", 32))
    parser.add_argument("--score-batch-size", type=int, default=env_int("SCORE_BATCH_SIZE", 128))
    parser.add_argument("--max-length", type=int, default=env_int("JOBBERT_MAX_LENGTH", 512))
    parser.add_argument("--cv-limit", type=int, default=env_optional_int("CV_LIMIT"))
    parser.add_argument("--jd-limit", type=int, default=env_optional_int("JD_LIMIT"))
    parser.add_argument("--random-state", type=int, default=env_int("RANDOM_STATE", 42))
    parser.add_argument("--bm25-k1", type=float, default=env_float("BM25_K1", 1.5))
    parser.add_argument("--bm25-b", type=float, default=env_float("BM25_B", 0.75))
    parser.add_argument(
        "--regex-tokenizer",
        action="store_true",
        help="Use simple regex tokenization for BM25 instead of the JobBERT tokenizer.",
    )
    parser.add_argument("--no-store-db", action="store_true")
    parser.add_argument("--no-write-results", action="store_true")
    args = parser.parse_args()

    return Config(
        run_name=run_name,
        algorithm=algorithm,
        dataset_dir=Path(args.dataset_dir),
        results_dir=Path(args.results_dir),
        model_name=args.model_name,
        top_k=max(1, args.top_k),
        batch_size=max(1, args.batch_size),
        score_batch_size=max(1, args.score_batch_size),
        max_length=max(1, args.max_length),
        cv_limit=args.cv_limit if args.cv_limit is None else max(1, args.cv_limit),
        jd_limit=args.jd_limit if args.jd_limit is None else max(1, args.jd_limit),
        random_state=args.random_state,
        bm25_k1=max(0.01, args.bm25_k1),
        bm25_b=min(1.0, max(0.0, args.bm25_b)),
        regex_tokenizer=args.regex_tokenizer or env_flag("BM25_REGEX_TOKENIZER", False),
        store_db=env_flag("STORE_DB", True) and not args.no_store_db,
        write_results=env_flag("WRITE_RESULTS", True) and not args.no_write_results,
    )


def chunks(items: list[str], batch_size: int) -> Iterable[tuple[int, list[str]]]:
    for start in range(0, len(items), batch_size):
        yield start, items[start : start + batch_size]


def get_jd_match_text(row: pd.Series) -> str:
    return get_jd_text(row, include_company=True)


def load_experiment_datasets(config: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        df_cv, df_jd = load_datasets(config.dataset_dir)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"{exc}\n"
            "6.x expects the shared matching datasets under Data/: jd.csv or JOB_DATA_FINAL.csv, "
            "plus mockcv.csv, cv.csv, or USER_DATA_FINAL.csv. The current ONS skills workbook is "
            "not a CV/JD matching dataset by itself."
        ) from exc

    df_cv = sample_dataframe(df_cv, config.cv_limit, random_state=config.random_state)
    df_jd = sample_dataframe(df_jd, config.jd_limit, random_state=config.random_state)
    if df_cv.empty or df_jd.empty:
        raise ValueError("Both CV and JD datasets must contain at least one row.")
    return df_cv, df_jd


def safe_model_name(model_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", model_name).strip("_") or "model"


def text_digest(texts: list[str], model_name: str, max_length: int) -> str:
    digest = hashlib.sha256()
    digest.update(model_name.encode("utf-8"))
    digest.update(str(max_length).encode("ascii"))
    for text in texts:
        data = text.encode("utf-8", errors="ignore")
        digest.update(str(len(data)).encode("ascii"))
        digest.update(b"\0")
        digest.update(data)
        digest.update(b"\0")
    return digest.hexdigest()[:16]


def embedding_cache_path(config: Config, label: str, texts: list[str]) -> Path:
    cache_dir = PROJECT_ROOT / ".cache" / "embeddings"
    cache_dir.mkdir(parents=True, exist_ok=True)
    digest = text_digest(texts, config.model_name, config.max_length)
    filename = f"{label}_{safe_model_name(config.model_name)}_{len(texts)}_{digest}.npy"
    return cache_dir / filename


class TransformerMeanPoolingEncoder:
    def __init__(self, model_name: str, max_length: int, device: str | None = None):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.model_name = model_name
        self.max_length = max_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=os.environ.get("HF_HOME"))
        self.model = AutoModel.from_pretrained(model_name, cache_dir=os.environ.get("HF_HOME"))
        self.model.to(self.device)
        self.model.eval()

    def encode(self, texts: list[str], batch_size: int) -> np.ndarray:
        vectors: list[np.ndarray] = []
        for _, batch in chunks(texts, batch_size):
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            with self.torch.no_grad():
                output = self.model(**encoded)
                attention_mask = encoded["attention_mask"].unsqueeze(-1).float()
                token_embeddings = output.last_hidden_state
                pooled = (token_embeddings * attention_mask).sum(dim=1)
                pooled = pooled / attention_mask.sum(dim=1).clamp(min=1e-9)
            vectors.append(pooled.detach().cpu().numpy())

        embeddings = np.vstack(vectors).astype(np.float32)
        if embeddings.ndim != 2:
            raise RuntimeError(f"Expected 2D embeddings, got shape={embeddings.shape}.")
        return embeddings


def load_embedding_model(config: Config) -> TransformerMeanPoolingEncoder:
    configure_torch_runtime()
    import torch
    try:
        return TransformerMeanPoolingEncoder(config.model_name, config.max_length)
    except RuntimeError as exc:
        message = str(exc).lower()
        if torch.cuda.is_available() and ("out of memory" in message or "cuda" in message):
            print("CUDA failed while loading JobBERT. Retrying on CPU.")
            return TransformerMeanPoolingEncoder(config.model_name, config.max_length, device="cpu")
        raise


def encode_texts(model: TransformerMeanPoolingEncoder, texts: list[str], config: Config) -> np.ndarray:
    return model.encode(texts, batch_size=config.batch_size)


def load_or_encode_embeddings(
    model,
    texts: list[str],
    config: Config,
    label: str,
) -> np.ndarray:
    cache_path = embedding_cache_path(config, label, texts)
    if cache_path.exists():
        embeddings = np.load(cache_path)
        if embeddings.shape[0] == len(texts):
            print(f"Loaded {label} embeddings from {cache_path}")
            return np.asarray(embeddings, dtype=np.float32)

    print(f"Encoding {len(texts)} {label} texts with {config.model_name}")
    embeddings = encode_texts(model, texts, config)
    np.save(cache_path, embeddings)
    print(f"Saved {label} embeddings to {cache_path}")
    return embeddings


def l2_normalize(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return embeddings / norms


def top_indices(scores: np.ndarray, top_k: int) -> np.ndarray:
    limit = min(top_k, len(scores))
    if limit <= 0:
        return np.array([], dtype=np.int64)
    if limit == len(scores):
        return np.argsort(scores)[::-1]
    candidate_idx = np.argpartition(scores, -limit)[-limit:]
    return candidate_idx[np.argsort(scores[candidate_idx])[::-1]]


def first_value(row: pd.Series, *names: str) -> str:
    for name in names:
        if name in row and pd.notna(row[name]) and str(row[name]).strip():
            return str(row[name]).strip()
    return ""


def build_match(
    df_cv: pd.DataFrame,
    df_jd: pd.DataFrame,
    cv_idx: int,
    jd_idx: int,
    rank: int,
    score: float,
    score_method: str,
    extra_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cv_row = df_cv.iloc[cv_idx]
    jd_row = df_jd.iloc[jd_idx]
    return {
        "cv_id": int(cv_idx),
        "cv_title": first_value(cv_row, "Tên ứng viên", "Vị trí ứng tuyển", "User Name", "Desired Job"),
        "jd_id": int(jd_idx),
        "jd_title": first_value(jd_row, "Vị trí cần tuyển", "Job Title"),
        "rank": int(rank),
        "score": float(score),
        "score_method": score_method,
        "cv_payload": cv_row.to_dict(),
        "jd_payload": jd_row.to_dict(),
        "meta": extra_meta or {},
    }


def dense_matches(
    df_cv: pd.DataFrame,
    df_jd: pd.DataFrame,
    cv_texts: list[str],
    jd_texts: list[str],
    config: Config,
) -> list[dict[str, Any]]:
    model = load_embedding_model(config)
    cv_embeddings = load_or_encode_embeddings(model, cv_texts, config, "cv")
    jd_embeddings = load_or_encode_embeddings(model, jd_texts, config, "jd")

    if config.algorithm == "cosine":
        cv_embeddings = l2_normalize(cv_embeddings)
        jd_embeddings = l2_normalize(jd_embeddings)
        score_method = "cosine"
    elif config.algorithm == "dot_product":
        score_method = "dot_product"
    else:
        raise ValueError(f"Unsupported dense algorithm: {config.algorithm}")

    matches: list[dict[str, Any]] = []
    progress = tqdm(total=len(df_cv), desc=f"Scoring CVs ({score_method})")
    for start in range(0, len(df_cv), config.score_batch_size):
        end = min(start + config.score_batch_size, len(df_cv))
        score_block = cv_embeddings[start:end] @ jd_embeddings.T
        for offset, scores in enumerate(score_block):
            cv_idx = start + offset
            for rank, jd_idx in enumerate(top_indices(scores, config.top_k), start=1):
                matches.append(
                    build_match(
                        df_cv,
                        df_jd,
                        cv_idx,
                        int(jd_idx),
                        rank,
                        float(scores[int(jd_idx)]),
                        score_method,
                        {"source": f"jobbert_{score_method}"},
                    )
                )
        progress.update(end - start)
    progress.close()
    return matches


def regex_tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def jobbert_tokenize_texts(texts: list[str], config: Config) -> tuple[list[list[str]], str]:
    if config.regex_tokenizer:
        return [regex_tokens(text) for text in tqdm(texts, desc="Regex tokenizing")], "regex"

    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(config.model_name, cache_dir=os.environ.get("HF_HOME"))
        tokenized: list[list[str]] = []
        for _, batch in chunks(texts, config.batch_size):
            encoded = tokenizer(
                batch,
                add_special_tokens=False,
                truncation=True,
                max_length=config.max_length,
            )
            tokenized.extend([[str(token_id) for token_id in ids] for ids in encoded["input_ids"]])
        return tokenized, f"{config.model_name} tokenizer"
    except Exception as exc:
        print(f"Could not load JobBERT tokenizer for BM25 ({exc}). Falling back to regex tokenization.")
        return [regex_tokens(text) for text in tqdm(texts, desc="Regex tokenizing")], "regex_fallback"


def build_bm25_index(tokenized_docs: list[list[str]], k1: float, b: float) -> BM25Index:
    postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
    doc_freq: dict[str, int] = defaultdict(int)
    doc_lens = np.zeros(len(tokenized_docs), dtype=np.float32)

    for doc_idx, tokens in enumerate(tqdm(tokenized_docs, desc="Building BM25 index")):
        counts = Counter(tokens)
        doc_lens[doc_idx] = sum(counts.values())
        for token, tf in counts.items():
            doc_freq[token] += 1
            postings[token].append((doc_idx, int(tf)))

    doc_count = max(1, len(tokenized_docs))
    avg_doc_len = float(doc_lens.mean()) if len(doc_lens) else 0.0
    if avg_doc_len <= 0:
        avg_doc_len = 1.0

    idf = {
        token: math.log(1.0 + (doc_count - freq + 0.5) / (freq + 0.5))
        for token, freq in doc_freq.items()
    }
    return BM25Index(dict(postings), idf, doc_lens, avg_doc_len, k1, b)


def bm25_scores(query_tokens: list[str], index: BM25Index) -> np.ndarray:
    scores = np.zeros(len(index.doc_lens), dtype=np.float32)
    for token, query_tf in Counter(query_tokens).items():
        postings = index.postings.get(token)
        if not postings:
            continue
        idf = index.idf[token]
        for doc_idx, tf in postings:
            length_norm = 1.0 - index.b + index.b * float(index.doc_lens[doc_idx]) / index.avg_doc_len
            denom = tf + index.k1 * length_norm
            scores[doc_idx] += float(query_tf) * idf * (tf * (index.k1 + 1.0) / denom)
    return scores


def bm25_matches(
    df_cv: pd.DataFrame,
    df_jd: pd.DataFrame,
    cv_texts: list[str],
    jd_texts: list[str],
    config: Config,
) -> list[dict[str, Any]]:
    jd_tokens, tokenizer_name = jobbert_tokenize_texts(jd_texts, config)
    cv_tokens, _ = jobbert_tokenize_texts(cv_texts, config)
    index = build_bm25_index(jd_tokens, config.bm25_k1, config.bm25_b)

    matches: list[dict[str, Any]] = []
    for cv_idx, query_tokens in enumerate(tqdm(cv_tokens, desc="Scoring CVs (BM25)")):
        scores = bm25_scores(query_tokens, index)
        for rank, jd_idx in enumerate(top_indices(scores, config.top_k), start=1):
            matches.append(
                build_match(
                    df_cv,
                    df_jd,
                    cv_idx,
                    int(jd_idx),
                    rank,
                    float(scores[int(jd_idx)]),
                    "bm25",
                    {
                        "source": "bm25",
                        "tokenizer": tokenizer_name,
                        "bm25_k1": config.bm25_k1,
                        "bm25_b": config.bm25_b,
                    },
                )
            )
    return matches


def run_info(config: Config, df_cv: pd.DataFrame, df_jd: pd.DataFrame) -> RunInfo:
    model_name = config.model_name if config.algorithm in DENSE_ALGORITHMS else f"BM25 with {config.model_name} tokenizer"
    params: dict[str, Any] = {
        "top_k": config.top_k,
        "cv_count": int(len(df_cv)),
        "jd_count": int(len(df_jd)),
        "cv_limit": config.cv_limit,
        "jd_limit": config.jd_limit,
        "random_state": config.random_state,
    }
    if config.algorithm in DENSE_ALGORITHMS:
        params.update(
            {
                "batch_size": config.batch_size,
                "score_batch_size": config.score_batch_size,
                "max_length": config.max_length,
                "embedding_model": config.model_name,
            }
        )
    else:
        params.update(
            {
                "bm25_k1": config.bm25_k1,
                "bm25_b": config.bm25_b,
                "tokenizer_model": config.model_name,
                "regex_tokenizer": config.regex_tokenizer,
                "max_length": config.max_length,
            }
        )

    return RunInfo(
        run_name=config.run_name,
        algorithm=config.algorithm,
        model_name=model_name,
        params=params,
        dataset_meta={
            "dataset_dir": str(config.dataset_dir),
            "cv_rows": int(len(df_cv)),
            "jd_rows": int(len(df_jd)),
        },
    )


def store_results(
    matches: list[dict[str, Any]],
    df_cv: pd.DataFrame,
    df_jd: pd.DataFrame,
    config: Config,
    started_monotonic: float,
) -> int | None:
    if not config.store_db:
        return None

    matches_by_cv: dict[int, list[dict[str, Any]]] = {}
    for match in matches:
        matches_by_cv.setdefault(int(match["cv_id"]), []).append(match)

    def ranked_matches(cv_idx: int) -> list[dict[str, Any]]:
        ranked = sorted(matches_by_cv.get(cv_idx, []), key=lambda item: item["rank"])
        return [
            {
                "jd_idx": int(match["jd_id"]),
                "rank": int(match["rank"]),
                "score": float(match["score"]),
                "meta": {
                    "source": match.get("meta", {}).get("source", config.algorithm),
                    "score_method": match.get("score_method"),
                    **(match.get("meta") or {}),
                },
            }
            for match in ranked
        ]

    conn = open_store()
    try:
        return store_match_run(
            conn,
            run_info(config, df_cv, df_jd),
            df_cv,
            df_jd,
            get_cv_text,
            get_jd_match_text,
            ranked_matches,
            top_k=min(config.top_k, len(df_jd)),
            started_monotonic=started_monotonic,
        )
    finally:
        conn.close()


def write_match_summary(matches: list[dict[str, Any]], config: Config, run_id: int | None) -> str | None:
    if not config.write_results:
        return None

    config.results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_label = f"run{run_id}" if run_id is not None else "local"
    path = config.results_dir / f"{config.run_name}_{run_label}_{timestamp}.csv"
    columns = ["cv_id", "cv_title", "rank", "jd_id", "jd_title", "score", "score_method"]
    rows = [{column: match.get(column) for column in columns} for match in matches]
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False)
    return str(path)


def run_matching(config: Config) -> dict[str, Any]:
    started_wall = time.time()
    started_monotonic = time.monotonic()

    print("Loading datasets...")
    df_cv, df_jd = load_experiment_datasets(config)
    cv_texts = [get_cv_text(row) for _, row in df_cv.iterrows()]
    jd_texts = [get_jd_match_text(row) for _, row in df_jd.iterrows()]

    if config.algorithm in DENSE_ALGORITHMS:
        matches = dense_matches(df_cv, df_jd, cv_texts, jd_texts, config)
    elif config.algorithm == "bm25":
        matches = bm25_matches(df_cv, df_jd, cv_texts, jd_texts, config)
    else:
        raise ValueError(f"Unsupported algorithm: {config.algorithm}")

    run_id = store_results(matches, df_cv, df_jd, config, started_monotonic)
    result_path = write_match_summary(matches, config, run_id)
    return {
        "cv_count": len(df_cv),
        "jd_count": len(df_jd),
        "match_count": len(matches),
        "run_id": run_id,
        "result_path": result_path,
        "elapsed_seconds": round(time.time() - started_wall, 3),
    }


def main(run_name: str, algorithm: str) -> None:
    config = parse_args(run_name, algorithm)
    result = run_matching(config)
    print(
        f"Matched {result['cv_count']} CVs against {result['jd_count']} JDs "
        f"({result['match_count']} rows) in {result['elapsed_seconds']}s. "
        f"Run ID: {result['run_id']}"
    )
    if result["result_path"]:
        print(f"Result CSV: {result['result_path']}")
