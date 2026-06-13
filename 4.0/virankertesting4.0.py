from __future__ import annotations

# ruff: noqa: E402

import argparse
import math
import os
import sys
import time
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

from pipeline import DATASET_DIR, RESULTS_DIR, get_cv_text, get_jd_text, load_datasets
from testingresult import RunInfo, env_flag, open_store, store_match_run


BGE_M3_MODEL = os.environ.get("BGE_M3_MODEL", "BAAI/bge-m3")
VIRANKER_MODEL = os.environ.get("VIRANKER_MODEL", "namdp-ptit/ViRanker")
DEFAULT_COLLECTION = "jd_hybrid_viranker_collection"


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name, "").strip()
    return default if not value else int(value)


def env_optional_int(name: str, default: int | None = None) -> int | None:
    value = os.environ.get(name, "").strip()
    if not value:
        return default
    return int(value)


@dataclass(frozen=True)
class Config:
    dataset_dir: Path
    results_dir: Path
    collection_name: str
    top_k: int
    batch_size: int
    upsert_batch_size: int
    reranker_batch_size: int
    reranker_max_length: int
    reranker_query_max_length: int | None
    reranker_normalize: bool
    prefetch_multiplier: int
    recreate_collection: bool
    use_fp16: bool
    qdrant_host: str
    qdrant_port: int
    qdrant_url: str | None
    qdrant_path: str | None
    qdrant_timeout: float
    qdrant_upsert_retries: int
    store_db: bool
    write_results: bool


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="Run BGE-M3 hybrid recall with ViRanker cross-encoder reranking for CV/JD matching."
    )
    parser.add_argument("--dataset-dir", default=str(DATASET_DIR))
    parser.add_argument("--results-dir", default=str(RESULTS_DIR))
    parser.add_argument("--collection", default=os.environ.get("QDRANT_COLLECTION", DEFAULT_COLLECTION))
    parser.add_argument("--top-k", type=int, default=env_int("TOP_K", 10))
    parser.add_argument("--batch-size", type=int, default=env_int("BGE_BATCH_SIZE", 16))
    parser.add_argument("--upsert-batch-size", type=int, default=env_int("QDRANT_UPSERT_BATCH_SIZE", 4))
    parser.add_argument("--reranker-batch-size", type=int, default=env_int("VIRANKER_BATCH_SIZE", 32))
    parser.add_argument("--reranker-max-length", type=int, default=env_int("VIRANKER_MAX_LENGTH", 1024))
    parser.add_argument(
        "--reranker-query-max-length",
        type=int,
        default=env_optional_int("VIRANKER_QUERY_MAX_LENGTH", 384),
        help="Maximum tokens reserved for the CV/query side before ViRanker truncates the pair.",
    )
    parser.add_argument("--raw-reranker-scores", action="store_true")
    parser.add_argument("--no-write-results", action="store_true")
    parser.add_argument("--prefetch-multiplier", type=int, default=env_int("PREFETCH_MULTIPLIER", 4))
    parser.add_argument("--keep-collection", action="store_true")
    parser.add_argument("--no-fp16", action="store_true")
    parser.add_argument("--qdrant-host", default=os.environ.get("QDRANT_HOST", "localhost"))
    parser.add_argument("--qdrant-port", type=int, default=env_int("QDRANT_PORT", 16340))
    parser.add_argument("--qdrant-url", default=os.environ.get("QDRANT_URL"))
    parser.add_argument("--qdrant-path", default=os.environ.get("QDRANT_PATH"))
    parser.add_argument("--qdrant-timeout", type=float, default=float(os.environ.get("QDRANT_TIMEOUT", "120")))
    parser.add_argument("--qdrant-upsert-retries", type=int, default=env_int("QDRANT_UPSERT_RETRIES", 3))
    args = parser.parse_args()
    query_max_length = args.reranker_query_max_length
    if query_max_length is not None and query_max_length <= 0:
        query_max_length = None

    return Config(
        dataset_dir=Path(args.dataset_dir),
        results_dir=Path(args.results_dir),
        collection_name=args.collection,
        top_k=max(1, args.top_k),
        batch_size=max(1, args.batch_size),
        upsert_batch_size=max(1, args.upsert_batch_size),
        reranker_batch_size=max(1, args.reranker_batch_size),
        reranker_max_length=max(1, args.reranker_max_length),
        reranker_query_max_length=query_max_length,
        reranker_normalize=env_flag("VIRANKER_NORMALIZE", True) and not args.raw_reranker_scores,
        prefetch_multiplier=max(1, args.prefetch_multiplier),
        recreate_collection=not args.keep_collection,
        use_fp16=not args.no_fp16,
        qdrant_host=args.qdrant_host,
        qdrant_port=args.qdrant_port,
        qdrant_url=args.qdrant_url,
        qdrant_path=args.qdrant_path,
        qdrant_timeout=max(1.0, args.qdrant_timeout),
        qdrant_upsert_retries=max(1, args.qdrant_upsert_retries),
        store_db=env_flag("STORE_DB", True),
        write_results=env_flag("WRITE_RESULTS", True) and not args.no_write_results,
    )


def require_bgem3_model():
    try:
        from FlagEmbedding import BGEM3FlagModel
    except ImportError as exc:
        raise RuntimeError("Missing dependency: install FlagEmbedding to run BGE-M3 matching.") from exc
    return BGEM3FlagModel


def require_viranker_model():
    try:
        from FlagEmbedding import FlagReranker
    except ImportError as exc:
        raise RuntimeError("Missing dependency: install FlagEmbedding to run ViRanker matching.") from exc
    return FlagReranker


def require_qdrant():
    try:
        from qdrant_client import QdrantClient, models
    except ImportError as exc:
        raise RuntimeError("Missing dependency: install qdrant-client to run Qdrant matching.") from exc
    return QdrantClient, models


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


def json_safe(value: Any) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, np.generic):
        return value.item()
    return value


def row_payload(row: pd.Series) -> dict[str, Any]:
    return {str(key): json_safe(value) for key, value in row.to_dict().items()}


def to_list(vector: Any) -> list:
    return vector.tolist() if hasattr(vector, "tolist") else list(vector)


def create_sparse_vector(sparse_data: dict[Any, Any], models):
    indices: list[int] = []
    values: list[float] = []

    for raw_key, raw_value in sparse_data.items():
        value = float(raw_value)
        if value <= 0:
            continue
        if isinstance(raw_key, str):
            if not raw_key.isdigit():
                continue
            raw_key = int(raw_key)
        indices.append(int(raw_key))
        values.append(value)

    return models.SparseVector(indices=indices, values=values)


def chunks(items: list[Any], batch_size: int) -> Iterable[tuple[int, list[Any]]]:
    for start in range(0, len(items), batch_size):
        yield start, items[start : start + batch_size]


def connect_qdrant(config: Config):
    QdrantClient, _ = require_qdrant()
    if config.qdrant_path:
        return QdrantClient(path=config.qdrant_path, timeout=config.qdrant_timeout)
    if config.qdrant_url:
        return QdrantClient(url=config.qdrant_url, timeout=config.qdrant_timeout)
    return QdrantClient(host=config.qdrant_host, port=config.qdrant_port, timeout=config.qdrant_timeout)


def qdrant_target(config: Config) -> str:
    if config.qdrant_path:
        return f"path={config.qdrant_path}"
    if config.qdrant_url:
        return config.qdrant_url
    return f"{config.qdrant_host}:{config.qdrant_port}"


def run_qdrant_startup_step(description: str, config: Config, step) -> Any:
    deadline = time.monotonic() + config.qdrant_timeout
    attempt = 1

    while True:
        try:
            return step()
        except Exception as exc:
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"Qdrant {description} failed after {config.qdrant_timeout:.0f}s "
                    f"for {qdrant_target(config)}: {type(exc).__name__}: {exc}"
                ) from exc

            delay_seconds = min(2 ** attempt, 10)
            print(
                f"Qdrant {description} failed on attempt {attempt}: "
                f"{type(exc).__name__}: {exc}. Retrying in {delay_seconds}s...",
                file=sys.stderr,
            )
            time.sleep(delay_seconds)
            attempt += 1


def prepare_collection_once(client, config: Config, models) -> None:
    exists = client.collection_exists(config.collection_name)
    if exists and not config.recreate_collection:
        return
    if exists:
        client.delete_collection(config.collection_name)

    client.create_collection(
        collection_name=config.collection_name,
        vectors_config={
            "dense": models.VectorParams(size=1024, distance=models.Distance.COSINE),
        },
        sparse_vectors_config={
            "sparse": models.SparseVectorParams(
                index=models.SparseIndexParams(on_disk=True),
            ),
        },
    )


def prepare_collection(client, config: Config, models) -> None:
    run_qdrant_startup_step("startup check", config, client.get_collections)
    run_qdrant_startup_step(
        "collection setup",
        config,
        lambda: prepare_collection_once(client, config, models),
    )


def encode_batch(model, texts: list[str], batch_size: int) -> dict[str, Any]:
    return model.encode(
        texts,
        batch_size=batch_size,
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=False,
    )


def build_reranker(config: Config):
    FlagReranker = require_viranker_model()
    kwargs: dict[str, Any] = {
        "use_fp16": config.use_fp16,
        "batch_size": config.reranker_batch_size,
        "max_length": config.reranker_max_length,
        "normalize": config.reranker_normalize,
        "cache_dir": os.environ.get("HF_HOME"),
    }
    if config.reranker_query_max_length is not None:
        kwargs["query_max_length"] = config.reranker_query_max_length
    kwargs = {key: value for key, value in kwargs.items() if value is not None}

    try:
        return FlagReranker(VIRANKER_MODEL, **kwargs)
    except TypeError:
        return FlagReranker(VIRANKER_MODEL, use_fp16=config.use_fp16)


def normalize_scores(scores: Any) -> list[float]:
    if np.isscalar(scores):
        return [float(scores)]
    if hasattr(scores, "tolist"):
        scores = scores.tolist()
    return [float(score) for score in scores]


def sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1 / (1 + z)
    z = math.exp(value)
    return z / (1 + z)


def compute_reranker_scores(reranker, pairs: list[list[str]], config: Config) -> list[float]:
    kwargs: dict[str, Any] = {
        "batch_size": config.reranker_batch_size,
        "max_length": config.reranker_max_length,
        "normalize": config.reranker_normalize,
    }
    if config.reranker_query_max_length is not None:
        kwargs["query_max_length"] = config.reranker_query_max_length

    try:
        return normalize_scores(reranker.compute_score(pairs, **kwargs))
    except TypeError:
        try:
            scores = normalize_scores(
                reranker.compute_score(
                    pairs,
                    batch_size=config.reranker_batch_size,
                    max_length=config.reranker_max_length,
                )
            )
        except TypeError:
            scores = normalize_scores(reranker.compute_score(pairs))
        return [sigmoid(score) for score in scores] if config.reranker_normalize else scores


def upsert_points_with_retry(client, config: Config, points: list[Any]) -> None:
    for attempt in range(1, config.qdrant_upsert_retries + 1):
        try:
            client.upsert(collection_name=config.collection_name, points=points, wait=True)
            return
        except Exception as exc:
            if attempt >= config.qdrant_upsert_retries:
                raise
            delay_seconds = min(2 ** attempt, 30)
            print(
                f"Qdrant upsert failed on attempt {attempt}/{config.qdrant_upsert_retries}: "
                f"{type(exc).__name__}: {exc}. Retrying in {delay_seconds}s...",
                file=sys.stderr,
            )
            time.sleep(delay_seconds)


def index_jobs(client, model, df_jd: pd.DataFrame, config: Config, models) -> None:
    jd_texts = [get_jd_text(row) for _, row in df_jd.iterrows()]
    progress = tqdm(total=len(jd_texts), desc="Indexing JDs")

    for start, text_batch in chunks(jd_texts, config.batch_size):
        output = encode_batch(model, text_batch, config.batch_size)
        points = []

        for offset, text in enumerate(text_batch):
            jd_idx = start + offset
            jd_row = df_jd.iloc[jd_idx]
            sparse = create_sparse_vector(output["lexical_weights"][offset], models)
            points.append(
                models.PointStruct(
                    id=int(jd_idx),
                    payload={
                        **row_payload(jd_row),
                        "_text": text,
                    },
                    vector={
                        "dense": to_list(output["dense_vecs"][offset]),
                        "sparse": sparse,
                    },
                )
            )

        for _, point_batch in chunks(points, config.upsert_batch_size):
            upsert_points_with_retry(client, config, point_batch)
        progress.update(len(text_batch))

    progress.close()


def match_cvs(client, model, df_cv: pd.DataFrame, config: Config, models) -> list[dict[str, Any]]:
    print(
        f"Initializing ViRanker model: {VIRANKER_MODEL} "
        f"(max_length={config.reranker_max_length}, batch_size={config.reranker_batch_size}, "
        f"normalize={config.reranker_normalize})"
    )
    reranker = build_reranker(config)

    cv_texts = [get_cv_text(row) for _, row in df_cv.iterrows()]
    matches: list[dict[str, Any]] = []
    prefetch_limit = max(config.top_k, config.top_k * config.prefetch_multiplier)
    progress = tqdm(total=len(cv_texts), desc="Matching CVs")

    for start, text_batch in chunks(cv_texts, config.batch_size):
        output = encode_batch(model, text_batch, config.batch_size)

        for offset, cv_text in enumerate(text_batch):
            cv_idx = start + offset
            cv_row = df_cv.iloc[cv_idx]
            sparse = create_sparse_vector(output["lexical_weights"][offset], models)

            results = client.query_points(
                collection_name=config.collection_name,
                prefetch=[
                    models.Prefetch(query=sparse, using="sparse", limit=prefetch_limit),
                    models.Prefetch(query=to_list(output["dense_vecs"][offset]), using="dense", limit=prefetch_limit),
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                with_payload=True,
                limit=prefetch_limit,
            )

            candidates = []
            for retrieval_rank, result in enumerate(results.points, start=1):
                jd_payload = result.payload or {}
                jd_text = jd_payload.get("_text", "")
                candidates.append((retrieval_rank, result, jd_payload, jd_text))

            if not candidates:
                continue

            pairs = [[cv_text, c[3]] for c in candidates]
            scores = compute_reranker_scores(reranker, pairs, config)
            if len(scores) != len(candidates):
                raise RuntimeError(
                    f"ViRanker returned {len(scores)} scores for {len(candidates)} candidate pairs."
                )

            scored_candidates = []
            for score, (retrieval_rank, result, jd_payload, _) in zip(scores, candidates):
                scored_candidates.append(
                    {
                        "jd_id": int(result.id),
                        "score": float(score),
                        "retrieval_rank": int(retrieval_rank),
                        "retrieval_score": float(result.score),
                        "jd_payload": jd_payload,
                    }
                )

            scored_candidates.sort(key=lambda x: x["score"], reverse=True)
            top_candidates = scored_candidates[:config.top_k]

            cv_title = first_value(cv_row, "Tên ứng viên", "Vị trí ứng tuyển", "User Name", "Desired Job")
            for rank, cand in enumerate(top_candidates, start=1):
                jd_payload = cand["jd_payload"]
                matches.append(
                    {
                        "cv_id": int(cv_idx),
                        "cv_title": cv_title,
                        "jd_id": cand["jd_id"],
                        "jd_title": first_payload_value(jd_payload, "Vị trí cần tuyển", "Job Title"),
                        "rank": rank,
                        "score": cand["score"],
                        "score_type": "sigmoid" if config.reranker_normalize else "raw",
                        "retrieval_rank": cand["retrieval_rank"],
                        "retrieval_score": cand["retrieval_score"],
                        "cv_payload": row_payload(cv_row),
                        "jd_payload": {key: value for key, value in jd_payload.items() if key != "_text"},
                    }
                )

        progress.update(len(text_batch))

    progress.close()
    return matches


def first_value(row: pd.Series, *names: str) -> str:
    for name in names:
        if name in row and pd.notna(row[name]) and str(row[name]).strip():
            return str(row[name]).strip()
    return ""


def first_payload_value(payload: dict[str, Any], *names: str) -> str:
    for name in names:
        value = payload.get(name)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def write_match_summary(matches: list[dict[str, Any]], config: Config, run_id: int | None) -> str | None:
    if not config.write_results:
        return None

    config.results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_label = f"run{run_id}" if run_id is not None else "local"
    path = config.results_dir / f"virankertesting4.0_{run_label}_{timestamp}.csv"
    columns = [
        "cv_id",
        "cv_title",
        "rank",
        "jd_id",
        "jd_title",
        "score",
        "score_type",
        "retrieval_rank",
        "retrieval_score",
    ]
    rows = [{column: match.get(column) for column in columns} for match in matches]
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False)
    return str(path)


def run_bgem3_qdrant(config: Config) -> dict[str, Any]:
    started_wall = time.time()
    started_monotonic = time.monotonic()
    _, models = require_qdrant()
    BGEM3FlagModel = require_bgem3_model()
    configure_torch_runtime()

    print("Loading datasets...")
    df_cv, df_jd = load_datasets(config.dataset_dir)
    if df_cv.empty or df_jd.empty:
        raise ValueError("Both CV and JD datasets must contain at least one row.")

    print(f"Initializing BGE-M3 model: {BGE_M3_MODEL}")
    model = BGEM3FlagModel(BGE_M3_MODEL, use_fp16=config.use_fp16)

    print("Connecting to Qdrant...")
    client = connect_qdrant(config)
    prepare_collection(client, config, models)

    index_jobs(client, model, df_jd, config, models)
    matches = match_cvs(client, model, df_cv, config, models)

    run_id = None
    if config.store_db:
        expected_top_k = min(config.top_k, len(df_jd))
        matches_by_cv: dict[int, list[dict[str, Any]]] = {}
        for match in matches:
            matches_by_cv.setdefault(int(match["cv_id"]), []).append(match)

        def ranked_matches(cv_idx: int) -> list[dict[str, Any]]:
            return [
                {
                    "jd_idx": int(match["jd_id"]),
                    "rank": int(match["rank"]),
                    "score": float(match["score"]),
                    "meta": {
                        "source": "hybrid_viranker",
                        "score_type": match.get("score_type"),
                        "retrieval_rank": match.get("retrieval_rank"),
                        "retrieval_score": match.get("retrieval_score"),
                        "reranker_model": VIRANKER_MODEL,
                        "reranker_max_length": config.reranker_max_length,
                        "reranker_query_max_length": config.reranker_query_max_length,
                    },
                }
                for match in sorted(matches_by_cv.get(cv_idx, []), key=lambda item: item["rank"])
            ]

        conn = open_store()
        try:
            run_id = store_match_run(
                conn,
                RunInfo(
                    run_name="virankertesting4.0",
                    algorithm="qdrant_hybrid_viranker",
                    model_name=f"{BGE_M3_MODEL} -> {VIRANKER_MODEL}",
                    params={
                        "top_k": config.top_k,
                        "batch_size": config.batch_size,
                        "upsert_batch_size": config.upsert_batch_size,
                        "reranker_batch_size": config.reranker_batch_size,
                        "reranker_max_length": config.reranker_max_length,
                        "reranker_query_max_length": config.reranker_query_max_length,
                        "reranker_normalize": config.reranker_normalize,
                        "prefetch_multiplier": config.prefetch_multiplier,
                        "qdrant_timeout": config.qdrant_timeout,
                        "qdrant_upsert_retries": config.qdrant_upsert_retries,
                        "collection": config.collection_name,
                        "retriever_model": BGE_M3_MODEL,
                        "reranker_model": VIRANKER_MODEL,
                    },
                    dataset_meta={
                        "dataset_dir": str(config.dataset_dir),
                        "cv_rows": len(df_cv),
                        "jd_rows": len(df_jd),
                    },
                ),
                df_cv,
                df_jd,
                get_cv_text,
                get_jd_text,
                ranked_matches,
                top_k=expected_top_k,
                started_monotonic=started_monotonic,
            )
        finally:
            conn.close()

    result_path = write_match_summary(matches, config, run_id)

    return {
        "cv_count": len(df_cv),
        "jd_count": len(df_jd),
        "match_count": len(matches),
        "run_id": run_id,
        "result_path": result_path,
        "elapsed_seconds": round(time.time() - started_wall, 3),
    }


def main() -> None:
    config = parse_args()
    result = run_bgem3_qdrant(config)
    print(
        f"Matched {result['cv_count']} CVs against {result['jd_count']} JDs "
        f"({result['match_count']} rows) in {result['elapsed_seconds']}s. "
        f"Run ID: {result['run_id']}"
    )
    if result["result_path"]:
        print(f"Result CSV: {result['result_path']}")


if __name__ == "__main__":
    main()
