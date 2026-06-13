from __future__ import annotations

# ruff: noqa: E402

import argparse
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
REPO_ROOT = PROJECT_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".cache" / "huggingface"))
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(PROJECT_ROOT / ".cache" / "sentence-transformers"))

import numpy as np
import pandas as pd
from tqdm import tqdm

from pipeline import DATASET_DIR, RESULTS_DIR, get_cv_text, get_jd_text, load_datasets
from pinecone_backend import (
    normalize_dense_vector,
    normalize_pinecone_index_name,
    pinecone_hybrid_vectors,
    pinecone_match_id,
    pinecone_match_metadata,
    pinecone_match_score,
    pinecone_matches,
    pinecone_metadata,
    pinecone_query,
    pinecone_sparse_payload,
    pinecone_upsert_with_retry,
    prepare_pinecone_index,
)
from testingresult import RunInfo, env_flag, open_store, store_match_run


BGE_M3_MODEL = os.environ.get("BGE_M3_MODEL", "BAAI/bge-m3")
DEFAULT_COLLECTION = "jd_bgem3_collection"


@dataclass(frozen=True)
class Config:
    dataset_dir: Path
    results_dir: Path
    vector_backend: str
    collection_name: str
    top_k: int
    batch_size: int
    upsert_batch_size: int
    prefetch_multiplier: int
    recreate_collection: bool
    use_fp16: bool
    qdrant_host: str
    qdrant_port: int
    qdrant_url: str | None
    qdrant_path: str | None
    qdrant_timeout: float
    qdrant_upsert_retries: int
    pinecone_index: str
    pinecone_host: str
    pinecone_index_host: str | None
    pinecone_api_key: str | None
    pinecone_cloud: str
    pinecone_region: str
    pinecone_namespace: str | None
    pinecone_alpha: float
    store_db: bool


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="Run BGE-M3 hybrid CV/JD matching with Qdrant dense, sparse, and ColBERT vectors."
    )
    parser.add_argument("--dataset-dir", default=str(DATASET_DIR))
    parser.add_argument("--results-dir", default=str(RESULTS_DIR))
    parser.add_argument(
        "--vector-backend",
        choices=("qdrant", "pinecone"),
        default=os.environ.get("VECTOR_BACKEND", "qdrant").lower(),
    )
    parser.add_argument("--collection", default=os.environ.get("QDRANT_COLLECTION", DEFAULT_COLLECTION))
    parser.add_argument("--top-k", type=int, default=int(os.environ.get("TOP_K", "5")))
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("BGE_BATCH_SIZE", "16")))
    parser.add_argument(
        "--upsert-batch-size",
        type=int,
        default=int(os.environ.get("QDRANT_UPSERT_BATCH_SIZE", "2")),
    )
    parser.add_argument("--prefetch-multiplier", type=int, default=int(os.environ.get("PREFETCH_MULTIPLIER", "4")))
    parser.add_argument("--keep-collection", action="store_true")
    parser.add_argument("--no-fp16", action="store_true")
    parser.add_argument("--qdrant-host", default=os.environ.get("QDRANT_HOST", "localhost"))
    parser.add_argument("--qdrant-port", type=int, default=int(os.environ.get("QDRANT_PORT", "16330")))
    parser.add_argument("--qdrant-url", default=os.environ.get("QDRANT_URL"))
    parser.add_argument("--qdrant-path", default=os.environ.get("QDRANT_PATH"))
    parser.add_argument("--qdrant-timeout", type=float, default=float(os.environ.get("QDRANT_TIMEOUT", "120")))
    parser.add_argument(
        "--qdrant-upsert-retries",
        type=int,
        default=int(os.environ.get("QDRANT_UPSERT_RETRIES", "3")),
    )
    parser.add_argument(
        "--pinecone-index",
        default=os.environ.get("PINECONE_INDEX", normalize_pinecone_index_name(DEFAULT_COLLECTION)),
    )
    parser.add_argument("--pinecone-host", default=os.environ.get("PINECONE_HOST", "http://localhost:5080"))
    parser.add_argument("--pinecone-index-host", default=os.environ.get("PINECONE_INDEX_HOST"))
    parser.add_argument("--pinecone-api-key", default=os.environ.get("PINECONE_API_KEY"))
    parser.add_argument("--pinecone-cloud", default=os.environ.get("PINECONE_CLOUD", "aws"))
    parser.add_argument("--pinecone-region", default=os.environ.get("PINECONE_REGION", "us-east-1"))
    parser.add_argument("--pinecone-namespace", default=os.environ.get("PINECONE_NAMESPACE", ""))
    parser.add_argument("--pinecone-alpha", type=float, default=float(os.environ.get("PINECONE_ALPHA", "0.75")))
    args = parser.parse_args()

    return Config(
        dataset_dir=Path(args.dataset_dir),
        results_dir=Path(args.results_dir),
        vector_backend=args.vector_backend,
        collection_name=args.collection,
        top_k=max(1, args.top_k),
        batch_size=max(1, args.batch_size),
        upsert_batch_size=max(1, args.upsert_batch_size),
        prefetch_multiplier=max(1, args.prefetch_multiplier),
        recreate_collection=not args.keep_collection,
        use_fp16=not args.no_fp16,
        qdrant_host=args.qdrant_host,
        qdrant_port=args.qdrant_port,
        qdrant_url=args.qdrant_url,
        qdrant_path=args.qdrant_path,
        qdrant_timeout=max(1.0, args.qdrant_timeout),
        qdrant_upsert_retries=max(1, args.qdrant_upsert_retries),
        pinecone_index=normalize_pinecone_index_name(args.pinecone_index),
        pinecone_host=args.pinecone_host,
        pinecone_index_host=args.pinecone_index_host,
        pinecone_api_key=args.pinecone_api_key,
        pinecone_cloud=args.pinecone_cloud,
        pinecone_region=args.pinecone_region,
        pinecone_namespace=args.pinecone_namespace or None,
        pinecone_alpha=min(1.0, max(0.0, args.pinecone_alpha)),
        store_db=env_flag("STORE_DB", True),
    )


def require_bgem3_model():
    try:
        from FlagEmbedding import BGEM3FlagModel
    except ImportError as exc:
        raise RuntimeError("Missing dependency: install FlagEmbedding to run BGE-M3 matching.") from exc
    return BGEM3FlagModel


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
            "colbert": models.VectorParams(
                size=1024,
                distance=models.Distance.COSINE,
                multivector_config=models.MultiVectorConfig(
                    comparator=models.MultiVectorComparator.MAX_SIM,
                ),
            ),
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
        return_colbert_vecs=True,
    )


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
                        "colbert": to_list(output["colbert_vecs"][offset]),
                    },
                )
            )

        for _, point_batch in chunks(points, config.upsert_batch_size):
            upsert_points_with_retry(client, config, point_batch)
        progress.update(len(text_batch))

    progress.close()


def match_cvs(client, model, df_cv: pd.DataFrame, config: Config, models) -> list[dict[str, Any]]:
    cv_texts = [get_cv_text(row) for _, row in df_cv.iterrows()]
    matches: list[dict[str, Any]] = []
    prefetch_limit = max(config.top_k, config.top_k * config.prefetch_multiplier)
    progress = tqdm(total=len(cv_texts), desc="Matching CVs")

    for start, text_batch in chunks(cv_texts, config.batch_size):
        output = encode_batch(model, text_batch, config.batch_size)

        for offset, _ in enumerate(text_batch):
            cv_idx = start + offset
            cv_row = df_cv.iloc[cv_idx]
            sparse = create_sparse_vector(output["lexical_weights"][offset], models)

            results = client.query_points(
                collection_name=config.collection_name,
                prefetch=[
                    models.Prefetch(query=sparse, using="sparse", limit=prefetch_limit),
                    models.Prefetch(query=to_list(output["dense_vecs"][offset]), using="dense", limit=prefetch_limit),
                ],
                query=to_list(output["colbert_vecs"][offset]),
                using="colbert",
                with_payload=True,
                limit=config.top_k,
            )

            cv_title = first_value(cv_row, "Tên ứng viên", "Vị trí ứng tuyển", "User Name", "Desired Job")
            for rank, result in enumerate(results.points, start=1):
                jd_payload = result.payload or {}
                matches.append(
                    {
                        "cv_id": int(cv_idx),
                        "cv_title": cv_title,
                        "jd_id": int(result.id),
                        "jd_title": first_payload_value(jd_payload, "Vị trí cần tuyển", "Job Title"),
                        "rank": rank,
                        "score": float(result.score),
                        "cv_payload": row_payload(cv_row),
                        "jd_payload": {key: value for key, value in jd_payload.items() if key != "_text"},
                    }
                )

        progress.update(len(text_batch))

    progress.close()
    return matches


def index_jobs_pinecone(index, model, df_jd: pd.DataFrame, config: Config) -> None:
    jd_texts = [get_jd_text(row) for _, row in df_jd.iterrows()]
    progress = tqdm(total=len(jd_texts), desc="Indexing JDs")

    for start, text_batch in chunks(jd_texts, config.batch_size):
        output = encode_batch(model, text_batch, config.batch_size)
        records = []

        for offset, text in enumerate(text_batch):
            jd_idx = start + offset
            jd_row = df_jd.iloc[jd_idx]
            records.append(
                {
                    "id": str(jd_idx),
                    "values": normalize_dense_vector(output["dense_vecs"][offset]),
                    "sparse_values": pinecone_sparse_payload(output["lexical_weights"][offset]),
                    "metadata": pinecone_metadata(
                        {
                            **row_payload(jd_row),
                            "_text": text,
                        }
                    ),
                }
            )

        for _, record_batch in chunks(records, config.upsert_batch_size):
            pinecone_upsert_with_retry(
                index,
                record_batch,
                namespace=config.pinecone_namespace,
                retries=config.qdrant_upsert_retries,
            )
        progress.update(len(text_batch))

    progress.close()


def match_cvs_pinecone(index, model, df_cv: pd.DataFrame, config: Config) -> list[dict[str, Any]]:
    cv_texts = [get_cv_text(row) for _, row in df_cv.iterrows()]
    matches: list[dict[str, Any]] = []
    progress = tqdm(total=len(cv_texts), desc="Matching CVs")

    for start, text_batch in chunks(cv_texts, config.batch_size):
        output = encode_batch(model, text_batch, config.batch_size)

        for offset, _ in enumerate(text_batch):
            cv_idx = start + offset
            cv_row = df_cv.iloc[cv_idx]
            dense, sparse = pinecone_hybrid_vectors(
                output["dense_vecs"][offset],
                pinecone_sparse_payload(output["lexical_weights"][offset]),
                config.pinecone_alpha,
            )

            response = pinecone_query(
                index,
                namespace=config.pinecone_namespace,
                top_k=config.top_k,
                dense_vector=dense,
                sparse_vector=sparse,
                include_metadata=True,
            )

            cv_title = first_value(cv_row, "Tên ứng viên", "Vị trí ứng tuyển", "User Name", "Desired Job")
            for rank, result in enumerate(pinecone_matches(response), start=1):
                jd_payload = pinecone_match_metadata(result)
                matches.append(
                    {
                        "cv_id": int(cv_idx),
                        "cv_title": cv_title,
                        "jd_id": int(pinecone_match_id(result)),
                        "jd_title": first_payload_value(jd_payload, "Vị trí cần tuyển", "Job Title"),
                        "rank": rank,
                        "score": pinecone_match_score(result),
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

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Initializing BGE-M3 model: {BGE_M3_MODEL} on {device.upper()}")
    model = BGEM3FlagModel(BGE_M3_MODEL, use_fp16=config.use_fp16, device=device)

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
                    "meta": {"source": "qdrant_bgem3"},
                }
                for match in sorted(matches_by_cv.get(cv_idx, []), key=lambda item: item["rank"])
            ]

        conn = open_store()
        try:
            run_id = store_match_run(
                conn,
                RunInfo(
                    run_name="bgmewdranttesting3.0",
                    algorithm="qdrant_bgem3_hybrid",
                    model_name=BGE_M3_MODEL,
                    params={
                        "top_k": config.top_k,
                        "batch_size": config.batch_size,
                        "upsert_batch_size": config.upsert_batch_size,
                        "prefetch_multiplier": config.prefetch_multiplier,
                        "qdrant_timeout": config.qdrant_timeout,
                        "qdrant_upsert_retries": config.qdrant_upsert_retries,
                        "collection": config.collection_name,
                    },
                    dataset_meta={"dataset_dir": str(config.dataset_dir)},
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

    return {
        "cv_count": len(df_cv),
        "jd_count": len(df_jd),
        "match_count": len(matches),
        "run_id": run_id,
        "elapsed_seconds": round(time.time() - started_wall, 3),
    }


def run_bgem3_pinecone(config: Config) -> dict[str, Any]:
    started_wall = time.time()
    started_monotonic = time.monotonic()
    BGEM3FlagModel = require_bgem3_model()
    configure_torch_runtime()

    print("Loading datasets...")
    df_cv, df_jd = load_datasets(config.dataset_dir)
    if df_cv.empty or df_jd.empty:
        raise ValueError("Both CV and JD datasets must contain at least one row.")

    print(f"Initializing BGE-M3 model: {BGE_M3_MODEL}")
    model = BGEM3FlagModel(BGE_M3_MODEL, use_fp16=config.use_fp16)

    print(f"Connecting to Pinecone Local: {config.pinecone_host}")
    index = prepare_pinecone_index(
        api_key=config.pinecone_api_key,
        index_name=config.pinecone_index,
        host=config.pinecone_host,
        index_host=config.pinecone_index_host,
        cloud=config.pinecone_cloud,
        region=config.pinecone_region,
        recreate_index=config.recreate_collection,
        timeout=config.qdrant_timeout,
    )

    index_jobs_pinecone(index, model, df_jd, config)
    matches = match_cvs_pinecone(index, model, df_cv, config)

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
                    "meta": {"source": "pinecone_bgem3_hybrid"},
                }
                for match in sorted(matches_by_cv.get(cv_idx, []), key=lambda item: item["rank"])
            ]

        conn = open_store()
        try:
            run_id = store_match_run(
                conn,
                RunInfo(
                    run_name="bgmewdranttesting3.0",
                    algorithm="pinecone_bgem3_hybrid",
                    model_name=BGE_M3_MODEL,
                    params={
                        "top_k": config.top_k,
                        "batch_size": config.batch_size,
                        "upsert_batch_size": config.upsert_batch_size,
                        "prefetch_multiplier": config.prefetch_multiplier,
                        "pinecone_index": config.pinecone_index,
                        "pinecone_namespace": config.pinecone_namespace,
                        "pinecone_alpha": config.pinecone_alpha,
                    },
                    dataset_meta={"dataset_dir": str(config.dataset_dir)},
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

    return {
        "cv_count": len(df_cv),
        "jd_count": len(df_jd),
        "match_count": len(matches),
        "run_id": run_id,
        "elapsed_seconds": round(time.time() - started_wall, 3),
    }


def main() -> None:
    config = parse_args()
    if config.vector_backend == "pinecone":
        result = run_bgem3_pinecone(config)
    else:
        result = run_bgem3_qdrant(config)
    print(
        f"Matched {result['cv_count']} CVs against {result['jd_count']} JDs "
        f"({result['match_count']} rows) in {result['elapsed_seconds']}s. "
        f"Run ID: {result['run_id']}"
    )


if __name__ == "__main__":
    main()
