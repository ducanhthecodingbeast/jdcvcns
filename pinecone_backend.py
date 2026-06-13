from __future__ import annotations

import math
import re
import sys
import time
from typing import Any


def normalize_pinecone_index_name(value: str) -> str:
    name = re.sub(r"[^a-z0-9-]+", "-", value.lower())
    name = re.sub(r"-+", "-", name).strip("-")
    if not name:
        name = "jdcvcns-index"
    return name[:45].strip("-") or "jdcvcns-index"


def require_pinecone():
    try:
        from pinecone import Pinecone as PineconeClient
        from pinecone import ServerlessSpec
    except ImportError as exc:
        raise RuntimeError("Missing dependency: install pinecone to run with VECTOR_BACKEND=pinecone.") from exc
    return PineconeClient, ServerlessSpec


def to_float_list(vector: Any) -> list[float]:
    if hasattr(vector, "tolist"):
        vector = vector.tolist()
    return [float(value) for value in vector]


def normalize_dense_vector(vector: Any) -> list[float]:
    values = to_float_list(vector)
    norm = math.sqrt(sum(value * value for value in values))
    if norm <= 0:
        return values
    return [value / norm for value in values]


def sparse_indices_values(sparse_data: dict[Any, Any]) -> tuple[list[int], list[float]]:
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

    return indices, values


def pinecone_sparse_payload(sparse_data: dict[Any, Any]) -> dict[str, list[int] | list[float]]:
    indices, values = sparse_indices_values(sparse_data)
    return {"indices": indices, "values": values}


def pinecone_hybrid_vectors(
    dense_vector: Any,
    sparse_vector: dict[str, list[int] | list[float]],
    alpha: float,
) -> tuple[list[float], dict[str, list[int] | list[float]]]:
    alpha = min(1.0, max(0.0, alpha))
    dense_values = [value * alpha for value in normalize_dense_vector(dense_vector)]

    sparse_indices = [int(value) for value in sparse_vector.get("indices", [])]
    sparse_values = [float(value) * (1.0 - alpha) for value in sparse_vector.get("values", [])]
    filtered_sparse_indices: list[int] = []
    filtered_sparse_values: list[float] = []
    for index, value in zip(sparse_indices, sparse_values):
        if value > 0:
            filtered_sparse_indices.append(index)
            filtered_sparse_values.append(value)

    return dense_values, {"indices": filtered_sparse_indices, "values": filtered_sparse_values}


def pinecone_metadata(payload: dict[str, Any]) -> dict[str, bool | int | float | str | list[str]]:
    metadata: dict[str, bool | int | float | str | list[str]] = {}
    for key, value in payload.items():
        if value is None:
            continue
        if isinstance(value, bool | int | float | str):
            metadata[str(key)] = value
        elif isinstance(value, list) and all(isinstance(item, str) for item in value):
            metadata[str(key)] = value
        else:
            metadata[str(key)] = str(value)
    return metadata


def pinecone_has_index(pc, index_name: str) -> bool:
    if hasattr(pc, "has_index"):
        return bool(pc.has_index(index_name))

    indexes = pc.list_indexes()
    if hasattr(indexes, "names"):
        return index_name in indexes.names()
    for item in indexes:
        if getattr(item, "name", None) == index_name:
            return True
        if isinstance(item, dict) and item.get("name") == index_name:
            return True
    return False


def get_model_value(model: Any, key: str, default: Any = None) -> Any:
    if isinstance(model, dict):
        return model.get(key, default)
    return getattr(model, key, default)


def pinecone_index_ready(description: Any) -> bool:
    status = get_model_value(description, "status", {})
    return bool(get_model_value(status, "ready", False))


def wait_for_pinecone_index(pc, index_name: str, timeout: float):
    deadline = time.monotonic() + timeout
    while True:
        description = pc.describe_index(index_name)
        if pinecone_index_ready(description):
            return description
        if time.monotonic() >= deadline:
            raise RuntimeError(f"Pinecone index {index_name} was not ready after {timeout:.0f}s.")
        time.sleep(5)


def validate_pinecone_index(description: Any, index_name: str, dimension: int) -> None:
    metric = get_model_value(description, "metric")
    if metric and str(metric).lower() != "dotproduct":
        raise RuntimeError(
            f"Pinecone index {index_name} uses metric={metric!r}; hybrid dense+sparse search requires dotproduct."
        )

    existing_dimension = get_model_value(description, "dimension")
    if existing_dimension is not None and int(existing_dimension) != dimension:
        raise RuntimeError(
            f"Pinecone index {index_name} uses dimension={existing_dimension}; expected {dimension}."
        )


def prepare_pinecone_index(
    *,
    api_key: str | None,
    index_name: str,
    host: str | None,
    index_host: str | None,
    cloud: str,
    region: str,
    recreate_index: bool,
    timeout: float,
    dimension: int = 1024,
):
    Pinecone, ServerlessSpec = require_pinecone()
    api_key = api_key or "pclocal"
    kwargs = {"api_key": api_key}
    if host:
        kwargs["host"] = host
    pc = Pinecone(**kwargs)

    if index_host:
        return pc.Index(host=index_host)

    if recreate_index and pinecone_has_index(pc, index_name):
        pc.delete_index(index_name)
        deadline = time.monotonic() + timeout
        while pinecone_has_index(pc, index_name):
            if time.monotonic() >= deadline:
                raise RuntimeError(f"Pinecone index {index_name} was not deleted after {timeout:.0f}s.")
            time.sleep(1)

    if not pinecone_has_index(pc, index_name):
        pc.create_index(
            name=index_name,
            vector_type="dense",
            dimension=dimension,
            metric="dotproduct",
            spec=ServerlessSpec(cloud=cloud, region=region),
            deletion_protection="disabled",
        )
    description = wait_for_pinecone_index(pc, index_name, timeout)
    validate_pinecone_index(description, index_name, dimension)
    index = pc.Index(index_name)

    return index


def pinecone_upsert_with_retry(
    index,
    records: list[dict[str, Any]],
    namespace: str | None,
    retries: int,
) -> None:
    for attempt in range(1, retries + 1):
        try:
            kwargs: dict[str, Any] = {"vectors": records}
            if namespace:
                kwargs["namespace"] = namespace
            index.upsert(**kwargs)
            return
        except Exception as exc:
            if attempt >= retries:
                raise
            delay_seconds = min(2**attempt, 30)
            print(
                f"Pinecone upsert failed on attempt {attempt}/{retries}: "
                f"{type(exc).__name__}: {exc}. Retrying in {delay_seconds}s...",
                file=sys.stderr,
            )
            time.sleep(delay_seconds)


def pinecone_query(
    index,
    *,
    namespace: str | None,
    top_k: int,
    dense_vector: list[float],
    sparse_vector: dict[str, list[int] | list[float]],
    include_metadata: bool = False,
):
    kwargs: dict[str, Any] = {
        "top_k": top_k,
        "vector": dense_vector,
        "include_metadata": include_metadata,
    }
    if namespace:
        kwargs["namespace"] = namespace
    if sparse_vector.get("indices"):
        kwargs["sparse_vector"] = sparse_vector
    return index.query(**kwargs)


def pinecone_matches(response: Any) -> list[Any]:
    if isinstance(response, dict):
        return list(response.get("matches") or [])
    return list(getattr(response, "matches", None) or [])


def pinecone_match_id(match: Any) -> str:
    value = match.get("id") if isinstance(match, dict) else getattr(match, "id", None)
    return str(value)


def pinecone_match_score(match: Any) -> float:
    value = match.get("score") if isinstance(match, dict) else getattr(match, "score", 0.0)
    return float(value)


def pinecone_match_metadata(match: Any) -> dict[str, Any]:
    value = match.get("metadata") if isinstance(match, dict) else getattr(match, "metadata", {})
    return dict(value or {})
