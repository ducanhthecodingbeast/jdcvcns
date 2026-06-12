# 4.0 independent test project

Goal: run `virankertesting4.0.py` as a complete version-4 test project with local PostgreSQL result storage, Qdrant retrieval, and ViRanker reranking setup.

This version follows the ViRanker design:
- BGE-M3 encodes JDs/CVs for first-stage hybrid recall.
- Qdrant combines dense and sparse recall with RRF.
- `namdp-ptit/ViRanker` reranks the recalled JD candidates as a Vietnamese cross-encoder.
- ViRanker defaults to `max_length=1024`, matching the paper's training/evaluation setting.
- Reranker scores are sigmoid-normalized by default for easier result comparison.

References:
- Paper: https://arxiv.org/pdf/2509.09131
- Model: https://huggingface.co/namdp-ptit/ViRanker

Shared exceptions:
- Data files are read from `../Data`: `jd.csv`, `cv.csv`, `mockcv.csv`.
- Mock CV generation and data preprocessing live in `../Dataset`.

Prepare shared data from the repo root first:

```bash
cd Dataset
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_api_key
./run.sh
cd ..
```

Run:

```bash
cd 4.0
./run.sh
```

Pass runner arguments after the launcher flags:

```bash
./run.sh -- --top-k 20 --prefetch-multiplier 6 --reranker-max-length 1024
./run.sh -- --raw-reranker-scores
```

`run.sh` defaults to safer Qdrant write settings:

```bash
QDRANT_UPSERT_BATCH_SIZE=1 QDRANT_TIMEOUT=300
```

Run fully in Docker Compose:

```bash
cd 4.0
../scripts/compose run --rm test
```

Core files:
- `virankertesting4.0.py`: BGE-M3 + Qdrant + ViRanker test entrypoint.
- `pipeline.py`: version-local dataset/text helpers.
- `testingresult.py`, `demoAPI/`: version-local PostgreSQL result storage.

Key environment variables:
- `BGE_M3_MODEL`: first-stage retriever, default `BAAI/bge-m3`.
- `VIRANKER_MODEL`: cross-encoder reranker, default `namdp-ptit/ViRanker`.
- `TOP_K`: final matches stored per CV.
- `PREFETCH_MULTIPLIER`: first-stage candidate pool size multiplier before reranking.
- `VIRANKER_MAX_LENGTH`: reranker pair max length, default `1024`.
- `VIRANKER_QUERY_MAX_LENGTH`: token budget reserved for the CV side, default `384`; set `0` to disable.
- `VIRANKER_NORMALIZE`: use sigmoid-normalized reranker scores when `1`.
- `WRITE_RESULTS`: write a lightweight CSV summary under `TestingResults/` when `1`.

Follow-up agents can add API/frontend wrappers here, but should keep the version-4 runtime independent from the other folders.
