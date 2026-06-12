# 6.0 independent test project

Goal: compare three CV/JD matching baselines on the same dataset snapshots:

- `jobberttesting6.0.py`: `TechWolf/JobBERT-v2` embeddings with cosine similarity.
- `jobberttesting6.1.py`: `TechWolf/JobBERT-v2` embeddings with dot product.
- `bm25testing6.2.py`: BM25 lexical ranking, using the JobBERT tokenizer by default.

Shared data is read from `../Data`. This project expects `jd.csv` or `JOB_DATA_FINAL.csv`, plus one CV source such as `mockcv.csv`, `cv.csv`, or `USER_DATA_FINAL.csv`.

Run the safest default variant first:

```bash
cd 6.0
./run.sh
```

If your system `python3` is too new for PyTorch wheels, use a supported Python:

```bash
PYTHON_BIN=python3.11 ./run.sh
```

Run one variant:

```bash
./run.sh 6.0 -- --top-k 20
./run.sh 6.1 -- --cv-limit 100 --jd-limit 100
./run.sh 6.2 -- --regex-tokenizer
./run.sh all
```

Run in Docker Compose:

```bash
cd 6.0
../scripts/compose run --rm test
```
