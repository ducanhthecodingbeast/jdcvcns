# 6.0 independent test project

Goal: compare three CV/JD matching baselines on the same dataset snapshots:

- `jobberttesting6.0.py`: `TechWolf/JobBERT-v2` embeddings with cosine similarity.
- `jobberttesting6.1.py`: `TechWolf/JobBERT-v2` embeddings with dot product.
- `bm25testing6.2.py`: BM25 lexical ranking, using the JobBERT tokenizer by default.

Shared data is read from `../Data`. The TalentCLEF server test expects `mockcv.small.csv` for CVs and `jd.csv` for JDs.

Each test matches every mock CV against all rows in `jd.csv`, stores the top 10 JDs per CV in Postgres, and writes `TestingResults/file.json` for `5matching.html`.

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

Open the static viewer after the server run writes JSON:

```bash
5matching.html
```

Run in Docker Compose:

```bash
cd 6.0
../scripts/compose run --rm test
```
