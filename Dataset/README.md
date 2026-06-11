# Shared data utilities

This folder is intentionally shared by `1.0`, `2.0`, and `3.0`.

Purpose:
- `data_preprocessing.py`: download/normalize raw Kaggle data into `../Data/cv.csv` and `../Data/jd.csv`.
- `mockcv.py`: generate or validate `../Data/mockcv.csv`.
- `localllm.py`: local Ollama client and prompt builders for mock CV generation.

Run preprocessing:

```bash
export KAGGLE_USERNAME=thecapybaracoder
export KAGGLE_API_TOKEN=KGAT_040b5e85a3503e30560d5a05ab5039b3
python3 Dataset/data_preprocessing.py
```

This project does not use `~/.kaggle/kaggle.json`; Kaggle credentials are read only from `KAGGLE_USERNAME` and `KAGGLE_KEY`.

Or with Docker Compose:

```bash
cd Dataset
docker compose run --rm preprocess
```

Generate/check mock CVs:

```bash
python3 -m Dataset.mockcv --check-only
python3 -m Dataset.mockcv --force
```

Or with Docker Compose:

```bash
cd Dataset
docker compose run --rm mockcv
MOCKCV_ARGS=--force docker compose run --rm mockcv
```

Follow-up agents should keep only data/mock/preprocessing concerns here. Version-specific test logic belongs inside `1.0`, `2.0`, or `3.0`.

Docker Compose note:

Use Docker Compose V2:

```bash
docker compose version
```

Do not use the old Python `docker-compose` package. If you see `Not supported URL scheme http+docker`, remove or bypass the old Python Compose package and run commands with `docker compose` instead of `docker-compose`.
