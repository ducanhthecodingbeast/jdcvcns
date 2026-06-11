# Shared data utilities

This folder is intentionally shared by `1.0`, `2.0`, and `3.0`.

Purpose:
- `data_preprocessing.py`: download/normalize raw Kaggle data into `../Data/cv.csv` and `../Data/jd.csv`.
- `mockcv.py`: generate or validate `../Data/mockcv.csv`.
- `localllm.py`: local Ollama client and prompt builders for mock CV generation.

Run preprocessing:

```bash
cd Dataset
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_api_key
python data_preprocessing.py
```

This project does not use `~/.kaggle/kaggle.json`; Kaggle credentials are read only from `KAGGLE_USERNAME` and `KAGGLE_KEY`.

Or with Docker Compose:

```bash
cd Dataset
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_api_key
../scripts/compose run --rm all
```

Generate/check mock CVs:

```bash
cd Dataset
source .venv/bin/activate
python -m mockcv --check-only
python -m mockcv --force
```

Or with Docker Compose:

```bash
cd Dataset
../scripts/compose run --rm preprocess
../scripts/compose run --rm mockcv
MOCKCV_ARGS=--force ../scripts/compose run --rm mockcv
```

Follow-up agents should keep only data/mock/preprocessing concerns here. Version-specific test logic belongs inside `1.0`, `2.0`, or `3.0`.

Docker Compose note:

Use Docker Compose V2:

```bash
../scripts/compose version
```

Do not use the old Python `docker-compose` package. If you see `Not supported URL scheme http+docker`, remove or bypass the old Python Compose package and run commands with `../scripts/compose` instead of `docker-compose`.

If your Python environment has old Docker packages, use this folder's virtual environment and upgrade to the latest packages without pinning versions:

```bash
source .venv/bin/activate
python -m pip uninstall docker-compose
python -m pip install --upgrade docker requests
```
