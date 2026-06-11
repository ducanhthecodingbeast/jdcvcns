# 1.0 independent test project

Goal: run `dotpdtesting1.0.py` as a complete version-1 test project without importing code from `JobCV-matching_CNS` or from `2.0`/`3.0`.

Shared exceptions:
- Data files are read from `../Data`: `jd.csv`, `cv.csv`, `mockcv.csv`.
- Mock CV generation and data preprocessing live in `../Dataset`.

Prepare shared data from the repo root first:

```bash
cd Dataset
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_api_key
python data_preprocessing.py
python -m mockcv --force
cd ..
```

Run:

```bash
cd 1.0
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
docker compose up -d
python dotpdtesting1.0.py
```

Run fully in Docker Compose:

```bash
cd 1.0
docker compose run --rm test
```

Core files:
- `dotpdtesting1.0.py`: version-1 test entrypoint.
- `pipeline.py`: version-local dataset/text/embedding helpers.
- `testingresult.py`, `demoAPI/`: version-local PostgreSQL result storage.

Follow-up agents can extend Docker/API integration here, but should not move shared runtime code back outside this folder.
