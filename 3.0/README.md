# 3.0 independent test project

Goal: run `bgmewdranttesting3.0.py` as a complete version-3 test project with local PostgreSQL result storage and Qdrant retrieval setup.

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
cd 3.0
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
docker compose up -d
python bgmewdranttesting3.0.py
```

Run fully in Docker Compose:

```bash
cd 3.0
docker compose run --rm test
```

Core files:
- `bgmewdranttesting3.0.py`: BGE-M3 + Qdrant test entrypoint.
- `pipeline.py`: version-local dataset/text helpers.
- `testingresult.py`, `demoAPI/`: version-local PostgreSQL result storage.

Follow-up agents can add API/frontend wrappers here, but should keep the version-3 runtime independent from the other folders.
