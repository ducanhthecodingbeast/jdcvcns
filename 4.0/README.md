# 4.0 independent test project

Goal: run `virankertesting4.0.py` as a complete version-4 test project with local PostgreSQL result storage, Qdrant retrieval, and ViRanker reranking setup.

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
cd 4.0
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
../scripts/compose up -d
python virankertesting4.0.py
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

Follow-up agents can add API/frontend wrappers here, but should keep the version-4 runtime independent from the other folders.
