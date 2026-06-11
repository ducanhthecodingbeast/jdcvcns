# 3.0 independent test project

Goal: run `bgmewdranttesting3.0.py` as a complete version-3 test project with local PostgreSQL result storage and Qdrant retrieval setup.

Shared exceptions:
- Data files are read from `../Data`: `jd.csv`, `cv.csv`, `mockcv.csv`.
- Mock CV generation and data preprocessing live in `../Dataset`.

Prepare shared data from the repo root first:

```bash
python3 Dataset/data_preprocessing.py
python3 -m Dataset.mockcv --force
```

Run:

```bash
cd 3.0
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
docker compose up -d
python3 bgmewdranttesting3.0.py
```

Core files:
- `bgmewdranttesting3.0.py`: BGE-M3 + Qdrant test entrypoint.
- `pipeline.py`: version-local dataset/text helpers.
- `testingresult.py`, `demoAPI/`: version-local PostgreSQL result storage.

Follow-up agents can add API/frontend wrappers here, but should keep the version-3 runtime independent from the other folders.
